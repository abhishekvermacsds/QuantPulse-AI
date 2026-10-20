"""Celery background tasks for model training, news classification, and data syncing."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from quantpulse.core.config import DEFAULT_SAMPLE_CSV, PROJECT_ROOT
from quantpulse.data.loader import MarketDataLoader
from quantpulse.db.models import (
    AssetPrice,
    SentimentRecord,
    VolatilityForecastRecord,
)
from quantpulse.db.session import SessionLocal
from quantpulse.ml.baseline import HistoricalVolatilityBaseline
from quantpulse.ml.dataset import (
    RobustFeatureScaler,
    VolatilityFeaturePipeline,
    VolatilitySequenceDataset,
    create_chronological_splits,
)
from quantpulse.ml.evaluator import calculate_regression_metrics
from quantpulse.ml.model import LSTMVolatilityForecaster
from quantpulse.ml.trainer import (
    TrainingConfig,
    VolatilityTrainer,
    save_checkpoint,
    set_seed,
)
from quantpulse.sentiment.finbert import FinBERTSentimentClassifier
from quantpulse.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


def train_volatility_pipeline(
    ticker: str = "SPY",
    epochs: int = 10,
    target_horizon_days: int = 5,
    sequence_length: int = 20,
    model_save_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train PyTorch LSTM volatility forecaster end-to-end and evaluate on test split."""
    set_seed(42)
    loader = MarketDataLoader(data_path=DEFAULT_SAMPLE_CSV)
    raw_df = loader.load()
    symbol = ticker.upper()
    df = raw_df[raw_df["Ticker"] == symbol].copy()
    if df.empty:
        raise ValueError(f"Ticker '{symbol}' not found in market data.")
    df.set_index("Date", inplace=True)

    pipeline = VolatilityFeaturePipeline(forward_horizon=target_horizon_days)
    features, target = pipeline.build_features(df)

    (train_X, train_y), (val_X, val_y), (test_X, test_y) = create_chronological_splits(
        features, target, train_ratio=0.70, val_ratio=0.15
    )

    scaler = RobustFeatureScaler().fit(train_X)
    train_ds = VolatilitySequenceDataset(
        scaler.transform(train_X), train_y.values, sequence_length=sequence_length
    )
    val_ds = VolatilitySequenceDataset(
        scaler.transform(val_X), val_y.values, sequence_length=sequence_length
    )
    test_ds = VolatilitySequenceDataset(
        scaler.transform(test_X), test_y.values, sequence_length=sequence_length
    )

    batch_size = 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        sequence_length=sequence_length,
        hidden_size=16,
        num_layers=1,
        device="cpu",
    )
    model = LSTMVolatilityForecaster(
        input_size=train_X.shape[1],
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
    )

    trainer = VolatilityTrainer(model, config)
    history = trainer.fit(train_loader, val_loader)

    # Evaluate predictions on unseen test split
    model.eval()
    test_preds = []
    test_actuals = []
    with torch.no_grad():
        for bx, by in test_loader:
            preds = model(bx.to("cpu"))
            test_preds.extend(preds.cpu().numpy().tolist())
            test_actuals.extend(by.numpy().tolist())

    y_pred = np.array(test_preds)
    y_true = np.array(test_actuals)
    test_metrics = calculate_regression_metrics(y_true, y_pred)

    # Baseline comparison
    baseline = HistoricalVolatilityBaseline(method="rolling", window_col="vol_21d")
    test_features_aligned = test_X.iloc[sequence_length - 1 : sequence_length - 1 + len(y_true)]
    baseline_preds = baseline.predict(test_features_aligned)
    baseline_metrics = calculate_regression_metrics(y_true, baseline_preds)

    # Save checkpoint
    out_dir = model_save_dir or (PROJECT_ROOT / "models" / "saved")
    out_path = out_dir / f"lstm_volatility_{ticker.lower()}.pt"
    save_checkpoint(model, scaler, config, out_path)

    return {
        "ticker": ticker,
        "epochs_completed": history["epochs_completed"],
        "model_path": str(out_path),
        "test_metrics": {
            "lstm_mae": float(test_metrics["MAE"]),
            "lstm_rmse": float(test_metrics["RMSE"]),
            "baseline_mae": float(baseline_metrics["MAE"]),
            "baseline_rmse": float(baseline_metrics["RMSE"]),
        },
    }


@celery_app.task(name="quantpulse.tasks.train_volatility_model", bind=True)
def task_train_volatility_model(
    self,
    ticker: str = "SPY",
    epochs: int = 10,
    horizon_days: int = 5,
    sequence_length: int = 20,
) -> Dict[str, Any]:
    """Train a PyTorch LSTM volatility forecasting model and persist performance metrics."""
    logger.info("Starting background LSTM training for %s (task_id: %s)", ticker, self.request.id)
    try:
        self.update_state(state="PROGRESS", meta={"status": "Loading market data and training LSTM..."})
    except Exception:
        pass

    result = train_volatility_pipeline(
        ticker=ticker,
        epochs=epochs,
        target_horizon_days=horizon_days,
        sequence_length=sequence_length,
    )

    test_metrics = result.get("test_metrics", {})
    mae = float(test_metrics.get("lstm_mae", 0.0))
    rmse = float(test_metrics.get("lstm_rmse", 0.0))
    baseline_rmse = float(test_metrics.get("baseline_rmse", 0.0))

    # Persist record in database
    db = SessionLocal()
    try:
        forecast_record = VolatilityForecastRecord(
            ticker=ticker,
            model_name="PyTorch_LSTM",
            sequence_length=sequence_length,
            target_horizon_days=horizon_days,
            predicted_volatility=rmse,
            actual_volatility=baseline_rmse,
            baseline_volatility=baseline_rmse,
            mae=mae,
            rmse=rmse,
        )
        db.add(forecast_record)
        db.commit()
        db.refresh(forecast_record)
        db_id = forecast_record.id
    except Exception as exc:
        logger.warning("Failed to persist VolatilityForecastRecord: %s", exc)
        db_id = None
        db.rollback()
    finally:
        db.close()

    return {
        "status": "COMPLETED",
        "ticker": ticker,
        "epochs": epochs,
        "db_record_id": db_id,
        "metrics": test_metrics,
        "model_path": result.get("model_path"),
    }


@celery_app.task(name="quantpulse.tasks.classify_news_batch", bind=True)
def task_classify_news_batch(
    self,
    headlines: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Asynchronously classify a batch of headlines with FinBERT and persist records.
    
    Includes memory management safeguard for 8 GB RAM systems by unloading model weights.
    """
    logger.info("Starting news classification batch of %d items (task_id: %s)", len(headlines), self.request.id)
    try:
        self.update_state(state="PROGRESS", meta={"status": "Classifying headlines with FinBERT..."})
    except Exception:
        pass

    classifier = FinBERTSentimentClassifier()
    texts = [h.get("headline", "") for h in headlines if h.get("headline")]

    results = []
    if texts:
        batch_result = classifier.classify_batch(texts)
        for original, item in zip(headlines, batch_result.items):
            results.append({
                "headline": original.get("headline"),
                "ticker": original.get("ticker"),
                "source": original.get("source", "Background Ingestion"),
                "published_at": original.get("published_at"),
                "predicted_label": item.label,
                "confidence_positive": item.probabilities.get("positive", 0.0),
                "confidence_negative": item.probabilities.get("negative", 0.0),
                "confidence_neutral": item.probabilities.get("neutral", 0.0),
                "compound_score": item.sentiment_score,
            })

    # Crucial 8 GB RAM safeguard: unload FinBERT transformer weights from memory immediately
    classifier.unload_model()

    # Persist in DB
    db = SessionLocal()
    persisted_count = 0
    try:
        for r in results:
            pub_dt = None
            if r.get("published_at"):
                try:
                    pub_dt = pd.to_datetime(r["published_at"]).to_pydatetime()
                except Exception:
                    pub_dt = datetime.utcnow()

            record = SentimentRecord(
                headline=r["headline"],
                ticker=r.get("ticker"),
                source=r.get("source"),
                published_at=pub_dt or datetime.utcnow(),
                predicted_label=r["predicted_label"],
                confidence_positive=r["confidence_positive"],
                confidence_negative=r["confidence_negative"],
                confidence_neutral=r["confidence_neutral"],
                compound_score=r["compound_score"],
            )
            db.add(record)
        db.commit()
        persisted_count = len(results)
    except Exception as exc:
        logger.warning("Failed to persist SentimentRecords: %s", exc)
        db.rollback()
    finally:
        db.close()

    # Summary aggregations
    pos = sum(1 for r in results if r["predicted_label"] == "positive")
    neg = sum(1 for r in results if r["predicted_label"] == "negative")
    neu = sum(1 for r in results if r["predicted_label"] == "neutral")

    return {
        "status": "COMPLETED",
        "total_processed": len(results),
        "persisted_records": persisted_count,
        "sentiment_breakdown": {
            "positive": pos,
            "negative": neg,
            "neutral": neu,
        },
        "results": results,
    }


@celery_app.task(name="quantpulse.tasks.sync_market_data")
def task_sync_market_data(ticker: Optional[str] = None) -> Dict[str, Any]:
    """Sync market data prices from sample CSV into the AssetPrice database table."""
    loader = MarketDataLoader(data_path=DEFAULT_SAMPLE_CSV)
    df = loader.load()
    if ticker:
        df = df[df["Ticker"].str.upper() == ticker.upper()]

    db = SessionLocal()
    inserted = 0
    try:
        existing_keys = {
            (row.ticker, row.date)
            for row in db.query(AssetPrice.ticker, AssetPrice.date).all()
        }

        records_to_insert = []
        for _, row in df.iterrows():
            row_date = row["Date"].date() if hasattr(row["Date"], "date") else row["Date"]
            if (row["Ticker"], row_date) not in existing_keys:
                records_to_insert.append(
                    AssetPrice(
                        ticker=row["Ticker"],
                        date=row_date,
                        open=float(row["Open"]),
                        high=float(row["High"]),
                        low=float(row["Low"]),
                        close=float(row["Close"]),
                        volume=float(row["Volume"]),
                        adjusted_close=float(row["Adj_Close"]),
                    )
                )

        if records_to_insert:
            db.bulk_save_objects(records_to_insert)
            db.commit()
            inserted = len(records_to_insert)
    except Exception as exc:
        logger.warning("Market data sync error: %s", exc)
        db.rollback()
    finally:
        db.close()

    return {
        "status": "COMPLETED",
        "ticker_filter": ticker or "ALL",
        "records_inserted": inserted,
    }
