"""Milestone 3 Quickstart Demonstration: PyTorch LSTM Volatility Forecasting Pipeline."""

import sys
from pathlib import Path

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from quantpulse.data.loader import MarketDataLoader
from quantpulse.ml.baseline import HistoricalVolatilityBaseline
from quantpulse.ml.dataset import (
    RobustFeatureScaler,
    VolatilityFeaturePipeline,
    VolatilitySequenceDataset,
    create_chronological_splits,
)
from quantpulse.ml.evaluator import compare_forecast_models
from quantpulse.ml.inference import VolatilityInferenceEngine
from quantpulse.ml.model import LSTMVolatilityForecaster
from quantpulse.ml.trainer import (
    TrainingConfig,
    VolatilityTrainer,
    save_checkpoint,
    set_seed,
)


def main():
    print("=" * 85)
    print("   QUANTPULSE AI — Autonomous Portfolio Risk & Volatility Optimization")
    print("   Milestone 3: PyTorch LSTM Volatility Forecasting Pipeline")
    print("=" * 85)

    # 1. Reproducibility
    set_seed(42)

    # 2. Ingest Historical Market Bars
    loader = MarketDataLoader()
    raw_df = loader.load()
    # Filter for S&P 500 benchmark ETF (SPY) for training volatility forecaster
    ticker = "SPY"
    asset_df = raw_df[raw_df["Ticker"] == ticker].copy().set_index("Date")

    print(f"\n[1] Ingested Market Observations for {ticker}:")
    print(f"    - Date Range: {asset_df.index.min().date()} to {asset_df.index.max().date()}")
    print(f"    - Observations: {len(asset_df)} daily bars")

    # 3. Feature Engineering & Forward Realized Volatility Target
    pipeline = VolatilityFeaturePipeline(forward_horizon=5)
    features_df, target_series = pipeline.build_features(asset_df)

    print(f"\n[2] Feature Pipeline (5-Day Forward Realized Volatility Target):")
    print(f"    - Extracted Features ({features_df.shape[1]}): {list(features_df.columns)}")
    print(f"    - Valid Chronological Samples: {len(features_df)}")

    # 4. Chronological Splitting (No Look-Ahead Leakage)
    (train_X, train_y), (val_X, val_y), (test_X, test_y) = create_chronological_splits(
        features_df, target_series, train_ratio=0.70, val_ratio=0.15
    )

    print(f"\n[3] Chronological Partitions:")
    print(f"    - Train (70%) : {train_X.index.min().date()} -> {train_X.index.max().date()} ({len(train_X)} samples)")
    print(f"    - Val   (15%) : {val_X.index.min().date()} -> {val_X.index.max().date()} ({len(val_X)} samples)")
    print(f"    - Test  (15%) : {test_X.index.min().date()} -> {test_X.index.max().date()} ({len(test_X)} samples)")

    # 5. Feature Scaling (Fitted STRICTLY on Train)
    scaler = RobustFeatureScaler().fit(train_X)
    train_scaled = scaler.transform(train_X)
    val_scaled = scaler.transform(val_X)
    test_scaled = scaler.transform(test_X)

    # 6. Sequence Datasets for LSTM
    seq_len = 20  # 20 trading days historical context window
    train_ds = VolatilitySequenceDataset(train_scaled, train_y.values, sequence_length=seq_len)
    val_ds = VolatilitySequenceDataset(val_scaled, val_y.values, sequence_length=seq_len)
    test_ds = VolatilitySequenceDataset(test_scaled, test_y.values, sequence_length=seq_len)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # 7. Model Initialization (Optimized for M1 Mac 8 GB RAM)
    config = TrainingConfig(
        epochs=35,
        batch_size=32,
        sequence_length=seq_len,
        learning_rate=0.003,
        hidden_size=32,
        num_layers=2,
        dropout=0.1,
        patience=8,
        seed=42,
        device="cpu",  # CPU is fastest for small sequences, fully compatible with Apple Silicon
    )

    model = LSTMVolatilityForecaster(
        input_size=train_X.shape[1],
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )

    print(f"\n[4] PyTorch Model Architecture:")
    print(f"    - Model: 2-Layer LSTM + Dense Regression Head + Softplus Output")
    print(f"    - Total Trainable Parameters: {model.count_parameters():,}")
    print(f"    - Memory Footprint: < 1.0 MB (ultra-lightweight for 8 GB RAM)")

    # 8. Model Training
    print("\n[5] Training Model with Validation Early Stopping...")
    trainer = VolatilityTrainer(model, config)
    history = trainer.fit(train_loader, val_loader)

    print(f"    - Epochs Completed: {history['epochs_completed']}")
    print(f"    - Best Validation Huber Loss: {history['best_val_loss']:.6f}")

    # 9. Evaluate on Unseen Out-of-Sample Test Data
    # Align test targets with test sequences
    y_test_ground_truth = test_y.values[seq_len - 1 :]

    # LSTM Predictions
    model.eval()
    lstm_preds_list = []
    with torch.no_grad():
        for bx, _ in test_loader:
            out = model(bx)
            lstm_preds_list.extend(out.cpu().numpy().flatten())
    lstm_test_preds = np.array(lstm_preds_list)

    # Baseline 1: Recent 21-day rolling volatility
    baseline_21d = HistoricalVolatilityBaseline(method="rolling", window_col="vol_21d")
    base_21d_preds = baseline_21d.predict(test_X)[seq_len - 1 :]

    # Baseline 2: Recent 5-day rolling volatility
    baseline_5d = HistoricalVolatilityBaseline(method="rolling", window_col="vol_5d")
    base_5d_preds = baseline_5d.predict(test_X)[seq_len - 1 :]

    eval_table = compare_forecast_models(
        y_true=y_test_ground_truth,
        model_predictions={
            "Historical_Vol_21d_Baseline": base_21d_preds,
            "Historical_Vol_5d_Baseline": base_5d_preds,
            "PyTorch_LSTM_Forecaster": lstm_test_preds,
        },
        baseline_name="Historical_Vol_21d_Baseline",
    )

    print("\n[6] Unseen Test Set Model Evaluation (Out-of-Sample):")
    print(eval_table.to_string())

    # 10. Checkpoint Persistence & Decoupled Inference Demo
    checkpoint_dir = Path(__file__).resolve().parent.parent / "models" / "saved"
    checkpoint_path = checkpoint_dir / f"lstm_volatility_{ticker.lower()}.pt"
    save_checkpoint(model, scaler, config, checkpoint_path)
    print(f"\n[7] Saved Checkpoint to: {checkpoint_path}")

    # Test standalone inference engine
    inference_engine = VolatilityInferenceEngine(checkpoint_path=checkpoint_path, device="cpu")
    latest_pred = inference_engine.predict_latest(asset_df)
    print(f"\n[8] Live Decoupled Inference Verification:")
    print(f"    - Asset: {ticker}")
    print(f"    - Most Recent Close Date: {asset_df.index.max().date()}")
    print(f"    - Forecasted Next 5-Day Realized Volatility: {latest_pred * 100:.2f}% (Annualized)")

    print("\n" + "=" * 85)
    print("   Milestone 3 Verification: SUCCESSFUL")
    print("=" * 85)


if __name__ == "__main__":
    main()
