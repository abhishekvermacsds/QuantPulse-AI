"""Database seeding utility for populating historical data, portfolios, and sample records."""

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional
import pandas as pd
from sqlalchemy.orm import Session

from quantpulse.core.config import DEFAULT_HEADLINES_CSV, DEFAULT_SAMPLE_CSV
from quantpulse.db.models import (
    AssetPrice,
    BacktestRecord,
    OptimizationRecord,
    Portfolio,
    SentimentRecord,
    VolatilityForecastRecord,
)
from quantpulse.db.session import SessionLocal, init_db

logger = logging.getLogger(__name__)


def seed_database(
    market_csv_path: Optional[Path] = None,
    headlines_csv_path: Optional[Path] = None,
    session: Optional[Session] = None,
) -> int:
    """Populate database with sample market prices, portfolio presets, and sample records.
    
    Returns total count of inserted asset price records.
    """
    db = session or SessionLocal()
    market_path = market_csv_path or DEFAULT_SAMPLE_CSV
    headlines_path = headlines_csv_path or DEFAULT_HEADLINES_CSV

    try:
        # 1. Initialize schema
        init_db()

        # 2. Seed Asset Prices if table is empty
        price_count = db.query(AssetPrice).count()
        if price_count == 0 and market_path.exists():
            logger.info("Seeding historical asset prices from %s", market_path)
            df = pd.read_csv(market_path)
            df.columns = [c.lower() for c in df.columns]
            if "adj_close" in df.columns and "adjusted_close" not in df.columns:
                df["adjusted_close"] = df["adj_close"]
            df["date"] = pd.to_datetime(df["date"]).dt.date

            records = []
            for _, row in df.iterrows():
                records.append(
                    AssetPrice(
                        ticker=str(row["ticker"]),
                        date=row["date"],
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                        adjusted_close=float(row["adjusted_close"]),
                    )
                )
            db.bulk_save_objects(records)
            db.commit()
            price_count = len(records)
            logger.info("Successfully seeded %d asset price bars.", price_count)
        else:
            logger.info("Asset prices already seeded (%d records exist).", price_count)

        # 3. Seed Default Portfolio if not exists
        core_portfolio = db.query(Portfolio).filter_by(name="Tech & Macro Core").first()
        if not core_portfolio:
            core_portfolio = Portfolio(
                name="Tech & Macro Core",
                description="Balanced portfolio combining Mega-Cap Tech, Financials, Energy, and US Treasuries",
                asset_tickers=["AAPL", "MSFT", "JPM", "XOM", "SPY", "TLT"],
                initial_capital=100000.0,
            )
            db.add(core_portfolio)
            db.commit()
            db.refresh(core_portfolio)
            logger.info("Created default portfolio: %s (id: %s)", core_portfolio.name, core_portfolio.id)

        # 4. Seed Sample Optimization Run if empty
        if db.query(OptimizationRecord).count() == 0 and core_portfolio:
            opt_record = OptimizationRecord(
                portfolio_id=core_portfolio.id,
                objective="max_sharpe",
                target_return=None,
                max_weight=0.35,
                optimal_weights={
                    "AAPL": 0.22,
                    "MSFT": 0.24,
                    "JPM": 0.14,
                    "XOM": 0.10,
                    "SPY": 0.15,
                    "TLT": 0.15,
                },
                expected_return=0.185,
                volatility=0.142,
                sharpe_ratio=1.02,
                diversification_ratio=1.45,
                effective_assets=5.1,
            )
            db.add(opt_record)
            db.commit()

        # 5. Seed Sample Volatility Forecast Record if empty
        if db.query(VolatilityForecastRecord).count() == 0:
            fc_record = VolatilityForecastRecord(
                ticker="SPY",
                model_name="PyTorch_LSTM",
                sequence_length=20,
                target_horizon_days=5,
                predicted_volatility=0.145,
                actual_volatility=0.151,
                baseline_volatility=0.162,
                mae=0.012,
                rmse=0.018,
            )
            db.add(fc_record)
            db.commit()

        # 6. Seed Sample Sentiment Headlines if empty
        if db.query(SentimentRecord).count() == 0 and headlines_path.exists():
            news_df = pd.read_csv(headlines_path)
            sentiment_records = []
            for _, row in news_df.iterrows():
                published_dt = None
                if "timestamp" in row and pd.notna(row["timestamp"]):
                    try:
                        published_dt = pd.to_datetime(row["timestamp"]).to_pydatetime()
                    except Exception:
                        published_dt = datetime.utcnow()

                sentiment_records.append(
                    SentimentRecord(
                        headline=str(row["headline"]),
                        ticker=str(row["ticker"]) if "ticker" in row and pd.notna(row["ticker"]) else None,
                        source="Sample News Wire",
                        published_at=published_dt or datetime.utcnow(),
                        predicted_label="positive" if "beat" in str(row["headline"]).lower() or "surges" in str(row["headline"]).lower() else "neutral",
                        confidence_positive=0.85 if "beat" in str(row["headline"]).lower() or "surges" in str(row["headline"]).lower() else 0.15,
                        confidence_negative=0.05,
                        confidence_neutral=0.10 if "beat" in str(row["headline"]).lower() or "surges" in str(row["headline"]).lower() else 0.80,
                        compound_score=0.80 if "beat" in str(row["headline"]).lower() or "surges" in str(row["headline"]).lower() else 0.05,
                    )
                )
            db.bulk_save_objects(sentiment_records)
            db.commit()
            logger.info("Seeded %d sample sentiment records.", len(sentiment_records))

        # 7. Seed Sample Backtest Record if empty
        if db.query(BacktestRecord).count() == 0 and core_portfolio:
            backtest_record = BacktestRecord(
                portfolio_id=core_portfolio.id,
                strategy_name="Max Sharpe Long-Only Rebalanced",
                start_date=date(2021, 1, 4),
                end_date=date(2023, 12, 29),
                initial_value=100000.0,
                final_value=138450.0,
                total_return=0.3845,
                cagr=0.1145,
                annualized_volatility=0.148,
                max_drawdown=-0.165,
                sharpe_ratio=0.89,
                sortino_ratio=1.12,
                calmar_ratio=0.69,
                equity_curve=[
                    {"date": "2021-01-04", "value": 100000.0},
                    {"date": "2022-01-03", "value": 115200.0},
                    {"date": "2023-01-03", "value": 108400.0},
                    {"date": "2023-12-29", "value": 138450.0},
                ],
            )
            db.add(backtest_record)
            db.commit()

        return price_count
    finally:
        if session is None:
            db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    count = seed_database()
    print(f"Database seeded successfully with {count} price records.")
