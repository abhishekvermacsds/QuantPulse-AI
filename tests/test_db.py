"""Integration tests for SQLAlchemy persistence models and database operations."""

from datetime import date, datetime
import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from quantpulse.db.models import (
    AssetPrice,
    BacktestRecord,
    Base,
    OptimizationRecord,
    Portfolio,
    SentimentRecord,
    VolatilityForecastRecord,
)
from quantpulse.db.seed import seed_database


@pytest.fixture
def in_memory_db():
    """Create an isolated in-memory SQLite database for test session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_asset_price_crud_and_unique_constraint(in_memory_db):
    """Verify AssetPrice insertion, querying, and unique (ticker, date) constraint."""
    p1 = AssetPrice(
        ticker="AAPL",
        date=date(2023, 1, 3),
        open=130.0,
        high=132.0,
        low=129.0,
        close=131.0,
        volume=5000000.0,
        adjusted_close=131.0,
    )
    in_memory_db.add(p1)
    in_memory_db.commit()

    retrieved = in_memory_db.query(AssetPrice).filter_by(ticker="AAPL").first()
    assert retrieved is not None
    assert retrieved.close == 131.0
    assert retrieved.to_dict()["ticker"] == "AAPL"

    # Attempt duplicate (ticker, date)
    duplicate = AssetPrice(
        ticker="AAPL",
        date=date(2023, 1, 3),
        open=131.0,
        high=133.0,
        low=130.0,
        close=132.0,
        volume=6000000.0,
        adjusted_close=132.0,
    )
    in_memory_db.add(duplicate)
    with pytest.raises(IntegrityError):
        in_memory_db.commit()
    in_memory_db.rollback()


def test_portfolio_relationships_and_cascades(in_memory_db):
    """Verify Portfolio creation with linked OptimizationRecord and BacktestRecord."""
    portfolio = Portfolio(
        name="All Weather Alpha",
        description="Diversified risk parity asset mix",
        asset_tickers=["SPY", "TLT", "GLD"],
        initial_capital=50000.0,
    )
    in_memory_db.add(portfolio)
    in_memory_db.commit()

    # Link optimization record
    opt = OptimizationRecord(
        portfolio_id=portfolio.id,
        objective="max_sharpe",
        max_weight=0.40,
        optimal_weights={"SPY": 0.40, "TLT": 0.35, "GLD": 0.25},
        expected_return=0.12,
        volatility=0.09,
        sharpe_ratio=0.89,
        diversification_ratio=1.35,
        effective_assets=2.8,
    )
    # Link backtest record
    bt = BacktestRecord(
        portfolio_id=portfolio.id,
        strategy_name="All Weather Backtest",
        start_date=date(2022, 1, 3),
        end_date=date(2022, 12, 30),
        initial_value=50000.0,
        final_value=53500.0,
        total_return=0.07,
        cagr=0.07,
        annualized_volatility=0.085,
        max_drawdown=-0.062,
        sharpe_ratio=0.82,
        equity_curve=[{"date": "2022-01-03", "value": 50000.0}],
    )
    in_memory_db.add_all([opt, bt])
    in_memory_db.commit()

    in_memory_db.refresh(portfolio)
    assert len(portfolio.optimization_records) == 1
    assert len(portfolio.backtest_records) == 1
    assert portfolio.optimization_records[0].optimal_weights["SPY"] == 0.40
    assert portfolio.backtest_records[0].final_value == 53500.0


def test_volatility_and_sentiment_records(in_memory_db):
    """Verify insertion and schema serialization for Volatility and Sentiment models."""
    vf = VolatilityForecastRecord(
        ticker="SPY",
        model_name="PyTorch_LSTM",
        sequence_length=20,
        target_horizon_days=5,
        predicted_volatility=0.142,
        actual_volatility=0.148,
        baseline_volatility=0.160,
        mae=0.011,
        rmse=0.015,
    )
    sent = SentimentRecord(
        headline="Apple beats revenue estimates on iPhone demand",
        ticker="AAPL",
        source="Reuters",
        predicted_label="positive",
        confidence_positive=0.92,
        confidence_negative=0.02,
        confidence_neutral=0.06,
        compound_score=0.90,
    )
    in_memory_db.add_all([vf, sent])
    in_memory_db.commit()

    assert vf.id is not None
    assert sent.id is not None
    assert vf.to_dict()["predicted_volatility"] == 0.142
    assert sent.to_dict()["predicted_label"] == "positive"


def test_seed_database_execution(in_memory_db):
    """Verify seed_database creates portfolios, prices, and sample sentiment records."""
    count = seed_database(session=in_memory_db)
    assert count > 0

    prices_count = in_memory_db.query(AssetPrice).count()
    assert prices_count == count

    portfolios = in_memory_db.query(Portfolio).all()
    assert len(portfolios) >= 1
    assert portfolios[0].name == "Tech & Macro Core"

    sentiments = in_memory_db.query(SentimentRecord).count()
    assert sentiments > 0
