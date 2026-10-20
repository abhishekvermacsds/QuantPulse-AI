"""SQLAlchemy 2.0 persistence models for QuantPulse AI."""

from datetime import date as dt_date
from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base declarative class for all QuantPulse database models."""
    pass


class AssetPrice(Base):
    """Historical daily OHLCV and adjusted close prices for portfolio universe assets."""

    __tablename__ = "asset_prices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
    date: Mapped[dt_date] = mapped_column(Date, index=True, nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)
    adjusted_close: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_asset_price_ticker_date"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert asset price record to dictionary."""
        return {
            "id": self.id,
            "ticker": self.ticker,
            "date": self.date.isoformat() if isinstance(self.date, dt_date) else str(self.date),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "adjusted_close": self.adjusted_close,
        }


class Portfolio(Base):
    """User-defined or system-managed investment portfolio entity."""

    __tablename__ = "portfolios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    asset_tickers: Mapped[List[str]] = mapped_column(JSON, nullable=False)
    initial_capital: Mapped[float] = mapped_column(Float, default=100000.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    optimization_records: Mapped[List["OptimizationRecord"]] = relationship(
        "OptimizationRecord", back_populates="portfolio", cascade="all, delete-orphan"
    )
    backtest_records: Mapped[List["BacktestRecord"]] = relationship(
        "BacktestRecord", back_populates="portfolio", cascade="all, delete-orphan"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert portfolio record to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "asset_tickers": self.asset_tickers,
            "initial_capital": self.initial_capital,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class OptimizationRecord(Base):
    """Historical convex portfolio optimization run and resulting optimal weights."""

    __tablename__ = "optimization_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("portfolios.id", ondelete="SET NULL"), nullable=True, index=True
    )
    objective: Mapped[str] = mapped_column(String(50), nullable=False)
    target_return: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_weight: Mapped[float] = mapped_column(Float, nullable=False)
    optimal_weights: Mapped[Dict[str, float]] = mapped_column(JSON, nullable=False)
    expected_return: Mapped[float] = mapped_column(Float, nullable=False)
    volatility: Mapped[float] = mapped_column(Float, nullable=False)
    sharpe_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    diversification_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    effective_assets: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationship
    portfolio: Mapped[Optional["Portfolio"]] = relationship(
        "Portfolio", back_populates="optimization_records"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert optimization record to dictionary."""
        return {
            "id": self.id,
            "portfolio_id": self.portfolio_id,
            "objective": self.objective,
            "target_return": self.target_return,
            "max_weight": self.max_weight,
            "optimal_weights": self.optimal_weights,
            "expected_return": self.expected_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "diversification_ratio": self.diversification_ratio,
            "effective_assets": self.effective_assets,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class VolatilityForecastRecord(Base):
    """PyTorch LSTM volatility forecasts and out-of-sample benchmark metrics."""

    __tablename__ = "volatility_forecast_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
    model_name: Mapped[str] = mapped_column(String(50), default="PyTorch_LSTM", nullable=False)
    sequence_length: Mapped[int] = mapped_column(Integer, default=20, nullable=False)
    target_horizon_days: Mapped[int] = mapped_column(Integer, default=5, nullable=False)
    predicted_volatility: Mapped[float] = mapped_column(Float, nullable=False)
    actual_volatility: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    baseline_volatility: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mae: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rmse: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert forecast record to dictionary."""
        return {
            "id": self.id,
            "ticker": self.ticker,
            "model_name": self.model_name,
            "sequence_length": self.sequence_length,
            "target_horizon_days": self.target_horizon_days,
            "predicted_volatility": self.predicted_volatility,
            "actual_volatility": self.actual_volatility,
            "baseline_volatility": self.baseline_volatility,
            "mae": self.mae,
            "rmse": self.rmse,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class SentimentRecord(Base):
    """FinBERT financial news headline classification and confidence scores."""

    __tablename__ = "sentiment_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    headline: Mapped[str] = mapped_column(Text, nullable=False)
    ticker: Mapped[Optional[str]] = mapped_column(String(20), index=True, nullable=True)
    source: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    published_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    predicted_label: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence_positive: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_negative: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_neutral: Mapped[float] = mapped_column(Float, nullable=False)
    compound_score: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert sentiment record to dictionary."""
        return {
            "id": self.id,
            "headline": self.headline,
            "ticker": self.ticker,
            "source": self.source,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "predicted_label": self.predicted_label,
            "confidence_positive": self.confidence_positive,
            "confidence_negative": self.confidence_negative,
            "confidence_neutral": self.confidence_neutral,
            "compound_score": self.compound_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class BacktestRecord(Base):
    """Historical backtest execution records and equity curve trajectories."""

    __tablename__ = "backtest_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("portfolios.id", ondelete="SET NULL"), nullable=True, index=True
    )
    strategy_name: Mapped[str] = mapped_column(String(100), nullable=False)
    start_date: Mapped[dt_date] = mapped_column(Date, nullable=False)
    end_date: Mapped[dt_date] = mapped_column(Date, nullable=False)
    initial_value: Mapped[float] = mapped_column(Float, default=100000.0, nullable=False)
    final_value: Mapped[float] = mapped_column(Float, nullable=False)
    total_return: Mapped[float] = mapped_column(Float, nullable=False)
    cagr: Mapped[float] = mapped_column(Float, nullable=False)
    annualized_volatility: Mapped[float] = mapped_column(Float, nullable=False)
    max_drawdown: Mapped[float] = mapped_column(Float, nullable=False)
    sharpe_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    sortino_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    calmar_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    equity_curve: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationship
    portfolio: Mapped[Optional["Portfolio"]] = relationship(
        "Portfolio", back_populates="backtest_records"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert backtest record to dictionary."""
        return {
            "id": self.id,
            "portfolio_id": self.portfolio_id,
            "strategy_name": self.strategy_name,
            "start_date": self.start_date.isoformat() if isinstance(self.start_date, dt_date) else str(self.start_date),
            "end_date": self.end_date.isoformat() if isinstance(self.end_date, dt_date) else str(self.end_date),
            "initial_value": self.initial_value,
            "final_value": self.final_value,
            "total_return": self.total_return,
            "cagr": self.cagr,
            "annualized_volatility": self.annualized_volatility,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "equity_curve": self.equity_curve,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
