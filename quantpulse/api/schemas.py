"""Pydantic request and response schemas for QuantPulse AI FastAPI backend."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "0.1.0"
    device: str
    memory_mb: float
    timestamp: str
    database: Optional[str] = None
    redis: Optional[str] = None


class AssetMetricsItem(BaseModel):
    ticker: str
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    var_95_historical: float
    var_95_cornish_fisher: float
    cvar_95_historical: float


class PortfolioMetricsResponse(BaseModel):
    start_date: str
    end_date: str
    trading_days: int
    asset_universe: List[str]
    metrics: List[AssetMetricsItem]


class CorrelationResponse(BaseModel):
    tickers: List[str]
    matrix: Dict[str, Dict[str, float]]


class OptimizationRequest(BaseModel):
    tickers: Optional[List[str]] = Field(default=None, description="Subset of asset tickers")
    objective: str = Field(default="all", description="min_volatility, max_sharpe, equal_weight, or all")
    max_weight: float = Field(default=0.35, gt=0.0, le=1.0, description="Maximum allocation limit per asset")
    risk_free_rate: float = Field(default=0.04, ge=0.0, description="Annual risk-free rate")
    include_frontier: bool = Field(default=True, description="Whether to include efficient frontier points")


class PortfolioAllocationProfile(BaseModel):
    objective: str
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    diversification_ratio: float
    effective_assets: float
    max_drawdown: float
    var_95_historical: float
    cvar_95_historical: float
    message: str


class OptimizationResponse(BaseModel):
    allocations: Dict[str, PortfolioAllocationProfile]
    frontier: Optional[List[Dict[str, Any]]] = None


class VolatilityForecastResponse(BaseModel):
    ticker: str
    as_of_date: str
    forecasted_volatility: float
    historical_vol_21d: float
    historical_vol_5d: float
    model: str
    context_bars: int


class SentimentHeadlineSchema(BaseModel):
    headline: str
    ticker: Optional[str] = None
    timestamp: Optional[str] = None
    label: str
    confidence: float
    sentiment_score: float
    probabilities: Dict[str, float]


class AssetSentimentSummarySchema(BaseModel):
    ticker: str
    headline_count: int
    mean_sentiment_score: float
    bullish_ratio: float
    bearish_ratio: float
    neutral_ratio: float


class SentimentOverviewResponse(BaseModel):
    total_headlines: int
    mean_score: float
    asset_summaries: List[AssetSentimentSummarySchema]
    recent_headlines: List[SentimentHeadlineSchema]


class ClassifyHeadlineRequest(BaseModel):
    headline: str = Field(..., min_length=2, description="Financial headline text")
    ticker: Optional[str] = Field(default=None, description="Optional associated ticker")


class ClassifyHeadlineResponse(BaseModel):
    item: SentimentHeadlineSchema


# Milestone 6: Background Tasks & Persistence Schemas
class TaskDispatchResponse(BaseModel):
    task_id: str
    status: str = "PENDING"
    message: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None


class TrainModelTaskRequest(BaseModel):
    ticker: str = Field(default="SPY", description="Target asset ticker")
    epochs: int = Field(default=10, ge=1, le=50, description="Training epochs")
    horizon_days: int = Field(default=5, ge=1, le=30, description="Forecast target horizon in trading days")


class BatchHeadlineItem(BaseModel):
    headline: str = Field(..., min_length=2)
    ticker: Optional[str] = None
    source: Optional[str] = "Manual Ingestion"
    published_at: Optional[str] = None


class BatchSentimentTaskRequest(BaseModel):
    headlines: List[BatchHeadlineItem]


class PortfolioCreateRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    description: Optional[str] = None
    asset_tickers: List[str] = Field(..., min_length=1)
    initial_capital: float = Field(default=100000.0, gt=0.0)


class PortfolioResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    asset_tickers: List[str]
    initial_capital: float
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class SeedDatabaseResponse(BaseModel):
    status: str = "success"
    message: str
    prices_seeded: int
