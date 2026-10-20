"""System-wide configuration, market constants, and persistence settings."""

import os
from pathlib import Path
from pydantic import BaseModel, Field

# Standard US equity trading days per calendar year
TRADING_DAYS_PER_YEAR: int = 252

# Default annual risk-free rate (approx. 4.0% annualized US Treasury yield)
DEFAULT_RISK_FREE_RATE: float = 0.04

# Default statistical confidence levels for risk calculations
DEFAULT_CONFIDENCE_LEVEL_VAR: float = 0.95
DEFAULT_CONFIDENCE_LEVEL_CVAR: float = 0.95

# Project directory paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
SAMPLE_DATA_DIR: Path = DATA_DIR / "sample"
DEFAULT_SAMPLE_CSV: Path = SAMPLE_DATA_DIR / "market_data.csv"
DEFAULT_HEADLINES_CSV: Path = SAMPLE_DATA_DIR / "financial_headlines.csv"

# Persistence & Infrastructure defaults (overridable via environment variables)
DEFAULT_DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{DATA_DIR}/quantpulse.db",
)

DEFAULT_REDIS_URL: str = os.getenv(
    "REDIS_URL",
    "redis://localhost:6379/0",
)

DEFAULT_CELERY_BROKER_URL: str = os.getenv(
    "CELERY_BROKER_URL",
    DEFAULT_REDIS_URL,
)

DEFAULT_CELERY_RESULT_BACKEND: str = os.getenv(
    "CELERY_RESULT_BACKEND",
    DEFAULT_REDIS_URL,
)

CELERY_TASK_ALWAYS_EAGER: bool = os.getenv(
    "CELERY_TASK_ALWAYS_EAGER", "False"
).lower() in ("true", "1", "yes")

CACHE_DEFAULT_TTL: int = int(os.getenv("CACHE_DEFAULT_TTL", "300"))


class SystemConfig(BaseModel):
    """Runtime configuration for data ingestion, risk calculations, and persistence."""

    trading_days_per_year: int = Field(
        default=TRADING_DAYS_PER_YEAR,
        gt=0,
        description="Number of trading days used to annualize daily statistics",
    )
    risk_free_rate: float = Field(
        default=DEFAULT_RISK_FREE_RATE,
        ge=0.0,
        description="Annualized benchmark risk-free rate",
    )
    default_confidence_level: float = Field(
        default=DEFAULT_CONFIDENCE_LEVEL_VAR,
        gt=0.0,
        lt=1.0,
        description="Default confidence level for Value at Risk and Expected Shortfall",
    )
    database_url: str = Field(
        default=DEFAULT_DATABASE_URL,
        description="SQLAlchemy database connection string",
    )
    redis_url: str = Field(
        default=DEFAULT_REDIS_URL,
        description="Redis connection URL for caching and broker",
    )
    celery_task_always_eager: bool = Field(
        default=CELERY_TASK_ALWAYS_EAGER,
        description="If True, Celery tasks execute synchronously for test isolation",
    )
