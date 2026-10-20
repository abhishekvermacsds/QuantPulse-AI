"""Data schemas and Pydantic models for financial sentiment analysis."""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class HeadlineItem(BaseModel):
    """Input representation of a financial news headline."""

    headline: str = Field(..., min_length=1, description="Financial headline text")
    ticker: Optional[str] = Field(default=None, description="Associated asset ticker if applicable")
    timestamp: Optional[str] = Field(default=None, description="ISO-8601 publication timestamp")


class HeadlineSentiment(BaseModel):
    """Classified sentiment output for an individual headline."""

    headline: str
    ticker: Optional[str] = None
    timestamp: Optional[str] = None
    label: str = Field(..., description="Predicted label: positive, negative, or neutral")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence of predicted class")
    probabilities: Dict[str, float] = Field(
        ..., description="Full probability distribution over positive, negative, and neutral"
    )
    sentiment_score: float = Field(
        ..., ge=-1.0, le=1.0, description="Compound score: P(positive) - P(negative)"
    )


class SentimentBatchResult(BaseModel):
    """Aggregate result of a batch sentiment inference run."""

    total_headlines: int
    items: List[HeadlineSentiment]
    mean_sentiment_score: float
    positive_count: int
    negative_count: int
    neutral_count: int


class AssetSentimentSummary(BaseModel):
    """Aggregated sentiment profile for a specific asset ticker."""

    ticker: str
    headline_count: int
    mean_sentiment_score: float
    positive_count: int
    negative_count: int
    neutral_count: int
    bullish_ratio: float = Field(
        ..., description="Proportion of positive headlines: positive / total"
    )
    bearish_ratio: float = Field(
        ..., description="Proportion of negative headlines: negative / total"
    )
    neutral_ratio: float = Field(
        ..., description="Proportion of neutral headlines: neutral / total"
    )
