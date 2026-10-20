"""FinBERT financial-news sentiment classification and asset aggregation module."""

from quantpulse.sentiment.models import (
    AssetSentimentSummary,
    HeadlineItem,
    HeadlineSentiment,
    SentimentBatchResult,
)
from quantpulse.sentiment.finbert import (
    FinBERTSentimentClassifier,
    ModelLoadError,
)
from quantpulse.sentiment.loader import HeadlineLoader
from quantpulse.sentiment.aggregator import (
    aggregate_sentiment_by_asset,
    sentiment_summary_dataframe,
    sentiment_time_series,
)

__all__ = [
    "AssetSentimentSummary",
    "HeadlineItem",
    "HeadlineSentiment",
    "SentimentBatchResult",
    "FinBERTSentimentClassifier",
    "ModelLoadError",
    "HeadlineLoader",
    "aggregate_sentiment_by_asset",
    "sentiment_summary_dataframe",
    "sentiment_time_series",
]
