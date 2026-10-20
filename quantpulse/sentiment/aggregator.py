"""Sentiment aggregation by asset, timestamped rolling windows, and summary metrics."""

from collections import defaultdict
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from quantpulse.sentiment.models import (
    AssetSentimentSummary,
    HeadlineSentiment,
    SentimentBatchResult,
)


def aggregate_sentiment_by_asset(
    results: Union[SentimentBatchResult, List[HeadlineSentiment]],
) -> Dict[str, AssetSentimentSummary]:
    """Aggregate headline sentiment scores by asset ticker.

    Keeps sentiment signals strictly descriptive and segregated from market return series.
    Does not produce automated buy/sell orders.

    Parameters
    ----------
    results : SentimentBatchResult or list of HeadlineSentiment

    Returns
    -------
    dict mapping ticker string to AssetSentimentSummary
    """
    items = results.items if isinstance(results, SentimentBatchResult) else results
    if not items:
        return {}

    grouped: Dict[str, List[HeadlineSentiment]] = defaultdict(list)
    for item in items:
        key = item.ticker.upper() if item.ticker else "GENERAL"
        grouped[key].append(item)

    summaries: Dict[str, AssetSentimentSummary] = {}
    for ticker, group_items in grouped.items():
        total = len(group_items)
        scores = [it.sentiment_score for it in group_items]
        pos_cnt = sum(1 for it in group_items if it.label == "positive")
        neg_cnt = sum(1 for it in group_items if it.label == "negative")
        neu_cnt = sum(1 for it in group_items if it.label == "neutral")

        summaries[ticker] = AssetSentimentSummary(
            ticker=ticker,
            headline_count=total,
            mean_sentiment_score=round(float(np.mean(scores)), 4),
            positive_count=pos_cnt,
            negative_count=neg_cnt,
            neutral_count=neu_cnt,
            bullish_ratio=round(pos_cnt / total, 4),
            bearish_ratio=round(neg_cnt / total, 4),
            neutral_ratio=round(neu_cnt / total, 4),
        )

    return summaries


def sentiment_time_series(
    results: Union[SentimentBatchResult, List[HeadlineSentiment]],
) -> pd.DataFrame:
    """Transform timestamped headline sentiments into a chronological DataFrame."""
    items = results.items if isinstance(results, SentimentBatchResult) else results
    records = []
    for it in items:
        records.append({
            "timestamp": pd.to_datetime(it.timestamp) if it.timestamp else pd.NaT,
            "ticker": it.ticker or "GENERAL",
            "headline": it.headline,
            "label": it.label,
            "confidence": it.confidence,
            "sentiment_score": it.sentiment_score,
            "prob_positive": it.probabilities.get("positive", 0.0),
            "prob_negative": it.probabilities.get("negative", 0.0),
            "prob_neutral": it.probabilities.get("neutral", 0.0),
        })

    df = pd.DataFrame(records)
    if "timestamp" in df.columns and df["timestamp"].notna().any():
        df.sort_values(by="timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def sentiment_summary_dataframe(
    summaries: Dict[str, AssetSentimentSummary],
) -> pd.DataFrame:
    """Format asset sentiment summaries as a clean presentation DataFrame."""
    rows = []
    for ticker, s in summaries.items():
        rows.append({
            "Asset": ticker,
            "Headlines": s.headline_count,
            "Sentiment Score": f"{s.mean_sentiment_score:+.2f}",
            "Bullish (%)": f"{s.bullish_ratio * 100:.1f}%",
            "Bearish (%)": f"{s.bearish_ratio * 100:.1f}%",
            "Neutral (%)": f"{s.neutral_ratio * 100:.1f}%",
        })
    return pd.DataFrame(rows)
