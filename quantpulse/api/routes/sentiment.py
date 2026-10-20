"""Financial news sentiment classification and asset attribution endpoints."""

from typing import List
from fastapi import APIRouter, HTTPException

from quantpulse.api.schemas import (
    AssetSentimentSummarySchema,
    ClassifyHeadlineRequest,
    ClassifyHeadlineResponse,
    SentimentHeadlineSchema,
    SentimentOverviewResponse,
)
from quantpulse.sentiment import (
    FinBERTSentimentClassifier,
    HeadlineItem,
    HeadlineLoader,
    aggregate_sentiment_by_asset,
)

router = APIRouter()
# Singleton instance with lazy loading (zero RAM allocated until first invocation)
_classifier = FinBERTSentimentClassifier(use_mock=False)


@router.get("/sentiment/headlines", response_model=SentimentOverviewResponse)
def get_sentiment_overview() -> SentimentOverviewResponse:
    """Retrieve timestamped sample headlines with FinBERT sentiment labels and asset aggregations."""
    try:
        headlines = HeadlineLoader.load_sample()
        batch_result = _classifier.classify_batch(headlines, batch_size=16)
        summaries_dict = aggregate_sentiment_by_asset(batch_result)

        asset_summaries = [
            AssetSentimentSummarySchema(
                ticker=s.ticker,
                headline_count=s.headline_count,
                mean_sentiment_score=s.mean_sentiment_score,
                bullish_ratio=round(s.bullish_ratio * 100.0, 1),
                bearish_ratio=round(s.bearish_ratio * 100.0, 1),
                neutral_ratio=round(s.neutral_ratio * 100.0, 1),
            )
            for s in summaries_dict.values()
        ]

        recent_items = [
            SentimentHeadlineSchema(
                headline=it.headline,
                ticker=it.ticker,
                timestamp=it.timestamp,
                label=it.label,
                confidence=round(it.confidence, 4),
                sentiment_score=round(it.sentiment_score, 4),
                probabilities=it.probabilities,
            )
            for it in batch_result.items
        ]

        return SentimentOverviewResponse(
            total_headlines=batch_result.total_headlines,
            mean_score=batch_result.mean_sentiment_score,
            asset_summaries=asset_summaries,
            recent_headlines=recent_items,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Sentiment processing failed: {str(exc)}",
        ) from exc


@router.post("/sentiment/classify", response_model=ClassifyHeadlineResponse)
def classify_headline(req: ClassifyHeadlineRequest) -> ClassifyHeadlineResponse:
    """Classify custom headline input into positive, negative, or neutral."""
    text = req.headline.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Headline cannot be empty or whitespace.")

    item_input = HeadlineItem(headline=text, ticker=req.ticker)
    try:
        res = _classifier.classify_headline(item_input)
        return ClassifyHeadlineResponse(
            item=SentimentHeadlineSchema(
                headline=res.headline,
                ticker=res.ticker,
                timestamp=res.timestamp,
                label=res.label,
                confidence=round(res.confidence, 4),
                sentiment_score=round(res.sentiment_score, 4),
                probabilities=res.probabilities,
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(exc)}") from exc
