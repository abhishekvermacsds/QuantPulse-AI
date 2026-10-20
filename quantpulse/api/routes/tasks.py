"""Celery background task dispatching and status monitoring routes."""

import logging
from typing import Any, Dict
from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException

from quantpulse.api.schemas import (
    BatchSentimentTaskRequest,
    TaskDispatchResponse,
    TaskStatusResponse,
    TrainModelTaskRequest,
)
from quantpulse.tasks.celery_app import celery_app
from quantpulse.tasks.tasks import (
    task_classify_news_batch,
    task_sync_market_data,
    task_train_volatility_model,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/tasks/volatility/train", response_model=TaskDispatchResponse)
def dispatch_volatility_training(request: TrainModelTaskRequest) -> TaskDispatchResponse:
    """Dispatch background training job for PyTorch LSTM volatility forecaster."""
    try:
        async_result = task_train_volatility_model.delay(
            ticker=request.ticker,
            epochs=request.epochs,
            horizon_days=request.horizon_days,
        )
        return TaskDispatchResponse(
            task_id=async_result.id,
            status=async_result.status,
            message=f"LSTM training dispatched for {request.ticker} ({request.epochs} epochs)",
        )
    except Exception as exc:
        logger.error("Failed to enqueue volatility training task: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to queue background training task: {str(exc)}",
        )


@router.post("/tasks/sentiment/batch", response_model=TaskDispatchResponse)
def dispatch_sentiment_batch(request: BatchSentimentTaskRequest) -> TaskDispatchResponse:
    """Dispatch asynchronous FinBERT classification for batch of headlines."""
    try:
        headlines_data = [item.model_dump() for item in request.headlines]
        async_result = task_classify_news_batch.delay(headlines=headlines_data)
        return TaskDispatchResponse(
            task_id=async_result.id,
            status=async_result.status,
            message=f"Enqueued FinBERT classification for {len(headlines_data)} headlines",
        )
    except Exception as exc:
        logger.error("Failed to enqueue sentiment batch task: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to queue background sentiment task: {str(exc)}",
        )


@router.post("/tasks/market-data/sync", response_model=TaskDispatchResponse)
def dispatch_market_data_sync(ticker: str = "SPY") -> TaskDispatchResponse:
    """Dispatch background synchronization of market data into PostgreSQL."""
    try:
        async_result = task_sync_market_data.delay(ticker=ticker)
        return TaskDispatchResponse(
            task_id=async_result.id,
            status=async_result.status,
            message=f"Market data sync dispatched for ticker {ticker}",
        )
    except Exception as exc:
        logger.error("Failed to enqueue market data sync task: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to queue background data sync task: {str(exc)}",
        )


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str) -> TaskStatusResponse:
    """Query execution status and output payload for any Celery task ID."""
    result = AsyncResult(task_id, app=celery_app)

    state = result.state
    output_result = None
    error_msg = None

    if state == "SUCCESS":
        output_result = result.result
    elif state == "FAILURE":
        error_msg = str(result.result)
    elif state == "PROGRESS":
        output_result = result.info

    return TaskStatusResponse(
        task_id=task_id,
        status=state,
        result=output_result,
        error=error_msg,
    )
