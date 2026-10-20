"""Celery asynchronous task queue and distributed workers."""

from quantpulse.tasks.celery_app import celery_app
from quantpulse.tasks.tasks import (
    task_classify_news_batch,
    task_sync_market_data,
    task_train_volatility_model,
)

__all__ = [
    "celery_app",
    "task_train_volatility_model",
    "task_classify_news_batch",
    "task_sync_market_data",
]
