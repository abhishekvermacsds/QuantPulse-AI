"""Celery application configuration and initialization."""

import os
from celery import Celery

from quantpulse.core.config import (
    CELERY_TASK_ALWAYS_EAGER,
    DEFAULT_CELERY_BROKER_URL,
    DEFAULT_CELERY_RESULT_BACKEND,
)

celery_app = Celery(
    "quantpulse_tasks",
    broker=os.getenv("CELERY_BROKER_URL", DEFAULT_CELERY_BROKER_URL),
    backend=os.getenv("CELERY_RESULT_BACKEND", DEFAULT_CELERY_RESULT_BACKEND),
    include=["quantpulse.tasks.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes maximum execution time
    worker_prefetch_multiplier=1,  # Fair distribution
    worker_concurrency=2,  # Memory conscious for 8 GB RAM machines
    task_always_eager=CELERY_TASK_ALWAYS_EAGER,
)
