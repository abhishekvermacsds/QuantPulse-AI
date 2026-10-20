"""Health check and runtime diagnostics endpoint."""

import os
import resource
from datetime import datetime, timezone
from fastapi import APIRouter
from sqlalchemy import text
import torch

from quantpulse import __version__
from quantpulse.api.schemas import HealthResponse
from quantpulse.core.cache import is_redis_available
from quantpulse.db.session import engine

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Retrieve service health status, memory footprint, compute device, and infrastructure health."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # macOS ru_maxrss is in bytes, Linux is in kilobytes
    raw_rss = usage.ru_maxrss
    if os.uname().sysname == "Darwin":
        memory_mb = round(raw_rss / (1024 * 1024), 2)
    else:
        memory_mb = round(raw_rss / 1024, 2)

    device_name = "mps" if torch.backends.mps.is_available() else "cpu"

    # Database connection test
    db_status = "connected"
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        db_status = "disconnected"

    # Redis connection test
    redis_status = "connected" if is_redis_available() else "disconnected"

    return HealthResponse(
        status="healthy",
        version=__version__,
        device=device_name,
        memory_mb=memory_mb,
        timestamp=datetime.now(timezone.utc).isoformat(),
        database=db_status,
        redis=redis_status,
    )
