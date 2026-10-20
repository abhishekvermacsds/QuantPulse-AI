"""FastAPI application factory for QuantPulse AI."""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from quantpulse import __version__
from quantpulse.api.routes import health, persistence, portfolio, sentiment, tasks, volatility


def create_app() -> FastAPI:
    """Instantiate and configure the FastAPI application."""
    app = FastAPI(
        title="QuantPulse AI Engine API",
        description="Autonomous Portfolio Risk, Convex Optimization, Volatility Forecasting & Sentiment Engine",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS configuration allowing local frontend servers and cloud deployments (Render, Vercel)
    cors_env = os.getenv("CORS_ORIGINS", "")
    extra_origins = [o.strip() for o in cors_env.split(",") if o.strip()]

    origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ] + extra_origins

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_origin_regex=r"^(https?://(localhost|127\.0\.0\.1)(:\d+)?|https://.*\.onrender\.com|https://.*\.vercel\.app|https://.*\.pages\.dev)$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routes
    app.include_router(health.router, prefix="/api", tags=["System"])
    app.include_router(portfolio.router, prefix="/api", tags=["Portfolio Analytics & Optimization"])
    app.include_router(volatility.router, prefix="/api", tags=["Volatility Forecasting"])
    app.include_router(sentiment.router, prefix="/api", tags=["Financial NLP Sentiment"])
    app.include_router(persistence.router, prefix="/api", tags=["Persistence & Portfolios"])
    app.include_router(tasks.router, prefix="/api", tags=["Background Tasks & Celery"])

    return app


app = create_app()
