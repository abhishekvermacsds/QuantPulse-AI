"""Integration and unit tests for FastAPI REST API endpoints."""

import pytest
from fastapi.testclient import TestClient

from quantpulse.api.app import app

client = TestClient(app)


def test_health_check_endpoint():
    """Verify system diagnostics and health status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "device" in data
    assert data["memory_mb"] > 0


def test_portfolio_metrics_endpoint():
    """Verify portfolio and asset performance metrics."""
    response = client.get("/api/portfolio/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["trading_days"] > 200
    assert len(data["asset_universe"]) >= 6
    assert len(data["metrics"]) >= 6

    # Verify first asset metrics structure
    first_asset = data["metrics"][0]
    assert "ticker" in first_asset
    assert "cagr" in first_asset
    assert "annualized_volatility" in first_asset
    assert "sharpe_ratio" in first_asset
    assert "max_drawdown" in first_asset


def test_correlation_matrix_endpoint():
    """Verify pairwise correlation matrix shape and symmetry."""
    response = client.get("/api/portfolio/correlation")
    assert response.status_code == 200
    data = response.json()
    tickers = data["tickers"]
    matrix = data["matrix"]

    assert len(tickers) >= 6
    for t in tickers:
        assert t in matrix
        # Diagonal element must be 1.0
        assert pytest.approx(matrix[t][t], rel=1e-3) == 1.0


def test_portfolio_optimize_endpoint_success():
    """Verify convex portfolio optimization with all objectives."""
    payload = {
        "objective": "all",
        "max_weight": 0.40,
        "risk_free_rate": 0.04,
        "include_frontier": True,
    }
    response = client.post("/api/portfolio/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "allocations" in data
    assert "equal_weight" in data["allocations"]
    assert "min_volatility" in data["allocations"]
    assert "max_sharpe" in data["allocations"]

    # Verify weight sum is 100%
    for alloc_name, profile in data["allocations"].items():
        w_sum = sum(profile["weights"].values())
        assert pytest.approx(w_sum, rel=1e-3) == 1.0
        # Verify no asset exceeds max_weight
        for w in profile["weights"].values():
            assert w <= 0.40 + 1e-4

    assert data["frontier"] is not None
    assert len(data["frontier"]) > 0


def test_portfolio_optimize_infeasible_max_weight():
    """Requesting max_weight < 1/N must return 400 Bad Request."""
    payload = {
        "max_weight": 0.10,  # 6 assets => 1/6 = 0.1667 > 0.10 => infeasible
    }
    response = client.post("/api/portfolio/optimize", json=payload)
    assert response.status_code == 400
    assert "Infeasible constraint" in response.json()["detail"]


def test_volatility_prediction_endpoint():
    """Verify volatility forecasting endpoint."""
    response = client.get("/api/volatility/predict?ticker=SPY")
    assert response.status_code == 200
    data = response.json()

    assert data["ticker"] == "SPY"
    assert data["forecasted_volatility"] > 0.0
    assert data["historical_vol_21d"] > 0.0
    assert "model" in data


def test_volatility_prediction_unknown_ticker():
    """Unknown ticker should return 404."""
    response = client.get("/api/volatility/predict?ticker=INVALID123")
    assert response.status_code == 404


def test_sentiment_headlines_overview():
    """Verify sample financial headlines and asset summaries."""
    response = client.get("/api/sentiment/headlines")
    assert response.status_code == 200
    data = response.json()

    assert data["total_headlines"] > 0
    assert len(data["recent_headlines"]) > 0
    assert len(data["asset_summaries"]) > 0


def test_sentiment_classify_custom_headline():
    """Verify single headline classification endpoint."""
    payload = {
        "headline": "Company reports record revenues beating Wall Street forecasts.",
        "ticker": "AAPL",
    }
    response = client.post("/api/sentiment/classify", json=payload)
    assert response.status_code == 200
    data = response.json()["item"]

    assert data["label"] == "positive"
    assert data["sentiment_score"] > 0.0
    assert data["ticker"] == "AAPL"


def test_sentiment_classify_empty_headline():
    """Empty headline must return 400 Bad Request."""
    payload = {"headline": "   "}
    response = client.post("/api/sentiment/classify", json=payload)
    assert response.status_code == 400
