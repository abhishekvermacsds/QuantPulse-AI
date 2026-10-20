"""Integration tests for Celery background tasks and API endpoints."""

import pytest
from fastapi.testclient import TestClient

from quantpulse.api.app import app
from quantpulse.tasks.celery_app import celery_app
from quantpulse.tasks.tasks import (
    task_classify_news_batch,
    task_sync_market_data,
    task_train_volatility_model,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def configure_celery_eager(monkeypatch):
    """Enable Celery eager mode during testing for synchronous execution."""
    old_eager = celery_app.conf.task_always_eager
    old_backend = celery_app.conf.result_backend
    celery_app.conf.update(
        task_always_eager=True,
        task_eager_propagates=True,
        result_backend="cache+memory://",
    )
    yield
    celery_app.conf.update(
        task_always_eager=old_eager,
        result_backend=old_backend,
    )


def test_task_sync_market_data_eager():
    """Verify market data sync background task executes and returns records."""
    result = task_sync_market_data.apply(kwargs={"ticker": "AAPL"}).get()
    assert result["status"] == "COMPLETED"
    assert result["ticker_filter"] == "AAPL"
    assert isinstance(result["records_inserted"], int)


def test_task_classify_news_batch_eager():
    """Verify asynchronous FinBERT news classification task with RAM unload safeguard."""
    sample_headlines = [
        {"headline": "Tech giant beats revenue forecast significantly.", "ticker": "MSFT"},
        {"headline": "Automaker recalls thousands of vehicles after probe.", "ticker": "TSLA"},
    ]
    result = task_classify_news_batch.apply(kwargs={"headlines": sample_headlines}).get()

    assert result["status"] == "COMPLETED"
    assert result["total_processed"] == 2
    assert "sentiment_breakdown" in result
    assert len(result["results"]) == 2
    assert result["results"][0]["predicted_label"] in ["positive", "neutral", "negative"]


def test_task_train_volatility_model_eager(tmp_path):
    """Verify background LSTM training task executes with small epochs."""
    result = task_train_volatility_model.apply(
        kwargs={
            "ticker": "SPY",
            "epochs": 1,
            "horizon_days": 5,
            "sequence_length": 10,
        }
    ).get()

    assert result["status"] == "COMPLETED"
    assert result["ticker"] == "SPY"
    assert "metrics" in result
    assert "lstm_mae" in result["metrics"]


def test_api_tasks_endpoints():
    """Verify task dispatch endpoints (/api/tasks/...)."""
    # 1. Market Data Sync Task
    res_sync = client.post("/api/tasks/market-data/sync?ticker=SPY")
    assert res_sync.status_code == 200
    sync_data = res_sync.json()
    assert "task_id" in sync_data

    # 2. News Batch Task
    res_sent = client.post(
        "/api/tasks/sentiment/batch",
        json={
            "headlines": [
                {"headline": "Federal Reserve maintains interest rates target range.", "ticker": "SPY"}
            ]
        },
    )
    assert res_sent.status_code == 200
    sent_data = res_sent.json()
    assert "task_id" in sent_data

    # 3. Task Status Check
    task_id = sent_data["task_id"]
    res_status = client.get(f"/api/tasks/{task_id}")
    assert res_status.status_code == 200
    status_data = res_status.json()
    assert status_data["task_id"] == task_id


def test_api_portfolios_and_seed_endpoints():
    """Verify database management endpoints (/api/portfolios and /api/db/seed)."""
    # Seed DB
    res_seed = client.post("/api/db/seed")
    assert res_seed.status_code == 200
    assert res_seed.json()["status"] == "success"

    # List Portfolios
    res_list = client.get("/api/portfolios")
    assert res_list.status_code == 200
    portfolios = res_list.json()
    assert isinstance(portfolios, list)

    # Create new portfolio
    unique_name = f"Test Growth Fund {pytest.approx(1)}"
    res_create = client.post(
        "/api/portfolios",
        json={
            "name": "Test Alpha Portfolio",
            "description": "Quant test allocation",
            "asset_tickers": ["AAPL", "MSFT", "SPY"],
            "initial_capital": 250000.0,
        },
    )
    # 201 Created or 400 if already exists in persistent SQLite
    assert res_create.status_code in [201, 400]
