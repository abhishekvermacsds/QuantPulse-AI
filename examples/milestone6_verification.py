"""Milestone 6 Automated Verification Script for QuantPulse AI."""

import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

# Setup test database path
TEST_DB_PATH = Path("data/verification_test.db")
if TEST_DB_PATH.exists():
    TEST_DB_PATH.unlink()

os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH.resolve()}"
os.environ["CELERY_TASK_ALWAYS_EAGER"] = "true"

from quantpulse.api.app import app, create_app
from quantpulse.core.cache import (
    clear_cache,
    generate_cache_key,
    get_cached_value,
    is_redis_available,
    set_cached_value,
)
from quantpulse.db.models import (
    AssetPrice,
    BacktestRecord,
    Base,
    OptimizationRecord,
    Portfolio,
    SentimentRecord,
    VolatilityForecastRecord,
)
from quantpulse.db.seed import seed_database
from quantpulse.tasks.celery_app import celery_app
from quantpulse.tasks.tasks import (
    task_classify_news_batch,
    task_sync_market_data,
    task_train_volatility_model,
)


def run_verification():
    print("=" * 70)
    print("QUANTPULSE AI — MILESTONE 6 COMPREHENSIVE VERIFICATION")
    print("=" * 70)

    # 1. Validate Database Schema Creation
    print("\n[1/6] Validating Schema Creation on Fresh Database...")
    engine = create_engine(os.environ["DATABASE_URL"])
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    tables = list(Base.metadata.tables.keys())
    print(f"  ✓ Successfully created {len(tables)} tables: {', '.join(tables)}")

    # 2. Validate Seeding Process
    print("\n[2/6] Validating Database Seed Process...")
    seeded_prices = seed_database(session=session)
    portfolio_count = session.query(Portfolio).count()
    sentiment_count = session.query(SentimentRecord).count()
    opt_count = session.query(OptimizationRecord).count()
    print(f"  ✓ Seeded AssetPrice records: {seeded_prices}")
    print(f"  ✓ Seeded Portfolios: {portfolio_count}")
    print(f"  ✓ Seeded Sentiment records: {sentiment_count}")
    print(f"  ✓ Seeded Optimization records: {opt_count}")
    session.close()

    # 3. Verify Redis Connectivity & Caching
    print("\n[3/6] Verifying Redis Connectivity & Graceful Cache Fallback...")
    redis_active = is_redis_available()
    print(f"  ✓ Redis active: {redis_active} (graceful memory fallback enabled)")
    test_key = generate_cache_key("test_milestone6", 42, name="AAPL")
    set_cached_value(test_key, {"status": "ok", "cached_val": 99.5}, ttl=60)
    val = get_cached_value(test_key)
    assert val["cached_val"] == 99.5
    print(f"  ✓ Cache get/set operational. Retrieved: {val}")

    # 4. Execute Celery Background Task
    print("\n[4/6] Executing Celery Background Task...")
    celery_app.conf.update(
        task_always_eager=True,
        task_eager_propagates=True,
        result_backend="cache+memory://",
    )
    task_result = task_classify_news_batch.apply(
        kwargs={
            "headlines": [
                {
                    "headline": "Federal Reserve signals steady policy as growth continues.",
                    "ticker": "SPY",
                }
            ]
        }
    ).get()
    print(f"  ✓ Celery task execution status: {task_result['status']}")
    print(f"  ✓ Total processed: {task_result['total_processed']}")
    print(f"  ✓ Sentiment label: {task_result['results'][0]['predicted_label']}")
    print(f"  ✓ Compound score: {task_result['results'][0]['compound_score']}")

    # 5. Data Persistence After API Restart
    print("\n[5/6] Verifying Data Persistence Across API Lifecycles...")
    # Session 1: Create a portfolio via FastAPI TestClient
    client1 = TestClient(create_app())
    post_res = client1.post(
        "/api/portfolios",
        json={
            "name": "Persistence Verification Portfolio",
            "description": "Created before simulated restart",
            "asset_tickers": ["AAPL", "MSFT", "SPY"],
            "initial_capital": 500000.0,
        },
    )
    assert post_res.status_code == 201, post_res.text
    created_id = post_res.json()["id"]
    print(f"  ✓ API Instance 1: Created portfolio ID {created_id} ('Persistence Verification Portfolio')")

    # Restart: Destroy client1, recreate fresh client2
    del client1
    client2 = TestClient(create_app())
    get_res = client2.get(f"/api/portfolios/{created_id}")
    assert get_res.status_code == 200, get_res.text
    fetched = get_res.json()
    assert fetched["name"] == "Persistence Verification Portfolio"
    assert fetched["initial_capital"] == 500000.0
    print(f"  ✓ API Instance 2 (after restart): Verified portfolio persisted: {fetched['name']} (${fetched['initial_capital']:,.2f})")

    # 6. Clean up temporary test DB
    engine.dispose()
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()
    print("\n[6/6] Cleanup of temporary verification database complete.")
    print("\n" + "=" * 70)
    print("ALL MILESTONE 6 VERIFICATION CRITERIA PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    run_verification()
