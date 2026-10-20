"""Integration tests for Redis caching and graceful fallback degradation."""

import pytest
from quantpulse.core.cache import (
    cache_response,
    clear_cache,
    delete_cached_value,
    generate_cache_key,
    get_cached_value,
    is_redis_available,
    set_cached_value,
)


def test_generate_cache_key_deterministic():
    """Verify deterministic hash key generation regardless of dict key order."""
    key1 = generate_cache_key("test", 1, 2, a="foo", b="bar")
    key2 = generate_cache_key("test", 1, 2, b="bar", a="foo")
    key3 = generate_cache_key("test", 1, 3, a="foo", b="bar")

    assert key1 == key2
    assert key1 != key3
    assert key1.startswith("quantpulse:test:")


def test_cache_get_set_delete_cycle():
    """Test standard cache lifecycle (set, get, delete)."""
    test_key = "quantpulse:test_cycle"
    data = {"metric": "sharpe", "value": 1.45, "assets": ["AAPL", "MSFT"]}

    set_cached_value(test_key, data, ttl=60)
    retrieved = get_cached_value(test_key)

    assert retrieved is not None
    assert retrieved["metric"] == "sharpe"
    assert retrieved["value"] == 1.45
    assert retrieved["assets"] == ["AAPL", "MSFT"]

    # Delete
    deleted = delete_cached_value(test_key)
    assert deleted is True
    assert get_cached_value(test_key) is None


def test_cache_response_decorator():
    """Verify decorator intercepts repeated calls and returns cached payload."""
    call_count = 0

    @cache_response(prefix="expensive_calc", ttl=60)
    def expensive_calculation(ticker: str, multiplier: float) -> dict:
        nonlocal call_count
        call_count += 1
        return {"ticker": ticker, "result": multiplier * 10}

    # Clear any previous cache
    clear_cache("quantpulse:expensive_calc:*")

    # First execution - computes
    res1 = expensive_calculation("AAPL", 2.0)
    assert res1["result"] == 20.0
    assert call_count == 1

    # Second execution with same parameters - served from cache
    res2 = expensive_calculation("AAPL", 2.0)
    assert res2["result"] == 20.0
    assert call_count == 1  # Not incremented!

    # Execution with different parameters - computes
    res3 = expensive_calculation("MSFT", 2.0)
    assert res3["result"] == 20.0
    assert call_count == 2


def test_redis_graceful_fallback():
    """Ensure is_redis_available returns bool without raising exceptions."""
    available = is_redis_available()
    assert isinstance(available, bool)
