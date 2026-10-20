"""Redis caching layer with graceful offline fallback and parameter hashing."""

import functools
import hashlib
import json
import logging
from typing import Any, Callable, Optional
import redis

from quantpulse.core.config import CACHE_DEFAULT_TTL, DEFAULT_REDIS_URL

logger = logging.getLogger(__name__)

# Global in-memory fallback cache for environments without a running Redis server
_memory_cache: dict[str, Any] = {}
_redis_client: Optional[redis.Redis] = None


def get_redis_client() -> Optional[redis.Redis]:
    """Obtain or lazily instantiate the Redis client connection pool."""
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = redis.Redis.from_url(
                DEFAULT_REDIS_URL,
                decode_responses=True,
                socket_timeout=1.0,
                socket_connect_timeout=1.0,
            )
            # Ping to verify active socket connection
            _redis_client.ping()
            logger.info("Connected to Redis cache at %s", DEFAULT_REDIS_URL)
        except Exception as exc:
            logger.debug("Redis unavailable (%s); using in-memory cache fallback.", exc)
            _redis_client = None
    return _redis_client


def is_redis_available() -> bool:
    """Check if the Redis service is accessible."""
    try:
        client = get_redis_client()
        return bool(client and client.ping())
    except Exception:
        return False


def generate_cache_key(prefix: str, *args: Any, **kwargs: Any) -> str:
    """Generate a deterministic MD5-based cache key from arbitrary arguments."""
    payload = {
        "args": [str(a) for a in args],
        "kwargs": {k: str(v) for k, v in sorted(kwargs.items())},
    }
    raw = json.dumps(payload, sort_keys=True)
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()
    return f"quantpulse:{prefix}:{digest}"


def get_cached_value(key: str) -> Optional[Any]:
    """Retrieve a cached value from Redis or memory fallback."""
    client = get_redis_client()
    if client:
        try:
            cached = client.get(key)
            if cached is not None:
                return json.loads(cached)
        except Exception as exc:
            logger.debug("Redis get failed for %s (%s); trying memory fallback.", key, exc)

    return _memory_cache.get(key)


def set_cached_value(key: str, value: Any, ttl: int = CACHE_DEFAULT_TTL) -> None:
    """Store a value in Redis or memory fallback with time-to-live."""
    client = get_redis_client()
    serialized = json.dumps(value, default=str)
    if client:
        try:
            client.setex(key, ttl, serialized)
            return
        except Exception as exc:
            logger.debug("Redis set failed for %s (%s); storing in memory.", key, exc)

    _memory_cache[key] = value


def delete_cached_value(key: str) -> bool:
    """Remove a specific key from cache."""
    client = get_redis_client()
    deleted = False
    if client:
        try:
            deleted = bool(client.delete(key))
        except Exception:
            pass
    if key in _memory_cache:
        del _memory_cache[key]
        deleted = True
    return deleted


def clear_cache(pattern: str = "quantpulse:*") -> int:
    """Clear keys matching pattern from Redis and in-memory fallback."""
    client = get_redis_client()
    count = 0
    if client:
        try:
            keys = client.keys(pattern)
            if keys:
                count = client.delete(*keys)
        except Exception:
            pass
    _memory_cache.clear()
    return count


def cache_response(prefix: str, ttl: int = CACHE_DEFAULT_TTL):
    """Decorator to cache function outputs based on arguments."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = generate_cache_key(prefix, *args, **kwargs)
            cached_data = get_cached_value(cache_key)
            if cached_data is not None:
                return cached_data

            result = func(*args, **kwargs)
            # Support Pydantic models or plain dicts/primitives
            cacheable_data = result.model_dump() if hasattr(result, "model_dump") else result
            set_cached_value(cache_key, cacheable_data, ttl=ttl)
            return result

        return wrapper
    return decorator
