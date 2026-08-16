import logging
import time
from collections.abc import Callable
from functools import wraps


def timed(label: str, logger: logging.Logger | None = None) -> Callable:
    """Decorator that logs entry, exit, and wall-clock duration of a function call."""

    def decorator(fn: Callable) -> Callable:
        _logger = logger or logging.getLogger(fn.__module__)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            _logger.info("%s: started", label)
            t0 = time.time()
            try:
                return fn(*args, **kwargs)
            finally:
                _logger.info("%s: completed in %.3fs", label, time.time() - t0)

        @wraps(fn)
        async def async_wrapper(*args, **kwargs):
            _logger.info("%s: started", label)
            t0 = time.time()
            try:
                return await fn(*args, **kwargs)
            finally:
                _logger.info("%s: completed in %.3fs", label, time.time() - t0)

        import asyncio

        return async_wrapper if asyncio.iscoroutinefunction(fn) else wrapper

    return decorator
