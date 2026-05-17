"""
Retry decorator for Ollama (and other) calls.

Usage:
    from shared.retry import retry

    @retry(max_retries=3, backoff=2.0)
    def my_ollama_call(...):
        ...
"""

import time
from functools import wraps

from shared.log import get_logger

logger = get_logger("retry")


def retry(max_retries: int = 3, backoff: float = 2.0):
    """
    Decorator that retries a function up to *max_retries* times with
    exponential backoff on any exception.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = backoff * (2 ** attempt)
                        logger.warning(
                            "[%s] Attempt %d/%d failed: %s — retrying in %.1fs…",
                            func.__name__,
                            attempt + 1,
                            max_retries,
                            e,
                            wait,
                        )
                        time.sleep(wait)
                    else:
                        logger.error(
                            "[%s] All %d attempts failed. Last error: %s",
                            func.__name__,
                            max_retries,
                            e,
                        )
                        raise

        return wrapper

    return decorator
