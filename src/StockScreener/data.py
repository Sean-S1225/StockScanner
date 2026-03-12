import yfinance as yf
import random
import time

try:
    from yfinance.exceptions import YFRateLimitError
except Exception:
    YFRateLimitError = None

def is_rate_limited(exc: Exception) -> bool:
    """Does the exception look to be a rate-limiting issue?

    Args:
        exc: The exception received

    Returns:
        True if the exception seems to be rate-limited related
    """

    msg = str(exc).lower()
    return (
        (YFRateLimitError is not None and isinstance(exc, YFRateLimitError))
        or "too many requests" in msg
        or "rate limited" in msg
        or "429" in msg
    )

def call_with_backoff(fn, *args, max_retries=5, base_sleep=2.0, max_sleep=60.0, **kwargs):
    """
    Retry only when the failure looks like a rate limit.
    Uses exponential backoff with a little jitter.
    """
    
    sleep_s = base_sleep

    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)

        except Exception as e:
            if not is_rate_limited(e):
                raise

            if attempt == max_retries:
                raise

            # jitter in [0.8, 1.2]
            jitter = random.uniform(0.8, 1.2)
            wait = min(sleep_s * jitter, max_sleep)
            print(f"Rate limited. Sleeping {wait:.1f}s before retry...")
            time.sleep(wait)
            sleep_s = min(sleep_s * 2, max_sleep)