"""
Token-bucket rate limiter respecting BingX API limits.
"""
import asyncio
import time
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """Async token-bucket rate limiter."""

    def __init__(self, rate: float, capacity: int) -> None:
        """
        Args:
            rate: tokens added per second
            capacity: max tokens (burst capacity)
        """
        self._rate = rate
        self._capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        async with self._lock:
            await self._refill()
            while self._tokens < tokens:
                wait = (tokens - self._tokens) / self._rate
                logger.debug("Rate limit: waiting %.2fs", wait)
                await asyncio.sleep(wait)
                await self._refill()
            self._tokens -= tokens

    async def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now


# Singleton limiters for BingX
ORDER_LIMITER = RateLimiter(rate=10, capacity=10)    # 10 orders/sec
REQUEST_LIMITER = RateLimiter(rate=200, capacity=200) # 2000/10s ≈ 200/s
