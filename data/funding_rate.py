"""
Funding rate tracker — polls BingX for current funding rates per symbol.
"""
import asyncio
import logging
import time
from typing import Dict, Optional

from exchange.bingx_client import BingXClient

logger = logging.getLogger(__name__)

POLL_INTERVAL_S = 300   # 5 minutes


class FundingRateTracker:
    """Maintains a cache of current funding rates for active symbols."""

    def __init__(self, client: BingXClient) -> None:
        self._client = client
        self._rates: Dict[str, float] = {}
        self._next_funding: Dict[str, int] = {}
        self._last_updated: Dict[str, float] = {}
        self._running = False

    async def start(self, symbols: list) -> None:
        self._running = True
        self._symbols = symbols
        # Initial fetch
        await self._poll_all()
        # Background loop
        while self._running:
            await asyncio.sleep(POLL_INTERVAL_S)
            await self._poll_all()

    async def stop(self) -> None:
        self._running = False

    async def _poll_all(self) -> None:
        tasks = [self._fetch_rate(sym) for sym in self._symbols]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _fetch_rate(self, symbol: str) -> None:
        try:
            data = await self._client.get_funding_rate(symbol)
            self._rates[symbol] = float(data.get("lastFundingRate", 0.0))
            self._next_funding[symbol] = int(data.get("nextFundingTime", 0))
            self._last_updated[symbol] = time.time()
            logger.debug("Funding rate %s: %.6f", symbol, self._rates[symbol])
        except Exception as exc:
            logger.warning("Funding rate fetch failed for %s: %s", symbol, exc)

    def get_rate(self, symbol: str) -> float:
        return self._rates.get(symbol, 0.0)

    def is_extreme(self, symbol: str, threshold: float = 0.001) -> tuple[bool, str]:
        """Return (is_extreme, direction) where direction is 'positive'/'negative'."""
        rate = self.get_rate(symbol)
        if rate > threshold:
            return True, "positive"
        if rate < -threshold:
            return True, "negative"
        return False, "neutral"
