"""
Market data layer — fetches OHLCV from BingX, caches it, and computes indicators.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional

import pandas as pd

from exchange.bingx_client import BingXClient
from strategy.indicators import atr, rsi

logger = logging.getLogger(__name__)

# BingX interval mapping
INTERVAL_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1H": "1h",
    "2H": "2h",
    "4H": "4h",
    "6H": "6h",
    "8H": "8h",
    "12H": "12h",
    "1D": "1d",
    "3D": "3d",
    "1W": "1w",
}


class MarketDataManager:
    """
    Manages OHLCV cache per symbol/timeframe pair.
    Fetches historical data on startup, then updates from WebSocket or REST.
    """

    def __init__(self, client: BingXClient, max_candles: int = 500) -> None:
        self._client = client
        self._max_candles = max_candles
        # Cache: {symbol: {interval: DataFrame}}
        self._cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._last_updated: Dict[str, float] = {}

    async def initialize(
        self, symbols: List[str], intervals: List[str] = ["1H", "4H"]
    ) -> None:
        """Fetch historical OHLCV for all symbols × intervals on startup."""
        tasks = [
            self._fetch_and_cache(sym, interval)
            for sym in symbols
            for interval in intervals
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Market data initialized for %d symbols", len(symbols))

    async def _fetch_and_cache(self, symbol: str, interval: str) -> None:
        try:
            bingx_interval = INTERVAL_MAP.get(interval, interval)
            raw = await self._client.get_klines(
                symbol, bingx_interval, limit=self._max_candles
            )
            df = self._parse_klines(raw, interval)
            self._cache.setdefault(symbol, {})[interval] = df
            self._last_updated[f"{symbol}_{interval}"] = time.time()
            logger.debug("Cached %d candles for %s %s", len(df), symbol, interval)
        except Exception as exc:
            logger.error("Failed to fetch %s %s: %s", symbol, interval, exc)

    @staticmethod
    def _parse_klines(raw: List, timeframe: str = "1H") -> pd.DataFrame:
        """Convert raw BingX klines list to DataFrame."""
        if not raw:
            return pd.DataFrame()

        rows = []
        for k in raw:
            # Demo mode returns dict, production returns list
            if isinstance(k, dict):
                rows.append(
                    {
                        "timestamp": int(k.get("time", k.get("openTime", 0))),
                        "open": float(k.get("open", 0)),
                        "high": float(k.get("high", 0)),
                        "low": float(k.get("low", 0)),
                        "close": float(k.get("close", 0)),
                        "volume": float(k.get("volume", 0)),
                    }
                )
            else:
                # Production format: [open_time, open, high, low, close, volume, ...]
                rows.append(
                    {
                        "timestamp": int(k[0]),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                    }
                )
        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        df.attrs["timeframe"] = timeframe
        return df

    def get_df(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """Return cached DataFrame for symbol/interval, or None if not ready."""
        return self._cache.get(symbol, {}).get(interval)

    def update_candle(self, symbol: str, interval: str, candle: dict) -> None:
        """Update the last candle in the cache from a WebSocket tick."""
        cache = self._cache.get(symbol, {})
        df = cache.get(interval)
        if df is None or df.empty:
            return

        ts = int(candle.get("t", 0))
        last_ts = int(df["timestamp"].iloc[-1])

        new_row = {
            "timestamp": ts,
            "open": float(candle.get("o", 0)),
            "high": float(candle.get("h", 0)),
            "low": float(candle.get("l", 0)),
            "close": float(candle.get("c", 0)),
            "volume": float(candle.get("v", 0)),
        }

        if ts == last_ts:
            # Update the existing last candle
            for col, val in new_row.items():
                df.at[df.index[-1], col] = val
        else:
            # Append new candle and trim to max size
            df = (
                pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                .tail(self._max_candles)
                .reset_index(drop=True)
            )

        df.attrs["timeframe"] = interval
        self._cache[symbol][interval] = df
        self._last_updated[f"{symbol}_{interval}"] = time.time()

    def is_stale(self, symbol: str, interval: str, threshold_s: float = 60.0) -> bool:
        key = f"{symbol}_{interval}"
        last = self._last_updated.get(key, 0.0)
        return (time.time() - last) > threshold_s

    async def refresh_if_stale(self, symbol: str, interval: str) -> None:
        if self.is_stale(symbol, interval):
            await self._fetch_and_cache(symbol, interval)
