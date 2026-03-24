"""
Order book processor — tracks bid/ask spread and depth.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class OrderBookProcessor:
    """Processes WebSocket depth updates and computes derived metrics."""

    def __init__(self, depth_levels: int = 20) -> None:
        self._depth_levels = depth_levels
        self._books: Dict[str, Dict] = {}   # symbol → {"bids": [...], "asks": [...]}

    def update(self, symbol: str, data: Dict[str, Any]) -> None:
        """Process a depth WebSocket message."""
        bids = sorted(
            [[float(p), float(q)] for p, q in data.get("bids", [])],
            key=lambda x: -x[0],
        )
        asks = sorted(
            [[float(p), float(q)] for p, q in data.get("asks", [])],
            key=lambda x: x[0],
        )
        self._books[symbol] = {"bids": bids[:self._depth_levels], "asks": asks[:self._depth_levels]}

    def best_bid(self, symbol: str) -> Optional[float]:
        book = self._books.get(symbol)
        if book and book["bids"]:
            return book["bids"][0][0]
        return None

    def best_ask(self, symbol: str) -> Optional[float]:
        book = self._books.get(symbol)
        if book and book["asks"]:
            return book["asks"][0][0]
        return None

    def mid_price(self, symbol: str) -> Optional[float]:
        bid = self.best_bid(symbol)
        ask = self.best_ask(symbol)
        if bid and ask:
            return (bid + ask) / 2
        return None

    def spread_pct(self, symbol: str) -> float:
        bid = self.best_bid(symbol)
        ask = self.best_ask(symbol)
        if bid and ask and bid > 0:
            return (ask - bid) / bid
        return 0.0

    def depth_imbalance(self, symbol: str, levels: int = 5) -> float:
        """
        Compute bid/ask imbalance at top `levels`.
        Positive → more bid pressure (bullish), negative → more ask pressure (bearish).
        """
        book = self._books.get(symbol, {})
        bids = book.get("bids", [])[:levels]
        asks = book.get("asks", [])[:levels]
        bid_vol = sum(q for _, q in bids)
        ask_vol = sum(q for _, q in asks)
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total
