"""
State manager — tracks open positions and order states in memory.
Reconciles with BingX exchange state periodically.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    position_side: str       # "LONG" | "SHORT"
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    strategy_name: str
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    partial_profit_taken: bool = False
    trailing_stop_active: bool = False
    exchange_position_id: Optional[str] = None
    sl_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    # Fill price tracking (H-7)
    requested_price: Optional[float] = None
    fill_price: Optional[float] = None
    slippage_bps: float = 0.0

    @property
    def open_hours(self) -> float:
        delta = datetime.now(timezone.utc) - self.opened_at
        return delta.total_seconds() / 3600

    def unrealized_pnl(self, current_price: float) -> float:
        if self.position_side == "LONG":
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

    def risk_usdt(self) -> float:
        return abs(self.entry_price - self.stop_loss) * self.quantity


def recalculate_sl_tp(
    fill_price: float,
    requested_price: float,
    original_sl: float,
    original_tp: float,
    position_side: str,
) -> tuple[float, float]:
    """
    Recalculate SL and TP relative to the actual fill price.
    Preserves the absolute distance from the original requested price.
    """
    sl_distance = abs(requested_price - original_sl)
    tp_distance = abs(requested_price - original_tp)

    if position_side == "LONG":
        new_sl = fill_price - sl_distance
        new_tp = fill_price + tp_distance
    else:
        new_sl = fill_price + sl_distance
        new_tp = fill_price - tp_distance

    return new_sl, new_tp


def compute_slippage_bps(fill_price: float, requested_price: float) -> float:
    """Compute slippage in basis points (always positive)."""
    if requested_price <= 0:
        return 0.0
    return abs(fill_price - requested_price) / requested_price * 10_000


class StateManager:
    """In-memory state store for positions."""

    def __init__(self) -> None:
        self._positions: Dict[str, Position] = {}  # key = symbol_side

    def open_position(self, pos: Position) -> None:
        key = f"{pos.symbol}_{pos.position_side}"
        if key in self._positions:
            logger.warning("Position already open: %s", key)
        self._positions[key] = pos
        logger.info(
            "Position opened: %s %s entry=%.4f qty=%.4f",
            pos.symbol, pos.position_side, pos.entry_price, pos.quantity,
        )

    def open_position_with_fill(
        self,
        pos: Position,
        fill_price: float,
        requested_price: float,
    ) -> Position:
        """
        Open a position using the real fill price.
        Recalculates SL/TP relative to fill_price and logs slippage.
        """
        slippage = compute_slippage_bps(fill_price, requested_price)
        new_sl, new_tp = recalculate_sl_tp(
            fill_price, requested_price, pos.stop_loss, pos.take_profit, pos.position_side,
        )

        pos.requested_price = requested_price
        pos.fill_price = fill_price
        pos.entry_price = fill_price
        pos.slippage_bps = slippage
        pos.stop_loss = new_sl
        pos.take_profit = new_tp

        logger.info(
            "Fill price adjustment: %s %s requested=%.4f fill=%.4f "
            "slippage=%.2f bps new_SL=%.4f new_TP=%.4f",
            pos.symbol, pos.position_side,
            requested_price, fill_price, slippage, new_sl, new_tp,
        )

        self.open_position(pos)
        return pos

    def close_position(self, symbol: str, position_side: str) -> Optional[Position]:
        key = f"{symbol}_{position_side}"
        pos = self._positions.pop(key, None)
        if pos:
            logger.info("Position closed: %s %s", symbol, position_side)
        return pos

    def update_stop_loss(self, symbol: str, position_side: str, new_sl: float) -> None:
        key = f"{symbol}_{position_side}"
        if key in self._positions:
            self._positions[key].stop_loss = new_sl
            logger.debug("SL updated: %s → %.4f", key, new_sl)

    def get_position(self, symbol: str, position_side: str) -> Optional[Position]:
        return self._positions.get(f"{symbol}_{position_side}")

    def all_positions(self) -> List[Position]:
        return list(self._positions.values())

    def open_symbols(self) -> List[str]:
        return list({pos.symbol for pos in self._positions.values()})

    def total_open_risk_usdt(self) -> float:
        return sum(pos.risk_usdt() for pos in self._positions.values())

    def count(self) -> int:
        return len(self._positions)
