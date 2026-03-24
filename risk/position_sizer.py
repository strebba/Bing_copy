"""
Position sizing — Fractional Kelly and Fixed Fractional methods.
"""
import logging
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)


class PositionSizer:
    """Calculate position size in USDT based on risk rules."""

    def __init__(
        self,
        max_risk_pct: float = settings.RISK_PER_TRADE,
        kelly_fraction: float = 0.5,
        max_single_pos_pct: float = settings.MAX_SINGLE_POSITION_PCT,
    ) -> None:
        self._max_risk_pct = max_risk_pct
        self._kelly_fraction = kelly_fraction
        self._max_single_pos_pct = max_single_pos_pct

    def kelly_position_size(
        self,
        equity: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        override_risk_pct: Optional[float] = None,
    ) -> float:
        """
        Half-Kelly position size (risk amount in USDT).
        f* = (p*b - q) / b, then apply fraction and hard cap.
        """
        if avg_loss <= 0 or equity <= 0:
            return 0.0
        b = avg_win / avg_loss
        q = 1.0 - win_rate
        kelly_full = (win_rate * b - q) / b
        kelly_applied = max(kelly_full * self._kelly_fraction, 0.0)
        risk_pct = min(
            kelly_applied,
            override_risk_pct or self._max_risk_pct,
            self._max_single_pos_pct,
        )
        return equity * risk_pct

    def fixed_fractional_size(
        self,
        equity: float,
        risk_pct: Optional[float] = None,
    ) -> float:
        """Fixed fraction of equity to risk per trade."""
        pct = min(risk_pct or self._max_risk_pct, self._max_single_pos_pct)
        return equity * max(pct, 0.0)

    def quantity_from_risk(
        self,
        risk_amount_usdt: float,
        entry_price: float,
        stop_loss_price: float,
        leverage: int = 1,
    ) -> float:
        """
        Convert a USDT risk amount to quantity (contracts/coins).
        quantity = risk_amount / |entry - stop_loss| * leverage
        """
        price_risk = abs(entry_price - stop_loss_price)
        if price_risk <= 0:
            return 0.0
        return risk_amount_usdt / price_risk

    def apply_size_scalar(self, base_size: float, scalar: float) -> float:
        """Apply a drawdown or regime scalar to a base size."""
        return max(base_size * scalar, 0.0)
