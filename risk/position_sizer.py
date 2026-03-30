"""
Position sizing — Fractional Kelly and Fixed Fractional methods.

All financial calculations use Decimal to prevent floating-point errors
that can cause incorrect position sizes in production trading.
"""

import logging
from decimal import Decimal
from typing import Optional, Union

from config import settings
from core.finance import (
    calculate_quantity_from_risk,
    round_money,
    round_quantity,
    to_decimal,
)

logger = logging.getLogger(__name__)


class PositionSizer:
    """Calculate position size in USDT based on risk rules."""

    def __init__(
        self,
        max_risk_pct: float = settings.RISK_PER_TRADE,
        kelly_fraction: float = 0.5,
        max_single_pos_pct: float = settings.MAX_SINGLE_POSITION_PCT,
    ) -> None:
        self._max_risk_pct = to_decimal(max_risk_pct)
        self._kelly_fraction = to_decimal(kelly_fraction)
        self._max_single_pos_pct = to_decimal(max_single_pos_pct)

    def kelly_position_size(
        self,
        equity: Union[float, Decimal],
        win_rate: Union[float, Decimal],
        avg_win: Union[float, Decimal],
        avg_loss: Union[float, Decimal],
        override_risk_pct: Optional[Union[float, Decimal]] = None,
    ) -> float:
        """
        Half-Kelly position size (risk amount in USDT).
        f* = (p*b - q) / b, then apply fraction and hard cap.

        Uses Decimal for precision in all calculations.
        """
        eq = to_decimal(equity)
        wr = to_decimal(win_rate)
        aw = to_decimal(avg_win)
        al = to_decimal(avg_loss)

        if al <= 0 or eq <= 0:
            return 0.0

        b = aw / al
        q = Decimal("1") - wr
        kelly_full = (wr * b - q) / b
        kelly_applied = max(kelly_full * self._kelly_fraction, Decimal("0"))

        override = to_decimal(override_risk_pct) if override_risk_pct else None
        risk_pct = min(
            kelly_applied,
            override or self._max_risk_pct,
            self._max_single_pos_pct,
        )

        result = eq * risk_pct
        return round_money(result)

    def fixed_fractional_size(
        self,
        equity: Union[float, Decimal],
        risk_pct: Optional[Union[float, Decimal]] = None,
    ) -> float:
        """Fixed fraction of equity to risk per trade."""
        eq = to_decimal(equity)
        rp = to_decimal(risk_pct) if risk_pct else None

        pct = min(rp or self._max_risk_pct, self._max_single_pos_pct)
        result = eq * max(pct, Decimal("0"))
        return round_money(result)

    def quantity_from_risk(
        self,
        risk_amount_usdt: Union[float, Decimal],
        entry_price: Union[float, Decimal],
        stop_loss_price: Union[float, Decimal],
        leverage: int = 1,
    ) -> float:
        """
        Convert a USDT risk amount to quantity (contracts/coins).
        quantity = risk_amount / |entry - stop_loss| * leverage

        Uses Decimal precision and returns quantity rounded to exchange precision.
        """
        return calculate_quantity_from_risk(
            risk_amount=risk_amount_usdt,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            leverage=leverage,
        )

    def apply_size_scalar(
        self,
        base_size: Union[float, Decimal],
        scalar: Union[float, Decimal],
    ) -> float:
        """Apply a drawdown or regime scalar to a base size."""
        base = to_decimal(base_size)
        scal = to_decimal(scalar)
        result = max(base * scal, Decimal("0"))
        return round_money(result)
