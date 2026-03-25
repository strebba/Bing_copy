"""
Core risk engine — pre-trade checks, in-trade management, portfolio controls.
"""
import logging
from typing import Dict, List, Optional, Tuple

from config import settings
from risk.correlation_guard import CorrelationGuard
from risk.drawdown_monitor import DrawdownMonitor
from risk.position_sizer import PositionSizer
from strategy.base_strategy import Signal, SignalDirection

logger = logging.getLogger(__name__)


class RiskEngine:
    """
    Defense-in-depth risk management:
      Level 1 — pre-trade checks
      Level 2 — in-trade monitoring (handled by engine via this class)
      Level 3 — portfolio controls (daily/weekly loss, open risk)
      Level 4 — circuit breakers (via DrawdownMonitor)
    """

    def __init__(
        self,
        dd_monitor: DrawdownMonitor,
        position_sizer: PositionSizer,
        open_positions: Optional[Dict] = None,
        correlation_guard: Optional[CorrelationGuard] = None,
    ) -> None:
        self._dd = dd_monitor
        self._sizer = position_sizer
        self._open_positions: Dict = open_positions or {}
        self._corr_guard: CorrelationGuard = correlation_guard or CorrelationGuard()

    # ── Level 1 — Full async pre-trade pipeline (with correlation) ────────────

    async def pre_trade_check(
        self,
        signal: Signal,
        equity: float,
        volume_24h: float,
        current_spread_pct: float,
        existing_positions_risk_usdt: float,
        open_positions: Optional[list] = None,
    ) -> Tuple[bool, str]:
        """
        Full async pre-trade check pipeline:
          1. Position count
          2. Daily / weekly loss limits
          3. Liquidity & spread checks
          4. Open risk cap
          5. Correlation guard  ← H-3

        Returns (approved, reason).
        """
        # Checks 1-4 via synchronous approve_signal
        approved, reason = self.approve_signal(
            signal=signal,
            equity=equity,
            volume_24h=volume_24h,
            current_spread_pct=current_spread_pct,
            existing_positions_risk_usdt=existing_positions_risk_usdt,
        )
        if not approved:
            return False, reason

        # Check 5 — Correlation guard
        if open_positions:
            is_ok, corr_reason = await self._corr_guard.check(
                new_pair=signal.symbol,
                new_direction=signal.direction.value,
                open_positions=open_positions,
            )
            if not is_ok:
                return False, f"Correlation guard rejected: {corr_reason}"

        return True, "OK"

    # ── Level 1 — Pre-trade ───────────────────────────────────────────────────

    def approve_signal(
        self,
        signal: Signal,
        equity: float,
        volume_24h: float,
        current_spread_pct: float,
        existing_positions_risk_usdt: float,
    ) -> tuple[bool, str]:
        """
        Return (approved, reason).
        Checks all pre-trade risk rules before order submission.
        """
        # Circuit breaker halt
        if self._dd.is_halted():
            return False, "Circuit breaker halted"

        # Position count
        if len(self._open_positions) >= settings.MAX_OPEN_POSITIONS:
            return False, f"Max open positions reached ({settings.MAX_OPEN_POSITIONS})"

        # Daily / weekly loss limits
        daily_pnl = self._dd.daily_pnl_pct()
        if daily_pnl <= -settings.DAILY_LOSS_LIMIT_PCT:
            return False, f"Daily loss limit hit ({daily_pnl:.1%})"

        weekly_pnl = self._dd.weekly_pnl_pct()
        if weekly_pnl <= -settings.WEEKLY_LOSS_LIMIT_PCT:
            return False, f"Weekly loss limit hit ({weekly_pnl:.1%})"

        # Liquidity check
        if volume_24h < settings.MIN_LIQUIDITY_24H_USD:
            return False, f"Insufficient liquidity ({volume_24h:.0f} USDT)"

        # Spread check
        if current_spread_pct > settings.MAX_SPREAD_PCT:
            return False, f"Spread too wide ({current_spread_pct:.4%})"

        # Open risk cap
        max_open_risk = equity * settings.MAX_OPEN_RISK_PCT
        if existing_positions_risk_usdt >= max_open_risk:
            return False, f"Max open risk exceeded ({existing_positions_risk_usdt:.0f} USDT)"

        return True, "OK"

    def calculate_order_size(
        self,
        signal: Signal,
        equity: float,
        win_rate: float = 0.55,
        avg_win: float = 1.5,
        avg_loss: float = 1.0,
    ) -> float:
        """Calculate final risk amount in USDT after all scalars."""
        # Base size from Kelly
        base_risk = self._sizer.kelly_position_size(
            equity=equity,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            override_risk_pct=signal.risk_pct,
        )

        # Apply circuit breaker scalar
        scalar = self._dd.size_multiplier()

        # Apply equity curve filter: if equity below 20-period SMA, reduce 50 %
        # (equity history not available here — caller should pass reduced scalar)

        return self._sizer.apply_size_scalar(base_risk, scalar)

    # ── Level 2 — In-trade ────────────────────────────────────────────────────

    def check_trailing_stop(
        self,
        position_side: str,
        entry_price: float,
        current_price: float,
        atr_value: float,
        atr_mult: float = 2.0,
        partial_profit_taken: bool = False,
    ) -> Optional[float]:
        """
        Return a new stop loss price if trailing stop should be updated, else None.
        Trailing activates after 1:1 R:R is reached.
        """
        if position_side == "LONG":
            rr_target = entry_price + atr_value   # 1:1 level
            if current_price >= rr_target and partial_profit_taken:
                return current_price - atr_mult * atr_value
        else:
            rr_target = entry_price - atr_value
            if current_price <= rr_target and partial_profit_taken:
                return current_price + atr_mult * atr_value
        return None

    def check_time_stop(
        self,
        position_open_hours: float,
        max_hold_hours: float = 48.0,
    ) -> bool:
        """Return True if the position should be closed due to time."""
        return position_open_hours >= max_hold_hours

    # ── Level 3 — Portfolio ───────────────────────────────────────────────────

    def compute_var(
        self,
        equity: float,
        confidence: float = 0.95,
        daily_vol: float = 0.02,
    ) -> float:
        """Parametric VaR at given confidence level (assumes normal distribution)."""
        z_map = {0.95: 1.645, 0.99: 2.326}
        z = z_map.get(confidence, 1.645)
        return equity * daily_vol * z

    def total_open_risk_usdt(self, positions: List[Dict]) -> float:
        """Sum of (entry - stop_loss) * quantity for all open positions."""
        total = 0.0
        for pos in positions:
            entry = pos.get("entry_price", 0.0)
            sl = pos.get("stop_loss", 0.0)
            qty = pos.get("quantity", 0.0)
            total += abs(entry - sl) * qty
        return total
