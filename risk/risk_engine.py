"""
Core risk engine — pre-trade checks, in-trade management, portfolio controls.
Includes daily reset logic (H-8).
"""
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import settings
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
        event_bus: Optional[object] = None,
    ) -> None:
        self._dd = dd_monitor
        self._sizer = position_sizer
        self._open_positions: Dict = open_positions or {}
        self._event_bus = event_bus

        # Daily counters (H-8)
        now = datetime.now(timezone.utc)
        self._current_trading_day = now.date()
        self._daily_pnl = 0.0
        self._daily_trade_count = 0
        self._daily_loss_limit_hit = False
        self._previous_day_pnl = 0.0

    # ── Daily reset (H-8) ────────────────────────────────────────────────────

    def _check_daily_reset(self) -> None:
        """Check if trading day has changed and reset daily counters."""
        today = datetime.now(timezone.utc).date()
        if today != self._current_trading_day:
            logger.info(
                "Daily reset: %s → %s (previous day PnL: %.2f)",
                self._current_trading_day, today, self._daily_pnl,
            )
            self._previous_day_pnl = self._daily_pnl
            self._daily_pnl = 0.0
            self._daily_trade_count = 0
            self._daily_loss_limit_hit = False
            old_day = self._current_trading_day
            self._current_trading_day = today

            # Publish event for other modules (e.g., PerformanceTracker)
            if self._event_bus is not None:
                from core.event_bus import Event, EventType
                try:
                    self._event_bus.publish_nowait(Event(
                        type=EventType.HEARTBEAT,  # Re-use existing type for sync pub
                        data={
                            "_daily_reset": True,
                            "date": today.isoformat(),
                            "previous_date": old_day.isoformat(),
                            "previous_day_pnl": self._previous_day_pnl,
                        },
                    ))
                except Exception as exc:
                    logger.warning("Failed to publish daily reset event: %s", exc)

    def record_daily_pnl(self, pnl: float) -> None:
        """Record PnL from a closed trade for daily tracking."""
        self._check_daily_reset()
        self._daily_pnl += pnl
        self._daily_trade_count += 1

    @property
    def daily_pnl(self) -> float:
        self._check_daily_reset()
        return self._daily_pnl

    @property
    def daily_trade_count(self) -> int:
        self._check_daily_reset()
        return self._daily_trade_count

    @property
    def daily_loss_limit_hit(self) -> bool:
        self._check_daily_reset()
        return self._daily_loss_limit_hit

    @property
    def current_trading_day(self):
        return self._current_trading_day

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
        # Check daily reset first
        self._check_daily_reset()

        # Circuit breaker halt
        if self._dd.is_halted():
            return False, "Circuit breaker halted"

        # Daily loss limit from local counter
        if self._daily_loss_limit_hit:
            return False, "Daily loss limit hit (local counter)"

        # Position count
        if len(self._open_positions) >= settings.MAX_OPEN_POSITIONS:
            return False, f"Max open positions reached ({settings.MAX_OPEN_POSITIONS})"

        # Daily / weekly loss limits (from drawdown monitor)
        daily_pnl = self._dd.daily_pnl_pct()
        if daily_pnl <= -settings.DAILY_LOSS_LIMIT_PCT:
            self._daily_loss_limit_hit = True
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
