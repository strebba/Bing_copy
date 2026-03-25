"""
Core risk engine — pre-trade checks, in-trade management, portfolio controls.
"""
import logging
import time
from typing import Any, Dict, List, Optional

from config import settings
from risk.drawdown_monitor import DrawdownMonitor
from risk.position_sizer import PositionSizer
from strategy.base_strategy import Signal, SignalDirection

logger = logging.getLogger(__name__)

# How long (seconds) to trust the locally-cached position count before
# hitting the exchange again for reconciliation.
_POSITION_CACHE_TTL = 5.0


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
        exchange_client: Optional[Any] = None,
        state_manager: Optional[Any] = None,
    ) -> None:
        self._dd = dd_monitor
        self._sizer = position_sizer
        self._open_positions: Dict = open_positions or {}
        self._client = exchange_client
        self._state = state_manager
        # Monotonic timestamp of the last successful reconciliation with exchange
        self._last_reconcile: float = 0.0

    # ── Position reconciliation ───────────────────────────────────────────────

    async def _reconcile_positions_if_stale(self) -> None:
        """
        Fetch real open positions from BingX and update local state if the
        cached data is older than _POSITION_CACHE_TTL seconds.

        Removes ghost positions (local but not on exchange) and refreshes
        self._open_positions so the position-count guard is accurate even
        after a bot restart or connectivity gap.
        """
        if self._client is None or self._state is None:
            # No exchange wiring — fall back to whatever is in _open_positions
            return

        now = time.monotonic()
        if now - self._last_reconcile < _POSITION_CACHE_TTL:
            # Cache is still fresh — skip the API call
            return

        try:
            exchange_positions = await self._client.get_positions()

            # Build the set of keys that actually exist on the exchange
            real_keys = {
                f"{p['symbol']}_{p['positionSide']}"
                for p in exchange_positions
                if abs(float(p.get("positionAmt", 0))) > 0
            }

            # Remove ghost positions from state_manager
            local_keys = {
                f"{pos.symbol}_{pos.position_side}"
                for pos in self._state.all_positions()
            }
            for ghost_key in local_keys - real_keys:
                sym, side = ghost_key.rsplit("_", 1)
                self._state.close_position(sym, side)
                logger.warning(
                    "RiskEngine: ghost position reconciled and removed: %s", ghost_key
                )

            # Keep _open_positions in sync with exchange reality
            self._open_positions = {k: {} for k in real_keys}
            self._last_reconcile = now

        except Exception as exc:
            logger.error(
                "RiskEngine: position reconciliation failed — using cached data: %s",
                exc,
            )

    # ── Level 1 — Pre-trade ───────────────────────────────────────────────────

    async def approve_signal(
        self,
        signal: Signal,
        equity: float,
        volume_24h: float,
        current_spread_pct: float,
        existing_positions_risk_usdt: float,
    ) -> tuple[bool, str]:
        """
        Return (approved, reason).
        Reconciles positions with exchange (with 5-second TTL cache) before
        applying all pre-trade risk rules.
        """
        # ── Reconcile FIRST so the position count is authoritative ────────────
        await self._reconcile_positions_if_stale()

        # Circuit breaker halt
        if self._dd.is_halted():
            return False, "Circuit breaker halted"

        # Position count — prefer state_manager count (post-reconciliation)
        position_count = (
            self._state.count()
            if self._state is not None
            else len(self._open_positions)
        )
        if position_count >= settings.MAX_OPEN_POSITIONS:
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

        # TP must exist and be on the correct side of entry
        if signal.take_profit <= 0:
            return False, "TP price missing (take_profit must be > 0)"

        if signal.direction == SignalDirection.LONG:
            if signal.take_profit <= signal.entry_price:
                return (
                    False,
                    f"TP must be above entry for LONG "
                    f"(tp={signal.take_profit}, entry={signal.entry_price})",
                )
        else:
            if signal.take_profit >= signal.entry_price:
                return (
                    False,
                    f"TP must be below entry for SHORT "
                    f"(tp={signal.take_profit}, entry={signal.entry_price})",
                )

        # Minimum R:R ratio
        if signal.risk_reward < 1.5:
            return False, f"R:R ratio too low ({signal.risk_reward:.2f} < 1.5)"

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
