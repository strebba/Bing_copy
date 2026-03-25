"""
Unit tests for critical fixes:
  C-1 — Position Count Guard with exchange reconciliation + cache TTL
  C-3 — asyncio.gather() Crash Propagation & supervised_gather
"""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from risk.drawdown_monitor import DrawdownMonitor
from risk.position_sizer import PositionSizer
from risk.risk_engine import RiskEngine, _POSITION_CACHE_TTL


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_engine(
    exchange_positions=None,
    local_position_count=0,
    client_raises=False,
):
    """
    Build a RiskEngine wired with mocked exchange client and state_manager.

    exchange_positions : list of dicts returned by client.get_positions()
    local_position_count : how many positions the state_manager reports
    client_raises : simulate network error in get_positions()
    """
    dd = DrawdownMonitor(10_000)
    sizer = PositionSizer()

    # Mock exchange client
    client = MagicMock()
    if client_raises:
        client.get_positions = AsyncMock(side_effect=RuntimeError("network error"))
    else:
        client.get_positions = AsyncMock(return_value=exchange_positions or [])

    # Mock state_manager
    state = MagicMock()
    state.count = MagicMock(return_value=local_position_count)
    state.all_positions = MagicMock(return_value=[])
    state.close_position = MagicMock()

    engine = RiskEngine(dd, sizer, exchange_client=client, state_manager=state)
    return engine, client, state


def _make_signal():
    signal = MagicMock()
    signal.risk_pct = 0.01
    return signal


def _make_5_exchange_positions():
    """5 positions with non-zero positionAmt."""
    return [
        {"symbol": f"BTC-USDT_{i}", "positionSide": "LONG", "positionAmt": "0.1"}
        for i in range(5)
    ]


# ── C-1: Position Count Guard ──────────────────────────────────────────────────

class TestPositionCountGuard:

    @pytest.mark.asyncio
    async def test_blocks_when_5_real_positions_and_max_5(self):
        """With 5 real positions on exchange and MAX=5, approve_signal must block."""
        exchange_pos = _make_5_exchange_positions()
        engine, client, state = _make_engine(
            exchange_positions=exchange_pos,
            local_position_count=5,   # state reflects real count after reconcile
        )

        # After reconciliation the state.count() returns 5
        state.count.return_value = 5

        with patch("risk.risk_engine.settings") as mock_settings:
            mock_settings.MAX_OPEN_POSITIONS = 5
            mock_settings.DAILY_LOSS_LIMIT_PCT = 0.03
            mock_settings.WEEKLY_LOSS_LIMIT_PCT = 0.05
            mock_settings.MIN_LIQUIDITY_24H_USD = 0
            mock_settings.MAX_SPREAD_PCT = 1.0
            mock_settings.MAX_OPEN_RISK_PCT = 1.0

            approved, reason = await engine.approve_signal(
                signal=_make_signal(),
                equity=10_000,
                volume_24h=1_000_000,
                current_spread_pct=0.0001,
                existing_positions_risk_usdt=0,
            )

        assert not approved
        assert "Max open positions" in reason

    @pytest.mark.asyncio
    async def test_stale_local_3_real_5_blocks(self):
        """
        Local state reports 3 positions (stale), exchange has 5.
        After reconciliation, guard must use the real count (5) and block.
        """
        exchange_pos = _make_5_exchange_positions()
        engine, client, state = _make_engine(
            exchange_positions=exchange_pos,
            local_position_count=3,   # stale local count BEFORE reconcile
        )

        # Simulate that reconciliation updates the count to the real value
        # (state.count is called AFTER reconcile so we set it to 5)
        state.count.return_value = 5

        with patch("risk.risk_engine.settings") as mock_settings:
            mock_settings.MAX_OPEN_POSITIONS = 5
            mock_settings.DAILY_LOSS_LIMIT_PCT = 0.03
            mock_settings.WEEKLY_LOSS_LIMIT_PCT = 0.05
            mock_settings.MIN_LIQUIDITY_24H_USD = 0
            mock_settings.MAX_SPREAD_PCT = 1.0
            mock_settings.MAX_OPEN_RISK_PCT = 1.0

            approved, reason = await engine.approve_signal(
                signal=_make_signal(),
                equity=10_000,
                volume_24h=1_000_000,
                current_spread_pct=0.0001,
                existing_positions_risk_usdt=0,
            )

        # Exchange was queried (cache was empty → stale)
        client.get_positions.assert_awaited_once()
        assert not approved
        assert "Max open positions" in reason

    @pytest.mark.asyncio
    async def test_cache_ttl_no_api_call_within_5s(self):
        """If last reconciliation was < 5 s ago, exchange API must NOT be called again."""
        engine, client, state = _make_engine(
            exchange_positions=[],
            local_position_count=0,
        )
        state.count.return_value = 0

        # Prime the cache — first call triggers API
        await engine._reconcile_positions_if_stale()
        assert client.get_positions.await_count == 1

        # Second call within the TTL window — must reuse cache
        await engine._reconcile_positions_if_stale()
        assert client.get_positions.await_count == 1  # still 1, no second call

    @pytest.mark.asyncio
    async def test_cache_ttl_calls_api_after_expiry(self):
        """After TTL expires, the next call must hit the exchange again."""
        engine, client, state = _make_engine(
            exchange_positions=[],
            local_position_count=0,
        )

        # Artificially age the cache beyond TTL
        engine._last_reconcile = time.monotonic() - (_POSITION_CACHE_TTL + 1)

        await engine._reconcile_positions_if_stale()
        assert client.get_positions.await_count == 1

    @pytest.mark.asyncio
    async def test_reconciliation_removes_ghost_positions(self):
        """Ghost positions (local but not on exchange) are removed from state."""
        # Exchange has ZERO real positions
        engine, client, state = _make_engine(
            exchange_positions=[],
            local_position_count=2,
        )

        # Simulate two ghost positions in local state
        ghost1 = MagicMock()
        ghost1.symbol = "BTC-USDT"
        ghost1.position_side = "LONG"
        ghost2 = MagicMock()
        ghost2.symbol = "ETH-USDT"
        ghost2.position_side = "SHORT"
        state.all_positions.return_value = [ghost1, ghost2]

        await engine._reconcile_positions_if_stale()

        # Both ghost positions must have been closed in state_manager
        assert state.close_position.call_count == 2

    @pytest.mark.asyncio
    async def test_reconciliation_failure_uses_cached_data(self):
        """On API error, the engine falls back to cached data without crashing."""
        engine, client, state = _make_engine(client_raises=True, local_position_count=2)
        state.count.return_value = 2

        # Should not raise even if exchange is unreachable
        await engine._reconcile_positions_if_stale()

        # Cache timestamp must remain 0 (no successful reconcile)
        assert engine._last_reconcile == 0.0

    @pytest.mark.asyncio
    async def test_no_client_skips_reconciliation(self):
        """Without exchange_client, reconciliation is silently skipped."""
        dd = DrawdownMonitor(10_000)
        sizer = PositionSizer()
        engine = RiskEngine(dd, sizer)   # no client, no state

        # Must not raise
        await engine._reconcile_positions_if_stale()


# ── C-3: Supervised Gather / Crash Propagation ────────────────────────────────

class TestSupervisedGather:
    """
    Tests for TradingEngine._supervised_gather().
    We instantiate TradingEngine with all dependencies mocked to avoid
    real network calls.
    """

    def _build_engine(self):
        """Build a TradingEngine with all I/O dependencies stubbed out."""
        from core.engine import TradingEngine

        with (
            patch("core.engine.BingXClient"),
            patch("core.engine.OrderManager"),
            patch("core.engine.BingXWebSocket"),
            patch("core.engine.MarketDataStream"),
            patch("core.engine.MarketDataManager"),
            patch("core.engine.OrderBookProcessor"),
            patch("core.engine.FundingRateTracker"),
            patch("core.engine.DrawdownMonitor"),
            patch("core.engine.PositionSizer"),
            patch("core.engine.CorrelationGuard"),
            patch("core.engine.EmergencyStop"),
            patch("core.engine.PortfolioManager"),
            patch("core.engine.StateManager"),
            patch("core.engine.EventBus"),
            patch("core.engine.TelegramAlerter"),
            patch("core.engine.RiskEngine"),
        ):
            engine = TradingEngine()

        # Provide workable defaults for mocked attributes
        engine._emergency = MagicMock()
        engine._emergency.is_active = False
        engine._emergency.reason = ""
        engine._emergency.activate = MagicMock()
        engine._event_bus = MagicMock()
        engine._event_bus.publish = AsyncMock()
        engine._alerter = MagicMock()
        engine._alerter.send = AsyncMock(return_value=True)
        engine._alerter.close = AsyncMock()
        engine._client = MagicMock()
        engine._client.get_positions = AsyncMock(return_value=[])
        engine._client.close = AsyncMock()
        engine._order_mgr = MagicMock()
        engine._emergency.close_all_positions = AsyncMock(return_value=0)
        engine._ws = MagicMock()
        engine._ws.stop = AsyncMock()
        engine._funding = MagicMock()
        engine._funding.stop = AsyncMock()
        return engine

    @pytest.mark.asyncio
    async def test_critical_task_crash_triggers_emergency_stop(self):
        """A crash in a critical task must activate the emergency stop."""
        engine = self._build_engine()

        async def crashing_task():
            raise RuntimeError("websocket died")

        async def healthy_task():
            await asyncio.sleep(10)  # will be cancelled

        with pytest.raises(RuntimeError, match="websocket died"):
            await engine._supervised_gather(
                crashing_task(),
                healthy_task(),
                names=["market_data", "some_other"],
            )

        engine._emergency.activate.assert_called_once()
        call_arg = engine._emergency.activate.call_args[0][0]
        assert "market_data" in call_arg

    @pytest.mark.asyncio
    async def test_non_critical_task_crash_does_not_trigger_emergency_stop(self):
        """A crash in a non-critical task must NOT activate emergency stop."""
        engine = self._build_engine()

        async def crashing_task():
            raise ValueError("non-critical failure")

        with pytest.raises(ValueError, match="non-critical failure"):
            await engine._supervised_gather(
                crashing_task(),
                names=["heartbeat_loop"],
            )

        engine._emergency.activate.assert_not_called()

    @pytest.mark.asyncio
    async def test_pending_tasks_are_cancelled_after_crash(self):
        """When one task crashes, the remaining pending tasks must be cancelled."""
        engine = self._build_engine()
        cancelled_flag = asyncio.Event()

        async def crashing_task():
            raise RuntimeError("boom")

        async def long_running_task():
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                cancelled_flag.set()
                raise

        with pytest.raises(RuntimeError):
            await engine._supervised_gather(
                crashing_task(),
                long_running_task(),
                names=["strategy_loop", "equity_loop"],
            )

        assert cancelled_flag.is_set(), "Pending task was not cancelled"

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates_cleanly(self):
        """CancelledError from outside must cancel all tasks without hanging."""
        engine = self._build_engine()

        async def slow_task():
            await asyncio.sleep(100)

        async def run():
            await engine._supervised_gather(
                slow_task(),
                slow_task(),
                names=["t1", "t2"],
            )

        task = asyncio.create_task(run())
        await asyncio.sleep(0.05)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_graceful_shutdown_sends_telegram_alert(self):
        """_graceful_shutdown must attempt to send a Telegram crash alert."""
        engine = self._build_engine()
        await engine._graceful_shutdown()
        engine._alerter.send.assert_awaited_once()
        msg = engine._alerter.send.call_args[0][0]
        assert "emergency stop" in msg.lower() or "crashed" in msg.lower()

    @pytest.mark.asyncio
    async def test_graceful_shutdown_closes_ws_and_client(self):
        """_graceful_shutdown must close WebSocket and REST connections."""
        engine = self._build_engine()
        await engine._graceful_shutdown()
        engine._ws.stop.assert_awaited_once()
        engine._funding.stop.assert_awaited_once()
        engine._client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_graceful_shutdown_attempts_position_close(self):
        """_graceful_shutdown must attempt to close all open positions."""
        engine = self._build_engine()
        await engine._graceful_shutdown()
        engine._client.get_positions.assert_awaited_once()
        engine._emergency.close_all_positions.assert_awaited_once()
