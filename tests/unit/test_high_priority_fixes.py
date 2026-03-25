"""
Unit tests for high-priority fixes:
  H-1 — EmergencyStop de-duplication (asyncio.Lock + _triggered flag)
  H-4 — Circuit Breaker Event Publishing (DrawdownMonitor → EventBus →
         PortfolioManager / TelegramAlerter)
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.event_bus import Event, EventBus, EventType
from risk.drawdown_monitor import CircuitBreakerLevel, DrawdownMonitor
from risk.emergency_stop import EmergencyStop


# ═══════════════════════════════════════════════════════════════════════════════
# H-1 — EmergencyStop De-duplication
# ═══════════════════════════════════════════════════════════════════════════════

class TestEmergencyStopDeduplication:
    """
    Verify that concurrent trigger() calls are safe:
      - only the first caller executes close-all
      - subsequent callers within the same cycle are silently discarded
      - reset() re-arms the stop for a new cycle
      - the Lock never deadlocks
    """

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_order_manager(n_positions: int = 3):
        """Return a mock order_manager and a matching position list."""
        om = MagicMock()
        om.close_position = AsyncMock(return_value={"status": "ok"})
        positions = [
            {"symbol": f"BTC-USDT_{i}", "positionSide": "LONG", "positionAmt": "0.1"}
            for i in range(n_positions)
        ]
        return om, positions

    # ── Tests ──────────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_first_trigger_sets_flag_and_closes_positions(self):
        """trigger() sets _triggered and closes all open positions."""
        es = EmergencyStop()
        om, positions = self._make_order_manager(3)

        closed = await es.trigger("test reason", order_manager=om, open_positions=positions)

        assert es.is_active is True
        assert es._triggered is True
        assert es._last_triggered is not None
        assert es.reason == "test reason"
        assert closed == 3
        assert om.close_position.await_count == 3

    @pytest.mark.asyncio
    async def test_three_concurrent_triggers_only_first_executes_close_all(self):
        """
        Three concurrent trigger() calls must result in exactly ONE close-all
        execution.  The other two must be discarded with a warning.
        """
        es = EmergencyStop()
        om, positions = self._make_order_manager(2)

        results = await asyncio.gather(
            es.trigger("trigger-1", order_manager=om, open_positions=positions),
            es.trigger("trigger-2", order_manager=om, open_positions=positions),
            es.trigger("trigger-3", order_manager=om, open_positions=positions),
        )

        # Exactly one execution returns the real closed count; the others return 0
        assert sum(results) == 2          # 2 positions closed once
        assert results.count(0) == 2      # two duplicates discarded
        assert om.close_position.await_count == 2   # called only for the first trigger

    @pytest.mark.asyncio
    async def test_duplicate_trigger_without_positions_returns_zero(self):
        """Second trigger with no order_manager still returns 0 (not an error)."""
        es = EmergencyStop()
        await es.trigger("first")
        result = await es.trigger("second")
        assert result == 0

    @pytest.mark.asyncio
    async def test_after_reset_trigger_executes_again(self):
        """After reset() the stop is re-armed and the next trigger() runs close-all."""
        es = EmergencyStop()
        om, positions = self._make_order_manager(1)

        # First cycle
        await es.trigger("cycle-1", order_manager=om, open_positions=positions)
        assert es.is_active is True

        # Manual reset (operator action)
        await es.reset()
        assert es.is_active is False
        assert es._triggered is False

        # Second cycle — must execute normally
        closed = await es.trigger("cycle-2", order_manager=om, open_positions=positions)
        assert es.is_active is True
        assert closed == 1
        assert om.close_position.await_count == 2   # 1 per cycle × 2 cycles

    @pytest.mark.asyncio
    async def test_lock_does_not_deadlock_with_timeout(self):
        """
        trigger() must complete within a generous timeout even under concurrency.
        Deadlocks would cause this test to hang and then raise TimeoutError.
        """
        es = EmergencyStop()
        om, positions = self._make_order_manager(5)

        async def run():
            await asyncio.gather(*[
                es.trigger(f"reason-{i}", order_manager=om, open_positions=positions)
                for i in range(10)
            ])

        await asyncio.wait_for(run(), timeout=5.0)
        # Only the first trigger should have closed positions
        assert om.close_position.await_count == 5

    @pytest.mark.asyncio
    async def test_is_active_reflects_triggered_state(self):
        """is_active mirrors _triggered so callers can gate on it."""
        es = EmergencyStop()
        assert es.is_active is False
        await es.trigger("x")
        assert es.is_active is True
        await es.reset()
        assert es.is_active is False

    def test_activate_sync_backward_compat(self):
        """sync activate() still works for non-concurrent call sites."""
        es = EmergencyStop()
        es.activate("legacy caller")
        assert es.is_active is True
        assert es.reason == "legacy caller"

    def test_activate_does_not_re_trigger(self):
        """sync activate() on an already-active stop is a no-op."""
        es = EmergencyStop()
        es.activate("first")
        first_ts = es._last_triggered
        es.activate("second")
        assert es.reason == "first"
        assert es._last_triggered == first_ts   # timestamp unchanged


# ═══════════════════════════════════════════════════════════════════════════════
# H-4 — Circuit Breaker Event Publishing
# ═══════════════════════════════════════════════════════════════════════════════

class TestCircuitBreakerEventPublishing:
    """
    Verify that DrawdownMonitor publishes CIRCUIT_BREAKER_LEVEL_CHANGE events
    when the circuit breaker transitions between levels, and that downstream
    subscribers (PortfolioManager, TelegramAlerter) react correctly.
    """

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_bus_with_spy():
        """Return a real EventBus with publish_nowait replaced by a MagicMock spy."""
        bus = EventBus()
        bus.publish_nowait = MagicMock()
        return bus

    @staticmethod
    def _last_event_data(bus: EventBus) -> dict:
        """Extract data dict from the most recent publish_nowait call."""
        assert bus.publish_nowait.called, "publish_nowait was never called"
        event: Event = bus.publish_nowait.call_args[0][0]
        return event.data

    @staticmethod
    def _all_event_data(bus: EventBus) -> list:
        """Extract data dicts from ALL publish_nowait calls."""
        return [call[0][0].data for call in bus.publish_nowait.call_args_list]

    # ── L1 trigger ─────────────────────────────────────────────────────────────

    def test_dd_minus_6pct_triggers_level1_event(self):
        """DD −4 % → −6 %: L1 event published with size_multiplier=0.75."""
        bus = self._make_bus_with_spy()
        monitor = DrawdownMonitor(initial_equity=10_000, event_bus=bus)

        # Start from −4 % (no level)
        monitor.update(9_600)   # −4 % — still NONE
        bus.publish_nowait.reset_mock()

        # Drop to −6 % — crosses L1 threshold of −5 %
        monitor.update(9_400)   # −6 %

        assert bus.publish_nowait.call_count == 1
        data = self._last_event_data(bus)
        assert data["previous_level"] == CircuitBreakerLevel.NONE.value
        assert data["new_level"] == CircuitBreakerLevel.LEVEL_1.value
        assert data["size_multiplier"] == pytest.approx(0.75)
        assert data["cooldown_hours"] == 0
        assert data["current_dd"] == pytest.approx(-0.06, abs=0.001)

    # ── L2 trigger ─────────────────────────────────────────────────────────────

    def test_dd_minus_11pct_triggers_level2_event(self):
        """DD −11 %: L2 event published with size_multiplier=0.5 and cooldown 12 h."""
        bus = self._make_bus_with_spy()
        monitor = DrawdownMonitor(initial_equity=10_000, event_bus=bus)

        monitor.update(8_900)   # −11 %

        # Should have gone NONE → L1 → L2 in one update? No — each call to
        # update() evaluates the *current* level; a single step from 10k to 8900
        # lands directly at L2 (−11 % ≤ −10 %).
        assert bus.publish_nowait.call_count == 1
        data = self._last_event_data(bus)
        assert data["new_level"] == CircuitBreakerLevel.LEVEL_2.value
        assert data["size_multiplier"] == pytest.approx(0.50)
        assert data["cooldown_hours"] == 12

    # ── Recovery ───────────────────────────────────────────────────────────────

    def test_recovery_from_level2_to_none_publishes_recovery_event(self):
        """
        Recovery from −11 % back to −4 % (above L1 threshold):
        level goes L2 → NONE; event must be published with new_level=NONE
        and size_multiplier=1.0.
        """
        bus = self._make_bus_with_spy()
        monitor = DrawdownMonitor(initial_equity=10_000, event_bus=bus)

        # Trigger L2
        monitor.update(8_900)   # −11 % → L2
        bus.publish_nowait.reset_mock()

        # Simulate recovery: push peak up first so drawdown can improve
        # (update() resets peak on new highs)
        monitor.update(10_000)  # back to peak → NONE
        monitor.update(10_600)  # new peak; DD = 0 %

        # Then drop only slightly — should be NONE
        # The recovery from L2 to NONE must have fired an event
        events = self._all_event_data(bus)
        recovery_events = [e for e in events if e["new_level"] == CircuitBreakerLevel.NONE.value]
        assert len(recovery_events) >= 1

        rec = recovery_events[0]
        assert rec["size_multiplier"] == pytest.approx(1.0)

    def test_escalation_publishes_one_event_per_step(self):
        """Each individual level transition fires exactly one event."""
        bus = self._make_bus_with_spy()
        monitor = DrawdownMonitor(initial_equity=10_000, event_bus=bus)

        monitor.update(9_400)   # −6 %  → L1
        monitor.update(8_900)   # −11 % → L2
        monitor.update(8_400)   # −16 % → L3
        monitor.update(7_900)   # −21 % → L4

        assert bus.publish_nowait.call_count == 4
        levels = [e["new_level"] for e in self._all_event_data(bus)]
        assert levels == [
            CircuitBreakerLevel.LEVEL_1.value,
            CircuitBreakerLevel.LEVEL_2.value,
            CircuitBreakerLevel.LEVEL_3.value,
            CircuitBreakerLevel.LEVEL_4.value,
        ]

    def test_no_event_when_level_unchanged(self):
        """Multiple updates at the same level must not fire repeated events."""
        bus = self._make_bus_with_spy()
        monitor = DrawdownMonitor(initial_equity=10_000, event_bus=bus)

        monitor.update(9_400)   # −6 % → L1
        bus.publish_nowait.reset_mock()

        monitor.update(9_350)   # still L1
        monitor.update(9_300)   # still L1
        assert bus.publish_nowait.call_count == 0

    def test_no_event_bus_does_not_raise(self):
        """DrawdownMonitor without event_bus must work exactly as before."""
        monitor = DrawdownMonitor(initial_equity=10_000)   # no bus
        level = monitor.update(9_400)
        assert level == CircuitBreakerLevel.LEVEL_1


# ═══════════════════════════════════════════════════════════════════════════════
# H-4 — PortfolioManager subscribes and applies size_multiplier
# ═══════════════════════════════════════════════════════════════════════════════

class TestPortfolioManagerCircuitBreakerSubscription:

    @pytest.mark.asyncio
    async def test_level1_event_sets_multiplier_to_075(self):
        """
        When a CIRCUIT_BREAKER_LEVEL_CHANGE event arrives with size_multiplier=0.75
        the PortfolioManager must store it and apply it to new signals.
        """
        from strategy.portfolio_manager import PortfolioManager

        bus = EventBus()
        pm = PortfolioManager(event_bus=bus)

        assert pm._cb_size_multiplier == pytest.approx(1.0)

        # Simulate event dispatch
        event = Event(
            type=EventType.CIRCUIT_BREAKER_LEVEL_CHANGE,
            data={
                "previous_level": "NONE",
                "new_level": "LEVEL_1",
                "size_multiplier": 0.75,
                "cooldown_hours": 0,
                "current_dd": -0.06,
            },
        )
        await pm._on_circuit_breaker_level_change(event)

        assert pm._cb_size_multiplier == pytest.approx(0.75)

    @pytest.mark.asyncio
    async def test_level2_event_sets_multiplier_to_050(self):
        from strategy.portfolio_manager import PortfolioManager

        bus = EventBus()
        pm = PortfolioManager(event_bus=bus)

        event = Event(
            type=EventType.CIRCUIT_BREAKER_LEVEL_CHANGE,
            data={
                "previous_level": "LEVEL_1",
                "new_level": "LEVEL_2",
                "size_multiplier": 0.50,
                "cooldown_hours": 12,
                "current_dd": -0.11,
            },
        )
        await pm._on_circuit_breaker_level_change(event)

        assert pm._cb_size_multiplier == pytest.approx(0.50)

    @pytest.mark.asyncio
    async def test_recovery_event_resets_multiplier_to_1(self):
        from strategy.portfolio_manager import PortfolioManager

        bus = EventBus()
        pm = PortfolioManager(event_bus=bus)
        pm._cb_size_multiplier = 0.50   # pre-condition: was at L2

        event = Event(
            type=EventType.CIRCUIT_BREAKER_LEVEL_CHANGE,
            data={
                "previous_level": "LEVEL_2",
                "new_level": "NONE",
                "size_multiplier": 1.0,
                "cooldown_hours": 0,
                "current_dd": -0.04,
            },
        )
        await pm._on_circuit_breaker_level_change(event)

        assert pm._cb_size_multiplier == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_event_bus_wiring_via_real_bus(self):
        """
        End-to-end: DrawdownMonitor publishes → EventBus dispatches →
        PortfolioManager._cb_size_multiplier is updated.
        """
        from strategy.portfolio_manager import PortfolioManager

        bus = EventBus()
        monitor = DrawdownMonitor(initial_equity=10_000, event_bus=bus)
        pm = PortfolioManager(event_bus=bus)

        assert pm._cb_size_multiplier == pytest.approx(1.0)

        # DrawdownMonitor publishes via publish_nowait; manually process the queue
        monitor.update(8_900)   # −11 % → L2 event queued

        # Drain the event bus queue
        event = bus._queue.get_nowait()
        assert event.type == EventType.CIRCUIT_BREAKER_LEVEL_CHANGE

        handlers = bus._handlers.get(EventType.CIRCUIT_BREAKER_LEVEL_CHANGE, [])
        for h in handlers:
            import asyncio as _asyncio
            result = h(event)
            if _asyncio.iscoroutine(result):
                await result

        assert pm._cb_size_multiplier == pytest.approx(0.50)


# ═══════════════════════════════════════════════════════════════════════════════
# H-4 — TelegramAlerter subscribes and sends notification
# ═══════════════════════════════════════════════════════════════════════════════

class TestTelegramAlerterCircuitBreakerSubscription:

    @pytest.mark.asyncio
    async def test_level_change_sends_telegram_message(self):
        """
        When _on_circuit_breaker_level_change is called the alerter must
        call send() with a non-empty message containing the new level.
        """
        from monitoring.alerting import TelegramAlerter

        alerter = TelegramAlerter(token="tok", chat_id="123")
        alerter.send = AsyncMock(return_value=True)

        event = Event(
            type=EventType.CIRCUIT_BREAKER_LEVEL_CHANGE,
            data={
                "previous_level": "NONE",
                "new_level": "LEVEL_1",
                "size_multiplier": 0.75,
                "cooldown_hours": 0,
                "current_dd": -0.06,
            },
        )
        await alerter._on_circuit_breaker_level_change(event)

        alerter.send.assert_awaited_once()
        msg: str = alerter.send.call_args[0][0]
        assert "LEVEL_1" in msg
        assert "0.75" in msg

    @pytest.mark.asyncio
    async def test_level2_message_includes_cooldown(self):
        from monitoring.alerting import TelegramAlerter

        alerter = TelegramAlerter(token="tok", chat_id="123")
        alerter.send = AsyncMock(return_value=True)

        event = Event(
            type=EventType.CIRCUIT_BREAKER_LEVEL_CHANGE,
            data={
                "previous_level": "LEVEL_1",
                "new_level": "LEVEL_2",
                "size_multiplier": 0.50,
                "cooldown_hours": 12,
                "current_dd": -0.11,
            },
        )
        await alerter._on_circuit_breaker_level_change(event)

        msg: str = alerter.send.call_args[0][0]
        assert "LEVEL_2" in msg
        assert "12" in msg   # cooldown hours

    @pytest.mark.asyncio
    async def test_recovery_message_indicates_normal_trading(self):
        from monitoring.alerting import TelegramAlerter

        alerter = TelegramAlerter(token="tok", chat_id="123")
        alerter.send = AsyncMock(return_value=True)

        event = Event(
            type=EventType.CIRCUIT_BREAKER_LEVEL_CHANGE,
            data={
                "previous_level": "LEVEL_2",
                "new_level": "NONE",
                "size_multiplier": 1.0,
                "cooldown_hours": 0,
                "current_dd": -0.04,
            },
        )
        await alerter._on_circuit_breaker_level_change(event)

        msg: str = alerter.send.call_args[0][0]
        # Recovery message should hint at normal operations resuming
        assert "NONE" in msg or "ripreso" in msg.lower() or "1.0" in msg

    @pytest.mark.asyncio
    async def test_event_bus_subscription_registered(self):
        """
        When event_bus is passed to the constructor the handler must be
        registered in the bus for CIRCUIT_BREAKER_LEVEL_CHANGE.
        """
        from monitoring.alerting import TelegramAlerter

        bus = EventBus()
        alerter = TelegramAlerter(token="tok", chat_id="123", event_bus=bus)

        handlers = bus._handlers.get(EventType.CIRCUIT_BREAKER_LEVEL_CHANGE, [])
        assert alerter._on_circuit_breaker_level_change in handlers
