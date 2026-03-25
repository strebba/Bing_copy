"""
Tests for TelegramAlerter — rate limiting, event bus integration, digest.
Telegram API is fully mocked; no real HTTP calls are made.
"""
import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from monitoring.alerting import RATE_LIMIT_MAX, TelegramAlerter


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_alerter(enabled: bool = True) -> TelegramAlerter:
    alerter = TelegramAlerter(
        token="fake_token" if enabled else "",
        chat_id="12345" if enabled else "",
    )
    return alerter


async def _send_with_mock(alerter: TelegramAlerter, message: str, status: int = 200) -> bool:
    """Helper: mock the aiohttp session and call alerter.send()."""
    mock_resp = AsyncMock()
    mock_resp.status = status
    mock_resp.text = AsyncMock(return_value="error body")
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.post = MagicMock(return_value=mock_resp)

    alerter._session = mock_session
    return await alerter.send(message)


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestRateLimiting:
    def test_not_rate_limited_initially(self):
        alerter = _make_alerter()
        assert not alerter._is_rate_limited()

    def test_rate_limited_after_max_sends(self):
        alerter = _make_alerter()
        now = time.time()
        # Simulate RATE_LIMIT_MAX messages sent within the window
        for _ in range(RATE_LIMIT_MAX):
            alerter._sent_timestamps.append(now)
        assert alerter._is_rate_limited()

    def test_old_timestamps_evicted(self):
        alerter = _make_alerter()
        old_ts = time.time() - 120  # 2 minutes ago — outside 60s window
        for _ in range(RATE_LIMIT_MAX):
            alerter._sent_timestamps.append(old_ts)
        # Should NOT be rate limited because all timestamps are old
        assert not alerter._is_rate_limited()

    @pytest.mark.asyncio
    async def test_send_respects_rate_limit(self):
        alerter = _make_alerter()
        now = time.time()
        for _ in range(RATE_LIMIT_MAX):
            alerter._sent_timestamps.append(now)
        result = await _send_with_mock(alerter, "should be dropped")
        assert result is False  # rate limited → dropped

    @pytest.mark.asyncio
    async def test_send_succeeds_when_not_rate_limited(self):
        alerter = _make_alerter()
        result = await _send_with_mock(alerter, "hello", status=200)
        assert result is True
        assert len(alerter._sent_timestamps) == 1


class TestDisabledAlerter:
    @pytest.mark.asyncio
    async def test_returns_false_when_disabled(self):
        alerter = _make_alerter(enabled=False)
        result = await alerter.send("test")
        assert result is False


class TestDigest:
    def test_add_to_digest(self):
        alerter = _make_alerter()
        alerter.add_to_digest("event 1")
        alerter.add_to_digest("event 2")
        assert len(alerter._digest_queue) == 2

    @pytest.mark.asyncio
    async def test_flush_digest_clears_queue(self):
        alerter = _make_alerter()
        alerter.add_to_digest("event 1")
        result = await _send_with_mock(alerter, "")  # pre-warm session mock

        # Patch send to capture call
        alerter.send = AsyncMock(return_value=True)
        await alerter.flush_digest()
        assert len(alerter._digest_queue) == 0
        alerter.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_flush_empty_digest_skips_send(self):
        alerter = _make_alerter()
        alerter.send = AsyncMock(return_value=True)
        result = await alerter.flush_digest()
        assert result is True
        alerter.send.assert_not_called()


class TestEventBusIntegration:
    """Simulate the full event → Telegram pipeline with a mock event bus."""

    @pytest.mark.asyncio
    async def test_trade_opened_event_sends_message(self):
        alerter = _make_alerter()
        alerter.send = AsyncMock(return_value=True)

        event = MagicMock()
        event.data = {
            "symbol": "BTC-USDT",
            "direction": "LONG",
            "strategy": "alpha",
            "entry": 50000.0,
            "sl": 48000.0,
            "tp": 54000.0,
            "risk_usdt": 20.0,
        }

        await alerter._on_position_opened(event)
        alerter.send.assert_called_once()
        msg = alerter.send.call_args[0][0]
        assert "BTC-USDT" in msg
        assert "LONG" in msg
        assert "TRADE OPENED" in msg

    @pytest.mark.asyncio
    async def test_emergency_stop_event_sends_message(self):
        alerter = _make_alerter()
        alerter.send = AsyncMock(return_value=True)

        event = MagicMock()
        event.data = {"reason": "max_drawdown"}
        await alerter._on_emergency_stop(event)

        alerter.send.assert_called_once()
        msg = alerter.send.call_args[0][0]
        assert "EMERGENCY STOP" in msg
        assert "max_drawdown" in msg

    @pytest.mark.asyncio
    async def test_circuit_breaker_event_sends_message(self):
        alerter = _make_alerter()
        alerter.send = AsyncMock(return_value=True)

        event = MagicMock()
        event.data = {"level": "LEVEL_2", "dd_pct": 12.5}
        await alerter._on_circuit_breaker(event)

        alerter.send.assert_called_once()
        msg = alerter.send.call_args[0][0]
        assert "CIRCUIT BREAKER" in msg
        assert "LEVEL_2" in msg

    @pytest.mark.asyncio
    async def test_position_closed_sl_hit(self):
        alerter = _make_alerter()
        alerter.send = AsyncMock(return_value=True)

        event = MagicMock()
        event.data = {
            "symbol": "ETH-USDT",
            "direction": "SHORT",
            "pnl": -15.0,
            "reason": "stop_loss",
        }
        await alerter._on_position_closed(event)

        msg = alerter.send.call_args[0][0]
        assert "STOP LOSS HIT" in msg

    @pytest.mark.asyncio
    async def test_position_closed_tp_hit(self):
        alerter = _make_alerter()
        alerter.send = AsyncMock(return_value=True)

        event = MagicMock()
        event.data = {
            "symbol": "SOL-USDT",
            "direction": "LONG",
            "pnl": 30.0,
            "reason": "take_profit",
        }
        await alerter._on_position_closed(event)

        msg = alerter.send.call_args[0][0]
        assert "TAKE PROFIT HIT" in msg

    @pytest.mark.asyncio
    async def test_heartbeat_stale_adds_to_digest(self):
        alerter = _make_alerter()
        # Simulate last heartbeat 10 minutes ago
        alerter._last_heartbeat_ts = time.time() - 600

        event = MagicMock()
        event.data = {"ts": time.time()}
        await alerter._on_heartbeat(event)

        assert len(alerter._digest_queue) == 1
        assert "stale" in alerter._digest_queue[0].lower()

    @pytest.mark.asyncio
    async def test_heartbeat_fresh_does_not_add_to_digest(self):
        alerter = _make_alerter()
        alerter._last_heartbeat_ts = time.time() - 10  # 10s ago — fresh

        event = MagicMock()
        event.data = {"ts": time.time()}
        await alerter._on_heartbeat(event)

        assert len(alerter._digest_queue) == 0

    def test_subscribe_to_event_bus_registers_handlers(self):
        alerter = _make_alerter()
        from core.event_bus import EventBus, EventType

        bus = EventBus()
        alerter.subscribe_to_event_bus(bus)

        # Verify all critical event types are subscribed
        assert EventType.EMERGENCY_STOP in bus._handlers
        assert EventType.CIRCUIT_BREAKER in bus._handlers
        assert EventType.POSITION_OPENED in bus._handlers
        assert EventType.POSITION_CLOSED in bus._handlers
        assert EventType.HEARTBEAT in bus._handlers
