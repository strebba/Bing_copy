"""Unit tests for EmergencyStop with retry logic (H-5)."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from risk.emergency_stop import EmergencyStop


def _make_position(symbol: str, side: str = "LONG", qty: float = 1.0) -> dict:
    return {"symbol": symbol, "positionSide": side, "positionAmt": str(qty)}


class TestEmergencyStopRetry:
    """H-5: Emergency stop with aggressive retry on close failures."""

    @pytest.mark.asyncio
    async def test_all_positions_close_first_attempt(self):
        """All 3 positions close successfully on first attempt."""
        order_mgr = AsyncMock()
        order_mgr.close_position.return_value = {"orderId": "123"}

        es = EmergencyStop()
        positions = [
            _make_position("BTC-USDT", "LONG", 0.1),
            _make_position("ETH-USDT", "SHORT", 1.0),
            _make_position("SOL-USDT", "LONG", 10.0),
        ]
        closed = await es.close_all_positions(order_mgr, positions)
        assert closed == 3
        assert order_mgr.close_position.call_count == 3

    @pytest.mark.asyncio
    async def test_second_position_fails_twice_then_succeeds(self):
        """3 positions: #2 fails 2 times then succeeds. All close eventually."""
        call_counts = {"ETH": 0}

        async def mock_close(symbol, pos_side, qty):
            if symbol == "ETH-USDT":
                call_counts["ETH"] += 1
                if call_counts["ETH"] <= 2:
                    raise RuntimeError("Simulated BingX timeout")
            return {"orderId": "ok"}

        order_mgr = AsyncMock()
        order_mgr.close_position.side_effect = mock_close

        es = EmergencyStop()
        # Use very short delays for testing
        es.BASE_RETRY_DELAY = 0.01

        positions = [
            _make_position("BTC-USDT", "LONG", 0.1),
            _make_position("ETH-USDT", "SHORT", 1.0),
            _make_position("SOL-USDT", "LONG", 10.0),
        ]
        closed = await es.close_all_positions(order_mgr, positions)
        assert closed == 3
        # ETH was called 3 times (2 failures + 1 success)
        assert call_counts["ETH"] == 3

    @pytest.mark.asyncio
    async def test_third_position_always_fails_sends_critical_alert(self):
        """3 positions: #1 ok, #2 fails 2x then ok, #3 always fails → critical alert."""
        call_counts = {"ETH": 0}

        async def mock_close(symbol, pos_side, qty):
            if symbol == "ETH-USDT":
                call_counts["ETH"] += 1
                if call_counts["ETH"] <= 2:
                    raise RuntimeError("Simulated timeout")
            if symbol == "SOL-USDT":
                raise RuntimeError("Permanent failure")
            return {"orderId": "ok"}

        order_mgr = AsyncMock()
        order_mgr.close_position.side_effect = mock_close

        alerting = AsyncMock()
        alerting.send = AsyncMock(return_value=True)

        es = EmergencyStop(alerting=alerting)
        es.BASE_RETRY_DELAY = 0.01

        positions = [
            _make_position("BTC-USDT", "LONG", 0.1),
            _make_position("ETH-USDT", "SHORT", 1.0),
            _make_position("SOL-USDT", "LONG", 10.0),
        ]
        closed = await es.close_all_positions(order_mgr, positions)

        # BTC and ETH closed, SOL did not
        assert closed == 2
        # Critical alert was sent
        alerting.send.assert_called_once()
        alert_msg = alerting.send.call_args[0][0]
        assert "MANUAL INTERVENTION REQUIRED" in alert_msg
        assert "SOL-USDT" in alert_msg

    @pytest.mark.asyncio
    async def test_no_alert_when_all_close(self):
        """No critical alert when all positions close successfully."""
        order_mgr = AsyncMock()
        order_mgr.close_position.return_value = {"orderId": "ok"}

        alerting = AsyncMock()
        es = EmergencyStop(alerting=alerting)

        positions = [_make_position("BTC-USDT")]
        await es.close_all_positions(order_mgr, positions)
        alerting.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_positions_returns_zero(self):
        """Empty positions list returns 0 immediately."""
        order_mgr = AsyncMock()
        es = EmergencyStop()
        closed = await es.close_all_positions(order_mgr, [])
        assert closed == 0
        order_mgr.close_position.assert_not_called()

    @pytest.mark.asyncio
    async def test_zero_quantity_positions_skipped(self):
        """Positions with zero quantity are skipped."""
        order_mgr = AsyncMock()
        order_mgr.close_position.return_value = {"orderId": "ok"}

        es = EmergencyStop()
        positions = [
            _make_position("BTC-USDT", "LONG", 0.0),
            _make_position("ETH-USDT", "LONG", 1.0),
        ]
        closed = await es.close_all_positions(order_mgr, positions)
        assert closed == 1

    @pytest.mark.asyncio
    async def test_activate_deactivate(self):
        es = EmergencyStop()
        assert not es.is_active
        es.activate("test reason")
        assert es.is_active
        assert es.reason == "test reason"
        es.deactivate()
        assert not es.is_active
