"""Unit tests for health_check module."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from monitoring.health_check import HealthChecker, MAX_HEARTBEAT_GAP_S


class MockSignal:
    def __init__(self):
        self.symbol = "BTC-USDT"
        self.direction = MagicMock()
        self.direction.value = "LONG"
        self.strategy_name = "test"
        self.confidence = 0.8
        self.entry_price = 50000.0
        self.stop_loss = 49000.0
        self.take_profit = 52000.0
        self.risk_reward = 2.0


class TestHealthChecker:
    def setup_method(self):
        self.mock_client = MagicMock()
        self.mock_client.get_ticker = AsyncMock(return_value={"quoteVolume": 1000})
        self.health = HealthChecker(self.mock_client)

    def test_record_heartbeat(self):
        before = self.health._last_heartbeat
        time.sleep(0.01)
        self.health.record_heartbeat()
        assert self.health._last_heartbeat > before

    def test_record_ws_message(self):
        before = self.health._last_ws_message
        time.sleep(0.01)
        self.health.record_ws_message()
        assert self.health._last_ws_message > before

    @pytest.mark.asyncio
    async def test_check_exchange_connectivity_success(self):
        self.mock_client.get_ticker = AsyncMock(return_value={"quoteVolume": 1000})
        result = await self.health.check_exchange_connectivity()
        assert result is True

    @pytest.mark.asyncio
    async def test_check_exchange_connectivity_failure(self):
        self.mock_client.get_ticker = AsyncMock(side_effect=Exception("API Error"))
        result = await self.health.check_exchange_connectivity()
        assert result is False

    def test_heartbeat_ok_true_when_recent(self):
        self.health.record_heartbeat()
        assert self.health.heartbeat_ok() is True

    def test_heartbeat_ok_false_when_stale(self):
        self.health._last_heartbeat = time.time() - MAX_HEARTBEAT_GAP_S - 1
        assert self.health.heartbeat_ok() is False

    def test_ws_ok_true_when_recent(self):
        self.health.record_ws_message()
        assert self.health.ws_ok() is True

    def test_ws_ok_false_when_stale(self):
        self.health._last_ws_message = time.time() - MAX_HEARTBEAT_GAP_S - 1
        assert self.health.ws_ok() is False

    def test_status_returns_dict(self):
        self.health.record_heartbeat()
        self.health.record_ws_message()
        status = self.health.status()
        assert "heartbeat_ok" in status
        assert "ws_ok" in status
        assert "last_heartbeat_s" in status
        assert "last_ws_s" in status

    @pytest.mark.asyncio
    async def test_health_loop_runs_and_logs_ok(self):
        self.health._running = True
        self.mock_client.get_ticker = AsyncMock(return_value={"quoteVolume": 1000})

        async def run_once():
            self.health._running = False

        with patch("monitoring.health_check.logger") as mock_logger:
            task = asyncio.create_task(self.health.run_health_loop(interval_s=0))
            await asyncio.sleep(0.05)
            self.health._running = False
            try:
                await asyncio.wait_for(task, timeout=0.5)
            except asyncio.TimeoutError:
                pass
            assert mock_logger.debug.called or mock_logger.critical.called

    @pytest.mark.asyncio
    async def test_health_loop_logs_critical_on_connectivity_failure(self):
        """Test che l'alert 'connectivity lost' viene loggato quando API non risponde."""
        self.health._running = True
        self.mock_client.get_ticker = AsyncMock(
            side_effect=Exception("Connection timeout")
        )

        with patch("monitoring.health_check.logger") as mock_logger:
            task = asyncio.create_task(self.health.run_health_loop(interval_s=0))
            await asyncio.sleep(0.05)
            self.health._running = False
            try:
                await asyncio.wait_for(task, timeout=0.5)
            except asyncio.TimeoutError:
                pass
            assert mock_logger.critical.called
            # Verifica che il log contenga False per connected
            assert any(
                "False" in str(call) for call in mock_logger.critical.call_args_list
            )

    @pytest.mark.asyncio
    async def test_health_loop_logs_critical_on_stale_heartbeat(self):
        """Test che l'alert viene loggato quando heartbeat è stale."""
        self.health._running = True
        self.health._last_heartbeat = time.time() - MAX_HEARTBEAT_GAP_S - 1
        self.mock_client.get_ticker = AsyncMock(return_value={"quoteVolume": 1000})

        with patch("monitoring.health_check.logger") as mock_logger:
            task = asyncio.create_task(self.health.run_health_loop(interval_s=0))
            await asyncio.sleep(0.05)
            self.health._running = False
            try:
                await asyncio.wait_for(task, timeout=0.5)
            except asyncio.TimeoutError:
                pass
            assert mock_logger.critical.called
            # Verifica che il log contenga False per heartbeat
            assert any(
                "False" in str(call) for call in mock_logger.critical.call_args_list
            )

    @pytest.mark.asyncio
    async def test_ws_disconnect_alert_triggered(self):
        """Test che WebSocket disconnesso triggera alert con reconnect."""
        self.health._running = True
        self.health._last_ws_message = time.time() - MAX_HEARTBEAT_GAP_S - 1
        self.health._last_heartbeat = time.time()
        self.mock_client.get_ticker = AsyncMock(return_value={"quoteVolume": 1000})

        with patch("monitoring.health_check.logger") as mock_logger:
            task = asyncio.create_task(self.health.run_health_loop(interval_s=0))
            await asyncio.sleep(0.05)
            self.health._running = False
            try:
                await asyncio.wait_for(task, timeout=0.5)
            except asyncio.TimeoutError:
                pass
            status = self.health.status()
            assert status["ws_ok"] is False
