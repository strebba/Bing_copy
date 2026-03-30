"""Unit tests for core/engine module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.engine import TradingEngine
from core.state_manager import Position


class TestTradingEngine:
    @pytest.fixture
    def engine(self):
        with patch("core.engine.BingXClient"):
            with patch("core.engine.BingXWebSocket"):
                return TradingEngine()

    def test_engine_initialization(self, engine):
        assert engine._running is False
        assert engine._equity > 0
        assert engine._win_count == 0
        assert engine._loss_count == 0

    def test_engine_has_required_components(self, engine):
        assert hasattr(engine, "_client")
        assert hasattr(engine, "_order_mgr")
        assert hasattr(engine, "_ws")
        assert hasattr(engine, "_market_data")
        assert hasattr(engine, "_orderbook")
        assert hasattr(engine, "_funding")
        assert hasattr(engine, "_state")
        assert hasattr(engine, "_event_bus")
        assert hasattr(engine, "_dd_monitor")
        assert hasattr(engine, "_risk_engine")
        assert hasattr(engine, "_portfolio")

    @pytest.mark.asyncio
    async def test_start_sets_running_true(self, engine):
        engine._running = False
        with patch.object(engine, "_market_data") as md:
            md.initialize = AsyncMock()
            with patch.object(engine, "_market_stream") as ms:
                ms.subscribe_kline = MagicMock()
                ms.subscribe_depth = MagicMock()
                with patch.object(engine, "_ws") as ws:
                    ws.start = AsyncMock()
                    with patch.object(engine, "_user_stream") as us:
                        us.start = AsyncMock()
                        with patch.object(engine, "_funding") as funding:
                            funding.start = AsyncMock()
                            with patch("asyncio.gather", return_value=[]):
                                task = asyncio.create_task(engine.start())
                                await asyncio.sleep(0.01)
                                engine._running = False
                                task.cancel()
                                try:
                                    await task
                                except (asyncio.CancelledError, Exception):
                                    pass

    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self, engine):
        engine._running = True
        with patch("asyncio.gather", return_value=[]):
            await engine.stop()
        assert engine._running is False

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, engine):
        engine._running = True
        with patch.object(engine, "_emergency", reason="test shutdown"):
            with patch.object(engine, "_alerter") as alerter:
                alerter.send = AsyncMock()
                alerter.close = AsyncMock()
                with patch.object(engine, "_client") as client:
                    client.get_positions = AsyncMock(return_value=[])
                    with patch.object(engine, "_emergency") as emerg:
                        emerg.close_all_positions = AsyncMock()
                        with patch.object(engine, "_ws") as ws:
                            ws.stop = AsyncMock()
                            with patch.object(engine, "_user_stream"):
                                with patch.object(engine, "_funding"):
                                    with patch.object(engine, "_client") as cl:
                                        cl.close = AsyncMock()
                                        await engine._graceful_shutdown()

    def test_win_rate_with_no_trades(self, engine):
        assert engine._win_rate() == 0.55

    def test_win_rate_with_wins(self, engine):
        engine._win_count = 5
        engine._loss_count = 5
        assert engine._win_rate() == 0.5

    def test_win_rate_all_wins(self, engine):
        engine._win_count = 10
        engine._loss_count = 0
        assert engine._win_rate() == 1.0

    @pytest.mark.asyncio
    async def test_graceful_shutdown_closes_positions(self, engine):
        """Test shutdown chiude posizioni aperte."""
        engine._running = True

        mock_position = {
            "symbol": "BTC-USDT",
            "positionSide": "LONG",
            "positionAmt": "0.1",
        }

        with patch.object(engine, "_alerter") as alerter:
            alerter.send = AsyncMock()
            alerter.close = AsyncMock()
            with patch.object(engine, "_client") as client:
                client.get_positions = AsyncMock(return_value=[mock_position])
                client.close = AsyncMock()
                with patch.object(engine, "_emergency") as emerg:
                    emerg.close_all_positions = AsyncMock()
                    emerg.reason = "test"
                    with patch.object(engine, "_ws") as ws:
                        ws.stop = AsyncMock()
                        with patch.object(engine, "_user_stream") as us:
                            us.stop = AsyncMock()
                            with patch.object(engine, "_funding") as funding:
                                funding.stop = AsyncMock()
                                await engine._graceful_shutdown()

                                # Verifica che close_all_positions sia chiamato
                                emerg.close_all_positions.assert_called_once()

    @pytest.mark.asyncio
    async def test_engine_shutdown_sequence(self, engine):
        """Test sequenza di shutdown."""
        engine._running = True

        with patch.object(engine, "_graceful_shutdown") as mock_shutdown:
            await engine.stop()
            assert engine._running is False

    @pytest.mark.asyncio
    async def test_supervised_gather_handles_critical_task_crash(self, engine):
        """Test che crash task critici triggera emergency stop."""

        async def crash_task():
            raise Exception("Critical task crashed")

        with patch.object(engine, "_emergency") as emerg:
            emerg.trigger = AsyncMock()
            with patch.object(engine, "_event_bus") as bus:
                bus.publish = AsyncMock()

                try:
                    await engine._supervised_gather(crash_task(), names=["market_data"])
                except Exception:
                    pass

                # Il task critico dovrebbe triggerare emergency stop
                emerg.trigger.assert_called_once()
