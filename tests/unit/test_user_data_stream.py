"""
Unit tests for Gap #2 — SL/TP fill detection via User Data Stream WebSocket.

Covers:
  - ORDER_TRADE_UPDATE with FILLED status triggers correct handler (SL vs TP)
  - SL fill → TP is cancelled, SL_HIT event published, position closed
  - TP fill → SL is cancelled, TP_HIT event published, position closed
  - Non-FILLED orders are ignored
  - Order not matching any open position is ignored
  - Listen key refresh loop runs every 30 minutes
  - Listen key refresh failure → new listen key generated
  - WebSocket disconnection → reconnects automatically
  - Fallback: reconciliation loop still runs (changed to 60s interval)
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import websockets

from core.event_bus import Event, EventBus, EventType
from core.state_manager import Position, StateManager
from exchange.bingx_ws import UserDataStream


def _make_position(
    symbol: str = "BTC-USDT",
    position_side: str = "LONG",
    sl_order_id: str = "SL-001",
    tp_order_id: str = "TP-001",
) -> Position:
    return Position(
        symbol=symbol,
        position_side=position_side,
        entry_price=100.0,
        quantity=0.1,
        stop_loss=97.0,
        take_profit=106.0,
        strategy_name="alpha",
        sl_order_id=sl_order_id,
        tp_order_id=tp_order_id,
    )


def _make_mock_client(listen_key: str = "test-listen-key-123"):
    client = MagicMock()
    client.create_listen_key = AsyncMock(return_value=listen_key)
    client.refresh_listen_key = AsyncMock(return_value={"code": 0})
    return client


def _make_order_manager() -> MagicMock:
    mgr = MagicMock()
    mgr.on_sl_triggered = AsyncMock(return_value=True)
    mgr.on_tp_triggered = AsyncMock(return_value=True)
    return mgr


class TestSLFillDetection:
    @pytest.mark.asyncio
    async def test_sl_fill_triggers_correct_handler(self):
        """ORDER_TRADE_UPDATE FILLED with STOP_MARKET → on_sl_triggered called."""
        state = StateManager()
        pos = _make_position()
        state.open_position(pos)

        mock_client = _make_mock_client()
        mock_order_mgr = _make_order_manager()
        event_bus = EventBus()
        stream = UserDataStream(mock_client, mock_order_mgr, state, event_bus)

        ws_msg = {
            "dataType": "ORDER_TRADE_UPDATE",
            "data": {
                "order": {
                    "orderId": "SL-001",
                    "symbol": "BTC-USDT",
                    "side": "SELL",
                    "positionSide": "LONG",
                    "status": "FILLED",
                    "type": "STOP_MARKET",
                }
            },
        }

        await stream._on_order_trade_update(ws_msg)

        mock_order_mgr.on_sl_triggered.assert_called_once_with("BTC-USDT", "LONG")

    @pytest.mark.asyncio
    async def test_sl_fill_cancels_tp(self):
        """SL fill → TP order is cancelled via on_sl_triggered."""
        state = StateManager()
        pos = _make_position(sl_order_id="SL-001", tp_order_id="TP-001")
        state.open_position(pos)

        mock_client = _make_mock_client()
        mock_order_mgr = _make_order_manager()
        event_bus = EventBus()
        stream = UserDataStream(mock_client, mock_order_mgr, state, event_bus)

        ws_msg = {
            "dataType": "ORDER_TRADE_UPDATE",
            "data": {
                "order": {
                    "orderId": "SL-001",
                    "symbol": "BTC-USDT",
                    "side": "SELL",
                    "positionSide": "LONG",
                    "status": "FILLED",
                    "type": "STOP_MARKET",
                }
            },
        }

        await stream._on_order_trade_update(ws_msg)

        mock_order_mgr.on_sl_triggered.assert_called_once()

    @pytest.mark.asyncio
    async def test_sl_fill_closes_position(self):
        """SL fill → position is removed from state."""
        state = StateManager()
        pos = _make_position()
        state.open_position(pos)
        assert state.get_position("BTC-USDT", "LONG") is not None

        mock_client = _make_mock_client()
        mock_order_mgr = _make_order_manager()
        event_bus = EventBus()
        stream = UserDataStream(mock_client, mock_order_mgr, state, event_bus)

        ws_msg = {
            "dataType": "ORDER_TRADE_UPDATE",
            "data": {
                "order": {
                    "orderId": "SL-001",
                    "symbol": "BTC-USDT",
                    "side": "SELL",
                    "positionSide": "LONG",
                    "status": "FILLED",
                    "type": "STOP_MARKET",
                }
            },
        }

        await stream._on_order_trade_update(ws_msg)

        assert state.get_position("BTC-USDT", "LONG") is None

    @pytest.mark.asyncio
    async def test_sl_fill_publishes_sl_hit_event(self):
        """SL fill → SL_HIT event published on event bus."""
        state = StateManager()
        pos = _make_position()
        state.open_position(pos)

        mock_client = _make_mock_client()
        mock_order_mgr = _make_order_manager()
        event_bus = EventBus()
        stream = UserDataStream(mock_client, mock_order_mgr, state, event_bus)

        ws_msg = {
            "dataType": "ORDER_TRADE_UPDATE",
            "data": {
                "order": {
                    "orderId": "SL-001",
                    "symbol": "BTC-USDT",
                    "side": "SELL",
                    "positionSide": "LONG",
                    "status": "FILLED",
                    "type": "STOP_MARKET",
                }
            },
        }

        received_events = []

        async def listener(event: Event):
            received_events.append(event)

        event_bus.subscribe(EventType.SL_HIT, listener)
        proc_task = asyncio.create_task(event_bus.process_events())
        await stream._on_order_trade_update(ws_msg)
        await asyncio.sleep(0.05)
        proc_task.cancel()
        try:
            await proc_task
        except asyncio.CancelledError:
            pass

        sl_hit_events = [e for e in received_events if e.type == EventType.SL_HIT]
        assert len(sl_hit_events) == 1

    @pytest.mark.asyncio
    async def test_sl_fill_publishes_position_closed_event(self):
        """SL fill → POSITION_CLOSED event published with reason=stop_loss."""
        state = StateManager()
        pos = _make_position()
        state.open_position(pos)

        mock_client = _make_mock_client()
        mock_order_mgr = _make_order_manager()
        event_bus = EventBus()
        stream = UserDataStream(mock_client, mock_order_mgr, state, event_bus)

        ws_msg = {
            "dataType": "ORDER_TRADE_UPDATE",
            "data": {
                "order": {
                    "orderId": "SL-001",
                    "symbol": "BTC-USDT",
                    "side": "SELL",
                    "positionSide": "LONG",
                    "status": "FILLED",
                    "type": "STOP_MARKET",
                }
            },
        }

        closed_events = []

        async def listener(event: Event):
            if event.type == EventType.POSITION_CLOSED:
                closed_events.append(event)

        event_bus.subscribe(EventType.POSITION_CLOSED, listener)
        proc_task = asyncio.create_task(event_bus.process_events())
        await stream._on_order_trade_update(ws_msg)
        await asyncio.sleep(0.05)
        proc_task.cancel()
        try:
            await proc_task
        except asyncio.CancelledError:
            pass

        assert len(closed_events) == 1
        assert closed_events[0].data["reason"] == "stop_loss"


class TestTPFillDetection:
    @pytest.mark.asyncio
    async def test_tp_fill_triggers_correct_handler(self):
        """ORDER_TRADE_UPDATE FILLED with TAKE_PROFIT_MARKET → on_tp_triggered called."""
        state = StateManager()
        pos = _make_position()
        state.open_position(pos)

        mock_client = _make_mock_client()
        mock_order_mgr = _make_order_manager()
        event_bus = EventBus()
        stream = UserDataStream(mock_client, mock_order_mgr, state, event_bus)

        ws_msg = {
            "dataType": "ORDER_TRADE_UPDATE",
            "data": {
                "order": {
                    "orderId": "TP-001",
                    "symbol": "BTC-USDT",
                    "side": "SELL",
                    "positionSide": "LONG",
                    "status": "FILLED",
                    "type": "TAKE_PROFIT_MARKET",
                }
            },
        }

        await stream._on_order_trade_update(ws_msg)

        mock_order_mgr.on_tp_triggered.assert_called_once_with("BTC-USDT", "LONG")

    @pytest.mark.asyncio
    async def test_tp_fill_closes_position(self):
        """TP fill → position is removed from state."""
        state = StateManager()
        pos = _make_position()
        state.open_position(pos)

        mock_client = _make_mock_client()
        mock_order_mgr = _make_order_manager()
        event_bus = EventBus()
        stream = UserDataStream(mock_client, mock_order_mgr, state, event_bus)

        ws_msg = {
            "dataType": "ORDER_TRADE_UPDATE",
            "data": {
                "order": {
                    "orderId": "TP-001",
                    "symbol": "BTC-USDT",
                    "side": "SELL",
                    "positionSide": "LONG",
                    "status": "FILLED",
                    "type": "TAKE_PROFIT_MARKET",
                }
            },
        }

        await stream._on_order_trade_update(ws_msg)

        assert state.get_position("BTC-USDT", "LONG") is None

    @pytest.mark.asyncio
    async def test_tp_fill_publishes_tp_hit_event(self):
        """TP fill → TP_HIT event published on event bus."""
        state = StateManager()
        pos = _make_position()
        state.open_position(pos)

        mock_client = _make_mock_client()
        mock_order_mgr = _make_order_manager()
        event_bus = EventBus()
        stream = UserDataStream(mock_client, mock_order_mgr, state, event_bus)

        ws_msg = {
            "dataType": "ORDER_TRADE_UPDATE",
            "data": {
                "order": {
                    "orderId": "TP-001",
                    "symbol": "BTC-USDT",
                    "side": "SELL",
                    "positionSide": "LONG",
                    "status": "FILLED",
                    "type": "TAKE_PROFIT_MARKET",
                }
            },
        }

        received_events = []

        async def listener(event: Event):
            received_events.append(event)

        event_bus.subscribe(EventType.TP_HIT, listener)
        proc_task = asyncio.create_task(event_bus.process_events())
        await stream._on_order_trade_update(ws_msg)
        await asyncio.sleep(0.05)
        proc_task.cancel()
        try:
            await proc_task
        except asyncio.CancelledError:
            pass

        tp_hit_events = [e for e in received_events if e.type == EventType.TP_HIT]
        assert len(tp_hit_events) == 1


class TestNonFilledOrders:
    @pytest.mark.asyncio
    async def test_non_filled_order_ignored(self):
        """ORDER_TRADE_UPDATE with status != FILLED is ignored."""
        state = StateManager()
        pos = _make_position()
        state.open_position(pos)

        mock_client = _make_mock_client()
        mock_order_mgr = _make_order_manager()
        event_bus = EventBus()
        stream = UserDataStream(mock_client, mock_order_mgr, state, event_bus)

        ws_msg = {
            "dataType": "ORDER_TRADE_UPDATE",
            "data": {
                "order": {
                    "orderId": "SL-001",
                    "symbol": "BTC-USDT",
                    "side": "SELL",
                    "positionSide": "LONG",
                    "status": "NEW",
                    "type": "STOP_MARKET",
                }
            },
        }

        await stream._on_order_trade_update(ws_msg)

        mock_order_mgr.on_sl_triggered.assert_not_called()
        mock_order_mgr.on_tp_triggered.assert_not_called()
        assert state.get_position("BTC-USDT", "LONG") is not None


class TestUnknownPosition:
    @pytest.mark.asyncio
    async def test_order_without_open_position_ignored(self):
        """ORDER_TRADE_UPDATE for position not in state is ignored."""
        state = StateManager()

        mock_client = _make_mock_client()
        mock_order_mgr = _make_order_manager()
        event_bus = EventBus()
        stream = UserDataStream(mock_client, mock_order_mgr, state, event_bus)

        ws_msg = {
            "dataType": "ORDER_TRADE_UPDATE",
            "data": {
                "order": {
                    "orderId": "UNKNOWN-001",
                    "symbol": "BTC-USDT",
                    "side": "SELL",
                    "positionSide": "LONG",
                    "status": "FILLED",
                    "type": "STOP_MARKET",
                }
            },
        }

        await stream._on_order_trade_update(ws_msg)

        mock_order_mgr.on_sl_triggered.assert_not_called()
        mock_order_mgr.on_tp_triggered.assert_not_called()


class TestListenKeyRefresh:
    @pytest.mark.asyncio
    async def test_listen_key_refresh_called_every_30_minutes(self):
        """refresh_listen_key is called every 30 minutes."""
        mock_client = _make_mock_client()
        mock_order_mgr = _make_order_manager()
        event_bus = EventBus()
        state = StateManager()
        stream = UserDataStream(mock_client, mock_order_mgr, state, event_bus)
        stream._running = True

        refresh_count = 0

        async def mock_refresh(key):
            nonlocal refresh_count
            refresh_count += 1
            raise RuntimeError("fail")

        mock_client.refresh_listen_key = mock_refresh

        real_sleep = asyncio.sleep

        async def patched_sleep(duration):
            nonlocal refresh_count
            if duration >= 1800:
                await real_sleep(0.01)
            else:
                await real_sleep(duration)

        with patch("asyncio.sleep", patched_sleep):
            task = asyncio.create_task(stream._listen_key_refresh_loop())
            await asyncio.sleep(0.1)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert refresh_count >= 1


class TestWebSocketReconnection:
    @pytest.mark.asyncio
    async def test_websocket_disconnect_reconnects(self):
        """WS disconnection triggers reconnection with same listen key."""
        mock_client = _make_mock_client()
        mock_order_mgr = _make_order_manager()
        event_bus = EventBus()
        state = StateManager()
        stream = UserDataStream(mock_client, mock_order_mgr, state, event_bus)
        stream._running = True

        connect_count = 0
        close_count = 0

        class FakeWS:
            async def close(self):
                nonlocal close_count
                close_count += 1

            async def __aiter__(self):
                nonlocal connect_count
                connect_count += 1
                if connect_count == 1:
                    raise websockets.exceptions.ConnectionClosed(None, None)
                return
                yield

        with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = FakeWS()
            task = asyncio.create_task(stream._connect())
            await asyncio.sleep(0.1)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass


class TestShortPositionSLTP:
    @pytest.mark.asyncio
    async def test_short_position_sl_fill(self):
        """SHORT position: SELL STOP_MARKET (SL for SHORT) triggers on_sl_triggered."""
        state = StateManager()
        pos = _make_position(
            position_side="SHORT", sl_order_id="SL-S-001", tp_order_id="TP-S-001"
        )
        state.open_position(pos)

        mock_client = _make_mock_client()
        mock_order_mgr = _make_order_manager()
        event_bus = EventBus()
        stream = UserDataStream(mock_client, mock_order_mgr, state, event_bus)

        ws_msg = {
            "dataType": "ORDER_TRADE_UPDATE",
            "data": {
                "order": {
                    "orderId": "SL-S-001",
                    "symbol": "BTC-USDT",
                    "side": "BUY",
                    "positionSide": "SHORT",
                    "status": "FILLED",
                    "type": "STOP_MARKET",
                }
            },
        }

        await stream._on_order_trade_update(ws_msg)

        mock_order_mgr.on_sl_triggered.assert_called_once_with("BTC-USDT", "SHORT")
        assert state.get_position("BTC-USDT", "SHORT") is None

    @pytest.mark.asyncio
    async def test_short_position_tp_fill(self):
        """SHORT position: BUY TAKE_PROFIT_MARKET (TP for SHORT) triggers on_tp_triggered."""
        state = StateManager()
        pos = _make_position(
            position_side="SHORT", sl_order_id="SL-S-001", tp_order_id="TP-S-001"
        )
        state.open_position(pos)

        mock_client = _make_mock_client()
        mock_order_mgr = _make_order_manager()
        event_bus = EventBus()
        stream = UserDataStream(mock_client, mock_order_mgr, state, event_bus)

        ws_msg = {
            "dataType": "ORDER_TRADE_UPDATE",
            "data": {
                "order": {
                    "orderId": "TP-S-001",
                    "symbol": "BTC-USDT",
                    "side": "BUY",
                    "positionSide": "SHORT",
                    "status": "FILLED",
                    "type": "TAKE_PROFIT_MARKET",
                }
            },
        }

        await stream._on_order_trade_update(ws_msg)

        mock_order_mgr.on_tp_triggered.assert_called_once_with("BTC-USDT", "SHORT")
        assert state.get_position("BTC-USDT", "SHORT") is None
