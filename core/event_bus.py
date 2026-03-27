"""
Simple in-process async event bus using asyncio.Queue.
Decouples producers (market data, WebSocket) from consumers (engine, monitoring).
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    CANDLE_CLOSE = "candle_close"
    SIGNAL_GENERATED = "signal_generated"
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    SL_HIT = "trade.sl_hit"
    TP_HIT = "trade.tp_hit"
    CIRCUIT_BREAKER = "circuit_breaker"
    # H-4: published whenever the circuit breaker moves to a new level
    # (escalation AND recovery).  Payload keys:
    #   previous_level, new_level, current_dd, size_multiplier,
    #   cooldown_hours, timestamp
    CIRCUIT_BREAKER_LEVEL_CHANGE = "circuit_breaker.level_change"
    EMERGENCY_STOP = "emergency_stop"
    HEARTBEAT = "heartbeat"
    FUNDING_RATE_UPDATE = "funding_rate_update"
    DEPTH_UPDATE = "depth_update"


@dataclass
class Event:
    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)


Handler = Callable[[Event], Any]


class EventBus:
    """Async publish-subscribe event bus."""

    def __init__(self) -> None:
        self._handlers: Dict[EventType, List[Handler]] = {}
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=10_000)

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    async def publish(self, event: Event) -> None:
        await self._queue.put(event)

    def publish_nowait(self, event: Event) -> None:
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("Event bus queue full, dropping event: %s", event.type)

    async def process_events(self) -> None:
        """Main event processing loop — run as a background task."""
        while True:
            event = await self._queue.get()
            handlers = self._handlers.get(event.type, [])
            for handler in handlers:
                try:
                    result = handler(event)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as exc:
                    logger.error("Event handler error for %s: %s", event.type, exc)
            self._queue.task_done()
