"""
Telegram alerting — sends trade, risk, and system notifications.

H-4 Fix: subscribes to CIRCUIT_BREAKER_LEVEL_CHANGE events and sends a
detailed Telegram notification whenever the breaker changes level.

Features:
  - Rate limiting: max 20 messages/minute (Telegram API limit)
  - Digest: low-priority events batched and sent hourly
  - Event bus integration: subscribes to engine events automatically
  - /status bot command: equity, DD, open positions, circuit breaker, uptime
"""
import asyncio
import logging
import time
from collections import deque
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import aiohttp

from config import settings

if TYPE_CHECKING:
    from core.event_bus import Event, EventBus

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
TELEGRAM_GET_UPDATES = "https://api.telegram.org/bot{token}/getUpdates"

# Rate limit: max 20 messages per minute
RATE_LIMIT_MAX = 20
RATE_LIMIT_WINDOW_S = 60.0

# Digest: flush every hour
DIGEST_INTERVAL_S = 3600

# Heartbeat: alert if no heartbeat for 5 minutes
HEARTBEAT_STALE_S = 300


class TelegramAlerter:
    """
    Async Telegram notifier with rate limiting, digest, and event bus integration.
    """

    def __init__(
        self,
        token: str = settings.TELEGRAM_BOT_TOKEN,
        chat_id: str = settings.TELEGRAM_CHAT_ID,
        event_bus: Optional["EventBus"] = None,
    ) -> None:
        self._token = token
        self._chat_id = chat_id
        self._enabled = bool(token and chat_id)
        self._session: Optional[aiohttp.ClientSession] = None

        # Rate limiting: track timestamps of sent messages (rolling window)
        self._sent_timestamps: deque = deque()

        # Digest queue for low-priority events
        self._digest_queue: List[str] = []

        # Heartbeat tracking
        self._last_heartbeat_ts: float = time.time()

        # Bot polling offset for /status command
        self._poll_offset: int = 0

        # Status callback: set by engine to provide live status data
        self._status_callback: Optional[Callable[[], Dict[str, Any]]] = None

        # Engine start time for uptime tracking
        self._start_time: float = time.time()

        # H-4: subscribe to circuit breaker level changes
        if event_bus is not None:
            from core.event_bus import EventType  # noqa: PLC0415
            event_bus.subscribe(
                EventType.CIRCUIT_BREAKER_LEVEL_CHANGE,
                self._on_circuit_breaker_level_change,
            )

    # ── Session ───────────────────────────────────────────────────────────────

    async def _ensure_session(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Rate limiting ─────────────────────────────────────────────────────────

    def _is_rate_limited(self) -> bool:
        """Return True if we have hit the 20 msg/min limit."""
        now = time.time()
        # Evict timestamps older than the rolling window
        while self._sent_timestamps and now - self._sent_timestamps[0] > RATE_LIMIT_WINDOW_S:
            self._sent_timestamps.popleft()
        return len(self._sent_timestamps) >= RATE_LIMIT_MAX

    def _record_send(self) -> None:
        self._sent_timestamps.append(time.time())

    # ── Core send ─────────────────────────────────────────────────────────────

    async def send(self, message: str, parse_mode: str = "Markdown") -> bool:
        """
        Send a Telegram message, respecting the rate limit.
        Returns True on success, False on failure or rate-limit drop.
        """
        if not self._enabled:
            logger.debug("Telegram not configured, skipping alert")
            return False

        if self._is_rate_limited():
            logger.warning("Telegram rate limit hit — message dropped: %.80s", message)
            return False

        await self._ensure_session()
        url = TELEGRAM_API.format(token=self._token)
        payload = {
            "chat_id": self._chat_id,
            "text": message,
            "parse_mode": parse_mode,
        }
        try:
            async with self._session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("Telegram error %d: %s", resp.status, body)
                    return False
                self._record_send()
                return True
        except Exception as exc:
            logger.error("Telegram send failed: %s", exc)
            return False

    async def _on_circuit_breaker_level_change(self, event: "Event") -> None:
        """Send a Telegram alert whenever the circuit breaker changes level (H-4)."""
        data = event.data
        prev = data.get("previous_level", "NONE")
        new = data.get("new_level", "NONE")
        dd_pct = data.get("current_dd", 0.0) * 100
        multiplier = data.get("size_multiplier", 1.0)
        cooldown_h = data.get("cooldown_hours", 0)

        if new == "NONE":
            emoji = "✅"
            action_line = "Trading ripreso normalmente (size 1.0x)"
        else:
            emoji = "⚠️"
            action = f"Size ridotto a `{multiplier:.2f}x`"
            if cooldown_h > 0:
                action += f" + cooldown `{cooldown_h}h`"
            action_line = action

        msg = (
            f"{emoji} *Circuit Breaker — {prev} → {new}*\n"
            f"Drawdown corrente: `{dd_pct:.2f}%`\n"
            f"Azione: {action_line}"
        )
        await self.send(msg)

    # ── Digest ────────────────────────────────────────────────────────────────

    def add_to_digest(self, message: str) -> None:
        """Enqueue a low-priority message for the hourly digest."""
        self._digest_queue.append(message)

    async def flush_digest(self) -> bool:
        """Send accumulated digest messages as a single message."""
        if not self._digest_queue:
            return True
        lines = self._digest_queue.copy()
        self._digest_queue.clear()
        header = f"📋 *Hourly Digest* ({len(lines)} events)\n"
        body = "\n".join(f"• {l}" for l in lines[-50:])  # cap at 50 lines
        return await self.send(header + body)

    async def digest_loop(self) -> None:
        """Background task: flush digest every hour."""
        while True:
            await asyncio.sleep(DIGEST_INTERVAL_S)
            try:
                await self.flush_digest()
            except Exception as exc:
                logger.error("Digest flush error: %s", exc)

    # ── /status command polling ───────────────────────────────────────────────

    def set_status_callback(self, cb: Callable[[], Dict[str, Any]]) -> None:
        """Register a callback that returns current system status."""
        self._status_callback = cb

    async def _get_updates(self) -> List[dict]:
        """Poll Telegram for new bot updates."""
        if not self._enabled:
            return []
        await self._ensure_session()
        url = TELEGRAM_GET_UPDATES.format(token=self._token)
        params = {"offset": self._poll_offset, "timeout": 1, "limit": 10}
        try:
            async with self._session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                return data.get("result", [])
        except Exception:
            return []

    async def _handle_status_command(self) -> None:
        """Reply to /status bot command with live system info."""
        status: Dict[str, Any] = {}
        if self._status_callback:
            try:
                status = self._status_callback()
            except Exception as exc:
                logger.error("Status callback error: %s", exc)

        uptime_s = int(time.time() - self._start_time)
        uptime_str = f"{uptime_s // 3600}h {(uptime_s % 3600) // 60}m"
        hb_age = int(time.time() - self._last_heartbeat_ts)
        last_hb = f"{hb_age}s ago" if hb_age < HEARTBEAT_STALE_S else f"⚠️ {hb_age}s ago (STALE)"

        equity = status.get("equity", "N/A")
        dd = status.get("drawdown_pct", None)
        dd_str = f"{dd:.2f}%" if dd is not None else "N/A"
        open_pos = status.get("open_positions", [])
        cb_level = status.get("circuit_breaker", "NONE")

        if open_pos:
            pos_lines = "\n".join(
                f"  • `{p['symbol']}` {p['side']} @ `{p['entry']:.4f}`"
                for p in open_pos[:10]
            )
        else:
            pos_lines = "  _None_"

        msg = (
            "📊 *System Status*\n"
            f"Equity: `{equity} USDT`\n"
            f"Drawdown: `{dd_str}`\n"
            f"Circuit Breaker: `{cb_level}`\n"
            f"\n*Open Positions ({len(open_pos)}):*\n{pos_lines}\n"
            f"\nUptime: `{uptime_str}`\n"
            f"Last Heartbeat: `{last_hb}`"
        )
        await self.send(msg)

    async def bot_polling_loop(self) -> None:
        """Background task: poll for /status command and respond."""
        while True:
            await asyncio.sleep(5)
            if not self._enabled:
                continue
            try:
                updates = await self._get_updates()
                for update in updates:
                    self._poll_offset = update["update_id"] + 1
                    msg = update.get("message", {})
                    text = msg.get("text", "").strip()
                    if text.startswith("/status"):
                        await self._handle_status_command()
            except Exception as exc:
                logger.error("Bot polling error: %s", exc)

    # ── Event bus integration ─────────────────────────────────────────────────

    def subscribe_to_event_bus(
        self,
        event_bus: Any,
        state_manager: Any = None,
        dd_monitor: Any = None,
    ) -> None:
        """
        Subscribe this alerter to all relevant engine events.
        Call this once during engine startup.
        """
        from core.event_bus import EventType

        # Register a status callback if we have state/dd references
        if state_manager is not None and dd_monitor is not None:
            def _status() -> Dict[str, Any]:
                positions = [
                    {
                        "symbol": p.symbol,
                        "side": p.position_side,
                        "entry": p.entry_price,
                    }
                    for p in state_manager.all_positions()
                ]
                return {
                    "equity": f"{dd_monitor.state.current_equity:.2f}",
                    "drawdown_pct": dd_monitor.current_drawdown() * 100,
                    "circuit_breaker": dd_monitor.state.circuit_level,
                    "open_positions": positions,
                }
            self.set_status_callback(_status)

        # Critical events — always send immediately
        event_bus.subscribe(EventType.EMERGENCY_STOP, self._on_emergency_stop)
        event_bus.subscribe(EventType.CIRCUIT_BREAKER, self._on_circuit_breaker)
        event_bus.subscribe(EventType.POSITION_OPENED, self._on_position_opened)
        event_bus.subscribe(EventType.POSITION_CLOSED, self._on_position_closed)

        # Low-priority events — digest / conditional
        event_bus.subscribe(EventType.HEARTBEAT, self._on_heartbeat)

        logger.info("TelegramAlerter subscribed to event bus")

    # ── Event handlers ────────────────────────────────────────────────────────

    async def _on_emergency_stop(self, event: Any) -> None:
        reason = event.data.get("reason", "unknown")
        msg = f"🚨 *EMERGENCY STOP*\nReason: `{reason}`"
        await self.send(msg)

    async def _on_circuit_breaker(self, event: Any) -> None:
        level = event.data.get("level", "UNKNOWN")
        dd = event.data.get("dd_pct", 0.0)
        msg = (
            f"⚠️ *CIRCUIT BREAKER — {level}*\n"
            f"Current Drawdown: `{dd:.2f}%`"
        )
        await self.send(msg)

    async def _on_position_opened(self, event: Any) -> None:
        d = event.data
        direction = d.get("direction", "")
        emoji = "📈" if direction == "LONG" else "📉"
        msg = (
            f"{emoji} *TRADE OPENED*\n"
            f"Symbol: `{d.get('symbol', '')}`\n"
            f"Direction: *{direction}*\n"
            f"Strategy: `{d.get('strategy', '')}`\n"
            f"Entry: `{d.get('entry', 0):.4f}`\n"
            f"SL: `{d.get('sl', 0):.4f}`\n"
            f"TP: `{d.get('tp', 0):.4f}`\n"
            f"Risk: `{d.get('risk_usdt', 0):.2f} USDT`"
        )
        await self.send(msg)

    async def _on_position_closed(self, event: Any) -> None:
        d = event.data
        pnl = d.get("pnl", 0.0)
        reason = d.get("reason", "")

        # Distinguish SL hit vs TP hit vs other
        reason_lower = reason.lower()
        if "sl" in reason_lower or reason == "stop_loss":
            emoji = "🛑"
            label = "STOP LOSS HIT"
        elif "tp" in reason_lower or reason == "take_profit":
            emoji = "🎯"
            label = "TAKE PROFIT HIT"
        else:
            emoji = "✅" if pnl >= 0 else "❌"
            label = "TRADE CLOSED"

        msg = (
            f"{emoji} *{label}*\n"
            f"Symbol: `{d.get('symbol', '')}`\n"
            f"Direction: *{d.get('direction', '')}*\n"
            f"PnL: `{pnl:+.2f} USDT`\n"
            f"Reason: `{reason}`"
        )
        await self.send(msg)

    async def _on_heartbeat(self, event: Any) -> None:
        """
        Update the last heartbeat timestamp.
        Add to digest only if heartbeat was stale for > 5 minutes.
        """
        now = time.time()
        age = now - self._last_heartbeat_ts
        self._last_heartbeat_ts = now

        if age > HEARTBEAT_STALE_S:
            self.add_to_digest(
                f"⚠️ Heartbeat was stale for {int(age)}s (recovered at "
                f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC)"
            )

    # ── Legacy alert helpers (kept for backwards compat) ─────────────────────
    async def alert_position_opened(
        self, symbol: str, direction: str, entry: float, sl: float, tp: float, strategy: str
    ) -> None:
        emoji = "🟢" if direction == "LONG" else "🔴"
        msg = (
            f"{emoji} *Position Opened*\n"
            f"Symbol: `{symbol}`\n"
            f"Direction: *{direction}*\n"
            f"Strategy: `{strategy}`\n"
            f"Entry: `{entry:.4f}`\n"
            f"SL: `{sl:.4f}`\n"
            f"TP: `{tp:.4f}`"
        )
        await self.send(msg)

    async def alert_position_closed(
        self, symbol: str, direction: str, pnl: float, reason: str
    ) -> None:
        emoji = "✅" if pnl > 0 else "❌"
        msg = (
            f"{emoji} *Position Closed*\n"
            f"Symbol: `{symbol}`\n"
            f"Direction: *{direction}*\n"
            f"PnL: `{pnl:+.2f} USDT`\n"
            f"Reason: `{reason}`"
        )
        await self.send(msg)

    async def alert_circuit_breaker(self, level: str, dd_pct: float) -> None:
        msg = (
            f"⚠️ *Circuit Breaker — {level}*\n"
            f"Current Drawdown: `{dd_pct:.1f}%`"
        )
        await self.send(msg)

    async def alert_emergency_stop(self, reason: str) -> None:
        msg = f"🚨 *EMERGENCY STOP*\nReason: `{reason}`"
        await self.send(msg)

    async def alert_daily_report(self, stats: dict) -> None:
        msg = (
            "📊 *Daily Performance Report*\n"
            f"Trades: `{stats.get('total_trades', 0)}`\n"
            f"Win Rate: `{stats.get('win_rate', 0):.1%}`\n"
            f"PnL: `{stats.get('net_pnl_usdt', 0):+.2f} USDT`\n"
            f"ROI: `{stats.get('roi_pct', 0):+.2f}%`\n"
            f"Max DD: `{stats.get('max_drawdown_pct', 0):.2f}%`\n"
            f"Sharpe: `{stats.get('sharpe_ratio', 0):.2f}`"
        )
        await self.send(msg)
