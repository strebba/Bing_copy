"""
Telegram alerting — sends trade, risk, and system notifications.
"""
import logging
from typing import Optional

import aiohttp

from config import settings

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramAlerter:
    """Send async Telegram messages."""

    def __init__(
        self,
        token: str = settings.TELEGRAM_BOT_TOKEN,
        chat_id: str = settings.TELEGRAM_CHAT_ID,
    ) -> None:
        self._token = token
        self._chat_id = chat_id
        self._enabled = bool(token and chat_id)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def send(self, message: str, parse_mode: str = "Markdown") -> bool:
        if not self._enabled:
            logger.debug("Telegram not configured, skipping alert")
            return False
        await self._ensure_session()
        url = TELEGRAM_API.format(token=self._token)
        payload = {
            "chat_id": self._chat_id,
            "text": message,
            "parse_mode": parse_mode,
        }
        try:
            async with self._session.post(url, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("Telegram error %d: %s", resp.status, body)
                    return False
                return True
        except Exception as exc:
            logger.error("Telegram send failed: %s", exc)
            return False

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
