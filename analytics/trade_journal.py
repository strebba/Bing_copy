"""
Trade journal — structured logging of every trade event.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class TradeJournal:
    """
    Appends trade events to a JSONL file for later analysis.
    """

    def __init__(self, log_path: str = "logs/trade_journal.jsonl") -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, data: Dict[str, Any]) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **data,
        }
        try:
            with self._path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.error("Trade journal write error: %s", exc)

    def log_signal(self, signal: Any) -> None:
        self.log("SIGNAL", {
            "symbol": signal.symbol,
            "direction": signal.direction.value,
            "strategy": signal.strategy_name,
            "confidence": signal.confidence,
            "entry": signal.entry_price,
            "sl": signal.stop_loss,
            "tp": signal.take_profit,
            "rr": round(signal.risk_reward, 2),
        })

    def log_order(self, order_type: str, data: Dict) -> None:
        self.log(f"ORDER_{order_type}", data)

    def log_trade_closed(
        self,
        symbol: str,
        direction: str,
        entry: float,
        exit_price: float,
        pnl: float,
        reason: str,
    ) -> None:
        self.log("TRADE_CLOSED", {
            "symbol": symbol,
            "direction": direction,
            "entry": entry,
            "exit": exit_price,
            "pnl_usdt": round(pnl, 4),
            "reason": reason,
        })
