"""
WAGMI Copy Trading Bot — Main Entry Point
"""
import asyncio
import logging
import signal
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from core.engine import TradingEngine
from monitoring.metrics import METRICS

# ── Logging setup ─────────────────────────────────────────────────────────────

LOG_DIR = Path(settings.LOG_FILE).parent
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(settings.LOG_FILE),
    ],
)

logger = logging.getLogger("wagmi.main")


async def main() -> None:
    logger.info("=" * 60)
    logger.info("  WAGMI Copy Trading Bot v1.0")
    logger.info("  Mode: %s", "DEMO (Paper Trading)" if settings.DEMO_MODE else "LIVE TRADING")
    logger.info("  Pairs: %s", ", ".join(settings.TRADING_PAIRS))
    logger.info("  Initial Capital: %.2f USDT", settings.INITIAL_CAPITAL)
    logger.info("  Max Leverage: %dx", settings.MAX_LEVERAGE)
    logger.info("=" * 60)

    if not settings.BINGX_API_KEY or not settings.BINGX_API_SECRET:
        logger.error(
            "BingX API credentials not set. Copy .env.example to .env and fill in the values."
        )
        sys.exit(1)

    # Start Prometheus metrics
    try:
        METRICS.start_server(settings.PROMETHEUS_PORT)
    except Exception as exc:
        logger.warning("Prometheus metrics server not started: %s", exc)

    engine = TradingEngine()

    # Graceful shutdown on SIGTERM / SIGINT
    loop = asyncio.get_event_loop()

    def handle_shutdown(sig_name: str) -> None:
        logger.info("Received %s — initiating graceful shutdown…", sig_name)
        asyncio.create_task(engine.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig, lambda s=sig.name: handle_shutdown(s)
        )

    try:
        await engine.start()
    except asyncio.CancelledError:
        logger.info("Engine task cancelled")
    except Exception as exc:
        logger.critical("Unhandled engine error: %s", exc, exc_info=True)
        raise
    finally:
        await engine.stop()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
