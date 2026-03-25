"""
BingX REST API client for Perpetual Futures.
Handles authentication, request signing, and all REST endpoints.
"""
import asyncio
import hashlib
import hmac
import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import aiohttp

from config import settings

logger = logging.getLogger(__name__)


# ── Retry decorator ────────────────────────────────────────────────────────


RETRYABLE_STATUSES = (429, 500, 502, 503, 504)


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    retryable_status: Tuple[int, ...] = RETRYABLE_STATUSES,
) -> Callable:
    """Decorator that adds exponential-backoff retry for transient API errors."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except aiohttp.ClientResponseError as e:
                    if e.status in retryable_status:
                        last_exception = e
                        delay = base_delay * (2 ** attempt)
                        if e.status == 429:
                            delay = max(delay, 10.0)
                        logger.warning(
                            "API call %s failed with %d, retry %d/%d in %.1fs",
                            func.__name__, e.status, attempt + 1, max_retries, delay,
                        )
                        await asyncio.sleep(delay)
                    else:
                        raise
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    last_exception = e
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "API call %s failed: %s, retry %d/%d in %.1fs",
                        func.__name__, e, attempt + 1, max_retries, delay,
                    )
                    await asyncio.sleep(delay)
            raise last_exception  # type: ignore[misc]

        return wrapper

    return decorator


# ── Safe order placement (check-before-retry) ─────────────────────────────


def with_order_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> Callable:
    """
    Retry decorator for order placement that checks positions before retrying
    to prevent duplicate orders on timeout.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self: "BingXClient", *args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None
            symbol = kwargs.get("symbol") or (args[0] if args else None)
            position_side = kwargs.get("position_side") or (args[2] if len(args) > 2 else None)

            for attempt in range(max_retries + 1):
                try:
                    return await func(self, *args, **kwargs)
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    last_exception = e
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "Order placement %s timed out: %s, checking positions...",
                        func.__name__, e,
                    )
                    # Check if the order actually went through
                    if symbol and position_side:
                        try:
                            positions = await self.get_positions(symbol)
                            for pos in positions:
                                if pos.get("positionSide") == position_side:
                                    qty = abs(float(pos.get("positionAmt", 0)))
                                    if qty > 0:
                                        logger.info(
                                            "Order for %s %s found in positions despite timeout, skipping retry",
                                            symbol, position_side,
                                        )
                                        return pos
                        except Exception as check_err:
                            logger.error("Failed to check positions: %s", check_err)

                    logger.warning(
                        "Order not found in positions, retry %d/%d in %.1fs",
                        attempt + 1, max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                except aiohttp.ClientResponseError as e:
                    if e.status in RETRYABLE_STATUSES:
                        last_exception = e
                        delay = base_delay * (2 ** attempt)
                        if e.status == 429:
                            delay = max(delay, 10.0)
                        logger.warning(
                            "Order placement failed with %d, retry %d/%d in %.1fs",
                            e.status, attempt + 1, max_retries, delay,
                        )
                        await asyncio.sleep(delay)
                    else:
                        raise
            raise last_exception  # type: ignore[misc]

        return wrapper

    return decorator


class BingXClient:
    """Async REST client for BingX Perpetual Futures API."""

    def __init__(
        self,
        api_key: str = settings.BINGX_API_KEY,
        api_secret: str = settings.BINGX_API_SECRET,
        base_url: str = settings.BINGX_BASE_URL,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._base_url = base_url
        self._session: Optional[aiohttp.ClientSession] = None

    # ── Session management ────────────────────────────────────────────────────

    async def __aenter__(self) -> "BingXClient":
        await self._ensure_session()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def _ensure_session(self) -> None:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=10)
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Signing ───────────────────────────────────────────────────────────────

    def _sign(self, params: Dict[str, Any]) -> str:
        query = urlencode(sorted(params.items()))
        return hmac.new(
            self._api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _build_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params["timestamp"] = int(time.time() * 1000)
        params["signature"] = self._sign(params)
        return params

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    async def _get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None, sign: bool = True
    ) -> Dict[str, Any]:
        await self._ensure_session()
        p = params or {}
        if sign:
            p = self._build_params(p)
        headers = {"X-BX-APIKEY": self._api_key}
        url = self._base_url + endpoint
        async with self._session.get(url, params=p, headers=headers) as resp:
            data = await resp.json()
            self._raise_if_error(data)
            return data

    async def _post(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        await self._ensure_session()
        p = self._build_params(params or {})
        headers = {
            "X-BX-APIKEY": self._api_key,
            "Content-Type": "application/x-www-form-urlencoded",
        }
        url = self._base_url + endpoint
        async with self._session.post(url, data=p, headers=headers) as resp:
            data = await resp.json()
            self._raise_if_error(data)
            return data

    async def _delete(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        await self._ensure_session()
        p = self._build_params(params or {})
        headers = {"X-BX-APIKEY": self._api_key}
        url = self._base_url + endpoint
        async with self._session.delete(url, params=p, headers=headers) as resp:
            data = await resp.json()
            self._raise_if_error(data)
            return data

    @staticmethod
    def _raise_if_error(data: Dict[str, Any]) -> None:
        code = data.get("code", 0)
        if code != 0:
            raise RuntimeError(f"BingX API error {code}: {data.get('msg', '')}")

    # ── Account endpoints ─────────────────────────────────────────────────────

    @with_retry()
    async def get_balance(self) -> Dict[str, Any]:
        return await self._get(settings.BINGX_ENDPOINTS["balance"])

    @with_retry()
    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        params = {}
        if symbol:
            params["symbol"] = symbol
        resp = await self._get(settings.BINGX_ENDPOINTS["positions"], params)
        return resp.get("data", {}).get("positions", [])

    @with_retry()
    async def get_open_orders(self, symbol: str) -> List[Dict]:
        resp = await self._get(
            settings.BINGX_ENDPOINTS["open_orders"], {"symbol": symbol}
        )
        return resp.get("data", {}).get("orders", [])

    # ── Market data endpoints ─────────────────────────────────────────────────

    @with_retry()
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[List]:
        params: Dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        resp = await self._get(settings.BINGX_ENDPOINTS["klines"], params, sign=False)
        return resp.get("data", [])

    @with_retry()
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        resp = await self._get(
            settings.BINGX_ENDPOINTS["ticker"], {"symbol": symbol}, sign=False
        )
        return resp.get("data", {})

    @with_retry()
    async def get_depth(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        resp = await self._get(
            settings.BINGX_ENDPOINTS["depth"],
            {"symbol": symbol, "limit": limit},
            sign=False,
        )
        return resp.get("data", {})

    @with_retry()
    async def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        resp = await self._get(
            settings.BINGX_ENDPOINTS["funding_rate"], {"symbol": symbol}, sign=False
        )
        return resp.get("data", {})

    # ── Order endpoints ───────────────────────────────────────────────────────

    @with_order_retry()
    async def place_order(
        self,
        symbol: str,
        side: str,            # "BUY" | "SELL"
        position_side: str,   # "LONG" | "SHORT"
        order_type: str,      # "MARKET" | "LIMIT" | "STOP_MARKET"
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        reduce_only: bool = False,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "positionSide": position_side,
            "type": order_type,
            "quantity": quantity,
            "reduceOnly": str(reduce_only).upper(),
        }
        if price is not None:
            params["price"] = price
        if stop_price is not None:
            params["stopPrice"] = stop_price
        if client_order_id:
            params["clientOrderID"] = client_order_id
        resp = await self._post(settings.BINGX_ENDPOINTS["place_order"], params)
        return resp.get("data", {})

    async def get_order_detail(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Query a single order by orderId to get fill details (avgPrice, executedQty)."""
        resp = await self._get(
            settings.BINGX_ENDPOINTS["order_detail"],
            {"symbol": symbol, "orderId": order_id},
        )
        return resp.get("data", {})

    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Cancel order — NO retry (order may already be filled)."""
        resp = await self._delete(
            settings.BINGX_ENDPOINTS["cancel_order"],
            {"symbol": symbol, "orderId": order_id},
        )
        return resp.get("data", {})

    @with_retry()
    async def set_leverage(self, symbol: str, leverage: int, side: str = "LONG") -> None:
        await self._post(
            settings.BINGX_ENDPOINTS["leverage"],
            {"symbol": symbol, "leverage": leverage, "side": side},
        )

    @with_retry()
    async def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED") -> None:
        await self._post(
            settings.BINGX_ENDPOINTS["margin_type"],
            {"symbol": symbol, "marginType": margin_type},
        )
