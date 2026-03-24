"""
BingX REST API client for Perpetual Futures.
Handles authentication, request signing, and all REST endpoints.
"""
import asyncio
import hashlib
import hmac
import logging
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import aiohttp

from config import settings

logger = logging.getLogger(__name__)


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

    async def get_balance(self) -> Dict[str, Any]:
        return await self._get(settings.BINGX_ENDPOINTS["balance"])

    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        params = {}
        if symbol:
            params["symbol"] = symbol
        resp = await self._get(settings.BINGX_ENDPOINTS["positions"], params)
        return resp.get("data", {}).get("positions", [])

    async def get_open_orders(self, symbol: str) -> List[Dict]:
        resp = await self._get(
            settings.BINGX_ENDPOINTS["open_orders"], {"symbol": symbol}
        )
        return resp.get("data", {}).get("orders", [])

    # ── Market data endpoints ─────────────────────────────────────────────────

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

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        resp = await self._get(
            settings.BINGX_ENDPOINTS["ticker"], {"symbol": symbol}, sign=False
        )
        return resp.get("data", {})

    async def get_depth(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        resp = await self._get(
            settings.BINGX_ENDPOINTS["depth"],
            {"symbol": symbol, "limit": limit},
            sign=False,
        )
        return resp.get("data", {})

    async def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        resp = await self._get(
            settings.BINGX_ENDPOINTS["funding_rate"], {"symbol": symbol}, sign=False
        )
        return resp.get("data", {})

    # ── Order endpoints ───────────────────────────────────────────────────────

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

    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        resp = await self._delete(
            settings.BINGX_ENDPOINTS["cancel_order"],
            {"symbol": symbol, "orderId": order_id},
        )
        return resp.get("data", {})

    async def set_leverage(self, symbol: str, leverage: int, side: str = "LONG") -> None:
        await self._post(
            settings.BINGX_ENDPOINTS["leverage"],
            {"symbol": symbol, "leverage": leverage, "side": side},
        )

    async def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED") -> None:
        await self._post(
            settings.BINGX_ENDPOINTS["margin_type"],
            {"symbol": symbol, "marginType": margin_type},
        )
