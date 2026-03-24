"""
Integration tests for the BingX API client.
These tests use the BingX DEMO environment and require valid DEMO API keys.
Set BINGX_API_KEY and BINGX_API_SECRET env vars before running.
Skip automatically if credentials are not set.
"""
import os

import pytest

pytestmark = pytest.mark.skipif(
    not (os.getenv("BINGX_API_KEY") and os.getenv("BINGX_API_SECRET")),
    reason="BingX credentials not configured",
)


@pytest.mark.asyncio
async def test_get_ticker():
    from exchange.bingx_client import BingXClient
    async with BingXClient() as client:
        ticker = await client.get_ticker("BTC-USDT")
        assert "lastPrice" in ticker or "last" in str(ticker)


@pytest.mark.asyncio
async def test_get_klines():
    from exchange.bingx_client import BingXClient
    async with BingXClient() as client:
        klines = await client.get_klines("BTC-USDT", "1h", limit=10)
        assert len(klines) > 0


@pytest.mark.asyncio
async def test_get_depth():
    from exchange.bingx_client import BingXClient
    async with BingXClient() as client:
        depth = await client.get_depth("BTC-USDT", limit=5)
        assert "bids" in depth or "asks" in depth


@pytest.mark.asyncio
async def test_get_balance():
    from exchange.bingx_client import BingXClient
    async with BingXClient() as client:
        balance = await client.get_balance()
        assert "data" in balance


@pytest.mark.asyncio
async def test_get_funding_rate():
    from exchange.bingx_client import BingXClient
    async with BingXClient() as client:
        fr = await client.get_funding_rate("BTC-USDT")
        assert isinstance(fr, dict)
