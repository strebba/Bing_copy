"""Unit tests for BingX API retry logic (H-6)."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
import aiohttp

from exchange.bingx_client import BingXClient, with_retry, with_order_retry


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_client() -> BingXClient:
    """Create a BingXClient with dummy credentials."""
    client = BingXClient.__new__(BingXClient)
    client._api_key = "test_key"
    client._api_secret = "test_secret"
    client._base_url = "https://test.example.com"
    client._session = None
    return client


def _make_response_error(status: int) -> aiohttp.ClientResponseError:
    """Create a ClientResponseError with given status."""
    return aiohttp.ClientResponseError(
        request_info=MagicMock(),
        history=(),
        status=status,
        message=f"Error {status}",
    )


# ── with_retry decorator tests ────────────────────────────────────────────


class TestWithRetry:
    """H-6: Retry logic for transient API failures."""

    @pytest.mark.asyncio
    async def test_429_retry_then_success(self):
        """Simulates 429 rate limit → retry → success on 2nd attempt."""
        call_count = 0

        @with_retry(max_retries=3, base_delay=0.01)
        async def flaky_call():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _make_response_error(429)
            return {"data": "ok"}

        result = await flaky_call()
        assert result == {"data": "ok"}
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_500_exhausts_retries_raises(self):
        """Simulates 500 × 4 (initial + 3 retries) → raises after all retries."""
        call_count = 0

        @with_retry(max_retries=3, base_delay=0.01)
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise _make_response_error(500)

        with pytest.raises(aiohttp.ClientResponseError) as exc_info:
            await always_fails()
        assert exc_info.value.status == 500
        assert call_count == 4  # 1 initial + 3 retries

    @pytest.mark.asyncio
    async def test_timeout_retry_then_success(self):
        """Simulates timeout → retry → success."""
        call_count = 0

        @with_retry(max_retries=3, base_delay=0.01)
        async def timeout_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise asyncio.TimeoutError()
            return {"data": "recovered"}

        result = await timeout_once()
        assert result == {"data": "recovered"}
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_non_retryable_status_raises_immediately(self):
        """Non-retryable status (e.g. 400) raises immediately without retry."""
        call_count = 0

        @with_retry(max_retries=3, base_delay=0.01)
        async def bad_request():
            nonlocal call_count
            call_count += 1
            raise _make_response_error(400)

        with pytest.raises(aiohttp.ClientResponseError) as exc_info:
            await bad_request()
        assert exc_info.value.status == 400
        assert call_count == 1  # No retry

    @pytest.mark.asyncio
    async def test_success_on_first_try_no_retry(self):
        """Successful call doesn't trigger any retry."""
        call_count = 0

        @with_retry(max_retries=3, base_delay=0.01)
        async def works_fine():
            nonlocal call_count
            call_count += 1
            return {"status": "success"}

        result = await works_fine()
        assert result == {"status": "success"}
        assert call_count == 1


# ── with_order_retry tests ─────────────────────────────────────────────────


class TestWithOrderRetry:
    """H-6: Order placement with position-check-before-retry."""

    @pytest.mark.asyncio
    async def test_timeout_order_found_in_positions_no_retry(self):
        """Timeout on place_order → check positions → order found → no retry."""
        client = _make_client()

        # Mock _post to raise timeout on first call
        call_count = 0
        original_post = None

        async def mock_post(endpoint, params=None):
            nonlocal call_count
            call_count += 1
            raise asyncio.TimeoutError()

        # Mock get_positions to return the position exists
        async def mock_get_positions(symbol=None):
            return [{"positionSide": "LONG", "positionAmt": "0.1", "symbol": "BTC-USDT"}]

        client._post = mock_post
        client.get_positions = mock_get_positions
        client._build_params = lambda p: p
        client._ensure_session = AsyncMock()

        result = await client.place_order(
            symbol="BTC-USDT",
            side="BUY",
            position_side="LONG",
            order_type="MARKET",
            quantity=0.1,
        )
        # Should return the position found (not retry)
        assert result["positionSide"] == "LONG"
        assert call_count == 1  # Only one call to _post

    @pytest.mark.asyncio
    async def test_timeout_order_not_found_retries(self):
        """Timeout on place_order → check positions → not found → retry → success."""
        client = _make_client()

        call_count = 0

        async def mock_post(endpoint, params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise asyncio.TimeoutError()
            return {"code": 0, "data": {"orderId": "12345"}}

        async def mock_get_positions(symbol=None):
            return []  # No position found

        client._post = mock_post
        client.get_positions = mock_get_positions
        client._build_params = lambda p: p
        client._ensure_session = AsyncMock()
        client._raise_if_error = lambda data: None

        # We need to bypass the decorator to test with mock internals
        # Instead, test the decorator pattern directly
        @with_order_retry(max_retries=2, base_delay=0.01)
        async def mock_place_order(self, symbol, side, position_side, order_type, quantity):
            nonlocal call_count
            # Reset to use our counter from mock_post
            pass

        # Simpler approach: test the behavior at a higher level
        post_count = 0

        async def counting_post(endpoint, params=None):
            nonlocal post_count
            post_count += 1
            if post_count == 1:
                raise asyncio.TimeoutError()
            return {"code": 0, "data": {"orderId": "99"}}

        client._post = counting_post

        result = await client.place_order(
            symbol="BTC-USDT",
            side="BUY",
            position_side="LONG",
            order_type="MARKET",
            quantity=0.1,
        )
        # Second call should succeed
        assert post_count == 2

    @pytest.mark.asyncio
    async def test_cancel_order_no_retry(self):
        """cancel_order should NOT have retry (may already be filled)."""
        client = _make_client()

        async def mock_delete(endpoint, params=None):
            raise asyncio.TimeoutError()

        client._delete = mock_delete
        client._build_params = lambda p: p
        client._ensure_session = AsyncMock()

        # cancel_order should raise immediately without retry
        with pytest.raises(asyncio.TimeoutError):
            await client.cancel_order("BTC-USDT", "12345")


# ── Integration-style: decorator applied to BingXClient methods ───────────


class TestBingXClientRetryIntegration:
    """Verify decorators are correctly applied to BingXClient methods."""

    def test_get_balance_has_retry(self):
        """get_balance should be wrapped with retry."""
        assert hasattr(BingXClient.get_balance, "__wrapped__")

    def test_get_positions_has_retry(self):
        assert hasattr(BingXClient.get_positions, "__wrapped__")

    def test_get_klines_has_retry(self):
        assert hasattr(BingXClient.get_klines, "__wrapped__")

    def test_get_ticker_has_retry(self):
        assert hasattr(BingXClient.get_ticker, "__wrapped__")

    def test_get_depth_has_retry(self):
        assert hasattr(BingXClient.get_depth, "__wrapped__")

    def test_get_funding_rate_has_retry(self):
        assert hasattr(BingXClient.get_funding_rate, "__wrapped__")

    def test_place_order_has_order_retry(self):
        assert hasattr(BingXClient.place_order, "__wrapped__")

    def test_cancel_order_has_no_retry(self):
        """cancel_order must NOT have retry decorator."""
        assert not hasattr(BingXClient.cancel_order, "__wrapped__")

    def test_set_leverage_has_retry(self):
        assert hasattr(BingXClient.set_leverage, "__wrapped__")

    def test_set_margin_type_has_retry(self):
        assert hasattr(BingXClient.set_margin_type, "__wrapped__")
