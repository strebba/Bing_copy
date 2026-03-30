"""Unit tests for data/market_data module."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from data.market_data import MarketDataManager, INTERVAL_MAP


class TestMarketDataManager:
    @pytest.fixture
    def client(self):
        return MagicMock()

    @pytest.fixture
    def mdm(self, client):
        return MarketDataManager(client, max_candles=500)

    def test_initialization(self, mdm, client):
        assert mdm._client is client
        assert mdm._max_candles == 500
        assert mdm._cache == {}
        assert mdm._last_updated == {}

    def test_interval_map(self):
        assert INTERVAL_MAP["1H"] == "1h"
        assert INTERVAL_MAP["4H"] == "4h"
        assert INTERVAL_MAP["1D"] == "1d"
        assert INTERVAL_MAP["1m"] == "1m"

    @pytest.mark.asyncio
    async def test_initialize(self, mdm):
        with patch.object(
            mdm, "_fetch_and_cache", new_callable=AsyncMock
        ) as mock_fetch:
            await mdm.initialize(["BTC-USDT"], intervals=["1H"])
            assert mock_fetch.call_count == 1

    def test_parse_klines_empty(self, mdm):
        result = MarketDataManager._parse_klines([])
        assert result.empty

    def test_parse_klines_valid(self, mdm):
        raw = [
            [1000, "50000", "50100", "49900", "50050", "1000", 2000],
            [2000, "50050", "50200", "50000", "50150", "1200", 3000],
        ]
        df = MarketDataManager._parse_klines(raw, "1H")
        assert len(df) == 2
        assert "timestamp" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns
        assert df.attrs["timeframe"] == "1H"

    def test_get_df_returns_none_when_not_cached(self, mdm):
        result = mdm.get_df("BTC-USDT", "1H")
        assert result is None

    def test_get_df_returns_dataframe_when_cached(self, mdm):
        df = pd.DataFrame({"a": [1, 2, 3]})
        mdm._cache["BTC-USDT"] = {"1H": df}
        result = mdm.get_df("BTC-USDT", "1H")
        assert result is df

    def test_update_candle_new_candle(self, mdm):
        df = pd.DataFrame(
            {
                "timestamp": [1000],
                "open": [50000],
                "high": [50100],
                "low": [49900],
                "close": [50050],
                "volume": [1000],
            }
        )
        mdm._cache["BTC-USDT"] = {"1H": df}

        candle = {
            "t": 2000,
            "o": "50050",
            "h": "50200",
            "l": "50000",
            "c": "50150",
            "v": "1200",
        }
        mdm.update_candle("BTC-USDT", "1H", candle)

        result = mdm.get_df("BTC-USDT", "1H")
        assert len(result) == 2
        assert result["timestamp"].iloc[-1] == 2000

    def test_update_candle_existing_candle(self, mdm):
        df = pd.DataFrame(
            {
                "timestamp": [1000],
                "open": [50000],
                "high": [50100],
                "low": [49900],
                "close": [50050],
                "volume": [1000],
            }
        )
        mdm._cache["BTC-USDT"] = {"1H": df}

        candle = {
            "t": 1000,
            "o": "50000",
            "h": "50200",
            "l": "49800",
            "c": "50100",
            "v": "1500",
        }
        mdm.update_candle("BTC-USDT", "1H", candle)

        result = mdm.get_df("BTC-USDT", "1H")
        assert len(result) == 1
        assert result["high"].iloc[0] == 50200

    def test_update_candle_ignores_empty_cache(self, mdm):
        mdm._cache["BTC-USDT"] = {}
        candle = {
            "t": 1000,
            "o": "50000",
            "h": "50100",
            "l": "49900",
            "c": "50050",
            "v": "1000",
        }
        mdm.update_candle("BTC-USDT", "1H", candle)
        assert "BTC-USDT" in mdm._cache

    def test_is_stale_true(self, mdm):
        mdm._last_updated["BTC-USDT_1H"] = time.time() - 100
        assert mdm.is_stale("BTC-USDT", "1H", threshold_s=60) is True

    def test_is_stale_false(self, mdm):
        mdm._last_updated["BTC-USDT_1H"] = time.time()
        assert mdm.is_stale("BTC-USDT", "1H", threshold_s=60) is False

    @pytest.mark.asyncio
    async def test_refresh_if_stale_fetches(self, mdm):
        mdm._last_updated["BTC-USDT_1H"] = time.time() - 100
        with patch.object(
            mdm, "_fetch_and_cache", new_callable=AsyncMock
        ) as mock_fetch:
            await mdm.refresh_if_stale("BTC-USDT", "1H")
            assert mock_fetch.called

    @pytest.mark.asyncio
    async def test_refresh_if_stale_skips_when_fresh(self, mdm):
        mdm._last_updated["BTC-USDT_1H"] = time.time()
        with patch.object(
            mdm, "_fetch_and_cache", new_callable=AsyncMock
        ) as mock_fetch:
            await mdm.refresh_if_stale("BTC-USDT", "1H")
            assert not mock_fetch.called

    def test_candle_aggregation_multiple_timeframes(self):
        """Test per candle aggregation su multipli timeframe."""
        from unittest.mock import MagicMock

        client = MagicMock()
        mdm = MarketDataManager(client, max_candles=100)

        # Crea dati candle per 1H
        df_1h = pd.DataFrame(
            {
                "timestamp": [1000, 2000, 3000],
                "open": [50000.0, 50100.0, 50200.0],
                "high": [50100.0, 50200.0, 50300.0],
                "low": [49900.0, 50000.0, 50100.0],
                "close": [50100.0, 50200.0, 50300.0],
                "volume": [1000.0, 1100.0, 1200.0],
            }
        )
        df_1h.attrs["timeframe"] = "1H"

        # Crea dati candle per 4H
        df_4h = pd.DataFrame(
            {
                "timestamp": [4000, 8000],
                "open": [50000.0, 51000.0],
                "high": [51000.0, 52000.0],
                "low": [49500.0, 50500.0],
                "close": [51000.0, 52000.0],
                "volume": [5000.0, 5500.0],
            }
        )
        df_4h.attrs["timeframe"] = "4H"

        mdm._cache["BTC-USDT"] = {"1H": df_1h, "4H": df_4h}

        # Verifica che entrambi i timeframe siano accessibili
        result_1h = mdm.get_df("BTC-USDT", "1H")
        result_4h = mdm.get_df("BTC-USDT", "4H")

        assert result_1h is not None
        assert result_4h is not None
        assert len(result_1h) == 3
        assert len(result_4h) == 2
        assert result_1h.attrs["timeframe"] == "1H"
        assert result_4h.attrs["timeframe"] == "4H"

    def test_candle_aggregation_high_low_consistency(self):
        """Test che high sia sempre >= low nelle candle aggregate."""
        from unittest.mock import MagicMock

        client = MagicMock()
        mdm = MarketDataManager(client, max_candles=100)

        df = pd.DataFrame(
            {
                "timestamp": [1000, 2000, 3000],
                "open": [50000.0, 50100.0, 50200.0],
                "high": [50200.0, 50300.0, 50400.0],
                "low": [49900.0, 50000.0, 50100.0],
                "close": [50100.0, 50200.0, 50300.0],
                "volume": [1000.0, 1100.0, 1200.0],
            }
        )

        mdm._cache["BTC-USDT"] = {"1H": df}

        result = mdm.get_df("BTC-USDT", "1H")

        # Verifica che high >= low per ogni riga
        for _, row in result.iterrows():
            assert row["high"] >= row["low"]
            assert row["high"] >= row["open"]
            assert row["high"] >= row["close"]
            assert row["low"] <= row["open"]
            assert row["low"] <= row["close"]

    def test_candle_aggregation_volume_sum(self):
        """Test che il volume sia sommato correttamente nelle candle."""
        from unittest.mock import MagicMock

        client = MagicMock()
        mdm = MarketDataManager(client, max_candles=100)

        df = pd.DataFrame(
            {
                "timestamp": [1000, 2000, 3000],
                "open": [50000.0, 50100.0, 50200.0],
                "high": [50100.0, 50200.0, 50300.0],
                "low": [49900.0, 50000.0, 50100.0],
                "close": [50100.0, 50200.0, 50300.0],
                "volume": [1000.0, 1100.0, 1200.0],
            }
        )

        mdm._cache["BTC-USDT"] = {"1H": df}

        result = mdm.get_df("BTC-USDT", "1H")

        # Verifica che il volume totale sia la somma
        total_volume = result["volume"].sum()
        assert total_volume == 3300.0
        assert result["volume"].iloc[0] == 1000.0
        assert result["volume"].iloc[1] == 1100.0
        assert result["volume"].iloc[2] == 1200.0

    def test_update_candle_aggregation_maintains_order(self):
        """Test che l'aggregazione mantenga l'ordine cronologico."""
        from unittest.mock import MagicMock

        client = MagicMock()
        mdm = MarketDataManager(client, max_candles=100)

        df = pd.DataFrame(
            {
                "timestamp": [1000, 2000, 3000],
                "open": [50000.0, 50100.0, 50200.0],
                "high": [50100.0, 50200.0, 50300.0],
                "low": [49900.0, 50000.0, 50100.0],
                "close": [50100.0, 50200.0, 50300.0],
                "volume": [1000.0, 1100.0, 1200.0],
            }
        )

        mdm._cache["BTC-USDT"] = {"1H": df}

        # Aggiungi nuova candle
        new_candle = {
            "t": 4000,
            "o": "50300",
            "h": "50400",
            "l": "50200",
            "c": "50400",
            "v": "1300",
        }
        mdm.update_candle("BTC-USDT", "1H", new_candle)

        result = mdm.get_df("BTC-USDT", "1H")

        # Verifica che l'ordine sia mantenuto
        timestamps = result["timestamp"].tolist()
        assert timestamps == [1000, 2000, 3000, 4000]
