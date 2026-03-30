"""Unit tests for data/orderbook module."""

import pytest

from data.orderbook import OrderBookProcessor


class TestOrderBookProcessor:
    def setup_method(self):
        self.obp = OrderBookProcessor(depth_levels=20)

    def test_initialization(self):
        assert self.obp._depth_levels == 20
        assert self.obp._books == {}

    def test_update_with_valid_data(self):
        data = {
            "bids": [["50000", "1.0"], ["49999", "2.0"]],
            "asks": [["50001", "1.5"], ["50002", "3.0"]],
        }
        self.obp.update("BTC-USDT", data)
        assert "BTC-USDT" in self.obp._books

    def test_best_bid(self):
        data = {
            "bids": [["50000", "1.0"], ["49999", "2.0"]],
            "asks": [["50001", "1.5"]],
        }
        self.obp.update("BTC-USDT", data)
        assert self.obp.best_bid("BTC-USDT") == 50000.0

    def test_best_ask(self):
        data = {
            "bids": [["50000", "1.0"]],
            "asks": [["50001", "1.5"], ["50002", "3.0"]],
        }
        self.obp.update("BTC-USDT", data)
        assert self.obp.best_ask("BTC-USDT") == 50001.0

    def test_mid_price(self):
        data = {
            "bids": [["50000", "1.0"]],
            "asks": [["50010", "1.0"]],
        }
        self.obp.update("BTC-USDT", data)
        assert self.obp.mid_price("BTC-USDT") == 50005.0

    def test_mid_price_returns_none_when_no_data(self):
        assert self.obp.mid_price("UNKNOWN") is None

    def test_spread_pct(self):
        data = {
            "bids": [["50000", "1.0"]],
            "asks": [["50010", "1.0"]],
        }
        self.obp.update("BTC-USDT", data)
        spread = self.obp.spread_pct("BTC-USDT")
        assert spread == 0.0002

    def test_spread_pct_returns_zero_when_no_data(self):
        assert self.obp.spread_pct("UNKNOWN") == 0.0

    def test_depth_imbalance_bullish(self):
        data = {
            "bids": [["50000", "10"], ["49999", "10"]],
            "asks": [["50001", "1"], ["50002", "1"]],
        }
        self.obp.update("BTC-USDT", data)
        imbalance = self.obp.depth_imbalance("BTC-USDT", levels=2)
        assert imbalance > 0

    def test_depth_imbalance_bearish(self):
        data = {
            "bids": [["50000", "1"], ["49999", "1"]],
            "asks": [["50001", "10"], ["50002", "10"]],
        }
        self.obp.update("BTC-USDT", data)
        imbalance = self.obp.depth_imbalance("BTC-USDT", levels=2)
        assert imbalance < 0

    def test_depth_imbalance_returns_zero_when_empty(self):
        assert self.obp.depth_imbalance("UNKNOWN") == 0.0

    def test_depth_imbalance_with_partial_levels(self):
        data = {
            "bids": [["50000", "10"], ["49999", "10"], ["49998", "10"]],
            "asks": [["50001", "1"], ["50002", "1"]],
        }
        self.obp.update("BTC-USDT", data)
        imbalance = self.obp.depth_imbalance("BTC-USDT", levels=2)
        assert imbalance > 0

    def test_bids_sorted_by_price_descending(self):
        data = {
            "bids": [["49999", "1.0"], ["50001", "1.0"], ["50000", "1.0"]],
            "asks": [],
        }
        self.obp.update("BTC-USDT", data)
        bids = self.obp._books["BTC-USDT"]["bids"]
        assert bids[0][0] == 50001.0
        assert bids[1][0] == 50000.0
        assert bids[2][0] == 49999.0

    def test_asks_sorted_by_price_ascending(self):
        data = {
            "bids": [],
            "asks": [["50002", "1.0"], ["50000", "1.0"], ["50001", "1.0"]],
        }
        self.obp.update("BTC-USDT", data)
        asks = self.obp._books["BTC-USDT"]["asks"]
        assert asks[0][0] == 50000.0
        assert asks[1][0] == 50001.0
        assert asks[2][0] == 50002.0

    def test_spread_calculation_bid_ask_difference(self):
        """Test base per spread calculation: differenza tra ask e bid."""
        data = {
            "bids": [["50000.0", "1.0"]],
            "asks": [["50010.0", "1.0"]],
        }
        self.obp.update("BTC-USDT", data)

        spread_pct = self.obp.spread_pct("BTC-USDT")
        expected_spread = (50010.0 - 50000.0) / 50000.0
        assert spread_pct == pytest.approx(expected_spread, rel=1e-10)
        assert spread_pct == 0.0002  # 0.02%

    def test_spread_with_tight_spread(self):
        """Test spread con spread molto stretto."""
        data = {
            "bids": [["50000.0", "1.0"]],
            "asks": [["50000.5", "1.0"]],
        }
        self.obp.update("BTC-USDT", data)

        spread_pct = self.obp.spread_pct("BTC-USDT")
        expected = (50000.5 - 50000.0) / 50000.0
        assert spread_pct == pytest.approx(expected, rel=1e-10)

    def test_spread_with_wide_spread(self):
        """Test spread con spread molto ampio."""
        data = {
            "bids": [["50000.0", "1.0"]],
            "asks": [["50100.0", "1.0"]],
        }
        self.obp.update("BTC-USDT", data)

        spread_pct = self.obp.spread_pct("BTC-USDT")
        expected = (50100.0 - 50000.0) / 50000.0
        assert spread_pct == pytest.approx(expected, rel=1e-10)
        assert spread_pct == 0.002  # 0.2%

    def test_spread_updates_with_new_orderbook_data(self):
        """Test che lo spread si aggiorna con nuovi dati orderbook."""
        # Primo aggiornamento
        data1 = {
            "bids": [["50000.0", "1.0"]],
            "asks": [["50010.0", "1.0"]],
        }
        self.obp.update("BTC-USDT", data1)
        spread1 = self.obp.spread_pct("BTC-USDT")

        # Secondo aggiornamento con prezzi diversi
        data2 = {
            "bids": [["50100.0", "1.0"]],
            "asks": [["50120.0", "1.0"]],
        }
        self.obp.update("BTC-USDT", data2)
        spread2 = self.obp.spread_pct("BTC-USDT")

        # Verifica che lo spread sia cambiato
        assert spread1 != spread2
        expected_spread2 = (50120.0 - 50100.0) / 50100.0
        assert spread2 == pytest.approx(expected_spread2, rel=1e-10)

    def test_spread_calculation_with_multiple_levels(self):
        """Test spread considera solo i migliori livelli."""
        data = {
            "bids": [
                ["50000.0", "1.0"],  # Miglior bid
                ["49999.0", "2.0"],
                ["49998.0", "3.0"],
            ],
            "asks": [
                ["50010.0", "1.0"],  # Miglior ask
                ["50011.0", "2.0"],
                ["50012.0", "3.0"],
            ],
        }
        self.obp.update("BTC-USDT", data)

        spread_pct = self.obp.spread_pct("BTC-USDT")
        # Dovrebbe usare solo i migliori livelli
        expected = (50010.0 - 50000.0) / 50000.0
        assert spread_pct == pytest.approx(expected, rel=1e-10)
