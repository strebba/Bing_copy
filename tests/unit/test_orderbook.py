"""Unit tests for data/orderbook module."""

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
