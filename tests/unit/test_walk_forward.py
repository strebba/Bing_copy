"""Unit tests for walk_forward module."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from analytics.performance import PerformanceTracker
from backtest.walk_forward import WalkForwardAnalyzer, WalkForwardWindow
from backtest.backtester import BacktestConfig
from strategy.base_strategy import BaseStrategy


class DummyStrategy(BaseStrategy):
    pass


class TestWalkForwardWindow:
    def test_is_score_returns_sharpe_when_tracker_exists(self):
        tracker = MagicMock(spec=PerformanceTracker)
        tracker.sharpe_ratio.return_value = 1.5
        window = WalkForwardWindow(
            window_id=0,
            train_start=0,
            train_end=100,
            test_start=100,
            test_end=120,
            test_tracker=tracker,
        )
        assert window.is_score == 1.5

    def test_is_score_returns_none_when_no_tracker(self):
        window = WalkForwardWindow(
            window_id=0,
            train_start=0,
            train_end=100,
            test_start=100,
            test_end=120,
        )
        assert window.is_score is None


class TestWalkForwardAnalyzer:
    def setup_method(self):
        self.strategy = MagicMock(spec=BaseStrategy)

    def _make_test_df(self, n_rows: int = 1000) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=n_rows, freq="1h")
        return pd.DataFrame(
            {
                "timestamp": dates,
                "open": 50000 + pd.Series(range(n_rows)) * 0.1,
                "high": 50100 + pd.Series(range(n_rows)) * 0.1,
                "high_": 50100 + pd.Series(range(n_rows)) * 0.1,
                "low": 49900 + pd.Series(range(n_rows)) * 0.1,
                "close": 50000 + pd.Series(range(n_rows)) * 0.1,
                "volume": 1000,
            }
        )

    def test_windows_created_correctly(self):
        analyzer = WalkForwardAnalyzer(
            strategy=self.strategy,
            n_windows=6,
            train_ratio=0.8,
        )
        df = self._make_test_df(600)

        with patch("backtest.walk_forward.Backtester") as MockBacktester:
            mock_tracker = MagicMock(spec=PerformanceTracker)
            mock_tracker.sharpe_ratio.return_value = 1.0
            MockBacktester.return_value.run.return_value = mock_tracker

            results = analyzer.run(df, "BTC-USDT")

        assert len(results) == 6

        w = results[0]
        assert w.window_id == 0
        assert w.train_start == 0
        assert w.test_end > w.test_start

    def test_train_test_split_80_20(self):
        analyzer = WalkForwardAnalyzer(
            strategy=self.strategy,
            n_windows=6,
            train_ratio=0.8,
        )
        df = self._make_test_df(600)

        with patch("backtest.walk_forward.Backtester") as MockBacktester:
            mock_tracker = MagicMock(spec=PerformanceTracker)
            mock_tracker.sharpe_ratio.return_value = 1.0
            MockBacktester.return_value.run.return_value = mock_tracker

            results = analyzer.run(df, "BTC-USDT")

        w = results[0]
        train_size = w.train_end - w.train_start
        test_size = w.test_end - w.test_start
        total = train_size + test_size
        assert train_size / total == pytest.approx(0.8, rel=0.01)
        assert test_size / total == pytest.approx(0.2, rel=0.01)

    def test_no_data_leakage_between_train_and_test(self):
        analyzer = WalkForwardAnalyzer(
            strategy=self.strategy,
            n_windows=6,
            train_ratio=0.8,
        )
        df = self._make_test_df(600)

        with patch("backtest.walk_forward.Backtester") as MockBacktester:
            mock_tracker = MagicMock(spec=PerformanceTracker)
            mock_tracker.sharpe_ratio.return_value = 1.0
            MockBacktester.return_value.run.return_value = mock_tracker

            results = analyzer.run(df, "BTC-USDT")

        for w in results:
            assert w.train_end == w.test_start, "Train/test should meet at split point"
            assert w.train_start < w.train_end
            assert w.test_start < w.test_end
            assert w.train_end < w.test_end

    def test_output_contains_metrics_for_each_window(self):
        analyzer = WalkForwardAnalyzer(
            strategy=self.strategy,
            n_windows=3,
            train_ratio=0.8,
        )
        df = self._make_test_df(1200)

        with patch("backtest.walk_forward.Backtester") as MockBacktester:
            mock_tracker = MagicMock(spec=PerformanceTracker)
            mock_tracker.sharpe_ratio.return_value = 1.2
            MockBacktester.return_value.run.return_value = mock_tracker

            results = analyzer.run(df, "BTC-USDT")

        assert all(w.train_tracker is not None for w in results)
        assert all(w.test_tracker is not None for w in results)

    def test_run_with_small_dataframe_skips_backtest(self):
        analyzer = WalkForwardAnalyzer(
            strategy=self.strategy,
            n_windows=6,
            train_ratio=0.8,
        )
        df = self._make_test_df(50)

        results = analyzer.run(df, "BTC-USDT")

        assert len(results) == 6
        for w in results:
            assert w.train_tracker is None
            assert w.test_tracker is None

    def test_custom_train_ratio(self):
        analyzer = WalkForwardAnalyzer(
            strategy=self.strategy,
            n_windows=4,
            train_ratio=0.7,
        )
        df = self._make_test_df(400)

        with patch("backtest.walk_forward.Backtester") as MockBacktester:
            mock_tracker = MagicMock(spec=PerformanceTracker)
            mock_tracker.sharpe_ratio.return_value = 1.0
            MockBacktester.return_value.run.return_value = mock_tracker

            results = analyzer.run(df, "BTC-USDT")

        w = results[0]
        train_size = w.train_end - w.train_start
        test_size = w.test_end - w.test_start
        total = train_size + test_size
        assert train_size / total == pytest.approx(0.7, rel=0.01)

    def test_six_windows_created_with_80_20_split(self):
        """Test che 6 finestre sono create correttamente con split 80/20."""
        analyzer = WalkForwardAnalyzer(
            strategy=self.strategy,
            n_windows=6,
            train_ratio=0.8,
        )
        # Dataset con almeno 6 finestre valide
        df = self._make_test_df(1800)

        with patch("backtest.walk_forward.Backtester") as MockBacktester:
            mock_tracker = MagicMock(spec=PerformanceTracker)
            mock_tracker.sharpe_ratio.return_value = 1.2
            MockBacktester.return_value.run.return_value = mock_tracker

            results = analyzer.run(df, "BTC-USDT")

        assert len(results) == 6

        # Verifica che ogni finestra sia creata correttamente
        for i, w in enumerate(results):
            assert w.window_id == i
            assert w.train_start < w.train_end
            assert w.test_start == w.train_end
            assert w.test_end > w.test_start

            # Verifica split 80/20
            train_size = w.train_end - w.train_start
            test_size = w.test_end - w.test_start
            total = train_size + test_size
            assert train_size / total == pytest.approx(0.8, rel=0.02)
            assert test_size / total == pytest.approx(0.2, rel=0.02)

    def test_no_data_leakage_across_windows(self):
        """Test che non ci sia data leakage tra train e test set."""
        analyzer = WalkForwardAnalyzer(
            strategy=self.strategy,
            n_windows=6,
            train_ratio=0.8,
        )
        df = self._make_test_df(1200)

        with patch("backtest.walk_forward.Backtester") as MockBacktester:
            mock_tracker = MagicMock(spec=PerformanceTracker)
            mock_tracker.sharpe_ratio.return_value = 1.0
            MockBacktester.return_value.run.return_value = mock_tracker

            results = analyzer.run(df, "BTC-USDT")

        for w in results:
            # Train e test devono essere contigui ma non sovrapposti
            assert w.train_end == w.test_start
            assert w.train_start < w.train_end
            assert w.test_start < w.test_end

            # Nessuna sovrapposizione
            assert w.train_start < w.test_start
            assert w.train_end <= w.test_end

    def test_metrics_output_for_each_window(self):
        """Test che output contiene metriche per ogni finestra."""
        analyzer = WalkForwardAnalyzer(
            strategy=self.strategy,
            n_windows=3,
            train_ratio=0.8,
        )
        df = self._make_test_df(1200)

        with patch("backtest.walk_forward.Backtester") as MockBacktester:
            mock_tracker = MagicMock(spec=PerformanceTracker)
            mock_tracker.sharpe_ratio.return_value = 1.5
            MockBacktester.return_value.run.return_value = mock_tracker

            results = analyzer.run(df, "BTC-USDT")

        # Ogni finestra deve avere sia train che test tracker
        assert len(results) == 3
        for w in results:
            assert w.train_tracker is not None
            assert w.test_tracker is not None
            assert w.train_tracker.sharpe_ratio() == 1.5
            assert w.test_tracker.sharpe_ratio() == 1.5
            assert w.is_score == 1.5
