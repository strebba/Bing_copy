"""Unit tests for trade_journal module."""

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from analytics.trade_journal import TradeJournal


class MockSignal:
    def __init__(self):
        self.symbol = "BTC-USDT"
        self.direction = MagicMock()
        self.direction.value = "LONG"
        self.strategy_name = "alpha"
        self.confidence = 0.85
        self.entry_price = 50000.0
        self.stop_loss = 49000.0
        self.take_profit = 52000.0
        self.risk_reward = 2.0


class TestTradeJournal:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.journal_path = os.path.join(self.temp_dir, "test_journal.jsonl")
        self.journal = TradeJournal(log_path=self.journal_path)

    def teardown_method(self):
        if os.path.exists(self.journal_path):
            os.remove(self.journal_path)
        shutil.rmtree(self.temp_dir)

    def test_log_creates_file(self):
        self.journal.log("TEST", {"key": "value"})
        assert os.path.exists(self.journal_path)

    def test_log_writes_jsonl(self):
        self.journal.log("TEST", {"key": "value", "num": 123})
        with open(self.journal_path) as f:
            line = f.readline()
        record = json.loads(line)
        assert record["event"] == "TEST"
        assert record["key"] == "value"
        assert record["num"] == 123
        assert "ts" in record

    def test_log_signal(self):
        signal = MockSignal()
        self.journal.log_signal(signal)
        with open(self.journal_path) as f:
            line = f.readline()
        record = json.loads(line)
        assert record["event"] == "SIGNAL"
        assert record["symbol"] == "BTC-USDT"
        assert record["direction"] == "LONG"
        assert record["strategy"] == "alpha"
        assert record["confidence"] == 0.85
        assert record["entry"] == 50000.0
        assert record["sl"] == 49000.0
        assert record["tp"] == 52000.0
        assert record["rr"] == 2.0

    def test_log_order(self):
        self.journal.log_order("SUBMIT", {"order_id": "12345"})
        with open(self.journal_path) as f:
            line = f.readline()
        record = json.loads(line)
        assert record["event"] == "ORDER_SUBMIT"
        assert record["order_id"] == "12345"

    def test_log_order_fill(self):
        self.journal.log_order_fill(
            symbol="BTC-USDT",
            direction="LONG",
            requested_price=50000.0,
            fill_price=50010.0,
            slippage_bps=2.0,
            executed_qty=0.1,
        )
        with open(self.journal_path) as f:
            line = f.readline()
        record = json.loads(line)
        assert record["event"] == "ORDER_FILL"
        assert record["symbol"] == "BTC-USDT"
        assert record["direction"] == "LONG"
        assert record["requested_price"] == 50000.0
        assert record["fill_price"] == 50010.0
        assert record["slippage_bps"] == 2.0
        assert record["executed_qty"] == 0.1

    def test_log_trade_closed_with_pnl(self):
        self.journal.log_trade_closed(
            symbol="BTC-USDT",
            direction="LONG",
            entry=50000.0,
            exit_price=51000.0,
            pnl=100.0,
            reason="take_profit",
        )
        with open(self.journal_path) as f:
            line = f.readline()
        record = json.loads(line)
        assert record["event"] == "TRADE_CLOSED"
        assert record["symbol"] == "BTC-USDT"
        assert record["direction"] == "LONG"
        assert record["entry"] == 50000.0
        assert record["exit"] == 51000.0
        assert record["pnl_usdt"] == 100.0
        assert record["reason"] == "take_profit"

    def test_log_trade_closed_with_slippage(self):
        self.journal.log_trade_closed(
            symbol="ETH-USDT",
            direction="SHORT",
            entry=3000.0,
            exit_price=2950.0,
            pnl=50.0,
            reason="stop_loss",
            requested_price=2955.0,
            fill_price=2950.0,
            slippage_bps=16.9,
        )
        with open(self.journal_path) as f:
            line = f.readline()
        record = json.loads(line)
        assert record["requested_price"] == 2955.0
        assert record["fill_price"] == 2950.0
        assert record["slippage_bps"] == 16.9

    def test_export_to_csv(self):
        self.journal.log_trade_closed(
            symbol="BTC-USDT",
            direction="LONG",
            entry=50000.0,
            exit_price=51000.0,
            pnl=100.0,
            reason="take_profit",
        )
        self.journal.log_trade_closed(
            symbol="ETH-USDT",
            direction="SHORT",
            entry=3000.0,
            exit_price=2900.0,
            pnl=100.0,
            reason="take_profit",
        )

        csv_path = os.path.join(self.temp_dir, "exports.csv")
        df = pd.read_json(self.journal_path, lines=True)
        df.to_csv(csv_path, index=False)

        assert os.path.exists(csv_path)
        df_csv = pd.read_csv(csv_path)
        assert len(df_csv) == 2
        assert "symbol" in df_csv.columns
        assert "pnl_usdt" in df_csv.columns
