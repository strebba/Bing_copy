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

    def test_trade_open_entry_registered_with_all_fields(self):
        """Trade aperto → entry registrata con tutti i campi."""
        signal = MockSignal()
        self.journal.log_signal(signal)

        with open(self.journal_path) as f:
            lines = f.readlines()

        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["event"] == "SIGNAL"
        assert record["symbol"] == "BTC-USDT"
        assert record["direction"] == "LONG"
        assert record["strategy"] == "alpha"
        assert record["confidence"] == 0.85
        assert record["entry"] == 50000.0
        assert record["sl"] == 49000.0
        assert record["tp"] == 52000.0
        assert record["rr"] == 2.0
        assert "ts" in record

    def test_trade_closed_updates_entry_with_pnl_slippage_duration(self):
        """Trade chiuso → entry aggiornata con PnL, slippage, durata."""
        # Simula un trade aperto
        self.journal.log_signal(MockSignal())

        # Chiude il trade con PnL e slippage
        self.journal.log_trade_closed(
            symbol="BTC-USDT",
            direction="LONG",
            entry=50000.0,
            exit_price=51000.0,
            pnl=100.0,
            reason="take_profit",
            requested_price=51000.0,
            fill_price=50995.0,
            slippage_bps=1.0,
        )

        with open(self.journal_path) as f:
            lines = f.readlines()

        # Verifica l'ultima entry (TRADE_CLOSED)
        closed_record = json.loads(lines[-1])
        assert closed_record["event"] == "TRADE_CLOSED"
        assert closed_record["pnl_usdt"] == 100.0
        assert closed_record["entry"] == 50000.0
        assert closed_record["exit"] == 51000.0
        assert closed_record["reason"] == "take_profit"
        assert closed_record["requested_price"] == 51000.0
        assert closed_record["fill_price"] == 50995.0
        assert closed_record["slippage_bps"] == 1.0

    def test_csv_export_with_pnl_and_slippage_data(self):
        """Export dei trade in formato CSV funzionante con PnL e slippage."""
        # Trade con PnL positivo e slippage
        self.journal.log_trade_closed(
            symbol="BTC-USDT",
            direction="LONG",
            entry=50000.0,
            exit_price=52000.0,
            pnl=200.0,
            reason="take_profit",
            requested_price=52000.0,
            fill_price=51990.0,
            slippage_bps=1.92,
        )

        # Trade con perdita
        self.journal.log_trade_closed(
            symbol="ETH-USDT",
            direction="SHORT",
            entry=3000.0,
            exit_price=3100.0,
            pnl=-100.0,
            reason="stop_loss",
            requested_price=3100.0,
            fill_price=3105.0,
            slippage_bps=1.61,
        )

        csv_path = os.path.join(self.temp_dir, "trades_export.csv")
        df = pd.read_json(self.journal_path, lines=True)
        df.to_csv(csv_path, index=False)

        assert os.path.exists(csv_path)
        df_csv = pd.read_csv(csv_path)
        assert len(df_csv) == 2

        # Verifica colonne PnL
        assert "pnl_usdt" in df_csv.columns
        pnl_values = df_csv["pnl_usdt"].tolist()
        assert 200.0 in pnl_values
        assert -100.0 in pnl_values

        # Verifica colonne slippage
        assert "slippage_bps" in df_csv.columns
        slippage_values = df_csv["slippage_bps"].tolist()
        assert 1.92 in slippage_values
        assert 1.61 in slippage_values

    def test_multiple_trades_logged_sequentially(self):
        """Test che più trade vengano loggati in sequenza nel file."""
        # Trade 1
        signal1 = MockSignal()
        self.journal.log_signal(signal1)

        # Trade 2
        signal2 = MockSignal()
        signal2.symbol = "ETH-USDT"
        signal2.direction.value = "SHORT"
        self.journal.log_signal(signal2)

        with open(self.journal_path) as f:
            lines = f.readlines()

        assert len(lines) == 2

        record1 = json.loads(lines[0])
        record2 = json.loads(lines[1])

        assert record1["symbol"] == "BTC-USDT"
        assert record2["symbol"] == "ETH-USDT"
