from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from sqlalchemy import Column, DateTime, Float, Integer, MetaData, String, Table, Text, create_engine, insert, select, update, text
from sqlalchemy.engine import Engine

from core.config import resolve_path


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(resolve_path("logs/trader.log"), encoding="utf-8")],
    )


metadata = MetaData()

trades_table = Table(
    "trades",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("created_at", DateTime, nullable=False),
    Column("symbol", String(32), nullable=False),
    Column("direction", String(8), nullable=False),
    Column("setup_type", String(64)),
    Column("confidence", Float),
    Column("probability", Float),
    Column("entry", Float),
    Column("sl", Float),
    Column("tp", Float),
    Column("volume", Float),
    Column("spread_points", Float),
    Column("slippage_points", Float),
    Column("setup_score", Float),
    Column("expected_r", Float),
    Column("regime", String(64)),
    Column("execution_risk", Float),
    Column("quality_tier", String(16)),
    Column("profit", Float),
    Column("exit_r", Float),
    Column("exit_price", Float),
    Column("exit_reason", String(32)),
    Column("closed_at", DateTime),
    Column("status", String(32)),
    Column("execution_details", Text),
    Column("model_decision", Text),
)

equity_table = Table(
    "equity_snapshots",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("created_at", DateTime, nullable=False),
    Column("equity", Float, nullable=False),
    Column("balance", Float),
    Column("drawdown", Float),
)

risk_events_table = Table(
    "risk_events",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("created_at", DateTime, nullable=False),
    Column("event_type", String(64), nullable=False),
    Column("message", Text, nullable=False),
)


@dataclass(frozen=True)
class TradeLog:
    symbol: str
    direction: str
    setup_type: str
    confidence: float
    probability: float
    entry: float
    sl: float
    tp: float
    volume: float
    spread_points: float
    slippage_points: float = 0.0
    setup_score: float | None = None
    expected_r: float | None = None
    regime: str | None = None
    execution_risk: float | None = None
    quality_tier: str | None = None
    profit: float | None = None
    exit_r: float | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    closed_at: datetime | None = None
    status: str = "submitted"
    execution_details: str = ""
    model_decision: str = ""


class TradingJournal:
    def __init__(self, database_url: str = "sqlite:///database/trading_journal.sqlite") -> None:
        if database_url.startswith("sqlite:///"):
            db_path = resolve_path(database_url.replace("sqlite:///", ""))
            db_path.parent.mkdir(parents=True, exist_ok=True)
            database_url = f"sqlite:///{db_path.as_posix()}"
        self.engine: Engine = create_engine(database_url, future=True)
        metadata.create_all(self.engine)
        self._migrate_sqlite_columns()

    def _migrate_sqlite_columns(self) -> None:
        additions = {
            "exit_price": "REAL",
            "exit_reason": "TEXT",
            "closed_at": "TIMESTAMP",
            "setup_score": "REAL",
            "expected_r": "REAL",
            "regime": "TEXT",
            "execution_risk": "REAL",
            "quality_tier": "TEXT",
        }
        with self.engine.begin() as conn:
            existing = {row[1] for row in conn.execute(text("PRAGMA table_info(trades)"))}
            for column, ddl_type in additions.items():
                if column not in existing:
                    conn.execute(text(f"ALTER TABLE trades ADD COLUMN {column} {ddl_type}"))

    def log_trade(self, trade: TradeLog) -> None:
        payload = asdict(trade) | {"created_at": datetime.now(timezone.utc)}
        with self.engine.begin() as conn:
            conn.execute(insert(trades_table), payload)

    def open_paper_trades(self, symbol: str | None = None) -> list[dict]:
        query = select(trades_table).where(trades_table.c.status == "paper_open")
        if symbol:
            query = query.where(trades_table.c.symbol == symbol)
        with self.engine.begin() as conn:
            return [dict(row._mapping) for row in conn.execute(query)]

    def update_trade_exit(
        self,
        trade_id: int,
        exit_price: float,
        profit: float,
        exit_r: float,
        exit_reason: str,
        status: str = "paper_closed",
    ) -> None:
        payload = {
            "exit_price": exit_price,
            "profit": profit,
            "exit_r": exit_r,
            "exit_reason": exit_reason,
            "closed_at": datetime.now(timezone.utc),
            "status": status,
        }
        with self.engine.begin() as conn:
            conn.execute(update(trades_table).where(trades_table.c.id == trade_id), payload)

    def log_equity(self, equity: float, balance: float | None = None, drawdown: float | None = None) -> None:
        with self.engine.begin() as conn:
            conn.execute(insert(equity_table), {"created_at": datetime.now(timezone.utc), "equity": equity, "balance": balance, "drawdown": drawdown})

    def log_risk_event(self, event_type: str, message: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(insert(risk_events_table), {"created_at": datetime.now(timezone.utc), "event_type": event_type, "message": message})
