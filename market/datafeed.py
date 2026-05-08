from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import MetaTrader5 as mt5
import pandas as pd

from core.mt5_connector import MT5Connector


REQUIRED_COLUMNS = ("time", "open", "high", "low", "close", "tick_volume", "spread")


@dataclass(frozen=True)
class MarketDataRequest:
    symbol: str
    timeframe: str | int = "M5"
    bars: int = 5000
    start: datetime | None = None
    end: datetime | None = None


class MT5DataFeed:
    def __init__(self, connector: MT5Connector | None = None) -> None:
        self.connector = connector or MT5Connector()

    def get_rates(self, request: MarketDataRequest) -> pd.DataFrame:
        self.connector.ensure_connected()
        self.connector.validate_symbol(request.symbol)
        timeframe = self.connector.timeframe(request.timeframe)

        if request.start and request.end:
            rates = mt5.copy_rates_range(request.symbol, timeframe, request.start, request.end)
        else:
            rates = mt5.copy_rates_from_pos(request.symbol, timeframe, 0, request.bars)

        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No rates returned for {request.symbol}: {mt5.last_error()}")
        return normalize_rates(pd.DataFrame(rates))


def normalize_rates(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Market data missing required columns: {missing}")
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], unit="s", utc=True, errors="coerce")
    out = out.dropna(subset=["time"]).sort_values("time").drop_duplicates("time")
    numeric = [c for c in out.columns if c != "time"]
    out[numeric] = out[numeric].apply(pd.to_numeric, errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out.reset_index(drop=True)


def validate_ohlcv(df: pd.DataFrame, columns: Iterable[str] = REQUIRED_COLUMNS) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing market columns: {missing}")
    if (df["high"] < df[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("Invalid OHLC data: high below candle body or low")
    if (df["low"] > df[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Invalid OHLC data: low above candle body or high")


def get_data(symbol: str = "XAUUSD", timeframe: str | int = "M5", bars: int = 5000) -> pd.DataFrame:
    connector = MT5Connector()
    connector.initialize()
    try:
        return MT5DataFeed(connector).get_rates(MarketDataRequest(symbol=symbol, timeframe=timeframe, bars=bars))
    finally:
        connector.shutdown()
