from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import MetaTrader5 as mt5


LOGGER = logging.getLogger(__name__)

TIMEFRAMES: dict[str, int] = {
    "M1": mt5.TIMEFRAME_M1,
    "M2": mt5.TIMEFRAME_M2,
    "M3": mt5.TIMEFRAME_M3,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


@dataclass(frozen=True)
class SymbolSnapshot:
    symbol: str
    bid: float
    ask: float
    spread_points: float
    point: float
    digits: int
    trade_contract_size: float


class MT5ConnectionError(RuntimeError):
    """Raised when the MT5 terminal cannot be reached or validated."""


class MT5Connector:
    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0) -> None:
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.connected = False

    def initialize(self) -> None:
        for attempt in range(1, self.max_retries + 1):
            if mt5.initialize():
                self.connected = True
                LOGGER.info("MT5 initialized")
                return
            LOGGER.warning("MT5 initialize failed attempt %s: %s", attempt, mt5.last_error())
            time.sleep(self.retry_delay)
        raise MT5ConnectionError(f"MT5 initialize failed: {mt5.last_error()}")

    def shutdown(self) -> None:
        if self.connected:
            mt5.shutdown()
            self.connected = False
            LOGGER.info("MT5 shutdown complete")

    def reconnect(self) -> None:
        self.shutdown()
        self.initialize()

    def ensure_connected(self) -> None:
        if self.connected and mt5.terminal_info() is not None:
            return
        self.reconnect()

    def timeframe(self, value: str | int) -> int:
        if isinstance(value, int):
            return value
        key = value.upper()
        if key not in TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe '{value}'. Valid: {sorted(TIMEFRAMES)}")
        return TIMEFRAMES[key]

    def validate_symbol(self, symbol: str) -> None:
        self.ensure_connected()
        info = mt5.symbol_info(symbol)
        if info is None:
            raise ValueError(f"Symbol not found in MT5: {symbol}")
        if not info.visible and not mt5.symbol_select(symbol, True):
            raise ValueError(f"Symbol exists but cannot be selected: {symbol}")

    def snapshot(self, symbol: str) -> SymbolSnapshot:
        self.validate_symbol(symbol)
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if info is None or tick is None:
            raise MT5ConnectionError(f"Missing symbol/tick snapshot for {symbol}")
        spread_points = (tick.ask - tick.bid) / info.point if info.point else float("inf")
        return SymbolSnapshot(
            symbol=symbol,
            bid=float(tick.bid),
            ask=float(tick.ask),
            spread_points=float(spread_points),
            point=float(info.point),
            digits=int(info.digits),
            trade_contract_size=float(info.trade_contract_size),
        )

    def spread_ok(self, symbol: str, max_spread_points: float) -> bool:
        return self.snapshot(symbol).spread_points <= max_spread_points


@contextmanager
def mt5_session(connector: MT5Connector | None = None) -> Iterator[MT5Connector]:
    connector = connector or MT5Connector()
    connector.initialize()
    try:
        yield connector
    finally:
        connector.shutdown()
