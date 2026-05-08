from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

import MetaTrader5 as mt5

from core.config import RiskConfig, TradingConfig
from core.mt5_connector import MT5Connector


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    direction: str
    volume: float
    sl: float
    tp: float
    comment: str = "AI_SMC"


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    order_id: int | None
    price: float | None
    retcode: int | None
    message: str
    raw: str


class ExecutionEngine:
    def __init__(
        self,
        connector: MT5Connector | None = None,
        trading_config: TradingConfig | None = None,
        risk_config: RiskConfig | None = None,
        retries: int = 2,
    ) -> None:
        self.connector = connector or MT5Connector()
        self.trading_config = trading_config or TradingConfig()
        self.risk_config = risk_config or RiskConfig()
        self.retries = retries

    def market_order(self, request: OrderRequest) -> ExecutionResult:
        self.connector.ensure_connected()
        tick = mt5.symbol_info_tick(request.symbol)
        if tick is None:
            return ExecutionResult(False, None, None, None, "missing tick", "")
        order_type = mt5.ORDER_TYPE_BUY if request.direction.lower() == "buy" else mt5.ORDER_TYPE_SELL
        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
        payload = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": request.symbol,
            "volume": request.volume,
            "type": order_type,
            "price": price,
            "sl": request.sl,
            "tp": request.tp,
            "deviation": self.risk_config.max_slippage_points,
            "magic": self.trading_config.magic_number,
            "comment": request.comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = None
        for attempt in range(self.retries + 1):
            result = mt5.order_send(payload)
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                return ExecutionResult(True, int(result.order), float(result.price), int(result.retcode), "filled", str(result._asdict()))
            LOGGER.warning("order_send attempt %s failed: %s", attempt + 1, result)
            time.sleep(0.5)
        retcode = int(result.retcode) if result is not None else None
        return ExecutionResult(False, None, None, retcode, "order failed", str(result))

    def partial_close(self, symbol: str, ticket: int, volume: float) -> ExecutionResult:
        position = next((p for p in mt5.positions_get(symbol=symbol) or [] if p.ticket == ticket), None)
        if position is None:
            return ExecutionResult(False, None, None, None, "position not found", "")
        direction = "sell" if position.type == mt5.POSITION_TYPE_BUY else "buy"
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return ExecutionResult(False, None, None, None, "missing tick", "")
        payload = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_SELL if direction == "sell" else mt5.ORDER_TYPE_BUY,
            "price": tick.bid if direction == "sell" else tick.ask,
            "deviation": self.risk_config.max_slippage_points,
            "magic": self.trading_config.magic_number,
            "comment": "AI_SMC_PARTIAL",
        }
        result = mt5.order_send(payload)
        ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        return ExecutionResult(ok, int(result.order) if ok else None, float(result.price) if ok else None, int(result.retcode) if result else None, "partial close" if ok else "partial close failed", str(result))

    def move_stop(self, ticket: int, sl: float, tp: float) -> ExecutionResult:
        payload = {"action": mt5.TRADE_ACTION_SLTP, "position": ticket, "sl": sl, "tp": tp}
        result = mt5.order_send(payload)
        ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        return ExecutionResult(ok, int(ticket), None, int(result.retcode) if result else None, "stop updated" if ok else "stop update failed", str(result))

    def move_to_breakeven(self, ticket: int) -> ExecutionResult:
        position = next((p for p in mt5.positions_get() or [] if p.ticket == ticket), None)
        if position is None:
            return ExecutionResult(False, None, None, None, "position not found", "")
        return self.move_stop(ticket, float(position.price_open), float(position.tp))

    def trailing_stop_hook(self, ticket: int, new_sl: float) -> ExecutionResult:
        position = next((p for p in mt5.positions_get() or [] if p.ticket == ticket), None)
        if position is None:
            return ExecutionResult(False, None, None, None, "position not found", "")
        return self.move_stop(ticket, new_sl, float(position.tp))


def execute_buy(symbol: str, lot: float, sl: float, tp: float):
    return ExecutionEngine().market_order(OrderRequest(symbol=symbol, direction="buy", volume=lot, sl=sl, tp=tp))
