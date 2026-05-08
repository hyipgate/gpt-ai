from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from backtesting.engine import gold_pnl
from core.logger import TradingJournal


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PaperExecutionConfig:
    contract_size: float = 100.0
    commission_per_lot: float = 7.0


class PaperTradeManager:
    def __init__(self, journal: TradingJournal, config: PaperExecutionConfig | None = None) -> None:
        self.journal = journal
        self.config = config or PaperExecutionConfig()

    def update_open_trades(self, symbol: str, bars: pd.DataFrame) -> int:
        if bars.empty:
            return 0
        closed = 0
        for trade in self.journal.open_paper_trades(symbol):
            opened_at = pd.Timestamp(trade["created_at"])
            opened_at = opened_at.tz_localize("UTC") if opened_at.tzinfo is None else opened_at.tz_convert("UTC")
            future = bars[pd.to_datetime(bars["time"], utc=True) > opened_at]
            if future.empty:
                continue
            result = self._resolve_exit(trade, future)
            if result is None:
                continue
            exit_price, exit_r, reason = result
            direction_value = 1 if trade["direction"] == "buy" else -1
            volume = float(trade["volume"] or 0.01)
            gross = gold_pnl(float(trade["entry"]), exit_price, direction_value, volume, self.config.contract_size)
            commission = self.config.commission_per_lot * volume
            profit = gross - commission
            self.journal.update_trade_exit(
                int(trade["id"]),
                exit_price=exit_price,
                profit=profit,
                exit_r=exit_r,
                exit_reason=reason,
            )
            LOGGER.info("paper trade closed id=%s reason=%s pnl=%.2f", trade["id"], reason, profit)
            closed += 1
        return closed

    def _resolve_exit(self, trade: dict, bars: pd.DataFrame) -> tuple[float, float, str] | None:
        direction = trade["direction"]
        entry = float(trade["entry"])
        sl = float(trade["sl"])
        tp = float(trade["tp"])
        risk = abs(entry - sl)
        if risk <= 0:
            return None

        for _, row in bars.iterrows():
            if direction == "buy":
                sl_hit = float(row["low"]) <= sl
                tp_hit = float(row["high"]) >= tp
                if sl_hit and tp_hit:
                    return sl, -1.0, "sl_first_conservative"
                if sl_hit:
                    return sl, -1.0, "sl"
                if tp_hit:
                    return tp, (tp - entry) / risk, "tp"
            else:
                sl_hit = float(row["high"]) >= sl
                tp_hit = float(row["low"]) <= tp
                if sl_hit and tp_hit:
                    return sl, -1.0, "sl_first_conservative"
                if sl_hit:
                    return sl, -1.0, "sl"
                if tp_hit:
                    return tp, (entry - tp) / risk, "tp"
        return None
