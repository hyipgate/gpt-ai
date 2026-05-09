from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.metrics import summarize_trades


@dataclass(frozen=True)
class BacktestSettings:
    initial_equity: float = 10_000.0
    risk_per_trade: float = 0.005
    reward_r: float = 2.0
    horizon: int = 48
    spread_points: float = 25.0
    slippage_points: float = 5.0
    point_value: float = 0.01
    commission_per_lot: float = 7.0
    confidence_threshold: float = 0.70
    lot_size: float = 0.01
    contract_size: float = 100.0
    sizing_mode: str = "fixed_lot"
    max_spread_points: float | None = None
    min_atr_points: float | None = None


def gold_pnl(entry: float, exit_price: float, direction: int, lot_size: float = 0.01, contract_size: float = 100.0) -> float:
    """Dollar PnL for XAUUSD. With 0.01 lot and 100oz contract, a $1 move equals $1."""
    return direction * (exit_price - entry) * contract_size * lot_size


class BacktestEngine:
    def __init__(self, settings: BacktestSettings | None = None) -> None:
        self.settings = settings or BacktestSettings()

    def run(self, df: pd.DataFrame, probabilities: pd.Series, directions: pd.Series) -> dict[str, pd.DataFrame | dict[str, float]]:
        equity = self.settings.initial_equity
        equity_records = []
        trades = []
        costs_points = self.settings.spread_points + self.settings.slippage_points
        atr = df.get("atr", (df["high"] - df["low"]).rolling(14).mean()).bfill()

        i = 0
        while i < len(df) - 1:
            equity_records.append({"time": df.iloc[i]["time"], "equity": equity})
            confidence = float(probabilities.iloc[i])
            direction = int(directions.iloc[i])
            if confidence < self.settings.confidence_threshold or direction == 0:
                i += 1
                continue
            spread = float(df.iloc[i].get("spread", 0.0))
            if self.settings.max_spread_points is not None and spread > self.settings.max_spread_points:
                i += 1
                continue

            risk_distance = max(float(atr.iloc[i]), self.settings.point_value)
            atr_points = risk_distance / max(self.settings.point_value, 1e-12)
            if self.settings.min_atr_points is not None and atr_points < self.settings.min_atr_points:
                i += 1
                continue
            risk_cash = equity * self.settings.risk_per_trade
            entry_mid = float(df.iloc[i]["close"])
            entry = entry_mid + direction * costs_points * self.settings.point_value
            stop = entry - direction * risk_distance
            target = entry + direction * risk_distance * self.settings.reward_r
            if self.settings.sizing_mode == "fixed_lot":
                volume_lots = self.settings.lot_size
            else:
                volume_lots = risk_cash / max(risk_distance * self.settings.contract_size, 1e-12)
            future_start = i + 1
            future_end = min(i + 1 + self.settings.horizon, len(df))
            future = df.iloc[future_start:future_end]
            exit_price = float(future.iloc[-1]["close"]) if not future.empty else entry
            exit_pos = i
            exit_r = 0.0

            for pos in range(future_start, future_end):
                row = df.iloc[pos]
                if direction > 0:
                    if row["low"] <= stop:
                        exit_price, exit_pos, exit_r = stop, pos, -1.0
                        break
                    if row["high"] >= target:
                        exit_price, exit_pos, exit_r = target, pos, self.settings.reward_r
                        break
                else:
                    if row["high"] >= stop:
                        exit_price, exit_pos, exit_r = stop, pos, -1.0
                        break
                    if row["low"] <= target:
                        exit_price, exit_pos, exit_r = target, pos, self.settings.reward_r
                        break
            else:
                exit_r = direction * (exit_price - entry) / risk_distance

            gross_pnl = gold_pnl(entry, exit_price, direction, volume_lots, self.settings.contract_size)
            commission = self.settings.commission_per_lot * volume_lots
            pnl = gross_pnl - commission
            equity += pnl
            trades.append(
                {
                    "entry_time": df.iloc[i]["time"],
                    "exit_time": df.iloc[exit_pos]["time"] if exit_pos < len(df) else df.iloc[-1]["time"],
                    "direction": "buy" if direction > 0 else "sell",
                    "direction_value": direction,
                    "volume_lots": volume_lots,
                    "entry": entry,
                    "exit": exit_price,
                    "sl": stop,
                    "tp": target,
                    "confidence": confidence,
                    "regime": df.iloc[i].get("regime", "unknown"),
                    "spread_points": spread,
                    "exit_r": exit_r,
                    "gross_pnl": gross_pnl,
                    "commission": commission,
                    "pnl": pnl,
                    "equity": equity,
                }
            )
            i = max(exit_pos + 1, i + 1)

        equity_curve = pd.DataFrame(equity_records)
        trades_df = pd.DataFrame(trades)
        return {"trades": trades_df, "equity_curve": equity_curve, "metrics": summarize_trades(trades_df, equity_curve)}


Backtest = BacktestEngine
