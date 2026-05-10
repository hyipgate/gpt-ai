from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from ai.chart_renderer import render_candlestick_snapshot
from ai.image_feedback import load_image_feedback, save_image_feedback
from ai.hybrid import build_hybrid_analysis
from backtesting.metrics import summarize_trades
from core.config import resolve_path


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
    symbol: str = ""
    timeframe: str = ""
    hybrid_enabled: bool = False
    hybrid_render_charts: bool = False
    hybrid_show_rsi: bool = True
    hybrid_use_trained_model: bool = True
    hybrid_trained_model_path: str = "ai/models/local_vision_model.pkl"
    hybrid_filter_trained_vision: bool = False
    hybrid_filter_threshold: float | None = None
    hybrid_lookback_candles: int = 200
    hybrid_chart_dir: str = "logs/chart_snapshots/backtests"
    export_vision_dataset: bool = False
    vision_dataset_dir: str = "logs/chart_snapshots/vision_dataset"
    vision_feedback_path: str = "ai/models/auto_image_feedback.csv"


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
        vision_feedback_rows = []
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
            hybrid_analysis = None
            if self.settings.hybrid_enabled:
                hybrid_analysis = build_hybrid_analysis(
                    df.iloc[: i + 1],
                    symbol=self.settings.symbol,
                    timeframe=self.settings.timeframe,
                    chart_dir=self.settings.hybrid_chart_dir,
                    lookback=self.settings.hybrid_lookback_candles,
                    render_chart=self.settings.hybrid_render_charts,
                    show_rsi=self.settings.hybrid_show_rsi,
                    use_trained_model=self.settings.hybrid_use_trained_model,
                    trained_model_path=self.settings.hybrid_trained_model_path,
                )
                if self.settings.hybrid_filter_trained_vision:
                    trained = hybrid_analysis.trained_vision
                    if trained is None:
                        i += 1
                        continue
                    threshold = (
                        self.settings.hybrid_filter_threshold
                        if self.settings.hybrid_filter_threshold is not None
                        else trained.threshold
                    )
                    if trained.probability < threshold:
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
            vision_dataset_path = ""
            if self.settings.export_vision_dataset:
                entry_time = pd.Timestamp(df.iloc[i]["time"])
                stamp = entry_time.tz_convert("UTC").strftime("%Y%m%d_%H%M%S") if entry_time.tzinfo else entry_time.strftime("%Y%m%d_%H%M%S")
                direction_name = "buy" if direction > 0 else "sell"
                filename = f"{self.settings.symbol}_{self.settings.timeframe}_{stamp}_{direction_name}.png"
                output_path = resolve_path(self.settings.vision_dataset_dir) / filename
                vision_dataset_path = str(
                    render_candlestick_snapshot(
                        df.iloc[: i + 1],
                        output_path,
                        self.settings.symbol,
                        self.settings.timeframe,
                        self.settings.hybrid_lookback_candles,
                        show_rsi=self.settings.hybrid_show_rsi,
                    )
                )
                vision_feedback_rows.append(
                    {
                        "image_path": vision_dataset_path,
                        "symbol": self.settings.symbol,
                        "timeframe": self.settings.timeframe,
                        "direction": direction_name,
                        "label": int(exit_r > 0),
                        "weight": 2.0,
                        "note": f"auto_backtest exit_r={exit_r:.4f} pnl={pnl:.4f}",
                    }
                )
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
                    "hybrid_context": json.dumps(hybrid_analysis.context.to_dict(), default=str) if hybrid_analysis else "",
                    "hybrid_local_vision": json.dumps(hybrid_analysis.local_vision.to_dict(), default=str) if hybrid_analysis else "",
                    "hybrid_trained_vision": json.dumps(hybrid_analysis.trained_vision.to_dict(), default=str) if hybrid_analysis and hybrid_analysis.trained_vision else "",
                    "hybrid_image_vision": json.dumps(hybrid_analysis.image_vision.to_dict(), default=str) if hybrid_analysis and hybrid_analysis.image_vision else "",
                    "hybrid_cnn_image_vision": json.dumps(hybrid_analysis.cnn_image_vision.to_dict(), default=str) if hybrid_analysis and hybrid_analysis.cnn_image_vision else "",
                    "hybrid_chart_path": hybrid_analysis.chart_path if hybrid_analysis else "",
                    "hybrid_prompt": hybrid_analysis.prompt if hybrid_analysis else "",
                    "vision_dataset_path": vision_dataset_path,
                }
            )
            i = max(exit_pos + 1, i + 1)

        equity_curve = pd.DataFrame(equity_records)
        trades_df = pd.DataFrame(trades)
        if vision_feedback_rows:
            existing_feedback = load_image_feedback(self.settings.vision_feedback_path)
            save_image_feedback(
                pd.concat([existing_feedback, pd.DataFrame(vision_feedback_rows)], ignore_index=True),
                self.settings.vision_feedback_path,
            )
        return {"trades": trades_df, "equity_curve": equity_curve, "metrics": summarize_trades(trades_df, equity_curve)}


Backtest = BacktestEngine
