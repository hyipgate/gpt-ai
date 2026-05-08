from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty or returns.std(ddof=0) == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / returns.std(ddof=0))


def profit_factor(pnl: pd.Series) -> float:
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = pnl[pnl < 0].abs().sum()
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def expectancy_r(exit_r: pd.Series) -> float:
    return float(exit_r.mean()) if not exit_r.empty else 0.0


def summarize_trades(trades: pd.DataFrame, equity_curve: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {"trades": 0, "winrate": 0.0, "profit_factor": 0.0, "expectancy_r": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}
    equity = equity_curve["equity"] if "equity" in equity_curve else pd.Series(dtype=float)
    returns = equity.pct_change() if not equity.empty else pd.Series(dtype=float)
    summary = {
        "trades": float(len(trades)),
        "winrate": float((trades["pnl"] > 0).mean()),
        "profit_factor": profit_factor(trades["pnl"]),
        "expectancy_r": expectancy_r(trades["exit_r"]),
        "max_drawdown": max_drawdown(equity),
        "sharpe": sharpe_ratio(returns),
    }
    for direction, count in trades.get("direction", pd.Series(dtype=str)).value_counts().items():
        summary[f"trades_{direction}"] = float(count)
    if "regime" in trades:
        for regime, count in trades["regime"].fillna("unknown").value_counts().items():
            summary[f"trades_regime_{regime}"] = float(count)
    return summary
