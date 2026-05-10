from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtesting.metrics import max_drawdown, profit_factor


@dataclass(frozen=True)
class MonteCarloConfig:
    simulations: int = 1000
    initial_equity: float = 10_000.0
    random_state: int = 42


def monte_carlo_trade_resample(trades: pd.DataFrame, config: MonteCarloConfig | None = None) -> pd.DataFrame:
    config = config or MonteCarloConfig()
    if trades.empty or "pnl" not in trades:
        return pd.DataFrame()
    rng = np.random.default_rng(config.random_state)
    pnl = trades["pnl"].to_numpy(dtype=float)
    rows = []
    for i in range(config.simulations):
        sample = rng.choice(pnl, size=len(pnl), replace=True)
        equity = pd.Series(config.initial_equity + np.cumsum(sample))
        rows.append(
            {
                "simulation": i,
                "ending_equity": float(equity.iloc[-1]),
                "total_pnl": float(sample.sum()),
                "max_drawdown": max_drawdown(equity),
                "profit_factor": profit_factor(pd.Series(sample)),
                "mean_trade": float(sample.mean()),
            }
        )
    return pd.DataFrame(rows)
