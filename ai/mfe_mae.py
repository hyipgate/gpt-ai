from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExcursionConfig:
    horizon: int = 48
    risk_atr_multiple: float = 1.0
    min_risk: float = 1e-8


def calculate_mfe_mae(
    df: pd.DataFrame,
    directions: pd.Series,
    config: ExcursionConfig | None = None,
) -> pd.DataFrame:
    """Calculate MFE/MAE and realized R path statistics for setup ranking.

    This function intentionally does not collapse outcomes into a binary
    win/loss label. It keeps the excursion profile used by expectancy models.
    """
    config = config or ExcursionConfig()
    atr = df.get("atr", (df["high"] - df["low"]).rolling(14).mean()).bfill()
    rows: list[dict[str, float | int]] = []
    for i in range(len(df)):
        direction = int(directions.iloc[i])
        entry = float(df.iloc[i]["close"])
        risk = max(float(atr.iloc[i]) * config.risk_atr_multiple, config.min_risk)
        future = df.iloc[i + 1 : i + 1 + config.horizon]
        if direction == 0 or future.empty:
            rows.append({"mfe_r": 0.0, "mae_r": 0.0, "terminal_r": 0.0, "risk_distance": risk, "bars_to_mfe": 0, "bars_to_mae": 0})
            continue

        if direction > 0:
            favorable = (future["high"] - entry) / risk
            adverse = (entry - future["low"]) / risk
            terminal_r = (float(future.iloc[-1]["close"]) - entry) / risk
        else:
            favorable = (entry - future["low"]) / risk
            adverse = (future["high"] - entry) / risk
            terminal_r = (entry - float(future.iloc[-1]["close"])) / risk

        mfe_r = float(favorable.max())
        mae_r = float(adverse.max())
        rows.append(
            {
                "mfe_r": mfe_r,
                "mae_r": mae_r,
                "terminal_r": float(terminal_r),
                "risk_distance": risk,
                "bars_to_mfe": int(favorable.reset_index(drop=True).idxmax()) + 1,
                "bars_to_mae": int(adverse.reset_index(drop=True).idxmax()) + 1,
            }
        )
    return pd.DataFrame(rows, index=df.index)


def expectancy_from_excursions(excursions: pd.DataFrame, reward_r: float = 1.5, stop_r: float = 1.0) -> pd.Series:
    """Continuous expected-R proxy from excursion profile.

    TP-before-SL is still represented, but non-terminal outcomes retain useful
    information through terminal R and MFE/MAE balance.
    """
    mfe = excursions["mfe_r"].fillna(0.0)
    mae = excursions["mae_r"].fillna(0.0)
    terminal = excursions["terminal_r"].fillna(0.0)
    tp_hit = mfe.ge(reward_r)
    sl_hit = mae.ge(stop_r)
    hard_outcome = np.select([tp_hit & ~sl_hit, sl_hit & ~tp_hit, tp_hit & sl_hit], [reward_r, -stop_r, -stop_r], default=np.nan)
    soft_outcome = (0.45 * terminal + 0.35 * mfe.clip(0, reward_r) - 0.20 * mae.clip(0, stop_r)).clip(-stop_r, reward_r)
    return pd.Series(hard_outcome, index=excursions.index).fillna(soft_outcome).astype(float)
