from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelConfig:
    horizon: int = 48
    reward_r: float = 2.0
    risk_atr_multiple: float = 1.0
    min_risk: float = 1e-8
    require_valid_setup: bool = True


def _direction(row: pd.Series) -> int:
    for column in ("choch_state", "bos_state", "structure_bias", "fvg_direction", "active_ob_direction"):
        value = int(row.get(column, 0))
        if value != 0:
            return 1 if value > 0 else -1
    return 0


def create_outcome_labels(df: pd.DataFrame, config: LabelConfig | None = None) -> pd.DataFrame:
    """Create TP-before-SL labels and MFE/MAE without using future data in features."""
    config = config or LabelConfig()
    records: list[dict[str, float | int]] = []
    atr = df.get("atr", (df["high"] - df["low"]).rolling(14).mean()).bfill()

    for i in range(len(df)):
        direction = _direction(df.iloc[i])
        entry = float(df.iloc[i]["close"])
        risk = max(float(atr.iloc[i]) * config.risk_atr_multiple, config.min_risk)
        future = df.iloc[i + 1 : i + 1 + config.horizon]
        if direction == 0 or future.empty:
            records.append({"label": 0, "direction": direction, "mfe_r": 0.0, "mae_r": 0.0, "exit_r": 0.0})
            continue

        if direction > 0:
            tp = entry + risk * config.reward_r
            sl = entry - risk
            tp_hits = future.index[future["high"] >= tp]
            sl_hits = future.index[future["low"] <= sl]
            mfe = (future["high"].max() - entry) / risk
            mae = (entry - future["low"].min()) / risk
        else:
            tp = entry - risk * config.reward_r
            sl = entry + risk
            tp_hits = future.index[future["low"] <= tp]
            sl_hits = future.index[future["high"] >= sl]
            mfe = (entry - future["low"].min()) / risk
            mae = (future["high"].max() - entry) / risk

        tp_first = len(tp_hits) > 0 and (len(sl_hits) == 0 or tp_hits[0] < sl_hits[0])
        sl_first = len(sl_hits) > 0 and not tp_first
        exit_r = config.reward_r if tp_first else (-1.0 if sl_first else float(np.clip(mfe - mae, -1.0, config.reward_r)))
        records.append({"label": int(tp_first), "direction": direction, "mfe_r": float(mfe), "mae_r": float(mae), "exit_r": exit_r})

    return pd.DataFrame(records, index=df.index)


def create_quant_labels(df: pd.DataFrame, config: LabelConfig | None = None) -> pd.DataFrame:
    """Institutional setup-quality labels.

    label=1 only when a valid setup reaches +2R before -1R and liquidity
    context supports the trade direction. This is a trade-quality target,
    not a next-candle direction label.
    """
    config = config or LabelConfig()
    records: list[dict[str, float | int]] = []
    atr = df.get("atr", (df["high"] - df["low"]).rolling(14).mean()).bfill()
    valid_setup = df.get("valid_setup", pd.Series(True, index=df.index)).astype(bool)

    for i in range(len(df)):
        row = df.iloc[i]
        direction = int(row.get("setup_direction", 0)) or _direction(row)
        entry = float(row["close"])
        risk = max(float(atr.iloc[i]) * config.risk_atr_multiple, config.min_risk)
        future = df.iloc[i + 1 : i + 1 + config.horizon]
        liquidity_support = bool(row.get("liquidity_support", 0))
        if not liquidity_support:
            if direction > 0:
                liquidity_support = bool(row.get("liquidity_sweep_low", False) or row.get("equal_low", False))
            elif direction < 0:
                liquidity_support = bool(row.get("liquidity_sweep_high", False) or row.get("equal_high", False))

        if direction == 0 or future.empty or (config.require_valid_setup and not bool(valid_setup.iloc[i])) or not liquidity_support:
            records.append({"label": 0, "direction": direction, "mfe_r": 0.0, "mae_r": 0.0, "exit_r": 0.0, "expectancy_label": 0.0})
            continue

        if direction > 0:
            tp = entry + risk * config.reward_r
            sl = entry - risk
            tp_hits = future.index[future["high"] >= tp]
            sl_hits = future.index[future["low"] <= sl]
            mfe = (future["high"].max() - entry) / risk
            mae = (entry - future["low"].min()) / risk
        else:
            tp = entry - risk * config.reward_r
            sl = entry + risk
            tp_hits = future.index[future["low"] <= tp]
            sl_hits = future.index[future["high"] >= sl]
            mfe = (entry - future["low"].min()) / risk
            mae = (future["high"].max() - entry) / risk

        tp_first = len(tp_hits) > 0 and (len(sl_hits) == 0 or tp_hits[0] < sl_hits[0])
        sl_first = len(sl_hits) > 0 and not tp_first
        exit_r = config.reward_r if tp_first else (-1.0 if sl_first else float(np.clip(mfe - mae, -1.0, config.reward_r)))
        label = int(tp_first and liquidity_support)
        records.append(
            {
                "label": label,
                "direction": direction,
                "mfe_r": float(mfe),
                "mae_r": float(mae),
                "exit_r": float(exit_r),
                "expectancy_label": float(exit_r),
            }
        )

    return pd.DataFrame(records, index=df.index)


def create_labels(df: pd.DataFrame, rr: float = 2.0, horizon: int = 48) -> np.ndarray:
    return create_outcome_labels(df, LabelConfig(horizon=horizon, reward_r=rr))["label"].to_numpy(dtype=int)
