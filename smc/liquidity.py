from __future__ import annotations

import numpy as np
import pandas as pd


def detect_liquidity(
    df: pd.DataFrame,
    lookback: int = 20,
    equal_level_tolerance_atr: float = 0.12,
) -> pd.DataFrame:
    """Detect equal highs/lows and liquidity sweeps against recent ranges."""
    out = df.copy()
    atr = out.get("atr", (out["high"] - out["low"]).rolling(14).mean()).bfill().fillna(0.0)
    tolerance = atr * equal_level_tolerance_atr

    recent_high = out["high"].rolling(lookback).max().shift(1)
    recent_low = out["low"].rolling(lookback).min().shift(1)
    out["equal_high"] = (out["high"] - recent_high).abs().le(tolerance)
    out["equal_low"] = (out["low"] - recent_low).abs().le(tolerance)

    out["liquidity_sweep_high"] = (out["high"] > recent_high) & (out["close"] < recent_high)
    out["liquidity_sweep_low"] = (out["low"] < recent_low) & (out["close"] > recent_low)
    out["nearest_liquidity_high"] = recent_high
    out["nearest_liquidity_low"] = recent_low
    out["distance_to_liquidity_high"] = (recent_high - out["close"]).abs()
    out["distance_to_liquidity_low"] = (out["close"] - recent_low).abs()
    out["liquidity_state"] = np.select(
        [out["liquidity_sweep_low"], out["liquidity_sweep_high"]],
        [1, -1],
        default=0,
    ).astype(int)
    return out
