from __future__ import annotations

import pandas as pd


def detect_swings(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Mark confirmed swing highs/lows using centered rolling extrema."""
    if window < 1:
        raise ValueError("window must be >= 1")
    out = df.copy()
    span = window * 2 + 1
    high_max = out["high"].rolling(span, center=True).max()
    low_min = out["low"].rolling(span, center=True).min()
    out["swing_high"] = out["high"].eq(high_max).fillna(False)
    out["swing_low"] = out["low"].eq(low_min).fillna(False)
    out.loc[: window - 1, ["swing_high", "swing_low"]] = False
    out.loc[len(out) - window :, ["swing_high", "swing_low"]] = False
    out["last_swing_high"] = out["high"].where(out["swing_high"]).ffill().shift(1)
    out["last_swing_low"] = out["low"].where(out["swing_low"]).ffill().shift(1)
    return out
