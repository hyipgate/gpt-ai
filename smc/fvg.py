from __future__ import annotations

import numpy as np
import pandas as pd


def detect_fvg(df: pd.DataFrame, min_gap: float | None = None, atr_fraction: float = 0.10) -> pd.DataFrame:
    """Detect three-candle fair value gaps and expose gap bounds."""
    out = df.copy()
    atr = out.get("atr", pd.Series(0.0, index=out.index))
    threshold = min_gap if min_gap is not None else atr.fillna(0.0) * atr_fraction
    bullish_gap = out["low"] - out["high"].shift(2)
    bearish_gap = out["low"].shift(2) - out["high"]
    out["bullish_fvg"] = bullish_gap.gt(threshold)
    out["bearish_fvg"] = bearish_gap.gt(threshold)
    out["fvg_direction"] = np.select([out["bullish_fvg"], out["bearish_fvg"]], [1, -1], default=0).astype(int)
    out["fvg_lower"] = np.where(out["bullish_fvg"], out["high"].shift(2), np.where(out["bearish_fvg"], out["high"], np.nan))
    out["fvg_upper"] = np.where(out["bullish_fvg"], out["low"], np.where(out["bearish_fvg"], out["low"].shift(2), np.nan))
    out["fvg_size"] = (out["fvg_upper"] - out["fvg_lower"]).abs().fillna(0.0)
    return out
