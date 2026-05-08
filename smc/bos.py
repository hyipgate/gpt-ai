from __future__ import annotations

import numpy as np
import pandas as pd


def detect_bos(df: pd.DataFrame) -> pd.DataFrame:
    """Detect break of structure against the previous confirmed swing level."""
    out = df.copy()
    if "last_swing_high" not in out or "last_swing_low" not in out:
        out["last_swing_high"] = out["high"].where(out.get("swing_high", False)).ffill().shift(1)
        out["last_swing_low"] = out["low"].where(out.get("swing_low", False)).ffill().shift(1)
    prev_close = out["close"].shift(1)
    out["bos_bullish"] = (out["close"] > out["last_swing_high"]) & (prev_close <= out["last_swing_high"])
    out["bos_bearish"] = (out["close"] < out["last_swing_low"]) & (prev_close >= out["last_swing_low"])
    out["bos_state"] = np.select([out["bos_bullish"], out["bos_bearish"]], [1, -1], default=0).astype(int)
    out["structure_bias"] = out["bos_state"].replace(0, np.nan).ffill().fillna(0).astype(int)
    return out
