from __future__ import annotations

import numpy as np
import pandas as pd


def detect_order_blocks(df: pd.DataFrame, lookback: int = 12) -> pd.DataFrame:
    """Find the last opposing candle before a BOS as the candidate order block."""
    out = df.copy()
    out["bullish_ob"] = False
    out["bearish_ob"] = False
    out["ob_low"] = np.nan
    out["ob_high"] = np.nan
    out["ob_direction"] = 0

    bearish_candle = out["close"] < out["open"]
    bullish_candle = out["close"] > out["open"]

    for idx in np.flatnonzero(out.get("bos_bullish", False).to_numpy()):
        window = out.iloc[max(0, idx - lookback) : idx]
        candidates = window[bearish_candle.iloc[window.index]]
        if not candidates.empty:
            ob_idx = candidates.index[-1]
            out.loc[ob_idx, ["bullish_ob", "ob_low", "ob_high", "ob_direction"]] = [
                True,
                out.loc[ob_idx, "low"],
                out.loc[ob_idx, "high"],
                1,
            ]

    for idx in np.flatnonzero(out.get("bos_bearish", False).to_numpy()):
        window = out.iloc[max(0, idx - lookback) : idx]
        candidates = window[bullish_candle.iloc[window.index]]
        if not candidates.empty:
            ob_idx = candidates.index[-1]
            out.loc[ob_idx, ["bearish_ob", "ob_low", "ob_high", "ob_direction"]] = [
                True,
                out.loc[ob_idx, "low"],
                out.loc[ob_idx, "high"],
                -1,
            ]

    active_ob = out["ob_direction"].replace(0, np.nan)
    out["active_ob_direction"] = active_ob.ffill().fillna(0).astype(int)
    out["active_ob_low"] = out["ob_low"].where(out["ob_direction"].ne(0)).ffill()
    out["active_ob_high"] = out["ob_high"].where(out["ob_direction"].ne(0)).ffill()
    return out
