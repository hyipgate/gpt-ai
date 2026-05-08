from __future__ import annotations

import numpy as np
import pandas as pd


def detect_choch(df: pd.DataFrame) -> pd.DataFrame:
    """Mark change of character when BOS flips against the active structure bias."""
    out = df.copy()
    prior_bias = out.get("structure_bias", pd.Series(0, index=out.index)).shift(1).fillna(0)
    out["choch_bullish"] = out["bos_bullish"].astype(bool) & (prior_bias < 0)
    out["choch_bearish"] = out["bos_bearish"].astype(bool) & (prior_bias > 0)
    out["choch_state"] = np.select([out["choch_bullish"], out["choch_bearish"]], [1, -1], default=0).astype(int)
    return out
