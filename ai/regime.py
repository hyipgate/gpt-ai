from __future__ import annotations

import numpy as np
import pandas as pd


REGIME_MAP = {
    0: "range",
    1: "trend_up",
    2: "trend_down",
    3: "high_vol_manipulation",
}


def detect_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Rule-based regime labels suitable as features or model segments."""
    out = df.copy()
    close = out["close"]
    atr = out.get("atr", (out["high"] - out["low"]).rolling(14).mean()).bfill()
    ema_fast = close.ewm(span=20, adjust=False).mean()
    ema_slow = close.ewm(span=80, adjust=False).mean()
    slope = ema_slow.diff(10) / close.replace(0, np.nan)
    returns = close.pct_change()
    realized_vol = returns.rolling(40, min_periods=20).std()
    atr_pct = atr / close.replace(0, np.nan)
    atr_pct_rank = atr_pct.rolling(250, min_periods=50).rank(pct=True).fillna(0.5)
    sweep = out.get("liquidity_state", pd.Series(0, index=out.index)).abs().gt(0)
    displacement = ((out["close"] - out["open"]).abs() / atr.replace(0, np.nan)).fillna(0)

    trend_up = (ema_fast > ema_slow) & slope.gt(0)
    trend_down = (ema_fast < ema_slow) & slope.lt(0)
    high_vol_manip = atr_pct_rank.gt(0.80) & (sweep | displacement.gt(1.5))
    regime_id = np.select([high_vol_manip, trend_up, trend_down], [3, 1, 2], default=0).astype(int)

    out["regime_id"] = regime_id
    out["regime"] = pd.Series(regime_id, index=out.index).map(REGIME_MAP)
    out["regime_trend_up"] = (out["regime_id"] == 1).astype(int)
    out["regime_trend_down"] = (out["regime_id"] == 2).astype(int)
    out["regime_range"] = (out["regime_id"] == 0).astype(int)
    out["regime_high_vol_manipulation"] = (out["regime_id"] == 3).astype(int)
    out["atr_percentile"] = atr_pct_rank
    out["realized_volatility"] = realized_vol.fillna(0)
    return out
