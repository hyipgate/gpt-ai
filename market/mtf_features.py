from __future__ import annotations

import pandas as pd

from market.multitimeframe import build_timeframe_structure


def _timeframe_signal_frame(df: pd.DataFrame, timeframe: str, config) -> pd.DataFrame:
    structured = build_timeframe_structure(df, config)
    out = structured[["time", "close"]].copy()
    out[f"mtf_{timeframe}_bias"] = structured.get("structure_bias", 0).fillna(0).astype(int)
    out[f"mtf_{timeframe}_choch"] = structured.get("choch_state", 0).fillna(0).astype(int)
    out[f"mtf_{timeframe}_bos"] = structured.get("bos_state", 0).fillna(0).astype(int)
    out[f"mtf_{timeframe}_close"] = structured["close"].astype(float)
    return out.drop(columns=["close"]).sort_values("time")


def add_mtf_backtest_features(
    base: pd.DataFrame,
    higher_timeframe_data: dict[str, pd.DataFrame],
    config,
) -> pd.DataFrame:
    """Attach higher-timeframe state to each base candle without lookahead."""
    out = base.sort_values("time").copy()
    for timeframe, higher_df in higher_timeframe_data.items():
        signals = _timeframe_signal_frame(higher_df, timeframe, config)
        out = pd.merge_asof(out, signals, on="time", direction="backward")
    return out


def mtf_alignment_series(
    df: pd.DataFrame,
    directions: pd.Series,
    timeframes: tuple[str, ...],
) -> tuple[pd.Series, pd.Series]:
    desired = directions.fillna(0).astype(int)
    score = pd.Series(0, index=df.index, dtype="int64")
    for timeframe in timeframes:
        column = f"mtf_{timeframe}_bias"
        if column not in df:
            continue
        bias = df[column].fillna(0).astype(int)
        score += (bias.eq(desired) & desired.ne(0)).astype(int)
        score -= (bias.eq(-desired) & desired.ne(0)).astype(int)
    aligned = score.ge(0) | desired.eq(0)
    return aligned, score
