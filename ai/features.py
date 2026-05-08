from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "bos_state",
    "choch_state",
    "fvg_direction",
    "fvg_size_atr",
    "ob_alignment",
    "atr",
    "atr_pct",
    "volatility_20",
    "tick_volume_zscore",
    "session_asia",
    "session_london",
    "session_new_york",
    "session_overlap",
    "trend_bias",
    "distance_to_liquidity_atr",
    "premium_discount",
    "candle_displacement",
    "body_to_range",
    "imbalance_ratio",
    "sweep_state",
    "equal_high",
    "equal_low",
    "spread",
    "setup_candidate",
    "setup_quality_rule_score",
    "setup_direction",
    "trend_strength",
    "hh_count_20",
    "ll_count_20",
    "volume_spike",
    "atr_percentile",
    "volatility_regime",
    "bos_choch_strength",
    "liquidity_support",
    "regime_trend_up",
    "regime_trend_down",
    "regime_range",
    "regime_high_vol_manipulation",
    "regime_id",
]


def add_technical_features(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["close"].shift(1)
    true_range = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr"] = true_range.rolling(atr_period, min_periods=atr_period).mean()
    out["atr_pct"] = out["atr"] / out["close"].replace(0, np.nan)
    out["returns"] = out["close"].pct_change()
    out["volatility_20"] = out["returns"].rolling(20, min_periods=10).std()
    volume_mean = out["tick_volume"].rolling(50, min_periods=10).mean()
    volume_std = out["tick_volume"].rolling(50, min_periods=10).std().replace(0, np.nan)
    out["tick_volume_zscore"] = (out["tick_volume"] - volume_mean) / volume_std

    candle_range = (out["high"] - out["low"]).replace(0, np.nan)
    body = (out["close"] - out["open"]).abs()
    out["body_to_range"] = body / candle_range
    out["candle_displacement"] = body / out["atr"].replace(0, np.nan)
    upper_wick = out["high"] - out[["open", "close"]].max(axis=1)
    lower_wick = out[["open", "close"]].min(axis=1) - out["low"]
    out["imbalance_ratio"] = (upper_wick - lower_wick) / candle_range
    return out


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    hours = pd.to_datetime(out["time"], utc=True).dt.hour
    out["session_asia"] = hours.between(0, 6).astype(int)
    out["session_london"] = hours.between(7, 11).astype(int)
    out["session_new_york"] = hours.between(12, 20).astype(int)
    out["session_overlap"] = hours.between(12, 16).astype(int)
    return out


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    def series(name: str, default: float = 0.0) -> pd.Series:
        value = out.get(name)
        if isinstance(value, pd.Series):
            return value
        return pd.Series(default, index=out.index)

    atr = out["atr"].replace(0, np.nan)
    structure_bias = series("structure_bias")
    active_ob_direction = series("active_ob_direction")
    out["trend_bias"] = structure_bias
    out["ob_alignment"] = np.where(
        active_ob_direction.eq(structure_bias),
        active_ob_direction,
        0,
    )
    dist_high = series("distance_to_liquidity_high", np.nan)
    dist_low = series("distance_to_liquidity_low", np.nan)
    out["distance_to_liquidity_atr"] = pd.concat([dist_high, dist_low], axis=1).min(axis=1) / atr
    rolling_high = out["high"].rolling(100, min_periods=20).max()
    rolling_low = out["low"].rolling(100, min_periods=20).min()
    equilibrium = (rolling_high + rolling_low) / 2
    range_size = (rolling_high - rolling_low).replace(0, np.nan)
    out["premium_discount"] = (out["close"] - equilibrium) / range_size
    out["fvg_size_atr"] = series("fvg_size") / atr
    out["sweep_state"] = series("liquidity_state")
    event_columns = [
        series("bos_state").abs(),
        series("choch_state").abs(),
        series("fvg_direction").abs(),
        series("liquidity_state").abs(),
    ]
    out["setup_candidate"] = pd.concat(event_columns, axis=1).max(axis=1).fillna(0).clip(0, 1)
    out["setup_quality_rule_score"] = series("setup_quality_rule_score")
    out["setup_direction"] = series("setup_direction")
    swing_high = series("swing_high").astype(bool)
    swing_low = series("swing_low").astype(bool)
    out["hh_count_20"] = (out["high"].where(swing_high).diff().gt(0)).rolling(20, min_periods=1).sum()
    out["ll_count_20"] = (out["low"].where(swing_low).diff().lt(0)).rolling(20, min_periods=1).sum()
    out["trend_strength"] = (out["hh_count_20"] - out["ll_count_20"]) / 20.0
    out["volume_spike"] = series("tick_volume_zscore").gt(2.0).astype(int)
    out["atr_percentile"] = series("atr_percentile", 0.5)
    out["volatility_regime"] = pd.cut(out["atr_percentile"], bins=[-0.01, 0.33, 0.66, 1.01], labels=[0, 1, 2]).astype(float)
    out["bos_choch_strength"] = (series("bos_state").abs() + 2 * series("choch_state").abs()) * out["candle_displacement"].clip(0, 3)
    out["liquidity_support"] = (
        (out["setup_direction"].gt(0) & (series("liquidity_sweep_low").astype(bool) | series("equal_low").astype(bool)))
        | (out["setup_direction"].lt(0) & (series("liquidity_sweep_high").astype(bool) | series("equal_high").astype(bool)))
    ).astype(int)
    for column in ("regime_trend_up", "regime_trend_down", "regime_range", "regime_high_vol_manipulation"):
        out[column] = series(column)
    out["regime_id"] = series("regime_id")
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return ML-ready numeric features with no object columns."""
    out = add_context_features(add_session_features(add_technical_features(df)))
    for column in FEATURE_COLUMNS:
        if column not in out:
            out[column] = 0
    features = out[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features.astype(float)
