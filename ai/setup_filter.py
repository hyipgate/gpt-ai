from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SetupFilterConfig:
    min_atr_points: float = 20.0
    max_spread_points: float = 80.0
    point_value: float = 0.01
    require_london_or_ny: bool = True
    liquidity_lookback: int = 20
    choch_lookback: int = 12
    inducement_lookback: int = 20


def _series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    value = df.get(column)
    if isinstance(value, pd.Series):
        return value
    return pd.Series(default, index=df.index)


def infer_setup_direction(df: pd.DataFrame) -> pd.Series:
    choch = _series(df, "choch_state").replace(0, np.nan).ffill()
    sweep = _series(df, "liquidity_state").replace(0, np.nan).ffill()
    bos = _series(df, "bos_state")
    fvg = _series(df, "fvg_direction")
    direction = choch.where(choch.ne(0), sweep)
    direction = direction.where(direction.ne(0), bos)
    direction = direction.where(direction.ne(0), fvg)
    return direction.fillna(0).astype(int)


def filter_valid_setups(df: pd.DataFrame, config: SetupFilterConfig | None = None) -> pd.DataFrame:
    """Return only institutional setup candidates used for ML samples.

    A row is valid only when liquidity/inducement, CHoCH, OB, session,
    volatility, and spread constraints are aligned. This prevents the model
    from learning random market states.
    """
    config = config or SetupFilterConfig()
    out = df.copy()
    if "session_london" not in out or "session_new_york" not in out:
        hours = pd.to_datetime(out["time"], utc=True).dt.hour
        out["session_london"] = hours.between(7, 11).astype(int)
        out["session_new_york"] = hours.between(12, 20).astype(int)

    bullish_liquidity = _series(out, "liquidity_sweep_low").astype(bool) | _series(out, "equal_low").astype(bool)
    bearish_liquidity = _series(out, "liquidity_sweep_high").astype(bool) | _series(out, "equal_high").astype(bool)
    recent_bullish_liquidity = bullish_liquidity.rolling(config.liquidity_lookback, min_periods=1).max().astype(bool)
    recent_bearish_liquidity = bearish_liquidity.rolling(config.liquidity_lookback, min_periods=1).max().astype(bool)
    recent_liquidity_or_inducement = (
        (bullish_liquidity | bearish_liquidity)
        .rolling(config.inducement_lookback, min_periods=1)
        .max()
        .astype(bool)
    )

    recent_choch = _series(out, "choch_state").replace(0, np.nan).ffill(limit=config.choch_lookback)
    direction = recent_choch.fillna(infer_setup_direction(out)).fillna(0).astype(int)
    choch_confirmed = recent_choch.notna()
    order_block_exists = _series(out, "active_ob_direction").ne(0) | _series(out, "bullish_ob").astype(bool) | _series(out, "bearish_ob").astype(bool)
    session_ok = _series(out, "session_london").eq(1) | _series(out, "session_new_york").eq(1)
    if not config.require_london_or_ny:
        session_ok = pd.Series(True, index=out.index)

    atr_points = _series(out, "atr") / max(config.point_value, 1e-12)
    spread = _series(out, "spread")
    volatility_ok = atr_points.ge(config.min_atr_points)
    spread_ok = spread.le(config.max_spread_points)

    liquidity_support = (
        (direction.gt(0) & recent_bullish_liquidity)
        | (direction.lt(0) & recent_bearish_liquidity)
    )
    ob_support = _series(out, "active_ob_direction").eq(direction) | _series(out, "active_ob_direction").eq(0)

    out["setup_direction"] = direction
    out["valid_setup"] = (
        direction.ne(0)
        & recent_liquidity_or_inducement
        & liquidity_support
        & choch_confirmed
        & order_block_exists
        & ob_support
        & session_ok
        & volatility_ok
        & spread_ok
    )
    out["setup_quality_rule_score"] = (
        recent_liquidity_or_inducement.astype(int)
        + choch_confirmed.astype(int)
        + order_block_exists.astype(int)
        + liquidity_support.astype(int)
        + session_ok.astype(int)
        + volatility_ok.astype(int)
        + spread_ok.astype(int)
    ) / 7.0
    return out.loc[out["valid_setup"]].copy()
