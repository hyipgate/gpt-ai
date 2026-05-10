from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LevelContext:
    price: float | None
    distance_atr: float | None


@dataclass(frozen=True)
class MarketContext:
    symbol: str
    timeframe: str
    candles: int
    last_time: str
    last_close: float
    trend: str
    momentum: str
    market_structure: str
    volatility: str
    rsi: float | None
    rsi_state: str
    body_pressure: str
    nearest_support: LevelContext
    nearest_resistance: LevelContext
    distance_from_equilibrium_atr: float | None
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _classify_trend(close: pd.Series, atr: float) -> str:
    window = close.tail(min(50, len(close))).astype(float)
    if len(window) < 10 or atr <= 0:
        return "unknown"
    x = np.arange(len(window), dtype=float)
    slope = float(np.polyfit(x, window.to_numpy(), 1)[0])
    net_change_atr = (float(window.iloc[-1]) - float(window.iloc[0])) / atr
    slope_atr = slope * len(window) / atr
    if net_change_atr > 1.2 and slope_atr > 0.8:
        return "bullish"
    if net_change_atr < -1.2 and slope_atr < -0.8:
        return "bearish"
    return "sideways"


def _classify_momentum(close: pd.Series, rsi_value: float | None, atr: float) -> str:
    if len(close) < 6 or atr <= 0:
        return "unknown"
    impulse_atr = (float(close.iloc[-1]) - float(close.iloc[-6])) / atr
    if rsi_value is not None and rsi_value >= 58 and impulse_atr > 0.35:
        return "bullish"
    if rsi_value is not None and rsi_value <= 42 and impulse_atr < -0.35:
        return "bearish"
    if impulse_atr > 0.8:
        return "bullish"
    if impulse_atr < -0.8:
        return "bearish"
    return "neutral"


def _classify_rsi(rsi_value: float | None) -> str:
    if rsi_value is None:
        return "unknown"
    if rsi_value >= 70:
        return "overbought"
    if rsi_value <= 30:
        return "oversold"
    if rsi_value >= 55:
        return "bullish"
    if rsi_value <= 45:
        return "bearish"
    return "neutral"


def _swing_levels(df: pd.DataFrame, window: int = 2) -> tuple[pd.Series, pd.Series]:
    highs = df["high"].astype(float)
    lows = df["low"].astype(float)
    swing_high = pd.Series(True, index=df.index)
    swing_low = pd.Series(True, index=df.index)
    for offset in range(1, window + 1):
        swing_high &= highs.gt(highs.shift(offset)) & highs.gt(highs.shift(-offset))
        swing_low &= lows.lt(lows.shift(offset)) & lows.lt(lows.shift(-offset))
    return highs.where(swing_high), lows.where(swing_low)


def _nearest_levels(df: pd.DataFrame, close: float, atr: float) -> tuple[LevelContext, LevelContext]:
    if "swing_high" in df and "swing_low" in df:
        highs = df["high"].where(df["swing_high"].astype(bool))
        lows = df["low"].where(df["swing_low"].astype(bool))
    else:
        highs, lows = _swing_levels(df)

    support_candidates = lows.dropna().astype(float)
    resistance_candidates = highs.dropna().astype(float)
    support_candidates = support_candidates[support_candidates <= close]
    resistance_candidates = resistance_candidates[resistance_candidates >= close]

    support = _safe_float(support_candidates.iloc[-1]) if not support_candidates.empty else None
    resistance = _safe_float(resistance_candidates.iloc[-1]) if not resistance_candidates.empty else None

    support_distance = None if support is None or atr <= 0 else abs(close - support) / atr
    resistance_distance = None if resistance is None or atr <= 0 else abs(resistance - close) / atr
    return (
        LevelContext(support, _safe_float(support_distance)),
        LevelContext(resistance, _safe_float(resistance_distance)),
    )


def _market_structure(df: pd.DataFrame) -> str:
    if "structure_bias" in df and int(df["structure_bias"].fillna(0).iloc[-1]) > 0:
        return "bullish_structure"
    if "structure_bias" in df and int(df["structure_bias"].fillna(0).iloc[-1]) < 0:
        return "bearish_structure"
    if "bos_state" in df and int(df["bos_state"].fillna(0).iloc[-1]) > 0:
        return "bullish_bos"
    if "bos_state" in df and int(df["bos_state"].fillna(0).iloc[-1]) < 0:
        return "bearish_bos"
    if "choch_state" in df and int(df["choch_state"].fillna(0).iloc[-1]) > 0:
        return "bullish_choch"
    if "choch_state" in df and int(df["choch_state"].fillna(0).iloc[-1]) < 0:
        return "bearish_choch"
    highs, lows = _swing_levels(df)
    last_highs = highs.dropna().tail(2).to_numpy(dtype=float)
    last_lows = lows.dropna().tail(2).to_numpy(dtype=float)
    if len(last_highs) == 2 and len(last_lows) == 2:
        if last_highs[-1] > last_highs[-2] and last_lows[-1] > last_lows[-2]:
            return "higher_high_higher_low"
        if last_highs[-1] < last_highs[-2] and last_lows[-1] < last_lows[-2]:
            return "lower_high_lower_low"
    return "range_or_unclear"


def build_market_context(
    df: pd.DataFrame,
    symbol: str = "",
    timeframe: str = "",
    lookback: int = 200,
) -> MarketContext:
    if df.empty:
        raise ValueError("Cannot build market context from an empty dataframe.")
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")

    frame = df.tail(lookback).copy()
    close = frame["close"].astype(float)
    last = frame.iloc[-1]
    atr = _safe_float(last.get("atr"))
    if atr is None or atr <= 0:
        atr = _safe_float((frame["high"].astype(float) - frame["low"].astype(float)).tail(14).mean()) or 0.0

    rsi_series = calculate_rsi(close)
    rsi_value = _safe_float(rsi_series.iloc[-1])
    last_close = float(close.iloc[-1])
    support, resistance = _nearest_levels(frame, last_close, atr)

    recent_range = frame["high"].tail(20).max() - frame["low"].tail(20).min()
    volatility_ratio = _safe_float(recent_range / atr) if atr > 0 else None
    if volatility_ratio is None:
        volatility = "unknown"
    elif volatility_ratio >= 8:
        volatility = "expanding"
    elif volatility_ratio <= 3:
        volatility = "compressed"
    else:
        volatility = "normal"

    candle_range = max(float(last["high"] - last["low"]), 1e-12)
    body_ratio = abs(float(last["close"] - last["open"])) / candle_range
    if body_ratio >= 0.65 and float(last["close"]) > float(last["open"]):
        body_pressure = "strong_bullish_close"
    elif body_ratio >= 0.65 and float(last["close"]) < float(last["open"]):
        body_pressure = "strong_bearish_close"
    elif body_ratio <= 0.25:
        body_pressure = "indecision"
    else:
        body_pressure = "balanced"

    notes = []
    trend = _classify_trend(close, atr)
    momentum = _classify_momentum(close, rsi_value, atr)
    rsi_state = _classify_rsi(rsi_value)
    structure = _market_structure(frame)
    if support.distance_atr is not None and support.distance_atr <= 1.0:
        notes.append("price_near_support")
    if resistance.distance_atr is not None and resistance.distance_atr <= 1.0:
        notes.append("price_near_resistance")
    if trend != "sideways" and momentum != "neutral" and trend != momentum:
        notes.append("trend_momentum_conflict")
    if volatility == "compressed":
        notes.append("range_compression")
    if volatility == "expanding":
        notes.append("volatility_expansion")

    equilibrium = (frame["high"].tail(100).max() + frame["low"].tail(100).min()) / 2
    distance_from_equilibrium = None if atr <= 0 else abs(last_close - float(equilibrium)) / atr

    last_time = str(last.get("time", frame.index[-1]))
    return MarketContext(
        symbol=symbol,
        timeframe=timeframe,
        candles=len(frame),
        last_time=last_time,
        last_close=last_close,
        trend=trend,
        momentum=momentum,
        market_structure=structure,
        volatility=volatility,
        rsi=rsi_value,
        rsi_state=rsi_state,
        body_pressure=body_pressure,
        nearest_support=support,
        nearest_resistance=resistance,
        distance_from_equilibrium_atr=_safe_float(distance_from_equilibrium),
        notes=notes,
    )
