from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ai.features import add_technical_features
from market.datafeed import MT5DataFeed, MarketDataRequest
from market.structure import detect_swings
from smc.bos import detect_bos
from smc.choch import detect_choch
from smc.fvg import detect_fvg
from smc.liquidity import detect_liquidity
from smc.order_blocks import detect_order_blocks


@dataclass(frozen=True)
class TimeframeContext:
    timeframe: str
    bias: int
    choch_state: int
    bos_state: int
    last_close: float
    last_time: pd.Timestamp


def build_timeframe_structure(df: pd.DataFrame, config) -> pd.DataFrame:
    out = add_technical_features(df)
    out = detect_swings(out, window=config.smc.swing_window)
    out = detect_bos(out)
    out = detect_choch(out)
    out = detect_order_blocks(out, lookback=config.smc.order_block_lookback)
    out = detect_fvg(out, atr_fraction=config.smc.fvg_min_atr_fraction)
    out = detect_liquidity(
        out,
        lookback=config.smc.liquidity_sweep_lookback,
        equal_level_tolerance_atr=config.smc.equal_level_tolerance_atr,
    )
    return out


def latest_context(df: pd.DataFrame, timeframe: str) -> TimeframeContext:
    row = df.iloc[-1]
    return TimeframeContext(
        timeframe=timeframe,
        bias=int(row.get("structure_bias", 0)),
        choch_state=int(row.get("choch_state", 0)),
        bos_state=int(row.get("bos_state", 0)),
        last_close=float(row["close"]),
        last_time=pd.Timestamp(row["time"]),
    )


def collect_mtf_context(
    feed: MT5DataFeed,
    symbol: str,
    timeframes: tuple[str, ...],
    config,
    bars: int = 1000,
) -> dict[str, TimeframeContext]:
    contexts: dict[str, TimeframeContext] = {}
    for timeframe in timeframes:
        df = feed.get_rates(MarketDataRequest(symbol=symbol, timeframe=timeframe, bars=bars))
        contexts[timeframe] = latest_context(build_timeframe_structure(df, config), timeframe)
    return contexts


def mtf_alignment(direction: str | None, contexts: dict[str, TimeframeContext]) -> tuple[bool, int]:
    if direction is None or not contexts:
        return True, 0
    desired = 1 if direction == "buy" else -1
    score = sum(1 for ctx in contexts.values() if ctx.bias == desired) - sum(1 for ctx in contexts.values() if ctx.bias == -desired)
    return score >= 0, score
