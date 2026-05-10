from __future__ import annotations

import argparse
import json

import pandas as pd

from ai.features import add_session_features
from ai.feedback import apply_feedback_labels
from ai.labels import LabelConfig, create_quant_labels
from ai.local_vision_model import local_vision_model_path, train_local_vision_model
from ai.regime import detect_market_regime
from ai.setup_filter import SetupFilterConfig, filter_valid_setups, infer_setup_direction
from app.live_trader import build_structure
from core.config import load_config, resolve_path
from core.mt5_connector import MT5Connector
from market.datafeed import MT5DataFeed, MarketDataRequest


def _load_history(path: str) -> pd.DataFrame:
    history_path = resolve_path(path)
    if not history_path.exists():
        return pd.DataFrame()
    return pd.read_csv(history_path)


def _merge_history(existing: pd.DataFrame, latest: pd.DataFrame, path: str) -> pd.DataFrame:
    merged = pd.concat([existing, latest], ignore_index=True)
    if "time" in merged:
        merged["time"] = pd.to_datetime(merged["time"], utc=True, errors="coerce")
        merged = merged.dropna(subset=["time"]).drop_duplicates(subset=["time"], keep="last")
        merged = merged.sort_values("time").reset_index(drop=True)
    else:
        merged = merged.drop_duplicates().reset_index(drop=True)
    output = resolve_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the no-API local chart vision model.")
    parser.add_argument("--symbol")
    parser.add_argument("--timeframe")
    parser.add_argument("--bars", type=int)
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--history-path", default=None)
    parser.add_argument("--feedback-path", default="ai/models/manual_vision_feedback.csv")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--no-history-cache", action="store_true")
    args = parser.parse_args()

    config = load_config()
    symbol = args.symbol or config.trading.symbol
    timeframe = args.timeframe or config.trading.timeframe
    bars = args.bars or config.trading.bars
    history_path = args.history_path or f"ai/models/local_vision_history_{symbol}_{timeframe}.csv"
    model_path = args.model_path or local_vision_model_path(symbol, timeframe)

    connector = MT5Connector()
    connector.initialize()
    try:
        raw = MT5DataFeed(connector).get_rates(MarketDataRequest(symbol=symbol, timeframe=timeframe, bars=bars))
    finally:
        connector.shutdown()

    if not args.no_history_cache:
        existing = _load_history(history_path)
        raw = _merge_history(existing, raw, history_path)
        print(f"History cache: {resolve_path(history_path)} rows={len(raw)}")

    structured = add_session_features(build_structure(raw, config))
    if config.research.use_regime_detection:
        structured = detect_market_regime(structured)
    else:
        structured["regime_id"] = 0
        structured["regime"] = "disabled"

    structured["valid_setup"] = False
    structured["setup_direction"] = 0
    structured["setup_quality_rule_score"] = 0.0
    if config.research.use_quant_research_pipeline and config.research.use_setup_filter:
        valid_setups = filter_valid_setups(
            structured,
            SetupFilterConfig(
                min_atr_points=config.risk.min_atr_points,
                max_spread_points=config.risk.max_spread_points,
                point_value=config.backtest.point_value,
                require_london_or_ny=config.research.require_london_or_ny_for_research,
            ),
        )
        for column in (
            "valid_setup",
            "setup_direction",
            "setup_quality_rule_score",
            "liquidity_support",
            "liquidity_or_inducement",
            "choch_confirmed",
            "order_block_exists",
            "ob_support",
            "session_ok",
            "volatility_ok",
            "spread_ok",
        ):
            if column in valid_setups:
                structured.loc[valid_setups.index, column] = valid_setups[column]
    else:
        structured["setup_direction"] = infer_setup_direction(structured)
        structured["valid_setup"] = structured["setup_direction"].ne(0)
        structured["setup_quality_rule_score"] = structured["valid_setup"].astype(float)

    outcomes = create_quant_labels(structured, LabelConfig(reward_r=2.0, require_valid_setup=True))
    mask = structured["valid_setup"].astype(bool) & outcomes["direction"].ne(0)
    labels = outcomes.loc[mask, "label"].astype(int)
    directions = outcomes["direction"].where(mask, 0).fillna(0).astype(int)
    labels, directions, sample_weight = apply_feedback_labels(
        labels,
        directions,
        structured,
        symbol=symbol,
        timeframe=timeframe,
        feedback_path=args.feedback_path,
    )

    payload = train_local_vision_model(
        structured,
        labels,
        directions,
        sample_weight=sample_weight,
        model_path=model_path,
        symbol=symbol,
        timeframe=timeframe,
        lookback=args.lookback,
        height=args.height,
        width=args.width,
        threshold=args.threshold,
    )
    output = resolve_path(model_path)
    print("Saved local vision model:", output)
    print(json.dumps(payload["metrics"], indent=2))


if __name__ == "__main__":
    main()
