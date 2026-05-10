from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from ai.features import add_session_features, build_features
from ai.local_vision_model import local_vision_model_path
from ai.regime import detect_market_regime
from ai.setup_filter import SetupFilterConfig, filter_valid_setups, infer_setup_direction
from app.live_trader import build_structure
from backtesting.engine import BacktestEngine, BacktestSettings, gold_pnl
from backtesting.monte_carlo import MonteCarloConfig, monte_carlo_trade_resample
from core.config import load_config, resolve_path
from core.mt5_connector import MT5Connector
from market.datafeed import MT5DataFeed, MarketDataRequest
from market.mtf_features import add_mtf_backtest_features, mtf_alignment_series


def direction_series(df: pd.DataFrame) -> pd.Series:
    choch = df.get("choch_state", 0)
    bos = df.get("bos_state", 0)
    fvg = df.get("fvg_direction", 0)
    sweep = df.get("liquidity_state", 0)
    direction = choch.where(choch.ne(0), bos)
    direction = direction.where(direction.ne(0), fvg)
    direction = direction.where(direction.ne(0), sweep)
    return direction.fillna(0).astype(int)


def add_research_breakdown_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    def series(name: str, default: float = 0.0) -> pd.Series:
        value = out.get(name)
        if isinstance(value, pd.Series):
            return value
        return pd.Series(default, index=out.index)

    out["direction_name"] = np.select(
        [out["setup_direction"].gt(0), out["setup_direction"].lt(0)],
        ["buy", "sell"],
        default="none",
    )
    out["session_name"] = np.select(
        [
            series("session_overlap").eq(1),
            series("session_london").eq(1),
            series("session_new_york").eq(1),
            series("session_asia").eq(1),
        ],
        ["overlap", "london", "new_york", "asia"],
        default="off_session",
    )
    out["ob_aligned"] = out.get("active_ob_direction", pd.Series(0, index=out.index)).eq(out["setup_direction"])
    out["sweep_type"] = np.select(
        [out.get("liquidity_sweep_low", False).astype(bool), out.get("liquidity_sweep_high", False).astype(bool)],
        ["sweep_low", "sweep_high"],
        default="none",
    )
    out["choch_type"] = np.select(
        [out.get("choch_state", 0).gt(0), out.get("choch_state", 0).lt(0)],
        ["choch_bullish", "choch_bearish"],
        default="none",
    )
    return out


def rules_profile_mask(df: pd.DataFrame, profile: str) -> pd.Series:
    base = df["valid_setup"].astype(bool) & df["ob_aligned"].astype(bool)
    if profile == "research_top":
        return base & (
            (
                df["direction_name"].eq("sell")
                & df["session_name"].eq("overlap")
                & df["regime"].isin(["trend_up", "trend_down", "range"])
            )
            | (
                df["direction_name"].eq("sell")
                & df["session_name"].eq("new_york")
                & df["regime"].eq("trend_down")
            )
            | (
                df["direction_name"].eq("buy")
                & df["session_name"].eq("overlap")
                & df["regime"].eq("high_vol_manipulation")
            )
        )
    if profile == "research_sells":
        return base & df["direction_name"].eq("sell") & df["session_name"].isin(["overlap", "new_york"]) & df["regime"].isin(["trend_down", "trend_up", "range"])
    if profile == "all_valid":
        return base
    raise ValueError("rules profile must be one of: research_top, research_sells, all_valid")


def model_probabilities(features: pd.DataFrame, model_path: str) -> tuple[pd.Series, float]:
    payload = joblib.load(resolve_path(model_path))
    model = payload["model"] if isinstance(payload, dict) else payload
    columns = payload.get("features") if isinstance(payload, dict) else list(features.columns)
    threshold = float(payload.get("threshold", 0.70)) if isinstance(payload, dict) else 0.70
    X = features.reindex(columns=columns, fill_value=0.0)
    return pd.Series(model.predict_proba(X)[:, 1], index=features.index, name="probability"), threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Run XAUUSD lot-based backtest.")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--bars", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--lot", type=float, default=None)
    parser.add_argument("--contract-size", type=float, default=None)
    parser.add_argument("--reward-r", type=float, default=1.5)
    parser.add_argument("--entry-mode", choices=["model", "rules"], default="model")
    parser.add_argument("--rules-profile", choices=["research_top", "research_sells", "all_valid"], default="research_top")
    parser.add_argument("--spread-points", type=float, default=None)
    parser.add_argument("--slippage-points", type=float, default=None)
    parser.add_argument("--max-spread-points", type=float, default=None)
    parser.add_argument("--hybrid-charts", action="store_true", help="Render a chart PNG for each backtest entry.")
    parser.add_argument("--no-rsi", action="store_true", help="Render/export chart images without the RSI panel.")
    parser.add_argument("--vision-filter", action="store_true", help="Use trained local vision as an entry filter.")
    parser.add_argument("--vision-threshold", type=float, default=None)
    parser.add_argument("--vision-model-path", default=None)
    parser.add_argument("--export-vision-dataset", action="store_true", help="Render backtest entry images and auto-label them for CNN training.")
    parser.add_argument("--vision-feedback-path", default="ai/models/auto_image_feedback.csv")
    args = parser.parse_args()

    config = load_config()
    symbol = args.symbol or config.trading.symbol
    timeframe = args.timeframe or config.trading.timeframe
    bars = args.bars or config.trading.bars
    lot_size = args.lot or config.backtest.lot_size
    contract_size = args.contract_size or config.backtest.contract_size
    vision_model_path = args.vision_model_path or local_vision_model_path(symbol, timeframe, config.hybrid.trained_model_path)

    print("PnL sanity check:")
    print(f"BUY  4700 -> 4701 @ {lot_size} lot = ${gold_pnl(4700, 4701, 1, lot_size, contract_size):.2f}")
    print(f"SELL 4701 -> 4700 @ {lot_size} lot = ${gold_pnl(4701, 4700, -1, lot_size, contract_size):.2f}")

    connector = MT5Connector()
    connector.initialize()
    try:
        feed = MT5DataFeed(connector)
        raw = feed.get_rates(MarketDataRequest(symbol=symbol, timeframe=timeframe, bars=bars))
        higher_timeframe_data = {
            higher_timeframe: feed.get_rates(MarketDataRequest(symbol=symbol, timeframe=higher_timeframe, bars=bars))
            for higher_timeframe in config.trading.higher_timeframes
            if higher_timeframe != timeframe
        }
    finally:
        connector.shutdown()

    structured = add_session_features(build_structure(raw, config))
    if higher_timeframe_data:
        structured = add_mtf_backtest_features(structured, higher_timeframe_data, config)
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
        setup_columns = (
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
        )
        for column in setup_columns:
            if column in valid_setups:
                structured.loc[valid_setups.index, column] = valid_setups[column]
    else:
        structured["setup_direction"] = infer_setup_direction(structured)
        structured["valid_setup"] = structured["setup_direction"].ne(0)
        structured["setup_quality_rule_score"] = structured["valid_setup"].astype(float)
    structured = add_research_breakdown_columns(structured)
    if args.entry_mode == "rules":
        allowed = rules_profile_mask(structured, args.rules_profile)
        probabilities = pd.Series(1.0, index=structured.index, name="rules_score").where(allowed, 0.0)
        threshold = 1.0
        directions = structured["setup_direction"].where(allowed, 0).astype(int)
    else:
        features = build_features(structured)
        probabilities, model_threshold = model_probabilities(features, config.paths.model_path)
        threshold = args.threshold if args.threshold is not None else model_threshold
        directions = structured["setup_direction"].where(structured["valid_setup"].astype(bool), 0).astype(int)

    if config.trading.require_mtf_alignment and higher_timeframe_data:
        mtf_ok, mtf_score = mtf_alignment_series(
            structured,
            directions,
            tuple(higher_timeframe_data.keys()),
        )
        structured["mtf_score"] = mtf_score
        structured["mtf_aligned"] = mtf_ok
        probabilities = probabilities.where(mtf_ok, 0.0)
        directions = directions.where(mtf_ok, 0).astype(int)
    else:
        structured["mtf_score"] = 0
        structured["mtf_aligned"] = True

    settings = BacktestSettings(
        initial_equity=config.backtest.initial_equity,
        risk_per_trade=config.risk.risk_per_trade,
        reward_r=args.reward_r,
        spread_points=config.backtest.spread_points if args.spread_points is None else args.spread_points,
        slippage_points=config.backtest.slippage_points if args.slippage_points is None else args.slippage_points,
        point_value=config.backtest.point_value,
        commission_per_lot=config.backtest.commission_per_lot,
        confidence_threshold=threshold,
        lot_size=lot_size,
        contract_size=contract_size,
        sizing_mode="fixed_lot",
        max_spread_points=(config.risk.max_spread_points if config.risk.use_spread_filter else None) if args.max_spread_points is None else args.max_spread_points,
        min_atr_points=config.risk.min_atr_points,
        symbol=symbol,
        timeframe=timeframe,
        hybrid_enabled=config.hybrid.enabled,
        hybrid_render_charts=args.hybrid_charts and config.hybrid.render_chart,
        hybrid_show_rsi=config.hybrid.show_rsi and not args.no_rsi,
        hybrid_use_trained_model=config.hybrid.use_trained_model,
        hybrid_trained_model_path=vision_model_path,
        hybrid_filter_trained_vision=args.vision_filter,
        hybrid_filter_threshold=args.vision_threshold,
        hybrid_lookback_candles=config.hybrid.lookback_candles,
        hybrid_chart_dir=f"{config.paths.chart_snapshot_dir}/backtests",
        export_vision_dataset=args.export_vision_dataset,
        vision_dataset_dir=f"{config.paths.chart_snapshot_dir}/vision_dataset/{symbol}_{timeframe}",
        vision_feedback_path=args.vision_feedback_path,
    )
    result = BacktestEngine(settings).run(structured, probabilities, directions)
    output_dir = resolve_path("ai/models")
    stem = "gold_backtest_rules" if args.entry_mode == "rules" else "gold_backtest"
    result["trades"].to_csv(output_dir / f"{stem}_trades.csv", index=False)
    result["equity_curve"].to_csv(output_dir / f"{stem}_equity.csv", index=False)
    mc = monte_carlo_trade_resample(result["trades"], MonteCarloConfig(initial_equity=config.backtest.initial_equity))
    if not mc.empty:
        mc.to_csv(output_dir / f"{stem}_monte_carlo.csv", index=False)
    (output_dir / f"{stem}_summary.json").write_text(json.dumps(result["metrics"], indent=2), encoding="utf-8")

    print("Backtest metrics:")
    print(json.dumps(result["metrics"], indent=2))
    print(f"Mode: {args.entry_mode} reward_r={args.reward_r} profile={args.rules_profile}")
    print(f"Saved: {output_dir / f'{stem}_trades.csv'}")
    if args.hybrid_charts:
        chart_count = int(result["trades"].get("hybrid_chart_path", pd.Series(dtype=str)).astype(bool).sum()) if not result["trades"].empty else 0
        print(f"Hybrid charts: {chart_count} saved under {resolve_path(f'{config.paths.chart_snapshot_dir}/backtests')}")
    else:
        print("Hybrid charts: disabled; pass --hybrid-charts to render PNG snapshots.")
    if args.export_vision_dataset:
        dataset_count = int(result["trades"].get("vision_dataset_path", pd.Series(dtype=str)).astype(bool).sum()) if not result["trades"].empty else 0
        print(f"Vision dataset: {dataset_count} labeled images saved under {resolve_path(f'{config.paths.chart_snapshot_dir}/vision_dataset/{symbol}_{timeframe}')}")
        print(f"Vision feedback: {resolve_path(args.vision_feedback_path)}")


if __name__ == "__main__":
    main()
