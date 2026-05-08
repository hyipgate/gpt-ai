from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from ai.features import add_session_features, build_features
from ai.regime import detect_market_regime
from ai.setup_filter import SetupFilterConfig, filter_valid_setups, infer_setup_direction
from app.live_trader import build_structure
from backtesting.engine import BacktestEngine, BacktestSettings, gold_pnl
from core.config import load_config, resolve_path
from core.mt5_connector import MT5Connector
from market.datafeed import MT5DataFeed, MarketDataRequest


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
    args = parser.parse_args()

    config = load_config()
    symbol = args.symbol or config.trading.symbol
    timeframe = args.timeframe or config.trading.timeframe
    bars = args.bars or config.trading.bars
    lot_size = args.lot or config.backtest.lot_size
    contract_size = args.contract_size or config.backtest.contract_size

    print("PnL sanity check:")
    print(f"BUY  4700 -> 4701 @ {lot_size} lot = ${gold_pnl(4700, 4701, 1, lot_size, contract_size):.2f}")
    print(f"SELL 4701 -> 4700 @ {lot_size} lot = ${gold_pnl(4701, 4700, -1, lot_size, contract_size):.2f}")

    connector = MT5Connector()
    connector.initialize()
    try:
        raw = MT5DataFeed(connector).get_rates(MarketDataRequest(symbol=symbol, timeframe=timeframe, bars=bars))
    finally:
        connector.shutdown()

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
    )
    result = BacktestEngine(settings).run(structured, probabilities, directions)
    output_dir = resolve_path("ai/models")
    stem = "gold_backtest_rules" if args.entry_mode == "rules" else "gold_backtest"
    result["trades"].to_csv(output_dir / f"{stem}_trades.csv", index=False)
    result["equity_curve"].to_csv(output_dir / f"{stem}_equity.csv", index=False)
    (output_dir / f"{stem}_summary.json").write_text(json.dumps(result["metrics"], indent=2), encoding="utf-8")

    print("Backtest metrics:")
    print(json.dumps(result["metrics"], indent=2))
    print(f"Mode: {args.entry_mode} reward_r={args.reward_r} profile={args.rules_profile}")
    print(f"Saved: {output_dir / f'{stem}_trades.csv'}")


if __name__ == "__main__":
    main()
