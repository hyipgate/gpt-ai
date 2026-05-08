from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd

from ai.features import build_features
from ai.regime import detect_market_regime
from ai.setup_filter import SetupFilterConfig, filter_valid_setups
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

    structured = detect_market_regime(build_structure(raw, config))
    valid_setups = filter_valid_setups(
        structured,
        SetupFilterConfig(
            min_atr_points=config.risk.min_atr_points,
            max_spread_points=config.risk.max_spread_points,
            point_value=config.backtest.point_value,
            require_london_or_ny=True,
        ),
    )
    structured["valid_setup"] = False
    structured["setup_direction"] = 0
    structured["setup_quality_rule_score"] = 0.0
    for column in ("valid_setup", "setup_direction", "setup_quality_rule_score"):
        if column in valid_setups:
            structured.loc[valid_setups.index, column] = valid_setups[column]
    features = build_features(structured)
    probabilities, model_threshold = model_probabilities(features, config.paths.model_path)
    threshold = args.threshold if args.threshold is not None else model_threshold
    directions = structured["setup_direction"].where(structured["valid_setup"].astype(bool), 0).astype(int)

    settings = BacktestSettings(
        initial_equity=config.backtest.initial_equity,
        risk_per_trade=config.risk.risk_per_trade,
        reward_r=2.0,
        spread_points=config.backtest.spread_points,
        slippage_points=config.backtest.slippage_points,
        point_value=config.backtest.point_value,
        commission_per_lot=config.backtest.commission_per_lot,
        confidence_threshold=threshold,
        lot_size=lot_size,
        contract_size=contract_size,
        sizing_mode="fixed_lot",
        max_spread_points=config.risk.max_spread_points,
        min_atr_points=config.risk.min_atr_points,
    )
    result = BacktestEngine(settings).run(structured, probabilities, directions)
    output_dir = resolve_path("ai/models")
    result["trades"].to_csv(output_dir / "gold_backtest_trades.csv", index=False)
    result["equity_curve"].to_csv(output_dir / "gold_backtest_equity.csv", index=False)
    (output_dir / "gold_backtest_summary.json").write_text(json.dumps(result["metrics"], indent=2), encoding="utf-8")

    print("Backtest metrics:")
    print(json.dumps(result["metrics"], indent=2))
    print(f"Saved: {output_dir / 'gold_backtest_trades.csv'}")


if __name__ == "__main__":
    main()
