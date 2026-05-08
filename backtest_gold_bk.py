from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd

from ai.features import add_technical_features, build_features
from backtesting.engine import BacktestEngine, BacktestSettings, gold_pnl
from core.config import load_config, resolve_path
from core.mt5_connector import MT5Connector
from market.datafeed import MT5DataFeed, MarketDataRequest
from market.structure import detect_swings
from smc.bos import detect_bos
from smc.choch import detect_choch
from smc.fvg import detect_fvg
from smc.liquidity import detect_liquidity
from smc.order_blocks import detect_order_blocks


def build_structure_bk(df: pd.DataFrame, config) -> pd.DataFrame:
    """Old structure pipeline used before quant setup/regime filters."""
    df = add_technical_features(df)
    df = detect_swings(df, window=config.smc.swing_window)
    df = detect_bos(df)
    df = detect_choch(df)
    df = detect_order_blocks(df, lookback=config.smc.order_block_lookback)
    df = detect_fvg(df, atr_fraction=config.smc.fvg_min_atr_fraction)
    df = detect_liquidity(
        df,
        lookback=config.smc.liquidity_sweep_lookback,
        equal_level_tolerance_atr=config.smc.equal_level_tolerance_atr,
    )
    return df


def infer_direction_bk(row: pd.Series) -> int:
    """Old direction logic: CHoCH, then BOS, then structure bias."""
    state = int(row.get("choch_state", 0) or row.get("bos_state", 0) or row.get("structure_bias", 0))
    if state > 0:
        return 1
    if state < 0:
        return -1
    return 0


def load_probabilities(features: pd.DataFrame, model_path: str) -> tuple[pd.Series, float]:
    payload = joblib.load(resolve_path(model_path))
    model = payload["model"] if isinstance(payload, dict) else payload
    columns = payload.get("features") if isinstance(payload, dict) else list(features.columns)
    threshold = float(payload.get("threshold", 0.70)) if isinstance(payload, dict) else 0.70
    X = features.reindex(columns=columns, fill_value=0.0)
    return pd.Series(model.predict_proba(X)[:, 1], index=features.index, name="probability"), threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Legacy XAUUSD backtest before quant research filters.")
    parser.add_argument("--symbol")
    parser.add_argument("--timeframe")
    parser.add_argument("--bars", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--lot", type=float)
    parser.add_argument("--contract-size", type=float)
    parser.add_argument("--ignore-risk-filters", action="store_true")
    args = parser.parse_args()

    config = load_config()
    symbol = args.symbol or config.trading.symbol
    timeframe = args.timeframe or config.trading.timeframe
    bars = args.bars or config.trading.bars
    lot_size = args.lot or config.backtest.lot_size
    contract_size = args.contract_size or config.backtest.contract_size

    print("Legacy backtest mode: CHoCH/BOS/structure_bias direction, no quant setup filter.")
    print(f"BUY  4700 -> 4701 @ {lot_size} lot = ${gold_pnl(4700, 4701, 1, lot_size, contract_size):.2f}")
    print(f"SELL 4701 -> 4700 @ {lot_size} lot = ${gold_pnl(4701, 4700, -1, lot_size, contract_size):.2f}")

    connector = MT5Connector()
    connector.initialize()
    try:
        raw = MT5DataFeed(connector).get_rates(MarketDataRequest(symbol=symbol, timeframe=timeframe, bars=bars))
    finally:
        connector.shutdown()

    structured = build_structure_bk(raw, config)
    features = build_features(structured)
    probabilities, model_threshold = load_probabilities(features, config.paths.model_path)
    threshold = args.threshold if args.threshold is not None else model_threshold
    directions = structured.apply(infer_direction_bk, axis=1).astype(int)

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
        max_spread_points=None if args.ignore_risk_filters else config.risk.max_spread_points,
        min_atr_points=None if args.ignore_risk_filters else config.risk.min_atr_points,
    )
    result = BacktestEngine(settings).run(structured, probabilities, directions)

    output_dir = resolve_path("ai/models")
    result["trades"].to_csv(output_dir / "gold_backtest_bk_trades.csv", index=False)
    result["equity_curve"].to_csv(output_dir / "gold_backtest_bk_equity.csv", index=False)
    (output_dir / "gold_backtest_bk_summary.json").write_text(json.dumps(result["metrics"], indent=2), encoding="utf-8")

    print("Backtest metrics:")
    print(json.dumps(result["metrics"], indent=2))
    print(f"Saved: {output_dir / 'gold_backtest_bk_trades.csv'}")


if __name__ == "__main__":
    main()
