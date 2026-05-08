from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai.features import add_session_features
from ai.labels import LabelConfig, create_quant_labels
from ai.regime import detect_market_regime
from ai.setup_filter import SetupFilterConfig, filter_valid_setups, infer_setup_direction
from app.live_trader import build_structure
from backtesting.metrics import profit_factor
from core.config import load_config, resolve_path
from core.mt5_connector import MT5Connector
from market.datafeed import MT5DataFeed, MarketDataRequest


def prepare_research_frame(symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
    config = load_config()
    connector = MT5Connector()
    connector.initialize()
    try:
        raw = MT5DataFeed(connector).get_rates(MarketDataRequest(symbol=symbol, timeframe=timeframe, bars=bars))
    finally:
        connector.shutdown()

    df = build_structure(raw, config)
    df = add_session_features(df)
    if config.research.use_regime_detection:
        df = detect_market_regime(df)
    else:
        df["regime_id"] = 0
        df["regime"] = "disabled"

    valid = filter_valid_setups(
        df,
        SetupFilterConfig(
            min_atr_points=config.risk.min_atr_points,
            max_spread_points=config.risk.max_spread_points,
            point_value=config.backtest.point_value,
            require_london_or_ny=config.research.require_london_or_ny_for_research,
        ),
    )
    df["setup_direction"] = infer_setup_direction(df)
    df["valid_setup"] = False
    df["setup_quality_rule_score"] = 0.0
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
        if column in valid:
            df.loc[valid.index, column] = valid[column]
    df["direction_name"] = np.select([df["setup_direction"].gt(0), df["setup_direction"].lt(0)], ["buy", "sell"], default="none")
    df["session_name"] = np.select(
        [
            df.get("session_overlap", 0).eq(1),
            df.get("session_london", 0).eq(1),
            df.get("session_new_york", 0).eq(1),
            df.get("session_asia", 0).eq(1),
        ],
        ["overlap", "london", "new_york", "asia"],
        default="off_session",
    )
    df["ob_aligned"] = df.get("active_ob_direction", pd.Series(0, index=df.index)).eq(df["setup_direction"])
    df["sweep_type"] = np.select(
        [df.get("liquidity_sweep_low", False).astype(bool), df.get("liquidity_sweep_high", False).astype(bool)],
        ["sweep_low", "sweep_high"],
        default="none",
    )
    df["choch_type"] = np.select(
        [df.get("choch_state", 0).gt(0), df.get("choch_state", 0).lt(0)],
        ["choch_bullish", "choch_bearish"],
        default="none",
    )
    return df


def attach_target_outcomes(df: pd.DataFrame, targets: list[float], horizon: int) -> pd.DataFrame:
    out = df.copy()
    for target in targets:
        labels = create_quant_labels(out, LabelConfig(horizon=horizon, reward_r=target, require_valid_setup=True))
        suffix = str(target).replace(".", "_")
        out[f"label_{suffix}r"] = labels["label"]
        out[f"mfe_{suffix}r"] = labels["mfe_r"]
        out[f"mae_{suffix}r"] = labels["mae_r"]
        out[f"exit_{suffix}r"] = labels["exit_r"]
    return out


def summarize_group(df: pd.DataFrame, group_cols: list[str], exit_col: str, label_col: str, min_trades: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = df.groupby(group_cols, dropna=False)
    for key, group in grouped:
        if len(group) < min_trades:
            continue
        exit_r = group[exit_col].astype(float)
        label = group[label_col].astype(int)
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = {column: value for column, value in zip(group_cols, key_tuple)}
        row.update(
            {
                "trades": len(group),
                "positive_rate": float(label.mean()),
                "expectancy_r": float(exit_r.mean()),
                "median_mfe_r": float(group.filter(like="mfe_").iloc[:, 0].median()) if any(c.startswith("mfe_") for c in group.columns) else 0.0,
                "median_mae_r": float(group.filter(like="mae_").iloc[:, 0].median()) if any(c.startswith("mae_") for c in group.columns) else 0.0,
                "profit_factor_r": profit_factor(exit_r),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["expectancy_r", "positive_rate", "trades"], ascending=False)


def run_breakdown(symbol: str, timeframe: str, bars: int, horizon: int, targets: list[float], min_trades: int) -> dict[str, object]:
    df = prepare_research_frame(symbol, timeframe, bars)
    df = attach_target_outcomes(df, targets, horizon)
    setups = df[df["valid_setup"].astype(bool)].copy()
    output_dir = resolve_path("ai/models/research_breakdown")
    output_dir.mkdir(parents=True, exist_ok=True)

    target = targets[-1]
    suffix = str(target).replace(".", "_")
    label_col = f"label_{suffix}r"
    exit_col = f"exit_{suffix}r"
    breakdown_specs = {
        "direction": ["direction_name"],
        "session": ["session_name"],
        "regime": ["regime"],
        "sweep": ["sweep_type"],
        "choch": ["choch_type"],
        "ob_alignment": ["ob_aligned"],
        "direction_session": ["direction_name", "session_name"],
        "direction_regime": ["direction_name", "regime"],
        "setup_combo": ["direction_name", "session_name", "regime", "sweep_type", "choch_type", "ob_aligned"],
    }
    summaries: dict[str, pd.DataFrame] = {}
    for name, cols in breakdown_specs.items():
        table = summarize_group(setups, cols, exit_col, label_col, min_trades)
        summaries[name] = table
        table.to_csv(output_dir / f"{name}.csv", index=False)

    target_rows = []
    for target in targets:
        suffix = str(target).replace(".", "_")
        target_rows.append(
            {
                "target_r": target,
                "valid_setups": int(len(setups)),
                "positive_rate": float(setups[f"label_{suffix}r"].mean()) if len(setups) else 0.0,
                "expectancy_r": float(setups[f"exit_{suffix}r"].mean()) if len(setups) else 0.0,
                "profit_factor_r": profit_factor(setups[f"exit_{suffix}r"]) if len(setups) else 0.0,
                "median_mfe_r": float(setups[f"mfe_{suffix}r"].median()) if len(setups) else 0.0,
                "median_mae_r": float(setups[f"mae_{suffix}r"].median()) if len(setups) else 0.0,
            }
        )
    target_table = pd.DataFrame(target_rows)
    target_table.to_csv(output_dir / "target_comparison.csv", index=False)
    setups.to_csv(output_dir / "valid_setups_with_outcomes.csv", index=False)

    best_combo = summaries["setup_combo"].head(10)
    recommendation = {
        "symbol": symbol,
        "timeframe": timeframe,
        "bars": bars,
        "valid_setups": int(len(setups)),
        "target_comparison": target_table.to_dict(orient="records"),
        "top_setup_combos": best_combo.to_dict(orient="records"),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(recommendation, indent=2, default=str), encoding="utf-8")
    return recommendation


def main() -> None:
    parser = argparse.ArgumentParser(description="Research breakdown for SMC setup edge.")
    parser.add_argument("--symbol")
    parser.add_argument("--timeframe")
    parser.add_argument("--bars", type=int, default=10000)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--targets", default="1.0,1.5,2.0")
    parser.add_argument("--min-trades", type=int, default=10)
    args = parser.parse_args()
    config = load_config()
    targets = [float(value.strip()) for value in args.targets.split(",") if value.strip()]
    result = run_breakdown(
        symbol=args.symbol or config.trading.symbol,
        timeframe=args.timeframe or config.trading.timeframe,
        bars=args.bars,
        horizon=args.horizon,
        targets=targets,
        min_trades=args.min_trades,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
