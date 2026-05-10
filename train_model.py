from __future__ import annotations

import argparse

from ai.features import build_features
from ai.labels import LabelConfig, create_outcome_labels, create_quant_labels
from ai.evaluate import EvaluationConfig, classification_metrics, optimize_threshold
from ai.expectancy_model import ExpectancyModel
from ai.regime import detect_market_regime
from ai.setup_filter import SetupFilterConfig, filter_valid_setups
from ai.mfe_mae import ExcursionConfig, calculate_mfe_mae, expectancy_from_excursions
from ai.train import feature_importance, make_model, select_informative_features, train_quality_bundle
from app.live_trader import build_structure
from backtesting.walkforward import WalkForwardConfig, walk_forward_validate
from backtesting.engine import BacktestSettings
from backtesting.engine import BacktestEngine
from core.config import load_config
from core.mt5_connector import MT5Connector
from market.datafeed import MT5DataFeed, MarketDataRequest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train setup-quality model with outcome labels.")
    parser.add_argument("--symbol")
    parser.add_argument("--timeframe")
    parser.add_argument("--bars", type=int)
    parser.add_argument("--model", choices=["xgboost", "lightgbm"], default="xgboost")
    parser.add_argument("--min-trades", type=int, default=20)
    args = parser.parse_args()

    config = load_config()
    symbol = args.symbol or config.trading.symbol
    timeframe = args.timeframe or config.trading.timeframe
    bars = args.bars or config.trading.bars

    connector = MT5Connector()
    connector.initialize()
    try:
        df = MT5DataFeed(connector).get_rates(MarketDataRequest(symbol=symbol, timeframe=timeframe, bars=bars))
    finally:
        connector.shutdown()

    structured = build_structure(df, config)
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
        from ai.setup_filter import infer_setup_direction

        structured["setup_direction"] = infer_setup_direction(structured)
        structured["valid_setup"] = structured["setup_direction"].ne(0)
        structured["setup_quality_rule_score"] = structured["valid_setup"].astype(float)

    X_full = build_features(structured)
    if config.research.use_quant_research_pipeline and config.research.use_quant_labels:
        outcomes = create_quant_labels(structured, LabelConfig(reward_r=2.0, require_valid_setup=True))
    else:
        outcomes = create_outcome_labels(structured, LabelConfig(reward_r=2.0, require_valid_setup=False))
    setup_mask = structured["valid_setup"].astype(bool) & outcomes["direction"].ne(0)
    X_candidates = X_full.loc[setup_mask]
    y_candidates = outcomes.loc[setup_mask, "label"].astype(int)
    regimes = structured.loc[setup_mask, "regime_id"].astype(int)
    directions = outcomes["direction"].where(setup_mask, 0).fillna(0).astype(int)
    excursions = calculate_mfe_mae(structured, directions, ExcursionConfig(horizon=48))
    expected_r = expectancy_from_excursions(excursions, reward_r=1.5).loc[setup_mask]
    if len(X_candidates) < 100:
        raise ValueError(f"Only {len(X_candidates)} valid setups found. Increase bars or loosen setup filter thresholds.")
    if config.research.use_feature_pruning:
        selected_features = select_informative_features(X_candidates, y_candidates, kind=args.model)
        X_candidates = X_candidates[selected_features]
    else:
        selected_features = list(X_candidates.columns)

    wf_metrics, candidate_probs = walk_forward_validate(
        X_candidates,
        y_candidates,
        lambda: make_model(args.model, y_candidates),
        WalkForwardConfig(
            train_size=max(200, int(len(X_candidates) * 0.55)),
            test_size=max(50, int(len(X_candidates) * 0.15)),
            step_size=max(50, int(len(X_candidates) * 0.15)),
        ),
    )
    full_probs = X_full["setup_candidate"].copy().astype(float) * 0.0
    full_probs.loc[candidate_probs.dropna().index] = candidate_probs.dropna()
    full_directions = directions
    backtest_settings = BacktestSettings(
        initial_equity=config.backtest.initial_equity,
        risk_per_trade=config.risk.risk_per_trade,
        reward_r=2.0,
        spread_points=config.backtest.spread_points,
        slippage_points=config.backtest.slippage_points,
        point_value=config.backtest.point_value,
        commission_per_lot=config.backtest.commission_per_lot,
        lot_size=config.backtest.lot_size,
        contract_size=config.backtest.contract_size,
        sizing_mode=config.backtest.sizing_mode,
        confidence_threshold=config.trading.confidence_threshold,
        max_spread_points=config.risk.max_spread_points if config.risk.use_spread_filter else None,
        min_atr_points=config.risk.min_atr_points,
    )
    threshold, threshold_table = optimize_threshold(
        structured,
        full_probs,
        full_directions,
        EvaluationConfig(min_trades=args.min_trades),
        backtest_settings,
    )
    bt_result = BacktestEngine(BacktestSettings(**(backtest_settings.__dict__ | {"confidence_threshold": threshold}))).run(
        structured,
        full_probs,
        full_directions,
    )
    output_dir = __import__("core.config", fromlist=["resolve_path"]).resolve_path("ai/models")
    wf_metrics.to_csv(output_dir / "walkforward_metrics.csv", index=False)
    threshold_table.to_csv(output_dir / "threshold_sweep.csv", index=False)
    full_probs.rename("probability").to_csv(output_dir / "walkforward_probabilities.csv", index=True)
    bt_result["trades"].to_csv(output_dir / "walkforward_trades.csv", index=False)
    bt_result["equity_curve"].to_csv(output_dir / "walkforward_equity.csv", index=False)
    evaluation = {
        "selected_threshold": threshold,
        "classification": classification_metrics(y_candidates, candidate_probs, threshold),
        "backtest": bt_result["metrics"],
        "candidate_count": int(setup_mask.sum()),
        "folds": wf_metrics.to_dict(orient="records"),
    }
    (output_dir / "evaluation_summary.json").write_text(__import__("json").dumps(evaluation, indent=2), encoding="utf-8")
    selected_threshold = float(evaluation["selected_threshold"])
    positives = int(y_candidates.sum())
    backtest_metrics = evaluation["backtest"]
    approved_for_production = True
    if config.research.enforce_production_gate:
        approved_for_production = (
            positives >= 20
            and backtest_metrics.get("trades", 0) >= args.min_trades
            and backtest_metrics.get("profit_factor", 0) > 1.05
            and backtest_metrics.get("expectancy_r", 0) > 0
        )
    model_path = config.paths.model_path if approved_for_production else "ai/models/rejected_candidate_model.pkl"
    bundle = train_quality_bundle(
        X_candidates,
        y_candidates,
        regimes,
        model_path,
        kind=args.model,
        threshold=selected_threshold,
    )
    model = bundle["setup_model"]
    expectancy_model = ExpectancyModel().fit(X_candidates, expected_r)
    expectancy_path = "ai/models/expectancy_model.pkl" if approved_for_production else "ai/models/rejected_expectancy_model.pkl"
    expectancy_model.save(expectancy_path)
    expectancy_metrics = expectancy_model.evaluate(X_candidates, expected_r)
    print("Selected threshold:", selected_threshold)
    print("Valid setups:", len(X_candidates), "positive_rate:", round(float(y_candidates.mean()), 4))
    print("Selected features:", selected_features)
    print("Walk-forward backtest:", evaluation["backtest"])
    print("Walk-forward classification:", evaluation["classification"])
    print("Production approved:", approved_for_production, "saved_model:", model_path)
    print("Expectancy model:", expectancy_metrics, "saved_model:", expectancy_path)
    print(feature_importance(model, list(X_candidates.columns)).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
