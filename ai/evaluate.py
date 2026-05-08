from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_fscore_support, roc_auc_score

from backtesting.engine import BacktestEngine, BacktestSettings
from backtesting.walkforward import WalkForwardConfig, walk_forward_validate
from core.config import resolve_path


@dataclass(frozen=True)
class EvaluationConfig:
    train_size: int = 2500
    test_size: int = 500
    step_size: int = 500
    min_trades: int = 20
    max_drawdown_floor: float = -0.12
    thresholds: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)


def classification_metrics(y: pd.Series, probabilities: pd.Series, threshold: float) -> dict[str, float]:
    aligned = pd.concat([y.rename("label"), probabilities.rename("probability")], axis=1).dropna()
    if aligned.empty:
        return {}
    predictions = (aligned["probability"] >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(aligned["label"], predictions, average="binary", zero_division=0)
    metrics = {
        "threshold": float(threshold),
        "samples": float(len(aligned)),
        "positive_rate": float(aligned["label"].mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "brier": float(brier_score_loss(aligned["label"], aligned["probability"])),
    }
    if aligned["label"].nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(aligned["label"], aligned["probability"]))
        metrics["average_precision"] = float(average_precision_score(aligned["label"], aligned["probability"]))
    return metrics


def optimize_threshold(
    df: pd.DataFrame,
    probabilities: pd.Series,
    directions: pd.Series,
    config: EvaluationConfig,
    backtest_settings: BacktestSettings,
) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    for threshold in config.thresholds:
        settings = BacktestSettings(**(asdict(backtest_settings) | {"confidence_threshold": threshold}))
        result = BacktestEngine(settings).run(df, probabilities.fillna(0.0), directions.fillna(0).astype(int))
        metrics = result["metrics"]
        trades = metrics.get("trades", 0.0)
        max_dd = metrics.get("max_drawdown", 0.0)
        score = -999.0
        if trades >= config.min_trades and max_dd >= config.max_drawdown_floor:
            score = (
                metrics.get("expectancy_r", 0.0) * 2.0
                + min(metrics.get("profit_factor", 0.0), 3.0)
                + metrics.get("sharpe", 0.0)
                + max_dd
            )
        rows.append({"threshold": threshold, "score": score, **metrics})
    table = pd.DataFrame(rows).sort_values(["score", "profit_factor", "expectancy_r"], ascending=False)
    if table.empty:
        return 0.70, table
    best = float(table.iloc[0]["threshold"])
    if table.iloc[0]["score"] <= -900:
        viable = table[table["trades"] > 0].sort_values(["expectancy_r", "profit_factor"], ascending=False)
        best = float(viable.iloc[0]["threshold"]) if not viable.empty else 0.70
    return best, table.sort_values("threshold")


def evaluate_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    directions: pd.Series,
    model_factory,
    output_dir: str | Path = "ai/models",
    evaluation_config: EvaluationConfig | None = None,
    backtest_settings: BacktestSettings | None = None,
) -> dict[str, object]:
    evaluation_config = evaluation_config or EvaluationConfig()
    backtest_settings = backtest_settings or BacktestSettings()
    wf_config = WalkForwardConfig(
        train_size=min(evaluation_config.train_size, max(200, int(len(X) * 0.5))),
        test_size=min(evaluation_config.test_size, max(50, int(len(X) * 0.15))),
        step_size=min(evaluation_config.step_size, max(50, int(len(X) * 0.15))),
    )
    fold_metrics, probabilities = walk_forward_validate(X, y, model_factory, wf_config)
    threshold, threshold_table = optimize_threshold(df, probabilities, directions, evaluation_config, backtest_settings)
    class_report = classification_metrics(y, probabilities, threshold)
    bt_result = BacktestEngine(BacktestSettings(**(asdict(backtest_settings) | {"confidence_threshold": threshold}))).run(
        df,
        probabilities.fillna(0.0),
        directions.fillna(0).astype(int),
    )

    out_dir = resolve_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(out_dir / "walkforward_metrics.csv", index=False)
    threshold_table.to_csv(out_dir / "threshold_sweep.csv", index=False)
    probabilities.rename("probability").to_csv(out_dir / "walkforward_probabilities.csv", index=True)
    bt_result["trades"].to_csv(out_dir / "walkforward_trades.csv", index=False)
    bt_result["equity_curve"].to_csv(out_dir / "walkforward_equity.csv", index=False)
    summary = {
        "selected_threshold": threshold,
        "classification": class_report,
        "backtest": bt_result["metrics"],
        "folds": fold_metrics.to_dict(orient="records"),
    }
    (out_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
