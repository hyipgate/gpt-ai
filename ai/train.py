from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from backtesting.walkforward import WalkForwardConfig, walk_forward_validate
from core.config import resolve_path
from ai.modeling import BalanceConfig, balanced_sample


@dataclass(frozen=True)
class TrainingResult:
    model_path: Path
    metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    threshold: float


def _imbalance_scale(y: pd.Series) -> float:
    positives = max(int(y.sum()), 1)
    negatives = max(int(len(y) - y.sum()), 1)
    return negatives / positives


def make_model(kind: str, y: pd.Series | None = None):
    kind = kind.lower()
    scale = _imbalance_scale(y) if y is not None else 1.0
    if kind == "lightgbm":
        return LGBMClassifier(
            n_estimators=350,
            learning_rate=0.03,
            max_depth=-1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
        )
    if kind == "xgboost":
        return XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale,
            random_state=42,
        )
    raise ValueError("kind must be 'xgboost' or 'lightgbm'")


def train_model(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    model_path: str | Path = "ai/models/model.pkl",
    kind: str = "xgboost",
    threshold: float = 0.70,
) -> object:
    y_series = pd.Series(y, index=X.index, name="label").astype(int)
    valid = X.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    X_train = X.loc[valid].astype(float)
    y_train = y_series.loc[valid]
    if y_train.nunique() < 2:
        raise ValueError("Training labels contain only one class; collect more varied setup outcomes.")

    model = make_model(kind, y_train)
    metrics, _ = walk_forward_validate(
        X_train,
        y_train,
        lambda: make_model(kind, y_train),
        WalkForwardConfig(train_size=max(200, int(len(X_train) * 0.5)), test_size=max(50, int(len(X_train) * 0.15)), step_size=max(50, int(len(X_train) * 0.15))),
    )
    model.fit(X_train, y_train)
    model_file = resolve_path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "features": list(X_train.columns),
        "kind": kind,
        "metrics": metrics.to_dict(orient="records"),
        "threshold": threshold,
    }
    joblib.dump(payload, model_file)

    preds = model.predict(X_train)
    report = classification_report(y_train, preds, output_dict=True, zero_division=0)
    (model_file.with_suffix(".report.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return model


def train_quality_bundle(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    regimes: pd.Series,
    model_path: str | Path = "ai/models/model.pkl",
    kind: str = "xgboost",
    threshold: float = 0.70,
    min_regime_samples: int = 80,
) -> dict[str, object]:
    """Train global, regime-segmented, and regime-classifier models."""
    y_series = pd.Series(y, index=X.index, name="label").astype(int)
    regime_series = pd.Series(regimes, index=X.index, name="regime_id").astype(int)
    valid = X.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    X = X.loc[valid].astype(float)
    y_series = y_series.loc[valid]
    regime_series = regime_series.loc[valid]
    if y_series.nunique() < 2:
        raise ValueError("Training labels contain only one class after setup filtering.")

    X_balanced, y_balanced = balanced_sample(X, y_series, BalanceConfig(max_negative_ratio=2.0))
    global_model = make_model(kind, y_balanced)
    global_model.fit(X_balanced, y_balanced)

    regime_models: dict[int, object] = {}
    for regime_id in sorted(regime_series.unique()):
        idx = regime_series[regime_series == regime_id].index
        if len(idx) < min_regime_samples or y_series.loc[idx].nunique() < 2:
            continue
        X_regime, y_regime = balanced_sample(X.loc[idx], y_series.loc[idx], BalanceConfig(max_negative_ratio=2.0))
        model = make_model(kind, y_regime)
        model.fit(X_regime, y_regime)
        regime_models[int(regime_id)] = model

    regime_classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        class_weight="balanced_subsample",
        random_state=42,
    )
    regime_classifier.fit(X, regime_series)

    model_file = resolve_path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "model": global_model,
        "setup_model": global_model,
        "regime_models": regime_models,
        "regime_classifier": regime_classifier,
        "features": list(X.columns),
        "kind": kind,
        "threshold": threshold,
        "objective": "trade_quality_score",
        "model_weights": {"global": 0.65, "regime": 0.35},
    }
    joblib.dump(payload, model_file)
    report = classification_report(y_series, global_model.predict(X), output_dict=True, zero_division=0)
    (model_file.with_suffix(".report.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return payload


def feature_importance(model: object, columns: list[str]) -> pd.DataFrame:
    values = getattr(model, "feature_importances_", np.zeros(len(columns)))
    return pd.DataFrame({"feature": columns, "importance": values}).sort_values("importance", ascending=False)


def select_informative_features(
    X: pd.DataFrame,
    y: pd.Series,
    kind: str = "xgboost",
    min_features: int = 12,
    max_features: int = 32,
) -> list[str]:
    X_sample, y_sample = balanced_sample(X, y, BalanceConfig(max_negative_ratio=2.0))
    probe = make_model(kind, y_sample)
    probe.fit(X_sample, y_sample)
    importance = feature_importance(probe, list(X.columns))
    useful = importance[importance["importance"] > 0]["feature"].tolist()
    if len(useful) < min_features:
        useful = importance["feature"].head(min_features).tolist()
    return useful[:max_features]
