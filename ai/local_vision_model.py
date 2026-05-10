from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from ai.chart_tensor import chart_feature_frame, chart_feature_row
from core.config import resolve_path


LEGACY_LOCAL_VISION_MODEL_PATH = "ai/models/local_vision_model.pkl"


@dataclass(frozen=True)
class TrainedVisionPrediction:
    probability: float
    passed: bool
    threshold: float
    model_path: str

    def to_dict(self) -> dict:
        return asdict(self)


def local_vision_model_path(symbol: str, timeframe: str, configured_path: str | Path | None = None) -> str:
    """Return the per-symbol/timeframe local vision model path unless explicitly overridden."""
    if configured_path and str(configured_path) != LEGACY_LOCAL_VISION_MODEL_PATH:
        return str(configured_path)
    safe_symbol = "".join(ch if ch.isalnum() else "_" for ch in symbol)
    safe_timeframe = "".join(ch if ch.isalnum() else "_" for ch in timeframe)
    return f"ai/models/local_vision_model_{safe_symbol}_{safe_timeframe}.pkl"


def train_local_vision_model(
    df: pd.DataFrame,
    labels: pd.Series,
    directions: pd.Series,
    sample_weight: pd.Series | None = None,
    model_path: str | Path = "ai/models/local_vision_model.pkl",
    symbol: str = "",
    timeframe: str = "",
    lookback: int = 96,
    height: int = 48,
    width: int = 48,
    threshold: float = 0.60,
) -> dict[str, object]:
    candidates = labels.index[directions.reindex(labels.index).fillna(0).astype(int).ne(0)]
    X = chart_feature_frame(df, candidates, directions, lookback=lookback, height=height, width=width)
    y = labels.reindex(X.index).astype(int)
    weights = sample_weight.reindex(X.index).fillna(1.0).astype(float) if sample_weight is not None else None
    if len(X) < 50:
        raise ValueError(f"Need at least 50 labeled chart samples; got {len(X)}.")
    if y.nunique() < 2:
        raise ValueError("Local vision labels contain only one class; collect more varied outcomes.")

    split = max(1, int(len(X) * 0.8))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    w_train = weights.iloc[:split] if weights is not None else None

    model = RandomForestClassifier(
        n_estimators=350,
        max_depth=12,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    train_probs = pd.Series(model.predict_proba(X_train)[:, 1], index=X_train.index)
    metrics: dict[str, float] = {
        "samples": float(len(X)),
        "train_samples": float(len(X_train)),
        "test_samples": float(len(X_test)),
        "positive_rate": float(y.mean()),
        "manual_weighted_samples": float(weights.gt(1.0).sum()) if weights is not None else 0.0,
        "threshold": float(threshold),
    }
    if len(X_test) > 0:
        test_probs = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)
        preds = test_probs.ge(threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
        metrics |= {
            "test_precision": float(precision),
            "test_recall": float(recall),
            "test_f1": float(f1),
        }
        if y_test.nunique() > 1:
            metrics["test_roc_auc"] = float(roc_auc_score(y_test, test_probs))
            metrics["test_average_precision"] = float(average_precision_score(y_test, test_probs))
    if y_train.nunique() > 1:
        metrics["train_average_precision"] = float(average_precision_score(y_train, train_probs))

    output = resolve_path(model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "lookback": lookback,
        "height": height,
        "width": width,
        "threshold": threshold,
        "columns": list(X.columns),
        "metrics": metrics,
        "symbol": symbol,
        "timeframe": timeframe,
        "objective": "local_chart_image_quality",
    }
    joblib.dump(payload, output)
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return payload


def predict_trained_local_vision(
    df: pd.DataFrame,
    direction: int,
    model_path: str | Path = "ai/models/local_vision_model.pkl",
    symbol: str = "",
    timeframe: str = "",
) -> TrainedVisionPrediction | None:
    path = resolve_path(model_path)
    if not path.exists() or direction == 0 or df.empty:
        return None
    payload = joblib.load(path)
    model_symbol = str(payload.get("symbol", "")).upper()
    model_timeframe = str(payload.get("timeframe", "")).upper()
    if model_symbol and symbol and model_symbol != symbol.upper():
        return None
    if model_timeframe and timeframe and model_timeframe != timeframe.upper():
        return None
    row = chart_feature_row(
        df,
        len(df) - 1,
        direction,
        lookback=int(payload.get("lookback", 96)),
        height=int(payload.get("height", 48)),
        width=int(payload.get("width", 48)),
    )
    columns = payload.get("columns") or [f"px_{i}" for i in range(len(row))]
    X = pd.DataFrame([row], columns=columns, dtype=np.float32)
    probability = float(payload["model"].predict_proba(X)[:, 1][0])
    threshold = float(payload.get("threshold", 0.60))
    return TrainedVisionPrediction(
        probability=probability,
        passed=probability >= threshold,
        threshold=threshold,
        model_path=str(path),
    )
