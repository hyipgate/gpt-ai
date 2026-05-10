from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from ai.image_feedback import load_image_feedback
from core.config import resolve_path


@dataclass(frozen=True)
class ImageVisionPrediction:
    probability: float
    passed: bool
    threshold: float
    model_path: str

    def to_dict(self) -> dict:
        return asdict(self)


def image_feature_row(image_path: str | Path, size: int = 64) -> np.ndarray:
    from PIL import Image, ImageOps

    path = resolve_path(image_path)
    with Image.open(path) as image:
        image = ImageOps.grayscale(image)
        image = ImageOps.fit(image, (size, size), method=Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return array.reshape(-1)


def _direction_suffix(direction: str) -> float:
    side = str(direction).lower()
    if side in {"buy", "long", "1"}:
        return 1.0
    if side in {"sell", "short", "-1"}:
        return -1.0
    return 0.0


def train_image_vision_model(
    feedback_path: str | Path = "ai/models/manual_image_feedback.csv",
    model_path: str | Path = "ai/models/image_vision_model.pkl",
    size: int = 64,
    threshold: float = 0.60,
) -> dict[str, object]:
    feedback = load_image_feedback(feedback_path)
    if len(feedback) < 10:
        raise ValueError(f"Need at least 10 labeled images; got {len(feedback)}.")
    if feedback["label"].nunique() < 2:
        raise ValueError("Image feedback contains only one class; add both good and bad examples.")

    rows = []
    labels = []
    weights = []
    kept = []
    for row in feedback.itertuples(index=False):
        image_path = resolve_path(row.image_path)
        if not image_path.exists():
            continue
        features = image_feature_row(image_path, size=size)
        features = np.append(features, _direction_suffix(row.direction)).astype(np.float32)
        rows.append(features)
        labels.append(int(row.label))
        weights.append(float(row.weight))
        kept.append(str(image_path))
    if len(rows) < 10:
        raise ValueError(f"Need at least 10 readable labeled images; got {len(rows)}.")

    X = pd.DataFrame(rows, dtype=np.float32)
    y = pd.Series(labels, dtype=int)
    sample_weight = pd.Series(weights, dtype=float)
    split = max(1, int(len(X) * 0.8))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    w_train = sample_weight.iloc[:split]

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    metrics: dict[str, float] = {
        "samples": float(len(X)),
        "positive_rate": float(y.mean()),
        "threshold": float(threshold),
        "image_size": float(size),
    }
    if len(X_test) > 0:
        probabilities = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)
        predictions = probabilities.ge(threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average="binary", zero_division=0)
        metrics |= {
            "test_precision": float(precision),
            "test_recall": float(recall),
            "test_f1": float(f1),
        }
        if y_test.nunique() > 1:
            metrics["test_roc_auc"] = float(roc_auc_score(y_test, probabilities))

    output = resolve_path(model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "size": size,
        "threshold": threshold,
        "metrics": metrics,
        "training_images": kept,
        "objective": "manual_screenshot_chart_quality",
    }
    joblib.dump(payload, output)
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return payload


def predict_image_vision(
    image_path: str | Path | None,
    direction: str | int = "",
    model_path: str | Path = "ai/models/image_vision_model.pkl",
) -> ImageVisionPrediction | None:
    if not image_path:
        return None
    path = resolve_path(model_path)
    if not path.exists():
        return None
    payload = joblib.load(path)
    features = image_feature_row(image_path, size=int(payload.get("size", 64)))
    features = np.append(features, _direction_suffix(str(direction))).astype(np.float32)
    X = pd.DataFrame([features], dtype=np.float32)
    probability = float(payload["model"].predict_proba(X)[:, 1][0])
    threshold = float(payload.get("threshold", 0.60))
    return ImageVisionPrediction(
        probability=probability,
        passed=probability >= threshold,
        threshold=threshold,
        model_path=str(path),
    )
