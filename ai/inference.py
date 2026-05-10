from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from core.config import resolve_path
from ai.modeling import confidence_from_probability, weighted_average


@dataclass(frozen=True)
class Prediction:
    probability: float
    confidence: float
    passed: bool


class ModelScorer:
    def __init__(self, model_path: str | Path = "ai/models/model.pkl", threshold: float = 0.70) -> None:
        payload = joblib.load(resolve_path(model_path))
        if isinstance(payload, dict) and "model" in payload:
            self.model = payload["model"]
            self.features = payload.get("features")
            #threshold = payload.get("threshold", threshold)
            self.threshold = threshold
            self.regime_models = payload.get("regime_models", {})
            self.regime_classifier = payload.get("regime_classifier")
            self.model_weights = payload.get("model_weights", {"global": 0.65, "regime": 0.35})
        else:
            self.model = payload
            self.features = None
            self.regime_models = {}
            self.regime_classifier = None
            self.model_weights = {"global": 1.0}
        self.threshold = threshold

    def predict_probability(self, features: pd.DataFrame) -> float:
        X = features.copy()
        if self.features:
            X = X.reindex(columns=self.features, fill_value=0.0)
        row = X.iloc[-1:]
        global_score = float(self.model.predict_proba(row)[:, 1][0])
        scores = {"global": global_score}
        if self.regime_models:
            regime_id = None
            if "regime_id" in row.columns:
                regime_id = int(row["regime_id"].iloc[0])
            elif self.regime_classifier is not None:
                regime_id = int(self.regime_classifier.predict(row)[0])
            model = self.regime_models.get(regime_id)
            if model is not None:
                scores["regime"] = float(model.predict_proba(row)[:, 1][0])
        return weighted_average(scores, self.model_weights)

    def score(self, features: pd.DataFrame) -> Prediction:
        probability = self.predict_probability(features)
        confidence = confidence_from_probability(probability)
        return Prediction(probability=probability, confidence=confidence, passed=probability >= self.threshold)


def predict(features, model_path: str | Path = "ai/models/model.pkl") -> float:
    frame = features if isinstance(features, pd.DataFrame) else pd.DataFrame(features)
    return ModelScorer(model_path).predict_probability(frame)
