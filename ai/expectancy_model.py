from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from core.config import resolve_path


@dataclass(frozen=True)
class ExpectancyPrediction:
    expected_r: float
    confidence: float
    model_dispersion: float


def make_expectancy_model(kind: str = "xgboost"):
    if kind == "xgboost":
        return XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.035,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=42,
        )
    if kind == "random_forest":
        return RandomForestRegressor(n_estimators=250, max_depth=7, min_samples_leaf=8, random_state=42, n_jobs=-1)
    raise ValueError("expectancy model kind must be xgboost or random_forest")


class ExpectancyModel:
    def __init__(self, model=None, features: list[str] | None = None) -> None:
        self.model = model or make_expectancy_model()
        self.features = features

    def fit(self, X: pd.DataFrame, expected_r: pd.Series) -> "ExpectancyModel":
        self.features = list(X.columns)
        self.model.fit(X, expected_r.astype(float))
        return self

    def predict_expected_return(self, X: pd.DataFrame) -> pd.Series:
        frame = X.reindex(columns=self.features, fill_value=0.0) if self.features else X
        return pd.Series(self.model.predict(frame), index=X.index, name="expected_r")

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        pred = self.predict_expected_return(X)
        return {"mae": float(mean_absolute_error(y, pred)), "r2": float(r2_score(y, pred)) if y.nunique() > 1 else 0.0}

    def save(self, path: str | Path) -> None:
        joblib.dump({"model": self.model, "features": self.features}, resolve_path(path))

    @classmethod
    def load(cls, path: str | Path) -> "ExpectancyModel":
        payload = joblib.load(resolve_path(path))
        return cls(payload["model"], payload.get("features"))


def confidence_from_expected_r(expected_r: pd.Series, dispersion: pd.Series | None = None) -> pd.Series:
    base = (expected_r.abs() / 2.0).clip(0, 1)
    if dispersion is not None:
        base = base * (1 - dispersion.clip(0, 1))
    return base.rename("expectancy_confidence")
