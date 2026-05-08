from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclass(frozen=True)
class WalkForwardConfig:
    train_size: int = 2000
    test_size: int = 500
    step_size: int = 500


def walk_forward_validate(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: Callable[[], object],
    config: WalkForwardConfig | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    config = config or WalkForwardConfig()
    rows: list[dict[str, float]] = []
    all_probs = pd.Series(index=X.index, dtype=float)

    start = 0
    fold = 1
    while start + config.train_size + config.test_size <= len(X):
        train_slice = slice(start, start + config.train_size)
        test_slice = slice(start + config.train_size, start + config.train_size + config.test_size)
        model = model_factory()
        model.fit(X.iloc[train_slice], y.iloc[train_slice])
        probs = pd.Series(model.predict_proba(X.iloc[test_slice])[:, 1], index=X.iloc[test_slice].index)
        all_probs.loc[probs.index] = probs
        y_test = y.iloc[test_slice]
        metrics = {"fold": float(fold), "positive_rate": float(y_test.mean()), "brier": brier_score_loss(y_test, probs)}
        if y_test.nunique() > 1:
            metrics["roc_auc"] = roc_auc_score(y_test, probs)
            metrics["average_precision"] = average_precision_score(y_test, probs)
        rows.append(metrics)
        start += config.step_size
        fold += 1
    return pd.DataFrame(rows), all_probs
