from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BalanceConfig:
    max_negative_ratio: float = 2.0
    random_state: int = 42


def balanced_sample(X: pd.DataFrame, y: pd.Series, config: BalanceConfig | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Downsample negative class without duplicating positives."""
    config = config or BalanceConfig()
    y = pd.Series(y, index=X.index).astype(int)
    positives = y[y == 1].index
    negatives = y[y == 0].index
    if len(positives) == 0 or len(negatives) == 0:
        return X, y
    max_negatives = int(len(positives) * config.max_negative_ratio)
    sampled_negatives = pd.Index(negatives).to_series().sample(
        n=min(len(negatives), max_negatives),
        random_state=config.random_state,
    ).index
    selected = positives.union(sampled_negatives).sort_values()
    return X.loc[selected], y.loc[selected]


def confidence_from_probability(probability: float) -> float:
    return float(abs(probability - 0.5) * 2.0)


def weighted_average(scores: dict[str, float], weights: dict[str, float] | None = None) -> float:
    if not scores:
        return 0.0
    weights = weights or {key: 1.0 for key in scores}
    total_weight = sum(weights.get(key, 0.0) for key in scores)
    if total_weight <= 0:
        return float(np.mean(list(scores.values())))
    return float(sum(scores[key] * weights.get(key, 0.0) for key in scores) / total_weight)
