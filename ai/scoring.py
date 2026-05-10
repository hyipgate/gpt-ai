from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SetupScore:
    setup_score: float
    expected_r: float
    confidence: float
    regime_quality: float
    execution_risk: float
    tier: str


def quality_tier(score: float) -> str:
    if score >= 0.85:
        return "A+"
    if score >= 0.72:
        return "A"
    if score >= 0.58:
        return "B"
    if score >= 0.42:
        return "C"
    return "Garbage"


def regime_quality_score(row: pd.Series) -> float:
    direction = int(row.get("setup_direction", 0))
    if direction > 0 and row.get("regime_trend_up", 0):
        return 0.85
    if direction < 0 and row.get("regime_trend_down", 0):
        return 0.85
    if row.get("regime_high_vol_manipulation", 0):
        return 0.55
    if row.get("regime_range", 0):
        return 0.45
    return 0.50


def execution_risk_score(row: pd.Series) -> float:
    spread = float(row.get("spread", 0.0))
    atr_points = float(row.get("atr", 0.0)) / 0.01 if float(row.get("atr", 0.0)) else 0.0
    spread_pressure = min(spread / max(atr_points, 1.0), 1.0)
    volatility_pressure = float(row.get("atr_percentile", 0.5))
    return float(np.clip(0.65 * spread_pressure + 0.35 * volatility_pressure, 0, 1))


def score_setup_row(row: pd.Series, probability: float, expected_r: float) -> SetupScore:
    rule_quality = float(row.get("setup_quality_rule_score", 0.0))
    liquidity = float(row.get("liquidity_proximity_score", row.get("liquidity_support", 0.0)))
    ob = 1.0 if bool(row.get("ob_aligned", False)) or float(row.get("ob_alignment", 0.0)) != 0 else 0.35
    displacement = min(float(row.get("candle_displacement", 0.0)) / 2.0, 1.0)
    regime_quality = regime_quality_score(row)
    execution_risk = execution_risk_score(row)
    expected_component = float(np.clip((expected_r + 1.0) / 3.0, 0, 1))
    score = (
        0.25 * probability
        + 0.25 * expected_component
        + 0.15 * rule_quality
        + 0.15 * regime_quality
        + 0.10 * liquidity
        + 0.07 * ob
        + 0.03 * displacement
        - 0.10 * execution_risk
    )
    score = float(np.clip(score, 0, 1))
    confidence = float(np.clip(abs(probability - 0.5) * 2.0 * 0.5 + abs(expected_r) / 3.0 * 0.5, 0, 1))
    return SetupScore(score, expected_r, confidence, regime_quality, execution_risk, quality_tier(score))


def score_setups(features: pd.DataFrame, probabilities: pd.Series, expected_r: pd.Series) -> pd.DataFrame:
    rows = []
    for idx, row in features.iterrows():
        score = score_setup_row(row, float(probabilities.loc[idx]), float(expected_r.loc[idx]))
        rows.append(score.__dict__)
    return pd.DataFrame(rows, index=features.index)
