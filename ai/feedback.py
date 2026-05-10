from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.config import resolve_path


FEEDBACK_COLUMNS = ["time", "symbol", "timeframe", "direction", "label", "weight", "note"]


def load_feedback(path: str | Path = "ai/models/manual_vision_feedback.csv") -> pd.DataFrame:
    feedback_path = resolve_path(path)
    if not feedback_path.exists():
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)
    out = pd.read_csv(feedback_path)
    for column in FEEDBACK_COLUMNS:
        if column not in out:
            out[column] = "" if column in {"time", "symbol", "timeframe", "direction", "note"} else 1
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    out["label"] = pd.to_numeric(out["label"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(3.0).clip(1.0, 20.0)
    return out.dropna(subset=["time"])[FEEDBACK_COLUMNS]


def save_feedback(feedback: pd.DataFrame, path: str | Path = "ai/models/manual_vision_feedback.csv") -> Path:
    output = resolve_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cleaned = feedback.copy()
    cleaned["time"] = pd.to_datetime(cleaned["time"], utc=True, errors="coerce")
    cleaned = cleaned.dropna(subset=["time"])
    cleaned = cleaned.sort_values("time").drop_duplicates(
        subset=["time", "symbol", "timeframe", "direction"],
        keep="last",
    )
    cleaned.to_csv(output, index=False)
    return output


def append_feedback(
    time: str,
    symbol: str,
    timeframe: str,
    direction: str,
    label: int,
    note: str = "",
    weight: float = 5.0,
    path: str | Path = "ai/models/manual_vision_feedback.csv",
) -> Path:
    existing = load_feedback(path)
    row = pd.DataFrame(
        [
            {
                "time": pd.Timestamp(time, tz="UTC") if pd.Timestamp(time).tzinfo is None else pd.Timestamp(time).tz_convert("UTC"),
                "symbol": symbol,
                "timeframe": timeframe,
                "direction": direction.lower(),
                "label": int(label),
                "weight": float(weight),
                "note": note,
            }
        ]
    )
    return save_feedback(pd.concat([existing, row], ignore_index=True), path)


def apply_feedback_labels(
    labels: pd.Series,
    directions: pd.Series,
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    feedback_path: str | Path = "ai/models/manual_vision_feedback.csv",
) -> tuple[pd.Series, pd.Series, pd.Series]:
    feedback = load_feedback(feedback_path)
    if feedback.empty:
        return labels, directions, pd.Series(1.0, index=labels.index, name="sample_weight")

    labels = labels.copy()
    directions = directions.copy()
    weights = pd.Series(1.0, index=labels.index, name="sample_weight")
    times = pd.to_datetime(df["time"], utc=True, errors="coerce")
    index_by_time = pd.Series(df.index, index=times)

    relevant = feedback[
        feedback["symbol"].astype(str).str.upper().eq(symbol.upper())
        & feedback["timeframe"].astype(str).str.upper().eq(timeframe.upper())
    ]
    for row in relevant.itertuples(index=False):
        if row.time not in index_by_time.index:
            continue
        idx = index_by_time.loc[row.time]
        side = 1 if str(row.direction).lower() in {"buy", "long", "1"} else -1
        labels.loc[idx] = int(row.label)
        directions.loc[idx] = side
        weights.loc[idx] = float(row.weight)
    return labels.astype(int), directions.astype(int), weights
