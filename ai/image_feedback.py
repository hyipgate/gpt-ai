from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.config import resolve_path


IMAGE_FEEDBACK_COLUMNS = [
    "image_path",
    "symbol",
    "timeframe",
    "direction",
    "label",
    "weight",
    "note",
]


def load_image_feedback(path: str | Path = "ai/models/manual_image_feedback.csv") -> pd.DataFrame:
    feedback_path = resolve_path(path)
    if not feedback_path.exists():
        return pd.DataFrame(columns=IMAGE_FEEDBACK_COLUMNS)
    out = pd.read_csv(feedback_path)
    for column in IMAGE_FEEDBACK_COLUMNS:
        if column not in out:
            out[column] = "" if column in {"image_path", "symbol", "timeframe", "direction", "note"} else 1
    out["label"] = pd.to_numeric(out["label"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(3.0).clip(1.0, 20.0)
    return out[IMAGE_FEEDBACK_COLUMNS]


def save_image_feedback(feedback: pd.DataFrame, path: str | Path = "ai/models/manual_image_feedback.csv") -> Path:
    output = resolve_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cleaned = feedback.copy()
    cleaned["image_path"] = cleaned["image_path"].astype(str)
    cleaned = cleaned[cleaned["image_path"].str.len().gt(0)]
    cleaned = cleaned.drop_duplicates(subset=["image_path", "direction"], keep="last")
    cleaned.to_csv(output, index=False)
    return output


def append_image_feedback(
    image_path: str | Path,
    symbol: str,
    timeframe: str,
    direction: str,
    label: int,
    note: str = "",
    weight: float = 5.0,
    path: str | Path = "ai/models/manual_image_feedback.csv",
) -> Path:
    resolved_image = resolve_path(image_path)
    if not resolved_image.exists():
        raise FileNotFoundError(f"Image not found: {resolved_image}")
    existing = load_image_feedback(path)
    row = pd.DataFrame(
        [
            {
                "image_path": str(resolved_image),
                "symbol": symbol,
                "timeframe": timeframe,
                "direction": direction.lower(),
                "label": int(label),
                "weight": float(weight),
                "note": note,
            }
        ]
    )
    return save_image_feedback(pd.concat([existing, row], ignore_index=True), path)
