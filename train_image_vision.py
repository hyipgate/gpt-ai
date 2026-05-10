from __future__ import annotations

import argparse
import json

import pandas as pd

from ai.image_vision_model import train_image_vision_model
from core.config import resolve_path


def _merge_feedback(manual_path: str, auto_path: str) -> str:
    from ai.image_feedback import load_image_feedback, save_image_feedback

    merged_path = "ai/models/combined_image_feedback.csv"
    manual = load_image_feedback(manual_path)
    auto = load_image_feedback(auto_path)
    save_image_feedback(pd.concat([auto, manual], ignore_index=True), merged_path)
    return merged_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the local screenshot/image vision model.")
    parser.add_argument("--feedback-path", default="ai/models/manual_image_feedback.csv")
    parser.add_argument("--auto-feedback-path", default="ai/models/auto_image_feedback.csv")
    parser.add_argument("--model-path", default="ai/models/image_vision_model.pkl")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.60)
    args = parser.parse_args()

    feedback_path = _merge_feedback(args.feedback_path, args.auto_feedback_path)
    payload = train_image_vision_model(
        feedback_path=feedback_path,
        model_path=args.model_path,
        size=args.size,
        threshold=args.threshold,
    )
    print("Saved image vision model:", resolve_path(args.model_path))
    print(json.dumps(payload["metrics"], indent=2))


if __name__ == "__main__":
    main()
