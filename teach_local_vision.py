from __future__ import annotations

import argparse

from ai.feedback import append_feedback


def main() -> None:
    parser = argparse.ArgumentParser(description="Teach the local chart vision model with your own correction.")
    parser.add_argument("--time", required=True, help="Signal candle time, for example 2026-05-08T16:25:00Z")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M5")
    parser.add_argument("--direction", choices=["buy", "sell"], required=True)
    parser.add_argument("--label", choices=["good", "bad", "1", "0"], required=True)
    parser.add_argument("--note", default="")
    parser.add_argument("--weight", type=float, default=5.0)
    parser.add_argument("--feedback-path", default="ai/models/manual_vision_feedback.csv")
    args = parser.parse_args()

    label = 1 if args.label in {"good", "1"} else 0
    path = append_feedback(
        time=args.time,
        symbol=args.symbol,
        timeframe=args.timeframe,
        direction=args.direction,
        label=label,
        note=args.note,
        weight=args.weight,
        path=args.feedback_path,
    )
    print(f"Saved feedback: {path}")


if __name__ == "__main__":
    main()
