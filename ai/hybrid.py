from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from ai.chart_context import MarketContext, build_market_context
from ai.chart_renderer import render_candlestick_snapshot
from ai.local_vision import LocalVisionSignal, analyze_local_chart_signal
from ai.local_vision_model import TrainedVisionPrediction, predict_trained_local_vision
from ai.image_vision_model import ImageVisionPrediction, predict_image_vision
from ai.cnn_image_vision_model import CnnVisionPrediction, predict_cnn_image_vision
from core.config import resolve_path


@dataclass(frozen=True)
class HybridAnalysis:
    context: MarketContext
    local_vision: LocalVisionSignal
    trained_vision: TrainedVisionPrediction | None
    image_vision: ImageVisionPrediction | None
    cnn_image_vision: CnnVisionPrediction | None
    chart_path: str | None
    prompt: str

    def to_dict(self) -> dict:
        return {
            "context": self.context.to_dict(),
            "local_vision": self.local_vision.to_dict(),
            "trained_vision": self.trained_vision.to_dict() if self.trained_vision else None,
            "image_vision": self.image_vision.to_dict() if self.image_vision else None,
            "cnn_image_vision": self.cnn_image_vision.to_dict() if self.cnn_image_vision else None,
            "chart_path": self.chart_path,
            "prompt": self.prompt,
        }


def build_vision_prompt(context: MarketContext) -> str:
    payload = context.to_dict()
    return (
        "Analyze this trading chart image together with the structured market context. "
        "Return JSON with: visual_bias, confidence, risk_notes, invalidation, and whether "
        "the image agrees with the numeric context. Structured context: "
        f"{payload}"
    )


def build_hybrid_analysis(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    chart_dir: str | Path = "logs/chart_snapshots",
    lookback: int = 200,
    render_chart: bool = True,
    show_rsi: bool = True,
    use_trained_model: bool = True,
    trained_model_path: str | Path = "ai/models/local_vision_model.pkl",
) -> HybridAnalysis:
    context = build_market_context(df, symbol=symbol, timeframe=timeframe, lookback=lookback)
    local_vision = analyze_local_chart_signal(context)
    direction = _latest_direction(df)
    trained_vision = (
        predict_trained_local_vision(
            df.tail(lookback),
            direction,
            trained_model_path,
            symbol=symbol,
            timeframe=timeframe,
        )
        if use_trained_model
        else None
    )
    chart_path = None
    if render_chart:
        stamp = _snapshot_stamp(context.last_time)
        safe_symbol = "".join(ch if ch.isalnum() else "_" for ch in symbol)
        safe_timeframe = "".join(ch if ch.isalnum() else "_" for ch in timeframe)
        output_path = resolve_path(chart_dir) / f"{safe_symbol}_{safe_timeframe}_{stamp}.png"
        chart_path = str(render_candlestick_snapshot(df, output_path, symbol, timeframe, lookback, show_rsi=show_rsi))
    image_vision = predict_image_vision(chart_path, direction=direction)
    cnn_image_vision = predict_cnn_image_vision(chart_path, direction=direction)
    return HybridAnalysis(
        context=context,
        local_vision=local_vision,
        trained_vision=trained_vision,
        image_vision=image_vision,
        cnn_image_vision=cnn_image_vision,
        chart_path=chart_path,
        prompt=build_vision_prompt(context),
    )


def _latest_direction(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    row = df.iloc[-1]
    for column in ("setup_direction", "choch_state", "bos_state", "structure_bias", "fvg_direction"):
        try:
            value = int(row.get(column, 0))
        except (TypeError, ValueError):
            value = 0
        if value != 0:
            return 1 if value > 0 else -1
    return 0


def _snapshot_stamp(value: str) -> str:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return "unknown_time"
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.strftime("%Y%m%d_%H%M%S")
