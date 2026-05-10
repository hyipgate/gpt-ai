from __future__ import annotations

from dataclasses import asdict, dataclass

from ai.chart_context import MarketContext


@dataclass(frozen=True)
class LocalVisionSignal:
    visual_bias: str
    signal: str
    confidence: float
    agrees_with_numeric_context: bool
    reasons: list[str]
    invalidation: str

    def to_dict(self) -> dict:
        return asdict(self)


def _side_from_text(value: str) -> int:
    text = str(value).lower()
    if "bullish" in text or text in {"buy", "long"}:
        return 1
    if "bearish" in text or text in {"sell", "short"}:
        return -1
    return 0


def analyze_local_chart_signal(context: MarketContext) -> LocalVisionSignal:
    score = 0.0
    reasons: list[str] = []

    trend_side = _side_from_text(context.trend)
    momentum_side = _side_from_text(context.momentum)
    structure_side = _side_from_text(context.market_structure)
    rsi_side = _side_from_text(context.rsi_state)
    body_side = _side_from_text(context.body_pressure)

    if trend_side:
        score += 0.22 * trend_side
        reasons.append(f"trend={context.trend}")
    if momentum_side:
        score += 0.25 * momentum_side
        reasons.append(f"momentum={context.momentum}")
    if structure_side:
        score += 0.25 * structure_side
        reasons.append(f"structure={context.market_structure}")
    if rsi_side:
        score += 0.14 * rsi_side
        reasons.append(f"rsi_state={context.rsi_state}")
    if body_side:
        score += 0.08 * body_side
        reasons.append(f"last_candle={context.body_pressure}")

    near_support = (
        context.nearest_support.distance_atr is not None
        and context.nearest_support.distance_atr <= 1.0
    )
    near_resistance = (
        context.nearest_resistance.distance_atr is not None
        and context.nearest_resistance.distance_atr <= 1.0
    )
    if near_support:
        score += 0.12
        reasons.append("price_near_support")
    if near_resistance:
        score -= 0.12
        reasons.append("price_near_resistance")

    if context.volatility == "compressed":
        score *= 0.70
        reasons.append("compressed_range_reduces_conviction")
    elif context.volatility == "expanding":
        score *= 1.12
        reasons.append("volatility_expansion")

    confidence = min(abs(score), 1.0)
    if confidence < 0.22:
        visual_bias = "neutral"
        signal = "no_trade"
    elif score > 0:
        visual_bias = "bullish"
        signal = "buy_watch"
    else:
        visual_bias = "bearish"
        signal = "sell_watch"

    numeric_sides = [side for side in (trend_side, momentum_side, structure_side, rsi_side) if side]
    if not numeric_sides or visual_bias == "neutral":
        agrees = True
    else:
        visual_side = 1 if visual_bias == "bullish" else -1
        agrees = sum(1 for side in numeric_sides if side == visual_side) >= max(1, len(numeric_sides) // 2)

    if signal == "buy_watch":
        invalidation = "close below nearest support or bullish momentum failure"
    elif signal == "sell_watch":
        invalidation = "close above nearest resistance or bearish momentum failure"
    else:
        invalidation = "wait for clear break, sweep, or momentum alignment"

    return LocalVisionSignal(
        visual_bias=visual_bias,
        signal=signal,
        confidence=round(float(confidence), 4),
        agrees_with_numeric_context=agrees,
        reasons=reasons,
        invalidation=invalidation,
    )
