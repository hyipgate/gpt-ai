from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path

from core.config import RiskConfig, resolve_path


@dataclass(frozen=True)
class RiskState:
    equity: float
    balance_start_of_day: float
    open_trades: int
    spread_points: float
    atr_points: float
    session: str
    confidence: float
    now: datetime
    news_blocked: bool = False


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reason: str
    position_size: float = 0.0


class RiskManager:
    def __init__(self, config: RiskConfig | None = None, allowed_sessions: tuple[str, ...] = ("london", "new_york", "overlap")) -> None:
        self.config = config or RiskConfig()
        self.allowed_sessions = set(allowed_sessions)

    def _kill_switch_active(self) -> bool:
        return resolve_path(self.config.kill_switch_file).exists()

    def session_name(self, now: datetime) -> str:
        hour = now.hour
        if 12 <= hour <= 16:
            return "overlap"
        if 7 <= hour <= 11:
            return "london"
        if 17 <= hour <= 20:
            return "new_york"
        if 0 <= hour <= 6:
            return "asia"
        return "off_session"

    def daily_loss_fraction(self, state: RiskState) -> float:
        if state.balance_start_of_day <= 0:
            return 0.0
        return max(0.0, (state.balance_start_of_day - state.equity) / state.balance_start_of_day)

    def position_size(self, equity: float, entry: float, stop: float, point: float, contract_size: float) -> float:
        stop_distance = abs(entry - stop)
        if stop_distance <= 0 or point <= 0 or contract_size <= 0:
            return 0.0
        cash_risk = equity * self.config.risk_per_trade
        raw_lots = cash_risk / (stop_distance * contract_size)
        return round(max(raw_lots, 0.0), 2)

    def validate(self, state: RiskState, threshold: float, proposed_size: float = 0.0) -> RiskDecision:
        if self._kill_switch_active():
            return RiskDecision(False, "kill switch file is active")
        if state.news_blocked:
            return RiskDecision(False, "news avoidance hook blocked trading")
        if self.daily_loss_fraction(state) >= self.config.max_daily_loss:
            return RiskDecision(False, "max daily loss reached")
        if state.open_trades >= self.config.max_open_trades:
            return RiskDecision(False, "max open trades reached")
        if self.config.use_spread_filter and state.spread_points > self.config.max_spread_points:
            return RiskDecision(False, "spread too wide")
        if not (self.config.min_atr_points <= state.atr_points <= self.config.max_atr_points):
            return RiskDecision(False, "volatility outside configured bounds")
        if state.session not in self.allowed_sessions:
            return RiskDecision(False, f"session not allowed: {state.session}")
        if state.confidence < threshold:
            return RiskDecision(False, "model probability below threshold")
        if proposed_size <= 0:
            return RiskDecision(False, "invalid position size")
        return RiskDecision(True, "approved", proposed_size)

    def can_trade(self, spread: float, confidence: float, open_positions: int) -> bool:
        state = RiskState(
            equity=100_000,
            balance_start_of_day=100_000,
            open_trades=open_positions,
            spread_points=spread,
            atr_points=self.config.min_atr_points,
            session=next(iter(self.allowed_sessions)),
            confidence=confidence,
            now=datetime.utcnow(),
        )
        return self.validate(state, threshold=0.70, proposed_size=0.01).approved
