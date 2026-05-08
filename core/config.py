from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TradingConfig:
    symbol: str = "XAUUSD"
    timeframe: str = "M5"
    higher_timeframes: tuple[str, ...] = ("M15", "H1")
    require_mtf_alignment: bool = False
    bars: int = 5000
    paper_mode: bool = True
    confidence_threshold: float = 0.70
    magic_number: int = 260507
    allowed_sessions: tuple[str, ...] = ("london", "new_york", "overlap")


@dataclass(frozen=True)
class RiskConfig:
    account_currency: str = "USD"
    risk_per_trade: float = 0.005
    max_daily_loss: float = 0.03
    max_open_trades: int = 2
    use_spread_filter: bool = False
    max_spread_points: float = 500
    min_atr_points: float = 20
    max_atr_points: float = 1000
    max_slippage_points: int = 200
    kill_switch_file: str = "config/kill_switch"


@dataclass(frozen=True)
class SMCConfig:
    swing_window: int = 5
    equal_level_tolerance_atr: float = 0.12
    liquidity_sweep_lookback: int = 20
    order_block_lookback: int = 12
    fvg_min_atr_fraction: float = 0.10


@dataclass(frozen=True)
class ResearchConfig:
    use_quant_research_pipeline: bool = True
    use_setup_filter: bool = True
    use_regime_detection: bool = True
    use_quant_labels: bool = True
    use_feature_pruning: bool = True
    enforce_production_gate: bool = True
    require_london_or_ny_for_research: bool = True


@dataclass(frozen=True)
class BacktestConfig:
    initial_equity: float = 10_000.0
    commission_per_lot: float = 7.0
    slippage_points: float = 0.0
    spread_points: float = 0.0
    point_value: float = 0.01
    lot_size: float = 0.01
    contract_size: float = 100.0
    sizing_mode: str = "fixed_lot"


@dataclass(frozen=True)
class PathConfig:
    model_path: str = "ai/models/model.pkl"
    database_url: str = "sqlite:///database/trading_journal.sqlite"


@dataclass(frozen=True)
class AppConfig:
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    smc: SMCConfig = field(default_factory=SMCConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    paths: PathConfig = field(default_factory=PathConfig)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(path: str | Path | None = None) -> AppConfig:
    load_dotenv(ROOT_DIR / ".env")
    config_path = Path(path) if path else ROOT_DIR / "config" / "default.yaml"
    raw = _load_yaml(config_path)

    env_override: dict[str, Any] = {
        "trading": {
            "symbol": os.getenv("TRADER_SYMBOL"),
            "timeframe": os.getenv("TRADER_TIMEFRAME"),
            "paper_mode": os.getenv("TRADER_PAPER_MODE"),
        },
        "paths": {"database_url": os.getenv("TRADER_DATABASE_URL")},
    }
    cleaned = {
        section: {k: v for k, v in values.items() if v is not None}
        for section, values in env_override.items()
        if any(v is not None for v in values.values())
    }
    raw = _merge_dict(raw, cleaned)
    if "higher_timeframes" in raw.get("trading", {}) and isinstance(raw["trading"]["higher_timeframes"], list):
        raw["trading"]["higher_timeframes"] = tuple(raw["trading"]["higher_timeframes"])
    if "allowed_sessions" in raw.get("trading", {}) and isinstance(raw["trading"]["allowed_sessions"], list):
        raw["trading"]["allowed_sessions"] = tuple(raw["trading"]["allowed_sessions"])
    if "paper_mode" in raw.get("trading", {}):
        value = raw["trading"]["paper_mode"]
        raw["trading"]["paper_mode"] = str(value).lower() in {"1", "true", "yes", "on"}
    if "require_mtf_alignment" in raw.get("trading", {}):
        value = raw["trading"]["require_mtf_alignment"]
        raw["trading"]["require_mtf_alignment"] = str(value).lower() in {"1", "true", "yes", "on"}
    for key, value in list(raw.get("research", {}).items()):
        if isinstance(value, bool):
            continue
        raw["research"][key] = str(value).lower() in {"1", "true", "yes", "on"}
    for key, value in list(raw.get("risk", {}).items()):
        if key.startswith("use_") and not isinstance(value, bool):
            raw["risk"][key] = str(value).lower() in {"1", "true", "yes", "on"}

    return AppConfig(
        trading=TradingConfig(**raw.get("trading", {})),
        risk=RiskConfig(**raw.get("risk", {})),
        smc=SMCConfig(**raw.get("smc", {})),
        research=ResearchConfig(**raw.get("research", {})),
        backtest=BacktestConfig(**raw.get("backtest", {})),
        paths=PathConfig(**raw.get("paths", {})),
    )


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT_DIR / path
