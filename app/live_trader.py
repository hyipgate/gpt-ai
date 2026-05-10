from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone

import MetaTrader5 as mt5
import pandas as pd

from ai.features import add_technical_features, build_features
from ai.hybrid import build_hybrid_analysis
from ai.inference import ModelScorer
from ai.local_vision_model import local_vision_model_path
from ai.expectancy_model import ExpectancyModel
from ai.regime import detect_market_regime
from ai.scoring import score_setups
from ai.setup_filter import SetupFilterConfig, filter_valid_setups, infer_setup_direction
from core.config import load_config, resolve_path
from core.execution_engine import ExecutionEngine, OrderRequest
from core.logger import TradeLog, TradingJournal, configure_logging
from core.mt5_connector import MT5Connector
from core.paper_trader import PaperExecutionConfig, PaperTradeManager
from core.risk_manager import RiskManager, RiskState
from market.datafeed import MT5DataFeed, MarketDataRequest
from market.multitimeframe import collect_mtf_context, mtf_alignment
from market.structure import detect_swings
from smc.bos import detect_bos
from smc.choch import detect_choch
from smc.fvg import detect_fvg
from smc.liquidity import detect_liquidity
from smc.order_blocks import detect_order_blocks


LOGGER = logging.getLogger(__name__)


def build_structure(df: pd.DataFrame, config) -> pd.DataFrame:
    df = add_technical_features(df)
    df = detect_swings(df, window=config.smc.swing_window)
    df = detect_bos(df)
    df = detect_choch(df)
    df = detect_order_blocks(df, lookback=config.smc.order_block_lookback)
    df = detect_fvg(df, atr_fraction=config.smc.fvg_min_atr_fraction)
    df = detect_liquidity(df, lookback=config.smc.liquidity_sweep_lookback, equal_level_tolerance_atr=config.smc.equal_level_tolerance_atr)
    return df


def infer_direction(row: pd.Series) -> str | None:
    if "valid_setup" in row and not bool(row.get("valid_setup", False)):
        return None
    if int(row.get("setup_direction", 0)) > 0:
        return "buy"
    if int(row.get("setup_direction", 0)) < 0:
        return "sell"
    if not any(
        int(row.get(column, 0)) != 0
        for column in ("choch_state", "bos_state", "fvg_direction", "liquidity_state")
    ):
        return None
    state = int(row.get("choch_state", 0) or row.get("bos_state", 0) or row.get("fvg_direction", 0) or row.get("liquidity_state", 0))
    if state > 0:
        return "buy"
    if state < 0:
        return "sell"
    return None


def account_snapshot() -> tuple[float, float, int]:
    account = mt5.account_info()
    positions = mt5.positions_get() or []
    if account is None:
        return 100_000.0, 100_000.0, len(positions)
    return float(account.equity), float(account.balance), len(positions)

def evaluate_market(
    connector: MT5Connector,
    journal: TradingJournal,
    symbol: str,
    timeframe: str,
    paper_mode: bool,
    skip_bar: pd.Timestamp | None = None,
) -> pd.Timestamp | None:

    config = load_config()

    feed = MT5DataFeed(connector)

    df = feed.get_rates(
        MarketDataRequest(
            symbol=symbol,
            timeframe=timeframe,
            bars=config.trading.bars,
        )
    )

    # =========================================================
    # USE ONLY CLOSED CANDLES
    # =========================================================
    df = df.iloc[:-1].copy()

    if len(df) < 100:
        return None

    # =========================================================
    # PAPER TRADE UPDATE
    # =========================================================
    if paper_mode:
        PaperTradeManager(
            journal,
            PaperExecutionConfig(
                contract_size=config.backtest.contract_size,
                commission_per_lot=config.backtest.commission_per_lot,
            ),
        ).update_open_trades(symbol, df)

    # =========================================================
    # BUILD MARKET STRUCTURE
    # =========================================================
    df = build_structure(df, config)

    if config.research.use_regime_detection:
        df = detect_market_regime(df)
    else:
        df["regime_id"] = 0
        df["regime"] = "disabled"

    # =========================================================
    # SETUP FILTER
    # =========================================================
    df["valid_setup"] = False
    df["setup_direction"] = 0
    df["setup_quality_rule_score"] = 0.0

    if (
        config.research.use_quant_research_pipeline
        and config.research.use_setup_filter
    ):

        valid_setups = filter_valid_setups(
            df,
            SetupFilterConfig(
                min_atr_points=config.risk.min_atr_points,
                max_spread_points=config.risk.max_spread_points,
                point_value=config.backtest.point_value,
                require_london_or_ny=config.research.require_london_or_ny_for_research,
            ),
        )

        setup_columns = (
            "valid_setup",
            "setup_direction",
            "setup_quality_rule_score",
            "liquidity_support",
            "liquidity_or_inducement",
            "choch_confirmed",
            "order_block_exists",
            "ob_support",
            "session_ok",
            "volatility_ok",
            "spread_ok",
        )

        for column in setup_columns:
            if column in valid_setups:
                df.loc[valid_setups.index, column] = valid_setups[column]

    else:
        df["setup_direction"] = infer_setup_direction(df)
        df["valid_setup"] = df["setup_direction"].ne(0)
        df["setup_quality_rule_score"] = (
            df["valid_setup"].astype(float)
        )

    # =========================================================
    # FEATURES
    # =========================================================
    features = build_features(df)

    hybrid_analysis = None
    if config.hybrid.enabled:
        try:
            hybrid_analysis = build_hybrid_analysis(
                df,
                symbol=symbol,
                timeframe=timeframe,
                chart_dir=config.paths.chart_snapshot_dir,
                lookback=config.hybrid.lookback_candles,
                render_chart=config.hybrid.render_chart,
                show_rsi=config.hybrid.show_rsi,
                use_trained_model=config.hybrid.use_trained_model,
                trained_model_path=local_vision_model_path(symbol, timeframe, config.hybrid.trained_model_path),
            )
        except Exception as exc:
            LOGGER.exception("hybrid analysis failed")
            journal.log_risk_event("hybrid_error", str(exc))

    scorer = ModelScorer(
        config.paths.model_path,
        threshold=config.trading.confidence_threshold,
    )

    prediction = scorer.score(features)

    # =========================================================
    # SAFE PROBABILITY
    # =========================================================
    latest_probability = float(
        prediction.probability[-1]
        if hasattr(prediction.probability, "__len__")
        else prediction.probability
    )

    # =========================================================
    # EXPECTANCY
    # =========================================================
    try:
        expectancy_model = ExpectancyModel.load(
            "ai/models/expectancy_model.pkl"
        )

        expected_r_series = (
            expectancy_model.predict_expected_return(
                features
            )
        )

    except Exception:
        expected_r_series = pd.Series(
            0.0,
            index=features.index,
        )

    # =========================================================
    # SETUP SCORING
    # =========================================================
    setup_scores = score_setups(
        features,
        pd.Series(
            prediction.probability,
            index=features.index,
        ),
        expected_r_series,
    )

    latest = df.iloc[-1]
    latest_score = setup_scores.iloc[-1]

    latest_time = pd.Timestamp(latest["time"])

    # =========================================================
    # PREVENT DUPLICATE BAR
    # =========================================================
    if (
        skip_bar is not None
        and latest_time == skip_bar
    ):
        LOGGER.info(
            "bar %s already processed; waiting",
            latest_time,
        )
        return latest_time

    direction = infer_direction(latest)
    if direction is None:
        journal.log_risk_event(
            "no_setup",
            f"No active structure direction "
            f"on bar {latest_time}",
        )
        return latest_time

    # =========================================================
    # MULTI TIMEFRAME
    # =========================================================
    mtf_context = collect_mtf_context(
        feed,
        symbol,
        tuple(config.trading.higher_timeframes),
        config,
        bars=min(config.trading.bars, 1200),
    )

    mtf_ok, mtf_score = mtf_alignment(
        direction,
        mtf_context,
    )

    # =========================================================
    # MARKET SNAPSHOT
    # =========================================================
    snapshot = connector.snapshot(symbol)

    # =========================================================
    # HARD SPREAD FILTER
    # =========================================================
    if (
        config.risk.max_spread_points is not None
        and snapshot.spread_points
        > config.risk.max_spread_points
    ):
        journal.log_risk_event(
            "spread_block",
            f"spread too high {snapshot.spread_points}",
        )
        return latest_time

    # =========================================================
    # ACCOUNT
    # =========================================================
    equity, balance, broker_open_trades = (
        account_snapshot()
    )

    paper_open_trades = (
        len(journal.open_paper_trades(symbol))
        if paper_mode
        else 0
    )

    open_trades = max(
        broker_open_trades,
        paper_open_trades,
    )

    atr_points = (
        float(latest.get("atr", 0.0))
        / snapshot.point
        if snapshot.point
        else 0.0
    )

    risk = RiskManager(
        config.risk,
        allowed_sessions=tuple(
            config.trading.allowed_sessions
        ),
    )

    session = risk.session_name(
        datetime.now(timezone.utc)
    )

    # =========================================================
    # ENTRY PRICE
    # =========================================================
    if paper_mode:
        entry = float(latest["close"])
    else:
        entry = (
            snapshot.ask
            if direction == "buy"
            else snapshot.bid
        )

    # =========================================================
    # STOP / TARGET
    # =========================================================
    stop_distance = max(
        float(latest.get("atr", 0.0)),
        snapshot.point
        * config.risk.min_atr_points,
    )

    if direction == "sell":
        sl = entry + stop_distance
        tp = entry - 2 * stop_distance
    else:
        sl = entry - stop_distance
        tp = entry + 2 * stop_distance

    # =========================================================
    # POSITION SIZE
    # =========================================================
    volume = (
        config.backtest.lot_size
        if paper_mode
        else risk.position_size_dynamic(
            equity,
            entry,
            sl,
            snapshot.point,
            snapshot.trade_contract_size,
            confidence=float(
                latest_score["confidence"]
            ),
            regime_quality=float(
                latest_score["regime_quality"]
            ),
            execution_risk=float(
                latest_score["execution_risk"]
            ),
            volatility_percentile=float(
                latest.get(
                    "atr_percentile",
                    0.5,
                )
            ),
        )
    )

    # =========================================================
    # RISK VALIDATION
    # =========================================================
    decision = risk.validate(
        RiskState(
            equity=equity,
            balance_start_of_day=balance,
            open_trades=open_trades,
            spread_points=snapshot.spread_points,
            atr_points=atr_points,
            session=session,
            confidence=latest_probability,
            now=datetime.now(timezone.utc),
        ),
        threshold=scorer.threshold,
        proposed_size=volume,
    )

    LOGGER.info(
        "bar=%s symbol=%s probability=%.3f direction=%s spread=%.1f session=%s",
        latest_time,
        symbol,
        latest_probability,
        direction,
        snapshot.spread_points,
        session,
    )

    # =========================================================
    # BLOCK CONDITIONS
    # =========================================================
    if (
        config.trading.require_mtf_alignment
        and not mtf_ok
    ):
        journal.log_risk_event(
            "mtf_block",
            f"MTF alignment blocked "
            f"{direction} on bar {latest_time}; "
            f"score={mtf_score}",
        )
        return latest_time

    if not decision.approved:
        journal.log_risk_event(
            "risk_block",
            f"{decision.reason} "
            f"on bar {latest_time}",
        )
        return latest_time

    # =========================================================
    # SETUP TYPE
    # =========================================================
    setup_type = (
        "choch"
        if latest.get("choch_state", 0)
        else "bos"
    )

    # =========================================================
    # PAPER MODE
    # =========================================================
    if paper_mode:

        journal.log_trade(
            TradeLog(
                symbol=symbol,
                direction=direction,
                setup_type=setup_type,
                confidence=latest_probability,
                probability=latest_probability,
                entry=entry,
                sl=sl,
                tp=tp,
                volume=volume,
                spread_points=snapshot.spread_points,
                setup_score=float(
                    latest_score["setup_score"]
                ),
                expected_r=float(
                    latest_score["expected_r"]
                ),
                regime=str(
                    latest.get(
                        "regime",
                        "unknown",
                    )
                ),
                execution_risk=float(
                    latest_score["execution_risk"]
                ),
                quality_tier=str(
                    latest_score["tier"]
                ),
                status="paper_open",
                model_decision=json.dumps(
                    {
                        "prediction": asdict(
                            prediction
                        ),
                        "setup_score": (
                            latest_score.to_dict()
                        ),
                        "mtf_score": mtf_score,
                        "hybrid": (
                            hybrid_analysis.to_dict()
                            if hybrid_analysis is not None
                            else None
                        ),
                    },
                    default=str,
                ),
            )
        )

        return latest_time

    # =========================================================
    # LIVE EXECUTION
    # =========================================================
    result = ExecutionEngine(
        connector,
        config.trading,
        config.risk,
    ).market_order(
        OrderRequest(
            symbol,
            direction,
            volume,
            sl,
            tp,
        )
    )

    journal.log_trade(
        TradeLog(
            symbol=symbol,
            direction=direction,
            setup_type=setup_type,
            confidence=latest_probability,
            probability=latest_probability,
            entry=result.price or entry,
            sl=sl,
            tp=tp,
            volume=volume,
            spread_points=snapshot.spread_points,
            setup_score=float(
                latest_score["setup_score"]
            ),
            expected_r=float(
                latest_score["expected_r"]
            ),
            regime=str(
                latest.get(
                    "regime",
                    "unknown",
                )
            ),
            execution_risk=float(
                latest_score["execution_risk"]
            ),
            quality_tier=str(
                latest_score["tier"]
            ),
            status=(
                "filled"
                if result.success
                else "rejected"
            ),
            execution_details=result.raw,
            model_decision=json.dumps(
                {
                    "prediction": asdict(prediction),
                    "setup_score": latest_score.to_dict(),
                    "mtf_score": mtf_score,
                    "hybrid": (
                        hybrid_analysis.to_dict()
                        if hybrid_analysis is not None
                        else None
                    ),
                },
                default=str,
            ),
        )
    )

    return latest_time


def run_once(symbol: str | None = None, timeframe: str | None = None, paper: bool | None = None) -> None:
    configure_logging()
    config = load_config()
    symbol = symbol or config.trading.symbol
    timeframe = timeframe or config.trading.timeframe
    paper_mode = config.trading.paper_mode if paper is None else paper
    connector = MT5Connector()
    journal = TradingJournal(config.paths.database_url)
    connector.initialize()
    try:
        evaluate_market(connector, journal, symbol, timeframe, paper_mode)
    finally:
        connector.shutdown()


def run_forever(
    symbol: str | None = None,
    timeframe: str | None = None,
    paper: bool | None = None,
    poll_seconds: int = 30,
) -> None:
    configure_logging()
    config = load_config()
    symbol = symbol or config.trading.symbol
    timeframe = timeframe or config.trading.timeframe
    paper_mode = config.trading.paper_mode if paper is None else paper
    connector = MT5Connector()
    journal = TradingJournal(config.paths.database_url)
    last_processed_bar: pd.Timestamp | None = None

    connector.initialize()
    LOGGER.info("continuous trader started symbol=%s timeframe=%s paper=%s poll_seconds=%s", symbol, timeframe, paper_mode, poll_seconds)
    try:
        while True:
            try:
                latest_bar = evaluate_market(connector, journal, symbol, timeframe, paper_mode, skip_bar=last_processed_bar)
                last_processed_bar = latest_bar or last_processed_bar
            except Exception as exc:
                LOGGER.exception("continuous trader cycle failed")
                journal.log_risk_event("runtime_error", str(exc))
                connector.reconnect()
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        LOGGER.info("continuous trader stopped by user")
    finally:
        connector.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol")
    parser.add_argument("--timeframe")
    parser.add_argument("--mode", choices=["paper", "live"], default=None)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    paper = None if args.mode is None else args.mode == "paper"
    if args.continuous:
        run_forever(args.symbol, args.timeframe, paper, args.poll_seconds)
    else:
        run_once(args.symbol, args.timeframe, paper)


if __name__ == "__main__":
    main()
