from __future__ import annotations

import argparse

from ai.features import build_features
from ai.inference import ModelScorer
from app.live_trader import build_structure
from core.config import load_config
from core.mt5_connector import MT5Connector
from market.datafeed import MT5DataFeed, MarketDataRequest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol")
    parser.add_argument("--timeframe")
    parser.add_argument("--bars", type=int, default=500)
    args = parser.parse_args()
    config = load_config()
    connector = MT5Connector()
    connector.initialize()
    try:
        df = MT5DataFeed(connector).get_rates(MarketDataRequest(args.symbol or config.trading.symbol, args.timeframe or config.trading.timeframe, args.bars))
    finally:
        connector.shutdown()
    features = build_features(build_structure(df, config))
    print(ModelScorer(config.paths.model_path, config.trading.confidence_threshold).score(features))


if __name__ == "__main__":
    main()
