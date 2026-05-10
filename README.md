# AI Smart Money Trading System for MetaTrader 5

This project is a modular trading framework, not a blind price-direction bot. The intended workflow is:

1. Rule-based Smart Money structure detection.
2. ML probability filtering for setup quality.
3. Strict risk management before any order is allowed.

The code supports MT5 data access, SMC feature generation, outcome-based labels, walk-forward validation, backtesting with costs, paper/live execution hooks, SQLite journaling, and a Streamlit dashboard.

## Quick Start

```powershell
pip install -r requirements.txt
python train_model.py --symbol XAUUSD --timeframe M5 --bars 5000
python run.py --mode paper --symbol XAUUSD --timeframe M5
streamlit run dashboard/streamlit_app.py
```

## Windows Shortcuts

Double-click the batch files in `scripts/`, or run them from PowerShell:

```powershell
scripts\install_requirements.bat
scripts\train_xauusd.bat
scripts\backtest_gold.bat
scripts\start_paper_trader.bat
scripts\start_dashboard.bat
scripts\status.bat
scripts\stop_paper_trader.bat
scripts\stop_dashboard.bat
```

Live trading requires MetaTrader 5 to be installed, logged in, and allowed to trade from Python. Keep `TRADER_PAPER_MODE=true` until the strategy has been validated on out-of-sample data and demo execution.

## Philosophy

The model scores whether a structurally valid setup has favorable expectancy. It does not predict the next candle in isolation, and it does not bypass risk controls.

## Research Pipeline

Data Layer
-> Setup Filter Layer
-> Regime Detection Layer
-> Feature Engine
-> ML Scoring Model
-> Risk Manager
-> Backtester
-> Execution Layer

The training pipeline now keeps only valid institutional setup rows before ML labeling: liquidity/inducement, CHoCH, order block context, London/New York session, volatility, and spread filters must pass. Labels are +2R before -1R trade-quality outcomes with MFE/MAE, not candle-direction labels.

## Hybrid Chart Context

The live trader now builds a hybrid analysis bundle on each evaluated closed candle:

- Structured context from OHLC/SMC features: trend, momentum, market structure, RSI state, volatility, nearby support/resistance, and risk notes.
- A local no-API visual signal layer that converts the chart context into visual bias, signal, confidence, reasons, and invalidation.
- A rendered candlestick + RSI PNG snapshot saved under `logs/chart_snapshots/`.
- A ready prompt that can be sent with the chart image to a vision model.

The bundle is stored inside `model_decision` for opened paper/live trades. Configure it in `config/default.yaml` under `hybrid` and `paths.chart_snapshot_dir`.

Backtests also add `hybrid_context`, `hybrid_local_vision`, `hybrid_trained_vision`, `hybrid_chart_path`, and `hybrid_prompt` columns to the trades CSV. By default the backtest stores structured context only; add `--hybrid-charts` to `backtest_gold.py` when you want one PNG per backtest entry.

Backtests also attach higher-timeframe context from `trading.higher_timeframes` and apply `trading.require_mtf_alignment` without lookahead, using only the latest higher-timeframe candle available at each base candle time.

Train the local chart vision model with:

```powershell
python train_local_vision.py --symbol XAUUSD --timeframe M5 --bars 5000
```

The script automatically maintains a separate history and model per symbol/timeframe, such as `ai/models/local_vision_history_XAUUSD_M5.csv` and `ai/models/local_vision_model_XAUUSD_M5.pkl`. Every run fetches the latest MT5 bars, merges them with the saved history, removes duplicate candle times, and retrains the model. Use `--no-history-cache` when you want to train only on the freshly fetched bars.

Teach it manually with corrections:

```powershell
python teach_local_vision.py --time 2026-05-08T16:25:00Z --symbol XAUUSD --timeframe M5 --direction buy --label bad --note "buy into resistance"
python train_local_vision.py --symbol XAUUSD --timeframe M5 --bars 5000
```

Manual feedback is stored in `ai/models/manual_vision_feedback.csv` and overrides the automatic TP-before-SL label with a higher training weight.

After training, Hybrid automatically includes `trained_vision` probabilities from `ai/models/local_vision_model.pkl`.

Use it as a real backtest filter with:

```powershell
python backtest_gold.py --symbol XAUUSD --timeframe M5 --bars 10000 --entry-mode rules --rules-profile all_valid --vision-filter --vision-threshold 0.55
```

You can also teach from chart screenshots:

```powershell
python teach_image_vision.py --image "C:\path\to\chart.png" --symbol XAUUSD --timeframe M5 --direction sell --label good --note "clean rejection from resistance"
python train_image_vision.py
```

Screenshot feedback is stored in `ai/models/manual_image_feedback.csv`. After training, Hybrid includes `image_vision` when a chart PNG is rendered.

For a stronger local visual model, train the CNN:

```powershell
python train_cnn_image_vision.py --epochs 30 --size 96
```

The CNN uses only local screenshots and labels, saves to `ai/models/cnn_image_vision_model.keras`, and Hybrid includes `cnn_image_vision` when a chart PNG is rendered.

To build a training image dataset from backtest outcomes without enabling Hybrid chart rendering:

```powershell
python backtest_gold.py --symbol XAUUSD --timeframe M5 --bars 10000 --entry-mode rules --rules-profile all_valid --reward-r 1.5 --export-vision-dataset
```

This saves entry screenshots under `logs/chart_snapshots/vision_dataset/` and labels them in `ai/models/auto_image_feedback.csv` using the backtest result: profitable trades are `good`, losing trades are `bad`. Manual screenshot feedback still overrides automatic feedback when both are merged for training.
