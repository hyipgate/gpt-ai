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
