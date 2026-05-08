@echo off
cd /d "%~dp0\.."
python backtest_gold.py --symbol XAUUSD --timeframe M5 --bars 10000 --lot 0.01 --contract-size 100
pause
