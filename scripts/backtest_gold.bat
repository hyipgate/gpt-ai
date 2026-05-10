@echo off
cd /d "%~dp0\.."
python backtest_gold.py --symbol XAUUSD --timeframe M5 --bars 10000 --entry-mode rules --rules-profile all_valid --reward-r 1.5
pause
