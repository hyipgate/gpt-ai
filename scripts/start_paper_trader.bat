@echo off
cd /d "%~dp0\.."
start "AI SMC Paper Trader" python run.py --mode paper --symbol XAUUSD --timeframe M5 --continuous --poll-seconds 30
