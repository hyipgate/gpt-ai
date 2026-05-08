@echo off
cd /d "%~dp0\.."
python run.py --mode paper --symbol XAUUSD --timeframe M5
pause
