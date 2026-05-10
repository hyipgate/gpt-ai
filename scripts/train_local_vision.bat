@echo off
cd /d "%~dp0.."
python train_local_vision.py --symbol XAUUSD --timeframe M5 --bars 5000
pause
