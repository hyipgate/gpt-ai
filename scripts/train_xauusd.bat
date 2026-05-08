@echo off
cd /d "%~dp0\.."
python train_model.py --symbol XAUUSD --timeframe M5 --bars 10000 --model xgboost --min-trades 300
pause
