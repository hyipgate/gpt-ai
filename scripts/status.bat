@echo off
cd /d "%~dp0\.."
echo === Trader Processes ===
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*run.py*--continuous*' } | Select-Object ProcessId,CommandLine | Format-Table -AutoSize"
echo.
echo === Dashboard Processes ===
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*streamlit*dashboard/streamlit_app.py*' -or $_.CommandLine -like '*streamlit*dashboard\\streamlit_app.py*' } | Select-Object ProcessId,CommandLine | Format-Table -AutoSize"
echo.
echo === Recent Log ===
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Content logs\trader.log -Tail 20"
pause
