@echo off
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*streamlit*dashboard/streamlit_app.py*' -or $_.CommandLine -like '*streamlit*dashboard\\streamlit_app.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Write-Host ('Stopped dashboard PID ' + $_.ProcessId) }"
pause
