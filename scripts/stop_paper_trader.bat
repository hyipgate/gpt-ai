@echo off
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*run.py*--continuous*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Write-Host ('Stopped trader PID ' + $_.ProcessId) }"
pause
