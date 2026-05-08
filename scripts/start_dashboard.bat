@echo off
cd /d "%~dp0\.."
start "AI SMC Dashboard" python -m streamlit run dashboard/streamlit_app.py --server.port 8501 --server.headless true
