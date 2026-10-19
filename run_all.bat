@echo off
REM Process everything in data\raw, then open the dashboard.
call .venv\Scripts\activate.bat
python scripts\doctor.py
python run_pipeline.py || exit /b 1
streamlit run web\app.py
