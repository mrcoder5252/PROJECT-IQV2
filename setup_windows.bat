@echo off
REM One-time setup: creates .venv, installs dependencies, downloads the spaCy model.
python -m venv .venv || exit /b 1
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt || exit /b 1
python -m spacy download en_core_web_sm
if not exist .env copy .env.example .env
echo.
echo Setup done. Edit .env (set PROJECTIQ_ADMIN_PASSKEY and optionally GEMINI_API_KEY),
echo put reports in data\raw\^<DEPT^>\^<Group N^>\..., then run run_all.bat
