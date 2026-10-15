@echo off
cd /d "%~dp0"
echo ForestWatch AI - setup, training and launch
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
python src\generate_dataset.py
python src\train_models.py
if errorlevel 1 (
  echo Training failed. Read the error above and check README.md.
  pause
  exit /b 1
)
python -m streamlit run app.py
pause
