@echo off
echo Starting MetaForge HEA Predictor...
echo.
if exist "%~dp0..\.venv\Scripts\python.exe" (
    "%~dp0..\.venv\Scripts\python.exe" "%~dp0app.py"
) else if exist "%~dp0..\pymatgenenv\Scripts\python.exe" (
    "%~dp0..\pymatgenenv\Scripts\python.exe" "%~dp0app.py"
) else (
    python "%~dp0app.py"
)
pause
