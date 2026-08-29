@echo off
echo Running MetaForge-Web tests...
cd /d "%~dp0"
if exist "..\.venv\Scripts\pytest.exe" (
    ..\.venv\Scripts\pytest.exe tests/
) else if exist "..\pymatgenenv\Scripts\pytest.exe" (
    ..\pymatgenenv\Scripts\pytest.exe tests/
) else (
    pytest tests/
)
pause
