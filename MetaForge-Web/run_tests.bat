@echo off
echo Running MetaForge-Web tests using pymatgenenv...
cd /d "%~dp0"
..\pymatgenenv\Scripts\pytest.exe tests/
pause
