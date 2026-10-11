@echo off
cd /d "%~dp0"
python bet_dashboard.py --open
if errorlevel 1 pause
