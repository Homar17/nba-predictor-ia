@echo off
cd /d "%~dp0"
call env_nba\Scripts\activate
start /B pythonw app.py