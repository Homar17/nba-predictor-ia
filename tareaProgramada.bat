@echo off
cd /d "%~dp0"
call env_nba\Scripts\activate
python run_pipeline.py