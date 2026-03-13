@echo off
REM ═══════════════════════════════════════════════════════
REM  BTC ML Trading Sentinel — Start Script
REM  Run from anywhere — just double-click!
REM ═══════════════════════════════════════════════════════

SET PYTHON="C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.10_3.10.3056.0_x64__qbz5n2kfra8p0\python3.10.exe"

echo.
echo  ============================================
echo   BTC ML Trading Sentinel
echo   Dashboard: http://localhost:8000
echo  ============================================
echo  Make sure .env exists in btc_rl_project\
echo.

REM Change to btc_rl_project so relative model paths resolve
cd /d "c:\Users\Sam-tech\Desktop\testing\btc_rl_project"

REM Add dashboard to PYTHONPATH so the import works
set PYTHONPATH=c:\Users\Sam-tech\Desktop\testing

%PYTHON% -m uvicorn dashboard.backend.app:app --host 0.0.0.0 --port 8000 --reload

pause
