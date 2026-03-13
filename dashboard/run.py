"""
run.py
======
Launch from the testing/ root folder:

    & "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.10_3.10.3056.0_x64__qbz5n2kfra8p0\python3.10.exe" dashboard/run.py

Or just double-click start.bat
"""
import subprocess, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # testing/
PROJECT = os.path.join(ROOT, "btc_rl_project")

# Must cd into btc_rl_project so model paths like "models/ml/..." resolve
os.chdir(PROJECT)
os.environ["PYTHONPATH"] = ROOT

cmd = [
    sys.executable, "-m", "uvicorn",
    "dashboard.backend.app:app",
    "--host", "0.0.0.0",
    "--port", "8000",
    "--reload",
]

print("=" * 55)
print("  BTC ML Trading Sentinel")
print("  Dashboard: http://localhost:8000")
print("  API docs:  http://localhost:8000/docs")
print("=" * 55)
print()

subprocess.run(cmd)
