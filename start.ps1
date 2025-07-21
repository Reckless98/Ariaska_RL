# ARIASKA_RL Startup Script for Windows
# This script sets UTF-8 encoding and starts the system

Write-Host "Starting ARIASKA_RL with UTF-8 encoding..." -ForegroundColor Green

# Set UTF-8 encoding for Python
$env:PYTHONIOENCODING = "utf-8"

# Start main.py
Write-Host "Launching main.py..." -ForegroundColor Yellow
python main.py

Write-Host "ARIASKA_RL session ended." -ForegroundColor Blue
