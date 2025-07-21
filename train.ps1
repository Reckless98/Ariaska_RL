# ARIASKA_RL Training Script for Windows
# This script sets UTF-8 encoding and runs robust training

Write-Host "Starting ARIASKA_RL Robust Training with UTF-8 encoding..." -ForegroundColor Green

# Set UTF-8 encoding for Python
$env:PYTHONIOENCODING = "utf-8"

# Start robust training
Write-Host "Launching robust_training.py..." -ForegroundColor Yellow
python robust_training.py

Write-Host "ARIASKA_RL training session ended." -ForegroundColor Blue
