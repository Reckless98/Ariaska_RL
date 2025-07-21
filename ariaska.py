#!/usr/bin/env python3
"""
ARIASKA_RL Launcher
Simple launcher that can be used as 'ariaska' command
"""

import os
import sys
from pathlib import Path

# Add the project directory to Python path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

# Set UTF-8 encoding for Windows
os.environ['PYTHONIOENCODING'] = 'utf-8'

if __name__ == "__main__":
    from ariaska_cli import main
    main()
