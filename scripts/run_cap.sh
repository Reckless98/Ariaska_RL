#!/bin/bash
cd /home/zer0/Projects/Ariaska_RL || exit 1
source .venv/bin/activate
rm -f logs/campaign_state.json
python -u ariaska_cli.py smart-train --env htb --target 10.129.4.210 --episodes 1 --steps 40 --difficulty normal --verbosity quiet
