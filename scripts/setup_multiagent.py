"""
fix_imports.py — ARIASKA Auto Import Fixer v1.0

Fixes broken import paths after folder restructuring.
Run from root of the project (where main.py is located).
"""

import os
import re

# Map incorrect → correct import patterns
IMPORT_FIXES = {
    r"from core\.teach\.teach\.teach import TeachModule": "from core.teach.teach import TeachModule",
    r"from core\.rl_agent import RLAgent": "from core.agents.red_agent import RedAgent",
    r"from core\.models\.stats_monitor import StatsMonitor": "from core.monitor.stats_monitor import StatsMonitor",
    r"from core\.logic\.output_interpreter import analyze_output": "from core.logic.output_interpreter import analyze_output",
    r"from core\.logic\.rule_engine import rule_based_selection": "from core.logic.rule_engine import rule_based_selection",
    r"from core\.logic\.chainbuilder import .*": "from core.logic.chainbuilder import build_and_store_chain",
    r"from core\.cyber_environment import CyberEnvironment": "from core.environment.cyber_environment import CyberEnvironment",
    r"from core\.teach import TeachModule": "from core.teach.teach import TeachModule",
    r"from core\.value_net import ValueNet": "from core.models.value_net import ValueNet",
    r"from core\.policy_net import PolicyNet": "from core.models.policy_net import PolicyNet",
}

TARGET_EXT = (".py",)

def fix_file_imports(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    modified = False

    for line in lines:
        new_line = line
        for pattern, replacement in IMPORT_FIXES.items():
            if re.search(pattern, line):
                new_line = re.sub(pattern, replacement, line)
                print(f"🔧 Fixed import in {filepath}:\n    {line.strip()} ➜ {new_line.strip()}")
                modified = True
        new_lines.append(new_line)

    if modified:
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

def fix_all_imports(start_dir="."):
    print("🔎 Scanning and fixing imports...")
    for root, _, files in os.walk(start_dir):
        for file in files:
            if file.endswith(TARGET_EXT):
                fix_file_imports(os.path.join(root, file))
    print("✅ Import fixing complete.")

if __name__ == "__main__":
    fix_all_imports()
