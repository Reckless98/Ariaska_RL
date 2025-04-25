# audit_ariaska.py — ARIASKA Auditor v3.4 (Now with GPT/Model/Phase Sanity Checks)

import os
import json
import traceback
from datetime import datetime
from rich.console import Console
from rich.table import Table
import sys
import subprocess
from collections import Counter

console = Console()
results = []
log_lines = []

# Path Configuration
project_root = os.path.abspath(os.path.dirname(__file__))
core_path = os.path.join(project_root, "core")
data_path = os.path.join(project_root, "data")
models_path = os.path.join(core_path, "models")
sys.path.insert(0, core_path)
sys.path.insert(0, models_path)

try:
    from core.agents.red_agent import RedAgent
    from core.teach.teach import TeachModule
    from core.logic.output_interpreter import analyze_output
    from core.logic.chainbuilder import build_and_store_chain
    from core.gpt_manager import GPTManager
except ImportError as imp_err:
    print("[FATAL] Failed to import core modules:", imp_err)
    sys.exit(1)

gpt_manager = GPTManager()

# Logging
def log(msg):
    log_lines.append(msg)
    console.print(msg)

def report(title, status, note=""):
    msg = f"[{title}] {status} {note}"
    log(msg)

# File Validation
def check_file(name):
    path = os.path.join(data_path, name)
    if not os.path.exists(path):
        report(name, "❌ MISSING", path)
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        report(name, "✅ OK")
        return data
    except Exception as e:
        report(name, "❌ CORRUPTED", str(e))
        return None

# Deep Check for model files
def check_model_files():
    files = ["policy_net.py", "value_net.py"]
    missing = []
    for fname in files:
        path = os.path.join(models_path, fname)
        if not os.path.exists(path):
            missing.append(fname)
    if missing:
        report("Model Files", "❌ MISSING", ", ".join(missing))
    else:
        report("Model Files", "✅ OK")

# Memory Checks
def scan_memory_consistency(memory):
    actions = memory.get("actions", [])
    bad = 0
    for entry in actions:
        if not entry.get("command") or "phase" not in entry.get("context", {}):
            bad += 1
    if bad:
        report("Memory Consistency", "❌ FAIL", f"{bad} malformed entries")
    else:
        report("Memory Consistency", "✅ OK")

def report_phase_distribution(memory):
    phases = [a.get("context", {}).get("phase", "unknown") for a in memory.get("actions", [])]
    phase_count = Counter(phases)
    phase_str = ", ".join(f"{p}: {c}" for p, c in phase_count.items())
    report("Memory Phase Stats", "✅ OK", phase_str)

# GPT Teach Injection
def test_teach_module():
    try:
        teach = TeachModule()
        teach.inject_from_gpt("nmap -p- 10.10.10.10", phase="recon")
        report("TeachModule", "✅ OK")
    except Exception as e:
        report("TeachModule", "❌ FAIL", str(e))
        traceback.print_exc()

def test_sgpt_cli():
    try:
        test_prompt = 'Return JSON: {"description": "desc", "when": "context", "why": "reason", "parameters": [], "param_descriptions": []}'
        response = gpt_manager.gpt_request(test_prompt, agent_id="DebugCLI")
        if response.strip().startswith("{"):
            report("GPTManager CLI", "✅ OK", "GPT response format valid")
        else:
            report("GPTManager CLI", "❌ FAIL", "Output not valid JSON")
    except Exception as e:
        report("GPTManager CLI", "❌ FAIL", str(e))

# RedAgent + GPT Feedback
def test_rlagent_gpt_interaction():
    try:
        agent = RedAgent(agent_id="TestLLM", role="Audit")
        agent.simulate_train(episodes=1)
        agent.train_on_batch()
        result = agent.generate_hint()
        if result:
            report("RedAgent-GPT Interaction", "✅ OK", f"Hint: {result[:80]}")
        else:
            report("RedAgent-GPT Interaction", "⚠ No Hint Generated")
        return agent
    except Exception as e:
        report("RedAgent-GPT Interaction", "❌ FAIL", str(e))
        traceback.print_exc()
        return None

def dry_model_output_test(agent):
    try:
        dummy = agent.encode_env_state(agent.env.reset()).unsqueeze(0)
        _, entropy = agent.policy_net(dummy)
        value, _ = agent.value_net(dummy)
        if entropy.item() > 0 and abs(value.item()) < 100:
            report("Model Output Sanity", "✅ OK", f"Entropy={entropy.item():.3f}, Value={value.item():.3f}")
        else:
            report("Model Output Sanity", "⚠ Suspicious Output", f"E={entropy.item()}, V={value.item()}")
    except Exception as e:
        report("Model Output Sanity", "❌ FAIL", str(e))

# GPT Chain Building
def test_chainbuilder(agent):
    try:
        build_and_store_chain(agent.memory, dry=True)
        report("ChainBuilder", "✅ OK")
    except Exception as e:
        report("ChainBuilder", "❌ FAIL", str(e))
        traceback.print_exc()

# History Output Analysis
def test_output_interpreter(history):
    if not history:
        report("History Analysis", "⚠️ EMPTY")
        return
    failures = 0
    for i, entry in enumerate(history[:50]):
        try:
            cmd = entry.get("command", "")
            out = entry.get("output", "")
            parsed = analyze_output(cmd, out)
            if parsed.get("phase") == "unknown":
                failures += 1
        except Exception as e:
            failures += 1
            traceback.print_exc()
    if failures:
        report("Output Interpreter", "❌ PARTIAL", f"{failures} unknown/failed out of 50")
    else:
        report("Output Interpreter", "✅ OK")

# Summary Writer
def final_summary():
    console.rule("[bold green]📋 ARIASKA AUDIT REPORT")
    table = Table(title="Subsystem Health Summary", show_lines=True)
    table.add_column("Module")
    table.add_column("Status")
    table.add_column("Details")
    for title, status, note in results:
        table.add_row(title, status, note)
    console.print(table)
    try:
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        outfile = os.path.join(desktop, f"ariaska_audit_log_{timestamp}.txt")
        with open(outfile, "w") as f:
            f.write("\n".join(log_lines))
        console.print(f"[green]📄 Full audit report saved to: {outfile}[/green]")
    except Exception as e:
        console.print(f"[red]⚠ Failed to write report: {e}[/red]")

# Launch Full Stack Audit
if __name__ == "__main__":
    console.rule("[bold magenta]🔍 ARIASKA RL PROJECT AUDIT — MODEL FIXED MODE")

    memory = check_file("memory.json")
    history_data = check_file("history.json")
    chain_data = check_file("chain_generated.json")

    check_model_files()
    test_sgpt_cli()
    test_teach_module()

    if memory:
        scan_memory_consistency(memory)
        report_phase_distribution(memory)
    if history_data:
        test_output_interpreter(history_data.get("history", []))

    agent = test_rlagent_gpt_interaction()
    if agent:
        dry_model_output_test(agent)
        test_chainbuilder(agent)

    final_summary()