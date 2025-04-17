# /core/rule_engine.py — ARIASKA TACTICAL RULE CORE v7.0

import re
import ast
import hashlib
from collections import defaultdict
from rich.console import Console
from core.output_interpreter import analyze_output

console = Console()

def normalize_context(ctx):
    if isinstance(ctx, str):
        try:
            return ast.literal_eval(ctx)
        except:
            return {}
    return ctx if isinstance(ctx, dict) else {}

def extract_template(command):
    ip_pattern = re.compile(r"(\b(?:\d{1,3}\.){3}\d{1,3}\b)")
    port_pattern = re.compile(r"\b\d{2,5}\b")
    hash_pattern = re.compile(r"\b[a-f0-9]{32,64}\b")
    tokens = command.strip().split()
    rebuilt = []
    for tok in tokens:
        if ip_pattern.search(tok):
            rebuilt.append(re.sub(ip_pattern, "{IP}", tok))
        elif port_pattern.fullmatch(tok):
            rebuilt.append("{PORT}")
        elif hash_pattern.match(tok):
            rebuilt.append("{HASH}")
        else:
            rebuilt.append(tok)
    return " ".join(rebuilt)

def substitute_placeholders(template, state=None):
    if not state:
        return template
    ip = state.get("target_ip", "10.10.10.10")
    port = str(state.get("open_ports", [80])[0])
    return template.replace("{IP}", ip).replace("{PORT}", port)

def detect_phase(command):
    command = command.lower()
    phase_tools = {
        "recon": ["nmap", "masscan", "amass", "gobuster", "ffuf", "recon-ng", "spiderfoot", "whois", "tshark", "dig"],
        "enumeration": ["enum4linux", "ldapsearch", "rpcclient", "smbclient", "curl", "nslookup", "openssl", "ftp"],
        "exploit": ["hydra", "sqlmap", "msfconsole", "exploit-db", "kerbrute", "nishang", "echo", "bash", "wget"],
        "privesc": ["linpeas", "winpeas", "pspy", "lse", "sudo", "capabilities", "getcap", "id"],
        "exfiltrate": ["scp", "wget", "curl", "nc", "tar", "zip", "ftp", "cat", "sshpass"],
        "persistence": ["reg add", "schtasks", "crontab", "rc.local", "backdoor", "echo"],
        "cleanup": ["rm", "truncate", "logrotate"]
    }
    for phase, tools in phase_tools.items():
        if any(tool in command for tool in tools):
            return phase
    return "unknown"
def score_entry(entry):
    """
    Evaluate tactical value of a history entry using reward, context, and output analysis.
    """
    command = entry.get("command", "")
    reward = entry.get("reward", 0)
    context = normalize_context(entry.get("context", {}))
    output = entry.get("output", "")
    output_info = analyze_output(command, output)

    blue_risk = context.get("blue_team_alert", 1.0) + 1.0
    stealth = 1.5 if context.get("stealth") else 1.0
    success_factor = 2.2 if output_info.get("success") else 0.6
    artifact_bonus = (len(output_info.get("artifacts", [])) + 1) * 1.1
    context_phase = context.get("phase", detect_phase(command))
    output_phase = output_info.get("phase", "unknown")
    phase_match = 1.2 if output_phase == context_phase else 0.95
    alert_penalty = 1.0 if context.get("blue_team_alert", 0.0) < 4.0 else 0.8

    return (reward * stealth * success_factor * artifact_bonus * phase_match * alert_penalty) / blue_risk


def rule_based_selection(history, current_state=None, min_reward_threshold=10):
    if not history or not isinstance(history, list):
        console.print("[yellow]⚠ Empty or invalid history passed to rule engine.[/yellow]")
        return fallback_recommendation(current_state)

    phase_groups = defaultdict(list)
    best_entry = None
    best_score = -9999

    for entry in history:
        reward = entry.get("reward", 0)
        if reward < min_reward_threshold:
            continue

        command = entry.get("command", "")
        context = normalize_context(entry.get("context", {}))
        output = entry.get("output", "")
        phase = context.get("phase", detect_phase(command))

        score = score_entry(entry)
        template = extract_template(command)
        phase_groups[phase].append((template, score, context, output))

    for phase, entries in phase_groups.items():
        for template, score, ctx, out in entries:
            if score > best_score:
                best_score = score
                best_entry = {
                    "template": template,
                    "context": ctx,
                    "phase": phase,
                    "output": out
                }

    if best_entry:
        cmd = substitute_placeholders(best_entry["template"], current_state)
        console.log(f"[green]🎯 Tactical Command Selected:[/green] {cmd}")
        return {
            "command": cmd,
            "phase": best_entry["phase"],
            "priority": round(best_score, 2),
            "artifacts": analyze_output(cmd, best_entry["output"]).get("artifacts", [])
        }

    return fallback_recommendation(current_state)


def fallback_recommendation(state=None):
    fallback = {
        "command": substitute_placeholders("nmap -sC -sV -p- {IP}", state),
        "phase": "recon",
        "priority": 0.8
    }
    console.print(f"[cyan]🔄 Fallback issued:[/cyan] {fallback['command']}")
    return fallback


# Debug only
if __name__ == "__main__":
    mock_history = [
        {
            "command": "nmap -p- 10.10.10.10",
            "reward": 25,
            "context": {"phase": "recon", "blue_team_alert": 2.0, "stealth": True},
            "output": "PORT   STATE SERVICE\n22/tcp open ssh\n80/tcp open http"
        },
        {
            "command": "hydra -L users.txt -P pass.txt ssh://10.10.10.10",
            "reward": 55,
            "context": {"phase": "exploit", "stealth": False},
            "output": "[22][ssh] host: 10.10.10.10 login: admin password: 123456"
        }
    ]
    current_state = {"target_ip": "10.10.10.99", "open_ports": [22, 80]}
    result = rule_based_selection(mock_history, current_state)
    console.print("[bold green]Recommendation Output:[/bold green]", result)
