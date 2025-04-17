# core/output_interpreter.py — v5.0 Live Signal Parser + Training Test Suite

import re
import json
import os
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from typing import Dict

console = Console()


class OutputSignal:
    def __init__(self, command: str, output: str):
        self.command = command.lower()
        self.output = output.lower()
        self.original_output = output
        self.result = {
            "phase": "unknown",
            "success": False,
            "artifacts": [],
            "hints": [],
            "entities": {},
            "risk_score": 0.0,
            "stealth_score": 1.0,
            "output_excerpt": output[:500]
        }

    def extract_entities(self):
        out = self.output
        entities = defaultdict(list)

        entities["ips"] = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", out)
        entities["ports"] = [{"port": int(p), "protocol": proto, "service": svc}
                             for p, proto, svc in re.findall(r"(\d{1,5})/(tcp|udp)\s+open\s+([\w\-]+)", out)]

        entities["users"] = list(set(re.findall(r"(?:user(?:name)?s?:?\s*)([a-zA-Z0-9_\-\.]+)", out)))
        entities["passwords"] = list(set(re.findall(r"(?:pass(?:word)?s?:?\s*)([^\s,]+)", out)))
        entities["hashes"] = list(set(re.findall(r"\b[a-f0-9]{32,64}\b", out)))
        entities["urls"] = list(set(re.findall(r"https?://[^\s\"']+", out)))
        entities["domains"] = list(set(re.findall(r"\b[a-zA-Z0-9\-.]+\.(com|net|org|local|io|int|edu|gov|mil)\b", out)))
        entities["paths"] = list(set(re.findall(r"(?:/[\w\-.]+)+", out)))

        entities["cves"] = list(set(re.findall(r"CVE-\d{4}-\d{4,7}", out)))
        entities["shares"] = re.findall(r"(?:\\\\|//)[\w\-]+\\[\w$]+", out)
        entities["tokens"] = re.findall(r"\beyJ[a-zA-Z0-9\-_]{10,}\b", out)
        entities["shell_prompts"] = re.findall(r"(?m)^[a-z_][\w\-]*@[\w\-]+:\S*[$#]", self.original_output)

        self.result["entities"] = dict(entities)

    def detect_phase_and_artifacts(self):
        cmd = self.command
        out = self.output

        phase = "unknown"
        success = False
        artifacts = []
        hints = []

        def mark(_phase, _artifacts, _hint):
            nonlocal phase, success, artifacts, hints
            phase = _phase
            success = True
            artifacts.extend(_artifacts)
            hints.append(_hint)

        if "nmap" in cmd or "masscan" in cmd:
            phase = "recon"
            if "open" in out:
                mark("recon", ["ports_discovered"], "Open ports identified via scan.")

        elif any(tool in cmd for tool in ["enum4linux", "ldapsearch", "smbclient", "rpcclient", "dnsenum", "dig", "host"]):
            phase = "enumeration"
            if "user" in out or "group" in out or "domain" in out:
                mark("enumeration", ["user_enum"], "Domain users/groups or LDAP data found.")

        elif any(tool in cmd for tool in ["msfconsole", "hydra", "sqlmap", "crackmapexec", "exploit"]):
            phase = "exploit"
            if "login:" in out or "shell" in out or "session opened" in out:
                mark("exploit", ["shell_access"], "Shell access or session confirmed.")

        elif any(tool in cmd for tool in ["linpeas", "winpeas", "pspy", "sudo -l", "lse", "capabilities"]):
            phase = "privesc"
            if any(keyword in out for keyword in ["root", "sudo", "capabilities", "setuid"]):
                mark("privesc", ["privesc_vector"], "Privilege escalation path suspected.")

        elif any(tool in cmd for tool in ["scp", "wget", "curl", "nc", "ftp", "exfil"]):
            phase = "exfiltrate"
            if any(success_str in out for success_str in ["transferred", "saved", "200 ok"]):
                mark("exfiltrate", ["data_exfiltrated"], "Data exfiltration confirmed.")

        elif any(tool in cmd for tool in ["crontab", "schtasks", "reg add", "rc.local", "startup"]):
            phase = "persistence"
            if "created" in out or "added" in out or "success" in out:
                mark("persistence", ["persistence_mechanism"], "Persistence method established.")

        self.result["phase"] = phase
        self.result["success"] = success
        self.result["artifacts"] = artifacts
        self.result["hints"] = hints

    def risk_and_stealth_scoring(self):
        out = self.output
        risk = 0.0
        stealth = 1.0

        if "error" in out or "failed" in out or "timeout" in out:
            risk += 0.5
            stealth -= 0.2
        if "unauthorized" in out or "denied" in out:
            risk += 1.0
            stealth -= 0.3
        if "alert" in out or "detected" in out or "honeypot" in out:
            risk += 2.0
            stealth -= 0.4
        if self.result["success"]:
            stealth += 0.2
            risk += 0.2

        self.result["risk_score"] = round(min(max(risk, 0), 10), 2)
        self.result["stealth_score"] = round(min(max(stealth, 0), 1), 2)

    def analyze(self) -> Dict:
        self.extract_entities()
        self.detect_phase_and_artifacts()
        self.risk_and_stealth_scoring()
        return self.result


def analyze_output(command: str, output: str, context: dict = None) -> dict:
    parser = OutputSignal(command, output)
    return parser.analyze()

# === Live Test Suite ===
if __name__ == "__main__":
    def run_tests():
        sample_file = "data/history.json"
        if not os.path.exists(sample_file):
            console.print("[red]⚠ No history.json found in data/. Run a simulation first.[/red]")
            return

        with open(sample_file, "r") as f:
            data = json.load(f)

        history = data.get("history", [])
        if not history:
            console.print("[yellow]⚠ No history entries found to test.[/yellow]")
            return

        console.rule("[bold cyan]🧪 Output Interpreter Live Test Suite")
        for i, entry in enumerate(history[:20]):
            cmd = entry.get("command", "")
            out = entry.get("output", "")
            parsed = analyze_output(cmd, out)

            t = Table(title=f"[green]#{i+1} {cmd[:60]}", show_header=True)
            t.add_column("Field", style="cyan", width=16)
            t.add_column("Value", style="magenta")

            t.add_row("Phase", parsed["phase"])
            t.add_row("Success", str(parsed["success"]))
            t.add_row("Artifacts", ", ".join(parsed["artifacts"]))
            t.add_row("Risk Score", str(parsed["risk_score"]))
            t.add_row("Stealth", str(parsed["stealth_score"]))
            t.add_row("Users", ", ".join(parsed["entities"].get("users", [])))
            t.add_row("Passwords", ", ".join(parsed["entities"].get("passwords", [])))
            t.add_row("Hashes", ", ".join(parsed["entities"].get("hashes", [])))
            console.print(t)

    run_tests()
