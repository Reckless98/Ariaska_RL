# output_interpreter.py — ARIASKA Signal Analyst v7.2 (GPT-Enhanced, Entity-Aware, Stealth-Rated)

import re
import json
import os
import subprocess
from collections import defaultdict
from typing import Dict, List
from rich.console import Console
from rich.table import Table

console = Console()

ENTITY_PATTERNS = {
    "ips": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "emails": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "ports": r"(\d{1,5})/(tcp|udp)\s+open\s+([\w\-]+)",
    "urls": r"https?://[^\s\"']+",
    "domains": r"\b[a-zA-Z0-9\-.]+\.(com|net|org|local|io|int|edu|gov|mil)\b",
    "users": r"(?:user(?:name)?s?:?\s*)([a-zA-Z0-9_\-\.]+)",
    "passwords": r"(?:pass(?:word)?s?:?\s*)([^\s,]+)",
    "hashes": r"\b[a-f0-9]{32,64}\b",
    "paths": r"(?:/[\w\-.]+)+",
    "tokens": r"\beyJ[a-zA-Z0-9\-_]{10,}\b",
    "cves": r"CVE-\d{4}-\d{4,7}",
    "shares": r"(?:\\\\|//)[\w\-]+\\[\w$]+",
    "prompts": r"(?m)^[a-z_][\w\-]*@[\w\-]+:\S*[$#]",
}

SIGNATURE_HOOKS = {
    "honeypot_trigger": ["fake_", "honeypot", "deception"],
    "blue_team_alert": ["traceback", "alert", "detected", "monitor"],
    "opsec_failure": ["unauthorized", "failed", "blocked", "suspicious"],
}


class OutputInterpreter:
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
            "signatures": [],
            "risk_score": 0.0,
            "stealth_score": 1.0,
            "confidence": 0.0,
            "output_excerpt": output[:700],
        }

    def extract_entities(self):
        out = self.output
        entities = defaultdict(list)

        for name, pattern in ENTITY_PATTERNS.items():
            if name == "ports":
                entities[name] = [
                    {"port": int(p), "protocol": proto, "service": svc}
                    for p, proto, svc in re.findall(pattern, out)
                ]
            else:
                entities[name] = list(set(re.findall(pattern, out)))

        self.result["entities"] = dict(entities)

    def detect_hooks(self):
        """
        Detect specific signal patterns useful for advanced stealth/risk modeling.
        """
        out = self.output
        triggered = []
        for name, patterns in SIGNATURE_HOOKS.items():
            if any(pat in out for pat in patterns):
                triggered.append(name)
        self.result["signatures"] = triggered

    def detect_phase_and_artifacts(self):
        cmd = self.command
        out = self.output
        phase, success, artifacts, hints = "unknown", False, [], []

        def mark(_phase, _artifacts, _hint):
            nonlocal phase, success, artifacts, hints
            phase = _phase
            success = True
            artifacts.extend(_artifacts)
            hints.append(_hint)

        rules = [
            ("recon", ["nmap", "masscan"], ["open"], ["ports_discovered"]),
            (
                "enumeration",
                ["enum4linux", "ldapsearch", "smbclient", "dig", "dnsenum"],
                ["user", "domain"],
                ["user_enum"],
            ),
            (
                "exploit",
                ["hydra", "sqlmap", "exploit", "msfconsole"],
                ["session", "login", "shell"],
                ["shell_access"],
            ),
            (
                "privesc",
                ["linpeas", "winpeas", "sudo -l", "pspy"],
                ["root", "capabilities", "setuid"],
                ["privesc_vector"],
            ),
            (
                "exfiltrate",
                ["scp", "wget", "curl", "ftp"],
                ["transferred", "saved", "200 ok"],
                ["data_exfiltrated"],
            ),
            (
                "persistence",
                ["crontab", "reg add", "schtasks", "rc.local"],
                ["created", "added"],
                ["persistence_mechanism"],
            ),
        ]

        for _phase, tools, signals, _artifacts in rules:
            if any(tool in cmd for tool in tools):
                if any(sig in out for sig in signals):
                    mark(_phase, _artifacts, f"{_phase} triggered by {tools[0]}")

        self.result["phase"] = phase
        self.result["success"] = success
        self.result["artifacts"] = artifacts
        self.result["hints"] = hints
        self.result["confidence"] = round(
            min((len(artifacts) + int(success) + len(hints)) / 5.0, 1.0), 2
        )

    def risk_and_stealth_scoring(self):
        out = self.output
        risk, stealth = 0.0, 1.0

        # Heuristic scoring based on language patterns
        risk_keywords = [
            "error",
            "failed",
            "timeout",
            "unauthorized",
            "denied",
            "alert",
            "detected",
            "honeypot",
            "traceback",
            "blocked",
        ]
        stealth_signals = [
            "200 ok",
            "silent",
            "no detection",
            "success",
            "authenticated",
            "undetected",
            "stealth",
        ]

        for word in risk_keywords:
            if word in out:
                risk += 0.6
                stealth -= 0.15

        for sig in stealth_signals:
            if sig in out:
                stealth += 0.12

        if self.result["success"]:
            stealth += 0.15
            risk += 0.2

        if "fake_" in out or "honeypot" in out:
            risk += 1.5
            stealth -= 0.4

        if "alert" in out and "traceback" in out:
            risk += 2.0
            stealth -= 0.5
            self.result["hints"].append("Blue Team trace alert likely triggered.")

        # Clamp values
        self.result["risk_score"] = round(min(max(risk, 0.0), 10.0), 2)
        self.result["stealth_score"] = round(min(max(stealth, 0.0), 1.0), 2)

    def gpt_phase_classification(self):
        prompt = f"""You are a cyber red team signal analyst. Analyze the following command and output:
Command: {self.command}
Output: {self.original_output[:1500]}

Return valid JSON with the following keys:
- phase (string)
- success (true/false)
- confidence (0.0–1.0)
- artifacts (list of strings)
- hint (concise explanation)"""

        try:
            result = subprocess.run(
                [
                    "sgpt",
                    "--model",
                    "gpt-4.1-nano",
                    "--temperature",
                    "0.45",
                    "--role",
                    "aria",
                    prompt,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=25,
            )
            raw = result.stdout.strip()
            gpt_data = json.loads(raw)

            self.result["phase"] = gpt_data.get("phase", self.result["phase"])
            self.result["success"] = gpt_data.get("success", self.result["success"])
            self.result["confidence"] = max(
                self.result["confidence"], gpt_data.get("confidence", 0.0)
            )
            self.result["artifacts"] = list(
                set(self.result["artifacts"] + gpt_data.get("artifacts", []))
            )
            if "hint" in gpt_data:
                self.result["hints"].append(gpt_data["hint"])
        except Exception as e:
            console.print(f"[yellow]⚠ GPT subprocess failed: {e}[/yellow]")
            self.openai_fallback(prompt)

    def openai_fallback(self, prompt: str):
        try:
            import openai

            openai.api_key = os.getenv("OPENAI_API_KEY", "")
            if not openai.api_key:
                raise ValueError("OPENAI_API_KEY is not set.")
            response = openai.ChatCompletion.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a red team analyst. Return clean JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.45,
                max_tokens=200,
            )
            raw = response.choices[0].message.content.strip()
            gpt_data = json.loads(raw)

            self.result["phase"] = gpt_data.get("phase", self.result["phase"])
            self.result["success"] = gpt_data.get("success", self.result["success"])
            self.result["confidence"] = max(
                self.result["confidence"], gpt_data.get("confidence", 0.0)
            )
            self.result["artifacts"] = list(
                set(self.result["artifacts"] + gpt_data.get("artifacts", []))
            )
            if "hint" in gpt_data:
                self.result["hints"].append(gpt_data["hint"])
        except Exception as e:
            console.print(f"[red]⚠ OpenAI fallback failed: {e}[/red]")

    def analyze(self, gpt_enhanced=True) -> Dict:
        self.extract_entities()
        self.detect_phase_and_artifacts()
        self.risk_and_stealth_scoring()
        if gpt_enhanced:
            self.gpt_phase_classification()

        # Final pass: normalize, tag, and optionally expand hints
        self.result["output_excerpt"] = self.original_output[:600].strip()
        self.result["tags"] = list(
            {
                tag
                for tag in self.result["artifacts"] + self.result["hints"]
                if isinstance(tag, str) and len(tag) < 100
            }
        )

        return self.result


def analyze_output(command: str, output: str, context: dict = None) -> dict:
    interpreter = OutputInterpreter(command, output)
    return interpreter.analyze(gpt_enhanced=True)


# 🧠 Optional Diagnostic Debug Block (for manual CLI inspection)
if __name__ == "__main__":
    test_cmd = "nmap -sC -sV -p- 10.10.10.10"
    test_out = """
Starting Nmap 7.94SVN ( https://nmap.org ) at 2025-04-20 23:11 CEST
Note: Host seems down. If it is really up, but blocking our ping probes, try -Pn
Nmap done: 1 IP address (0 host up) scanned in 3.31 seconds
"""
    result = analyze_output(test_cmd, test_out)
    console = Console()
    table = Table(title="Output Analysis Summary")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="magenta")

    for k, v in result.items():
        summary = str(v)
        if isinstance(v, list):
            summary = ", ".join(map(str, v[:3])) + ("..." if len(v) > 3 else "")
        elif isinstance(v, dict):
            summary = json.dumps(v)[:100] + "..."
        elif isinstance(v, str):
            summary = v[:80] + ("..." if len(v) > 80 else "")
        table.add_row(k, summary)

    console.print(table)
