#!/usr/bin/env python3
"""V2 Dataset Extraction — High-quality training data from Ariaska traces.

Improvements over v1:
  - Richer context from agent_records (reward breakdown, decision_source, discoveries)
  - True retrieval-aware reasoning samples from paired trace segments
  - Better synthetic augmentation for underrepresented cases
  - Quality scoring per sample
  - Deduplication by content hash
  - Proper train/val/holdout split (80/10/10)
  - JSON schema stress tests from real messy outputs
  
Output: ariaska_ai/dataset/v2/ directory with per-task and split JSONL files.
"""

import json
import hashlib
import os
import random
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

TRACE_DIR = Path(__file__).resolve().parents[2] / "traces"
POSTMORTEM_DIR = Path(__file__).resolve().parents[2] / "postmortems"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "dataset" / "v2"
SEED = 42
MAX_PER_TASK = 12000
MIN_QUALITY = 0.4

PHASES = [
    "RECON", "ENUMERATION", "EXPLOITATION", "PRIVILEGE_ESCALATION",
    "LATERAL_MOVEMENT", "POST_EXPLOITATION", "EXFILTRATION", "CLOSEOUT"
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

TOOL_CATEGORIES = {
    "nmap": "network_scanning", "masscan": "network_scanning",
    "nikto": "web_scanning", "gobuster": "web_discovery", "dirb": "web_discovery",
    "ffuf": "web_fuzzing", "wfuzz": "web_fuzzing",
    "hydra": "brute_force", "medusa": "brute_force",
    "sqlmap": "sql_injection", "curl": "web_interaction", "wget": "web_interaction",
    "ssh": "remote_access", "ftp": "file_transfer", "smbclient": "smb_access",
    "enum4linux": "smb_enumeration", "crackmapexec": "network_attack",
    "msfconsole": "exploitation_framework", "searchsploit": "exploit_search",
    "tshark": "packet_analysis", "tcpdump": "packet_analysis",
    "linpeas": "privilege_escalation", "winpeas": "privilege_escalation",
    "getcap": "capability_check", "find": "file_search",
    "cat": "file_read", "id": "user_check", "whoami": "user_check",
    "sudo": "privilege_check", "python3": "scripting",
}

# ── System Prompts (concise, structured, Ariaska-specific) ─────────────────

SYSTEM_PROMPTS = {
    "phase_classification": (
        "You are Ariaska, a cybersecurity AI coprocessor for authorized penetration testing. "
        "Classify the current attack phase from the engagement state. "
        "Respond in JSON: {\"phase\": \"<PHASE>\", \"confidence\": <0.0-1.0>, \"reasoning\": \"<brief>\"}"
    ),
    "next_step": (
        "You are Ariaska, a cybersecurity AI coprocessor. Given the engagement state, "
        "suggest the best next action. Respond in JSON:\n"
        "{\"action\": \"<exact command>\", \"reasoning\": \"<tactical justification>\", "
        "\"phase_fit\": <0.0-1.0>, \"alternatives\": [\"<alt1>\", \"<alt2>\"]}"
    ),
    "tool_output_parse": (
        "You are Ariaska, a cybersecurity AI coprocessor. Parse the tool output into structured findings. "
        "Respond in JSON:\n"
        "{\"discoveries\": [{\"type\": \"<port|service|version|credential|vuln|shell|file|user>\", "
        "\"value\": \"<finding>\", \"confidence\": <0.0-1.0>}], "
        "\"phase_impact\": \"<stay|advance>\", \"summary\": \"<brief>\"}"
    ),
    "state_summary": (
        "You are Ariaska, a cybersecurity AI coprocessor. Summarize the engagement state concisely. "
        "Respond in JSON:\n"
        "{\"phase\": \"<current>\", \"discoveries\": {\"ports\": [], \"services\": [], "
        "\"credentials\": [], \"vulns\": [], \"shells\": []}, "
        "\"progress\": \"<good|moderate|stalled>\", \"blockers\": [], \"next_priority\": \"<action>\"}"
    ),
    "retry_or_pivot": (
        "You are Ariaska, a cybersecurity AI coprocessor. A command failed or was unproductive. "
        "Decide the next move. Respond in JSON:\n"
        "{\"decision\": \"RETRY|PIVOT|ESCALATE\", \"action\": \"<next command>\", "
        "\"reasoning\": \"<why>\", \"confidence\": <0.0-1.0>}"
    ),
    "postmortem": (
        "You are Ariaska, a cybersecurity AI coprocessor. Analyze the completed engagement. "
        "Respond in JSON:\n"
        "{\"root_cause\": \"<what blocked progress>\", \"missed_signals\": [\"<signal>\"], "
        "\"corrected_path\": [\"<step>\"], \"key_lesson\": \"<brief>\"}"
    ),
    "command_validate": (
        "You are Ariaska, a cybersecurity AI coprocessor. Validate this command for the current state. "
        "Respond in JSON:\n"
        "{\"valid\": true|false, \"reasoning\": \"<why>\", \"alternative\": \"<better command if invalid>\"}"
    ),
    "evidence_check": (
        "You are Ariaska, a cybersecurity AI coprocessor. Given the current evidence and target state, "
        "determine if the evidence is sufficient to proceed. Respond in JSON:\n"
        "{\"sufficient\": true|false, \"missing\": [\"<what's needed>\"], "
        "\"confidence\": <0.0-1.0>, \"recommendation\": \"<brief>\"}"
    ),
    "retrieval_reasoning": (
        "You are Ariaska, a cybersecurity AI coprocessor. Using the current state plus retrieved "
        "prior experience, synthesize tactical guidance. Respond in JSON:\n"
        "{\"synthesis\": \"<integrated reasoning>\", \"from_current\": \"<what current state shows>\", "
        "\"from_memory\": \"<what prior experience suggests>\", \"action\": \"<recommended command>\", "
        "\"confidence\": <0.0-1.0>}"
    ),
}


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
        text = self.messages[-1]["content"] if self.messages else ""
        return hashlib.md5(text.encode()).hexdigest()


# ── Trace Loading ──────────────────────────────────────────────────────────

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


def extract_agent_record(step: dict) -> dict | None:
    """Get the primary agent record from a step."""
    records = step.get("agent_records", [])
    if not records:
        return None
    # Pick the record with highest reward, or first
    best = max(records, key=lambda r: r.get("reward", 0))
    return best


def build_evidence_snapshot(steps: list[dict], up_to: int) -> dict:
    """Build cumulative evidence from steps[0:up_to]."""
    evidence = {"ports": set(), "services": set(), "credentials": set(),
                "vulns": set(), "shells": set(), "users": set(),
                "files": set(), "versions": set()}
    for s in steps[:up_to]:
        for rec in s.get("agent_records", []):
            for d in rec.get("discoveries", []):
                if isinstance(d, str):
                    dtype = d.split(":")[0].lower() if ":" in d else "other"
                    val = d.split(":", 1)[1] if ":" in d else d
                    if "port" in dtype:
                        evidence["ports"].add(val)
                    elif "service" in dtype or "version" in dtype:
                        evidence["services"].add(val)
                    elif "cred" in dtype or "password" in dtype:
                        evidence["credentials"].add(val)
                    elif "vuln" in dtype:
                        evidence["vulns"].add(val)
                    elif "shell" in dtype:
                        evidence["shells"].add(val)
                    elif "user" in dtype:
                        evidence["users"].add(val)
                    elif "file" in dtype or "sensitive" in dtype:
                        evidence["files"].add(val)
                    else:
                        evidence["versions"].add(val)
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
    
    # Evidence snapshot
    ev = build_evidence_snapshot(steps, idx)
    for k, v in ev.items():
        if v:
            parts.append(f"{k.capitalize()}: {', '.join(str(x) for x in v[:5])}")
    
    # Recent commands (last 5)
    recent = []
    for s in steps[max(0, idx-5):idx]:
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
    score = 0.3  # base
    reward = record.get("reward", 0)
    if reward > 20:
        score += 0.3
    elif reward > 5:
        score += 0.2
    elif reward > 0:
        score += 0.1
    
    # Has discoveries
    discoveries = record.get("discoveries", [])
    if discoveries:
        score += 0.15
    
    # Has reward breakdown
    if record.get("reward_breakdown"):
        score += 0.05
    
    # Has real command (not empty/noop)
    cmd = record.get("command", "")
    if cmd and len(cmd) > 5:
        score += 0.1
    
    # Has stdout snippet (real execution data)
    if record.get("stdout_snippet"):
        score += 0.1

    return min(1.0, score)


# ── Sample Generators ──────────────────────────────────────────────────────

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
            
            # Build evidence-based reasoning
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
                "reasoning": reasoning
            })
            
            samples.append(Sample(
                task="phase_classification",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPTS["phase_classification"]},
                    {"role": "user", "content": f"Classify the attack phase:\n\n{ctx}"},
                    {"role": "assistant", "content": response},
                ],
                metadata={"phase": phase, "step": i, "source": fname},
                quality=q,
            ))
    return samples


def gen_next_step(traces: list[tuple[str, list[dict]]]) -> list[Sample]:
    """Next-step suggestions from successful steps."""
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
            
            # Build reasoning from decision source and context
            source = rec.get("decision_source", "")
            reasoning = f"Using {source}: " if source else ""
            
            breakdown = rec.get("reward_breakdown", {})
            if breakdown:
                top_components = sorted(
                    [(k, v) for k, v in breakdown.items() if k != "total" and v > 0],
                    key=lambda x: x[1], reverse=True
                )[:3]
                reasoning += "scored well on " + ", ".join(f"{k} ({v:.1f})" for k, v in top_components)
            else:
                reasoning += f"targets {PHASE_OBJECTIVES.get(phase, 'current objectives')}"
            
            phase_fit = min(1.0, reward / 25.0)
            
            # Build alternatives from nearby successful steps
            alts = []
            for j in range(max(0, i-3), min(len(steps), i+3)):
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
                "alternatives": alts
            })
            
            samples.append(Sample(
                task="next_step",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPTS["next_step"]},
                    {"role": "user", "content": f"Current engagement state:\n{ctx}\n\nSuggest the next action."},
                    {"role": "assistant", "content": response},
                ],
                metadata={"phase": phase, "reward": reward, "command": cmd[:100], "source": fname},
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
            
            # Parse discoveries into structured format
            parsed = []
            for d in discoveries:
                if isinstance(d, str):
                    dtype = d.split(":")[0] if ":" in d else "other"
                    val = d.split(":", 1)[1] if ":" in d else d
                    dtype_mapped = dtype.replace("_info", "").replace("_found", "")
                    if "port" in dtype_mapped:
                        dtype_mapped = "port"
                    elif "service" in dtype_mapped or "version" in dtype_mapped:
                        dtype_mapped = "service"
                    elif "cred" in dtype_mapped or "password" in dtype_mapped:
                        dtype_mapped = "credential"
                    elif "vuln" in dtype_mapped:
                        dtype_mapped = "vuln"
                    elif "shell" in dtype_mapped:
                        dtype_mapped = "shell"
                    elif "file" in dtype_mapped or "sensitive" in dtype_mapped:
                        dtype_mapped = "file"
                    elif "user" in dtype_mapped:
                        dtype_mapped = "user"
                    parsed.append({
                        "type": dtype_mapped,
                        "value": val.strip(),
                        "confidence": round(rec.get("confidence", 0.8), 2)
                    })
            
            reward = rec.get("reward", 0)
            phase_impact = "advance" if reward > 15 else "stay"
            
            response = json.dumps({
                "discoveries": parsed[:8],
                "phase_impact": phase_impact,
                "summary": f"Found {len(parsed)} item(s) from {cmd.split()[0] if cmd else 'tool'} output"
            })
            
            q = quality_score(step, rec)
            # Boost for having actual stdout
            q = min(1.0, q + 0.15)
            
            samples.append(Sample(
                task="tool_output_parse",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPTS["tool_output_parse"]},
                    {"role": "user", "content": f"Parse this tool output:\n\nCommand: {cmd[:150]}\n\nOutput:\n{stdout[:800]}"},
                    {"role": "assistant", "content": response},
                ],
                metadata={"phase": phase, "tool": cmd.split()[0] if cmd else "", "n_discoveries": len(parsed), "source": fname},
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
            
            # Determine progress
            if total_reward > 50:
                progress = "good"
            elif total_reward > 10:
                progress = "moderate"
            else:
                progress = "stalled"
            
            # Determine blockers
            blockers = []
            if phase in ("EXPLOITATION", "PRIVILEGE_ESCALATION") and not ev["credentials"] and not ev["vulns"]:
                blockers.append("No credentials or vulnerabilities discovered yet")
            if phase == "RECON" and i > 10 and not ev["ports"]:
                blockers.append("Port scanning not yielding results")
            
            # Determine next priority
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
            
            response = json.dumps({
                "phase": phase,
                "discoveries": {k: v[:5] for k, v in ev.items()},
                "progress": progress,
                "blockers": blockers,
                "next_priority": next_priority
            })
            
            samples.append(Sample(
                task="state_summary",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPTS["state_summary"]},
                    {"role": "user", "content": f"Summarize the engagement state:\n\n{ctx}"},
                    {"role": "assistant", "content": response},
                ],
                metadata={"phase": phase, "step": i, "reward_so_far": total_reward, "source": fname},
                quality=0.6 + (0.2 if ev["credentials"] or ev["shells"] else 0) + (0.1 if blockers else 0),
            ))
    return samples


def gen_retry_pivot(traces: list[tuple[str, list[dict]]]) -> list[Sample]:
    """Retry/pivot decisions from failure→success transitions."""
    samples = []
    for fname, steps in traces:
        for i in range(1, len(steps)):
            prev_rec = extract_agent_record(steps[i-1])
            curr_rec = extract_agent_record(steps[i])
            if not prev_rec or not curr_rec:
                continue
            
            prev_reward = prev_rec.get("reward", 0)
            curr_reward = curr_rec.get("reward", 0)
            prev_cmd = prev_rec.get("command", "")
            curr_cmd = curr_rec.get("command", "")
            
            if not prev_cmd or not curr_cmd:
                continue
            
            # Low→high transition (failure then success)
            if prev_reward <= 1.0 and curr_reward >= 3.0:
                prev_tool = prev_cmd.split()[0] if prev_cmd else ""
                curr_tool = curr_cmd.split()[0] if curr_cmd else ""
                
                if prev_tool == curr_tool:
                    decision = "RETRY"
                    reasoning = f"Same tool ({curr_tool}) with adjusted parameters"
                elif TOOL_CATEGORIES.get(prev_tool) == TOOL_CATEGORIES.get(curr_tool):
                    decision = "PIVOT"
                    reasoning = f"Switched from {prev_tool} to {curr_tool} (same category)"
                else:
                    decision = "ESCALATE"
                    reasoning = f"Changed approach from {TOOL_CATEGORIES.get(prev_tool, prev_tool)} to {TOOL_CATEGORIES.get(curr_tool, curr_tool)}"
                
                ctx = build_context_str(steps[i-1], steps, i-1)
                confidence = min(1.0, curr_reward / 20.0)
                
                response = json.dumps({
                    "decision": decision,
                    "action": curr_cmd[:200],
                    "reasoning": reasoning,
                    "confidence": round(confidence, 2)
                })
                
                samples.append(Sample(
                    task="retry_or_pivot",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPTS["retry_or_pivot"]},
                        {"role": "user", "content": (
                            f"This command was unproductive:\n"
                            f"Command: {prev_cmd[:150]}\n"
                            f"Reward: {prev_reward:.1f}\n\n"
                            f"State:\n{ctx}\n\n"
                            f"What should we do next?"
                        )},
                        {"role": "assistant", "content": response},
                    ],
                    metadata={"decision": decision, "prev_reward": prev_reward, "curr_reward": curr_reward, "source": fname},
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
        
        # Build user prompt
        user_prompt = (
            f"Analyze this penetration testing engagement:\n\n"
            f"Run: {pm.get('run_id', 'unknown')}\n"
            f"Summary: {summary}\n"
        )
        if wins:
            user_prompt += f"Successes: {'; '.join(wins[:5])}\n"
        if fails:
            user_prompt += f"Failures: {'; '.join(fails[:5])}\n"
        
        # Build root cause from failures
        root_cause = fails[0] if fails else "Unable to advance past initial phase"
        
        # Missed signals from skill cards
        missed = []
        for s in skills[:5]:
            if s.get("if_condition"):
                missed.append(s["if_condition"])
        
        # Corrected path from experiments
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
            "key_lesson": key_lesson[:200]
        })
        
        q = 0.5
        if wins and fails:
            q = 0.8  # high quality — has both success and failure data
        elif wins or fails:
            q = 0.6
        if skills:
            q += 0.1
        
        samples.append(Sample(
            task="postmortem",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS["postmortem"]},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": response},
            ],
            metadata={"run_id": pm.get("run_id", ""), "has_skills": bool(skills), "source": "postmortem"},
            quality=min(1.0, q),
        ))
    return samples


def gen_command_validate(traces: list[tuple[str, list[dict]]]) -> list[Sample]:
    """Command validation — valid and invalid examples."""
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
            ctx = build_context_str(step, steps, i)
            
            valid = reward > 0
            tool = cmd.split()[0] if cmd else ""
            tool_cat = TOOL_CATEGORIES.get(tool, "other")
            
            reasoning = (
                f"{tool} ({tool_cat}) {'aligns with' if valid else 'does not fit'} "
                f"{phase} phase ({PHASE_OBJECTIVES.get(phase, 'objectives')[:60]})"
            )
            
            result = {"valid": valid, "reasoning": reasoning}
            if not valid:
                # Find a nearby valid command as alternative
                for j in range(max(0, i-3), min(len(steps), i+3)):
                    if j == i:
                        continue
                    alt_rec = extract_agent_record(steps[j])
                    if alt_rec and alt_rec.get("reward", 0) > 0:
                        result["alternative"] = alt_rec.get("command", "")[:150]
                        break
            else:
                result["alternative"] = ""
            
            response = json.dumps(result)
            
            samples.append(Sample(
                task="command_validate",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPTS["command_validate"]},
                    {"role": "user", "content": f"Validate this command:\nCommand: {cmd[:150]}\nPhase: {phase}\n\n{ctx}"},
                    {"role": "assistant", "content": response},
                ],
                metadata={"valid": valid, "phase": phase, "reward": reward, "source": fname},
                quality=quality_score(step, rec),
            ))
    return samples


def gen_evidence_check(traces: list[tuple[str, list[dict]]]) -> list[Sample]:
    """Evidence sufficiency checks at phase transition points."""
    samples = []
    for fname, steps in traces:
        prev_phase = None
        for i, step in enumerate(steps):
            phase = step.get("phase_after", step.get("phase_before", "RECON"))
            if prev_phase and phase != prev_phase:
                # Phase transition — check if evidence was sufficient
                ev = build_evidence_snapshot(steps, i)
                ctx = build_context_str(step, steps, i)
                
                # Determine sufficiency based on phase requirements
                missing = []
                sufficient = True
                
                if phase == "EXPLOITATION":
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
                elif phase == "POST_EXPLOITATION":
                    if not ev["shells"]:
                        missing.append("shell access")
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
                    "recommendation": recommendation
                })
                
                samples.append(Sample(
                    task="evidence_check",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPTS["evidence_check"]},
                        {"role": "user", "content": f"Is the evidence sufficient to proceed?\n\nTransition: {prev_phase} → {phase}\n\n{ctx}"},
                        {"role": "assistant", "content": response},
                    ],
                    metadata={"from_phase": prev_phase, "to_phase": phase, "sufficient": sufficient, "source": fname},
                    quality=0.7 + (0.2 if missing else 0),
                ))
            prev_phase = phase
    return samples


# ── Synthetic Augmentation ─────────────────────────────────────────────────

def gen_synthetic_schema_stress() -> list[Sample]:
    """Synthetic: noisy/ambiguous inputs with perfect JSON output."""
    templates = [
        # Messy nmap output
        {
            "input": "Parse this:\n\nnmap scan results for 10.10.10.5\n22/tcp open ssh OpenSSH 7.9p1\n80/tcp open http Apache/2.4.38\n3306/tcp closed mysql\nSome garbage here\n\nNot shown: 997 filtered ports",
            "output": {"discoveries": [
                {"type": "port", "value": "22/tcp ssh", "confidence": 0.95},
                {"type": "service", "value": "OpenSSH 7.9p1", "confidence": 0.95},
                {"type": "port", "value": "80/tcp http", "confidence": 0.95},
                {"type": "service", "value": "Apache/2.4.38", "confidence": 0.95},
            ], "phase_impact": "stay", "summary": "Found 2 open services: SSH and HTTP"},
        },
        # Gobuster output with errors
        {
            "input": "Parse this:\n\ngobuster dir -u http://10.10.10.5 -w /usr/share/wordlists/dirb/common.txt\nError: the server returns 403 for Root\n/admin (Status: 200) [Size: 1234]\n/login (Status: 302) [Size: 0]\n/backup (Status: 200) [Size: 52428]\n/config.php (Status: 200) [Size: 0]\nProgress: 4614/4614 100%",
            "output": {"discoveries": [
                {"type": "file", "value": "/admin (200)", "confidence": 0.9},
                {"type": "file", "value": "/login (302)", "confidence": 0.85},
                {"type": "file", "value": "/backup (200, 52KB)", "confidence": 0.95},
                {"type": "file", "value": "/config.php (200)", "confidence": 0.9},
            ], "phase_impact": "stay", "summary": "Found 4 web paths including admin panel and backup file"},
        },
        # Hydra brute force result
        {
            "input": "Parse this:\n\nhydra -l admin -P rockyou.txt 10.10.10.5 ssh\n[DATA] attacking ssh://10.10.10.5:22/\n[22][ssh] host: 10.10.10.5   login: admin   password: password123\n1 of 1 target successfully completed, 1 valid password found",
            "output": {"discoveries": [
                {"type": "credential", "value": "admin:password123 (ssh)", "confidence": 0.99},
            ], "phase_impact": "advance", "summary": "Valid SSH credentials found: admin/password123"},
        },
        # LinPEAS escalation hints
        {
            "input": "Parse this:\n\nlinpeas output (truncated):\n════════════════════════╣ Interesting Files ╠═══\n╔══════════╣ SUID - Check easy privesc\n-rwsr-xr-x 1 root root 121456 /usr/bin/python3.8\n-rwsr-xr-x 1 root root 67816 /usr/bin/su\n════════════════════════╣ Capabilities ╠═══\n/usr/bin/python3.8 = cap_setuid+ep",
            "output": {"discoveries": [
                {"type": "vuln", "value": "python3.8 has cap_setuid capability", "confidence": 0.95},
                {"type": "vuln", "value": "python3.8 SUID binary", "confidence": 0.9},
            ], "phase_impact": "advance", "summary": "Python3.8 has cap_setuid — direct root escalation path"},
        },
        # FTP anonymous access
        {
            "input": "Parse this:\n\nftp 10.10.10.5\nConnected to 10.10.10.5.\n220 vsFTPd 3.0.3\nName: anonymous\n331 Please specify the password.\nPassword:\n230 Login successful.\nftp> ls\n200 PORT command successful.\n150 Here comes the directory listing.\n-rw-r--r--    1 0        0          12576 Jun 12 2021 backup.zip\n226 Directory send OK.",
            "output": {"discoveries": [
                {"type": "credential", "value": "anonymous FTP access", "confidence": 0.95},
                {"type": "service", "value": "vsFTPd 3.0.3", "confidence": 0.95},
                {"type": "file", "value": "backup.zip via FTP", "confidence": 0.9},
            ], "phase_impact": "advance", "summary": "Anonymous FTP access with backup.zip file available"},
        },
        # SSH banner grab
        {
            "input": "Parse this:\n\ncurl -v http://10.10.10.5\n* Connected to 10.10.10.5\n> GET / HTTP/1.1\n< HTTP/1.1 200 OK\n< Server: Werkzeug/1.0.1 Python/3.8.10\n< Content-Type: text/html\n<html>\n<!-- TODO: remove debug endpoint at /debug -->\n<title>Dashboard</title>",
            "output": {"discoveries": [
                {"type": "service", "value": "Werkzeug/1.0.1 Python/3.8.10", "confidence": 0.95},
                {"type": "file", "value": "/debug endpoint (from HTML comment)", "confidence": 0.85},
            ], "phase_impact": "stay", "summary": "Python Werkzeug server with hidden /debug endpoint"},
        },
        # PCAP analysis
        {
            "input": "Parse this:\n\ntshark -r capture.pcap -Y 'ftp' -T fields -e ftp.request.command -e ftp.request.arg\nUSER\tnathan\nPASS\tBuck3tH4TF0RM3!\nRETR\tsecret.txt",
            "output": {"discoveries": [
                {"type": "credential", "value": "nathan:Buck3tH4TF0RM3! (FTP)", "confidence": 0.99},
                {"type": "file", "value": "secret.txt transferred via FTP", "confidence": 0.9},
            ], "phase_impact": "advance", "summary": "FTP credentials captured from PCAP: nathan/Buck3tH4TF0RM3!"},
        },
    ]
    
    samples = []
    for t in templates:
        response = json.dumps(t["output"])
        samples.append(Sample(
            task="tool_output_parse",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS["tool_output_parse"]},
                {"role": "user", "content": t["input"]},
                {"role": "assistant", "content": response},
            ],
            metadata={"synthetic": True, "source": "schema_stress"},
            quality=0.95,
        ))
    return samples


def gen_synthetic_edge_cases() -> list[Sample]:
    """Synthetic: rare phases, conflicting evidence, deadlock patterns."""
    cases = []
    
    # LATERAL_MOVEMENT (rare phase)
    for target in ["10.10.10.20", "192.168.1.100", "172.16.0.5"]:
        cases.append(Sample(
            task="next_step",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS["next_step"]},
                {"role": "user", "content": (
                    f"Current engagement state:\n"
                    f"Phase: LATERAL_MOVEMENT\nTarget: {target}\nStep: 45/100\n"
                    f"Shells: user@10.10.10.5\nCredentials: admin:pass123\n\n"
                    f"Suggest the next action."
                )},
                {"role": "assistant", "content": json.dumps({
                    "action": f"ssh admin@{target} -o StrictHostKeyChecking=no",
                    "reasoning": "Reuse discovered credentials against new target for lateral movement",
                    "phase_fit": 0.92,
                    "alternatives": [
                        f"crackmapexec smb {target} -u admin -p pass123",
                        f"nmap -sV -p 22,80,445 {target}"
                    ]
                })},
            ],
            metadata={"phase": "LATERAL_MOVEMENT", "synthetic": True},
            quality=0.9,
        ))
    
    # CLOSEOUT phase (rare)
    cases.append(Sample(
        task="phase_classification",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPTS["phase_classification"]},
            {"role": "user", "content": (
                "Classify the attack phase:\n\n"
                "Phase: CLOSEOUT\nTarget: 10.10.10.5\nStep: 95/100\n"
                "Shells: root@10.10.10.5\nCredentials: admin:pass, nathan:Buck3t\n"
                "Ports: 22, 80, 21\nFlags captured: user.txt, root.txt"
            )},
            {"role": "assistant", "content": json.dumps({
                "phase": "CLOSEOUT",
                "confidence": 0.98,
                "reasoning": "Both flags captured, root shell obtained — engagement complete"
            })},
        ],
        metadata={"phase": "CLOSEOUT", "synthetic": True},
        quality=0.9,
    ))
    
    # Conflicting evidence
    cases.append(Sample(
        task="evidence_check",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPTS["evidence_check"]},
            {"role": "user", "content": (
                "Is the evidence sufficient to proceed?\n\n"
                "Transition: ENUMERATION → EXPLOITATION\n\n"
                "Phase: ENUMERATION\nTarget: 10.10.10.5\nStep: 30/100\n"
                "Ports: 22, 80, 443\nServices: OpenSSH 7.9, Apache 2.4.38\n"
                "No credentials found. No vulnerabilities confirmed.\n"
                "Gobuster found /admin but 403 forbidden."
            )},
            {"role": "assistant", "content": json.dumps({
                "sufficient": False,
                "missing": ["confirmed vulnerability or credential", "access to /admin panel"],
                "confidence": 0.35,
                "recommendation": "Continue enumeration: try different wordlists on /admin, check for default credentials, scan for known CVEs on Apache 2.4.38"
            })},
        ],
        metadata={"phase": "ENUMERATION", "synthetic": True},
        quality=0.95,
    ))
    
    # Deadlock/stagnation detection
    cases.append(Sample(
        task="retry_or_pivot",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPTS["retry_or_pivot"]},
            {"role": "user", "content": (
                "This command was unproductive:\n"
                "Command: gobuster dir -u http://10.10.10.5 -w /usr/share/wordlists/dirb/common.txt\n"
                "Reward: 0.0\n\n"
                "State:\nPhase: ENUMERATION\nStep: 25/100\n"
                "Note: This is the 4th directory brute-force attempt with different wordlists.\n"
                "All returned 403 or no results.\n\n"
                "What should we do next?"
            )},
            {"role": "assistant", "content": json.dumps({
                "decision": "ESCALATE",
                "action": "nikto -h http://10.10.10.5 -Tuning 123bde",
                "reasoning": "Directory brute-forcing consistently fails after 4 attempts. Escalating to vulnerability scanning with nikto to find alternative attack vectors",
                "confidence": 0.82
            })},
        ],
        metadata={"decision": "ESCALATE", "synthetic": True},
        quality=0.95,
    ))
    
    # HTB Cap-style PCAP attack chain (from real successful engagement)
    cases.append(Sample(
        task="retrieval_reasoning",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPTS["retrieval_reasoning"]},
            {"role": "user", "content": (
                "Current state:\nPhase: ENUMERATION\nTarget: 10.129.5.115\nStep: 8\n"
                "Ports: 21, 22, 80\nServices: vsFTPd 3.0.3, OpenSSH 8.2, gunicorn\n"
                "Web: found /download/0 returns PCAP file\n\n"
                "Prior experience: HTB Cap machine — web app serves network captures. "
                "PCAP at /data/0 (or /download/0) contains FTP credentials in cleartext. "
                "User nathan:Buck3tH4TF0RM3! found via tshark. SSH → user flag. "
                "Python3.8 cap_setuid → root.\n\n"
                "Synthesize guidance."
            )},
            {"role": "assistant", "content": json.dumps({
                "synthesis": "Web-served PCAP files are a signature pattern for credential leakage. Download the capture and extract FTP credentials using tshark, then pivot to SSH.",
                "from_current": "Web server at port 80 serves PCAP downloads; FTP service on port 21 suggests FTP traffic in captures",
                "from_memory": "HTB Cap pattern: /download/0 or /data/0 → PCAP → tshark for FTP USER/PASS → SSH access → cap_setuid privesc",
                "action": "tshark -r /tmp/capture.pcap -Y 'ftp.request.command == USER || ftp.request.command == PASS' -T fields -e ftp.request.arg",
                "confidence": 0.95
            })},
        ],
        metadata={"phase": "ENUMERATION", "synthetic": True, "pattern": "pcap_credential_leak"},
        quality=0.98,
    ))
    
    # Privilege escalation via SUID/capabilities
    for binary, method in [
        ("python3", "python3 -c 'import os; os.setuid(0); os.system(\"/bin/bash\")'"),
        ("find", "find / -exec /bin/sh -p \\; -quit"),
        ("vim", "vim -c ':!/bin/sh'"),
        ("nmap", "nmap --interactive\n!sh"),
    ]:
        cases.append(Sample(
            task="next_step",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS["next_step"]},
                {"role": "user", "content": (
                    f"Current engagement state:\n"
                    f"Phase: PRIVILEGE_ESCALATION\nTarget: 10.10.10.5\nStep: 50/100\n"
                    f"Shells: user@10.10.10.5\n"
                    f"Vuln: {binary} has SUID or cap_setuid capability\n\n"
                    f"Suggest the next action."
                )},
                {"role": "assistant", "content": json.dumps({
                    "action": method,
                    "reasoning": f"Exploit {binary} SUID/capability to escalate to root",
                    "phase_fit": 0.97,
                    "alternatives": ["sudo -l", "cat /etc/shadow"]
                })},
            ],
            metadata={"phase": "PRIVILEGE_ESCALATION", "synthetic": True, "binary": binary},
            quality=0.95,
        ))
    
    # Metasploitable 2 patterns
    ms2_services = [
        ("vsftpd 2.3.4", "searchsploit vsftpd 2.3.4", "Known backdoor in vsftpd 2.3.4 — port 6200 shell"),
        ("Samba 3.0.20", "use exploit/multi/samba/usermap_script", "Samba username map script RCE"),
        ("Apache Tomcat", "msfconsole -q -x 'use exploit/multi/http/tomcat_mgr_upload'", "Default Tomcat manager credentials allow WAR deployment"),
        ("UnrealIRCd", "msfconsole -q -x 'use exploit/unix/irc/unreal_ircd_3281_backdoor'", "UnrealIRCd 3.2.8.1 backdoor"),
    ]
    for service, cmd, reasoning in ms2_services:
        cases.append(Sample(
            task="next_step",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS["next_step"]},
                {"role": "user", "content": (
                    f"Current engagement state:\n"
                    f"Phase: EXPLOITATION\nTarget: 172.28.0.10\nStep: 20\n"
                    f"Services: {service}\n"
                    f"This is a Metasploitable lab target.\n\n"
                    f"Suggest the next action."
                )},
                {"role": "assistant", "content": json.dumps({
                    "action": cmd,
                    "reasoning": reasoning,
                    "phase_fit": 0.95,
                    "alternatives": ["nmap --script vuln 172.28.0.10"]
                })},
            ],
            metadata={"phase": "EXPLOITATION", "synthetic": True, "target_family": "metasploitable"},
            quality=0.95,
        ))
    
    return cases


def gen_synthetic_retrieval_reasoning() -> list[Sample]:
    """Synthetic: retrieval-aware reasoning combining state + memory."""
    patterns = [
        {
            "state": "Phase: RECON\nTarget: 10.10.10.X\nPorts: 22, 80\nServices: unknown",
            "memory": "Similar target had Werkzeug debug console at /console. Check for Python web frameworks.",
            "synthesis": "Port 80 may host a Python web framework. Check for debug endpoints (/console, /debug) and framework fingerprints before deep scanning.",
            "from_current": "HTTP on port 80 needs investigation, SSH on 22 as fallback",
            "from_memory": "Prior target with same port profile had Werkzeug debug exposure",
            "action": "curl -s http://10.10.10.X/ | grep -iE 'werkzeug|flask|django|debug|console'",
            "confidence": 0.75,
        },
        {
            "state": "Phase: ENUMERATION\nTarget: 10.10.10.X\nPorts: 21, 22, 80, 3306\nServices: MySQL 5.7, Apache, vsFTPd",
            "memory": "MySQL with default credentials found on lab targets. Try root:root, root:mysql, admin:admin.",
            "synthesis": "Multiple attack vectors: FTP anonymous, MySQL default creds, web application. MySQL default credentials are a quick check with high payoff.",
            "from_current": "4 services discovered — MySQL, Apache, FTP, SSH",
            "from_memory": "MySQL on lab targets often has default or weak credentials",
            "action": "mysql -h 10.10.10.X -u root -p'' -e 'SHOW DATABASES;' 2>/dev/null || mysql -h 10.10.10.X -u root -proot -e 'SHOW DATABASES;'",
            "confidence": 0.7,
        },
        {
            "state": "Phase: PRIVILEGE_ESCALATION\nShell: www-data@target\nSUID: /usr/bin/env",
            "memory": "env SUID escalation: env /bin/sh -p gives root immediately via GTFOBins.",
            "synthesis": "Direct root escalation via env SUID binary. This is a well-known GTFOBins technique with near-certain success.",
            "from_current": "www-data shell with env having SUID bit",
            "from_memory": "env SUID is a guaranteed root path per GTFOBins",
            "action": "/usr/bin/env /bin/sh -p",
            "confidence": 0.98,
        },
    ]
    
    samples = []
    for p in patterns:
        for target_suffix in ["5", "15", "25", "50"]:
            state = p["state"].replace("10.10.10.X", f"10.10.10.{target_suffix}")
            action = p["action"].replace("10.10.10.X", f"10.10.10.{target_suffix}")
            
            response = json.dumps({
                "synthesis": p["synthesis"],
                "from_current": p["from_current"],
                "from_memory": p["from_memory"],
                "action": action,
                "confidence": p["confidence"],
            })
            
            samples.append(Sample(
                task="retrieval_reasoning",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPTS["retrieval_reasoning"]},
                    {"role": "user", "content": (
                        f"Current state:\n{state}\n\n"
                        f"Prior experience: {p['memory']}\n\n"
                        f"Synthesize guidance."
                    )},
                    {"role": "assistant", "content": response},
                ],
                metadata={"synthetic": True, "source": "retrieval_reasoning"},
                quality=0.92,
            ))
    return samples


# ── Dataset Assembly ───────────────────────────────────────────────────────

def deduplicate(samples: list[Sample]) -> list[Sample]:
    """Remove duplicates by content hash."""
    seen = set()
    unique = []
    for s in samples:
        h = s.content_hash()
        if h not in seen:
            seen.add(h)
            unique.append(s)
    return unique


def split_dataset(samples: list[Sample], seed: int = 42) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Split into train (80%), val (10%), holdout (10%)."""
    rng = random.Random(seed)
    rng.shuffle(samples)
    n = len(samples)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    return samples[:train_end], samples[train_end:val_end], samples[val_end:]


def write_jsonl(samples: list[Sample], path: Path) -> None:
    """Write samples as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")


def main():
    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Ariaska Dataset V2 — High-Quality Extraction")
    print("=" * 60)
    
    # Load data
    print("\n[1/6] Loading traces...")
    traces = load_all_traces()
    print(f"  Loaded {len(traces)} trace files")
    
    print("[2/6] Loading postmortems...")
    postmortems = load_all_postmortems()
    print(f"  Loaded {len(postmortems)} postmortems")
    
    # Generate samples per task
    print("\n[3/6] Generating samples from real data...")
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
        samples = gen_fn()
        # Filter by quality
        samples = [s for s in samples if s.quality >= MIN_QUALITY]
        # Deduplicate
        samples = deduplicate(samples)
        # Cap
        if len(samples) > MAX_PER_TASK:
            # Keep highest quality
            samples.sort(key=lambda s: s.quality, reverse=True)
            samples = samples[:MAX_PER_TASK]
        task_samples[task_name] = samples
        print(f"  {task_name}: {len(samples)} samples")
    
    # Synthetic augmentation
    print("\n[4/6] Generating targeted synthetic data...")
    synthetic_schema = gen_synthetic_schema_stress()
    synthetic_edge = gen_synthetic_edge_cases()
    synthetic_retrieval = gen_synthetic_retrieval_reasoning()
    
    print(f"  Schema stress: {len(synthetic_schema)}")
    print(f"  Edge cases: {len(synthetic_edge)}")
    print(f"  Retrieval reasoning: {len(synthetic_retrieval)}")
    
    # Merge synthetic into task pools
    for s in synthetic_schema:
        task_samples.setdefault(s.task, []).append(s)
    for s in synthetic_edge:
        task_samples.setdefault(s.task, []).append(s)
    for s in synthetic_retrieval:
        task_samples.setdefault(s.task, []).append(s)
    
    # Write per-task files
    print("\n[5/6] Writing per-task JSONL files...")
    all_samples = []
    stats = {}
    for task_name, samples in sorted(task_samples.items()):
        write_jsonl(samples, OUTPUT_DIR / f"{task_name}.jsonl")
        all_samples.extend(samples)
        n_real = sum(1 for s in samples if not s.metadata.get("synthetic"))
        n_synth = sum(1 for s in samples if s.metadata.get("synthetic"))
        avg_q = sum(s.quality for s in samples) / max(1, len(samples))
        stats[task_name] = {"total": len(samples), "real": n_real, "synthetic": n_synth, "avg_quality": round(avg_q, 3)}
        print(f"  {task_name}: {len(samples)} (real: {n_real}, synthetic: {n_synth}, avg_q: {avg_q:.3f})")
    
    # Split into train/val/holdout
    print("\n[6/6] Splitting train/val/holdout...")
    all_samples = deduplicate(all_samples)
    train, val, holdout = split_dataset(all_samples, SEED)
    
    write_jsonl(train, OUTPUT_DIR / "train.jsonl")
    write_jsonl(val, OUTPUT_DIR / "val.jsonl")
    write_jsonl(holdout, OUTPUT_DIR / "holdout.jsonl")
    
    print(f"\n  Train: {len(train)}")
    print(f"  Val: {len(val)}")
    print(f"  Holdout: {len(holdout)}")
    print(f"  Total unique: {len(all_samples)}")
    
    # Write stats
    stats_path = OUTPUT_DIR / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "total_samples": len(all_samples),
            "train": len(train),
            "val": len(val),
            "holdout": len(holdout),
            "per_task": stats,
        }, f, indent=2)
    
    print(f"\n  Stats written to {stats_path}")
    print(f"\nDone! Dataset in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
