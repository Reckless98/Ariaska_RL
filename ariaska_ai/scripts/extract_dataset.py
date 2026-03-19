#!/usr/bin/env python3
"""Extract training dataset from Ariaska trace data for Qwen3.5-4B fine-tuning.

Reads from traces/ and postmortems/ to build conversation-format training data
across 9 task families that cover the full Ariaska decision pipeline.

Output: JSONL files in ariaska_ai/dataset/ ready for SFT with transformers/trl.
"""

import json
import os
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any

# ── Constants ──────────────────────────────────────────────────────────────────
TRACE_DIR = Path(__file__).resolve().parents[2] / "traces"
POSTMORTEM_DIR = Path(__file__).resolve().parents[2] / "postmortems"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "dataset"
MIN_REWARD_THRESHOLD = 2.0   # Only include steps with meaningful positive reward
MAX_SAMPLES_PER_FAMILY = 8000
SEED = 42

PHASES = [
    "RECON", "ENUMERATION", "EXPLOITATION", "PRIVILEGE_ESCALATION",
    "LATERAL_MOVEMENT", "POST_EXPLOITATION", "EXFILTRATION", "CLOSEOUT"
]

ATTACK_TOOLS = {
    "nmap", "nikto", "gobuster", "dirb", "wpscan", "hydra", "sqlmap",
    "searchsploit", "msfconsole", "curl", "wget", "ssh", "ftp", "smbclient",
    "enum4linux", "crackmapexec", "metasploit", "hashcat", "john",
    "tshark", "tcpdump", "netcat", "nc", "wafw00f", "whatweb", "nuclei",
    "ffuf", "wfuzz", "burpsuite", "responder", "impacket", "linpeas",
    "winpeas", "pspy", "chisel", "socat", "scp", "rsync", "python3",
    "perl", "php", "bash", "sh", "cat", "find", "grep", "ls", "id", "whoami",
    "uname", "ifconfig", "ip", "arp", "route", "getcap", "sudo"
}


@dataclass
class TrainingSample:
    """A single training sample in chat format."""
    task_family: str
    system_prompt: str
    user_prompt: str
    assistant_response: str
    metadata: dict = field(default_factory=dict)

    def to_chat_format(self) -> dict:
        """Convert to HuggingFace chat format for SFT."""
        return {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt},
                {"role": "assistant", "content": self.assistant_response},
            ],
            "task_family": self.task_family,
            "metadata": self.metadata,
        }


# ── System Prompts by Task Family ─────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "phase_classification": (
        "You are Ariaska, a cybersecurity AI coprocessor specialized in "
        "authorized penetration testing. Classify the current attack phase "
        "based on the engagement state. Respond with ONLY the phase name and "
        "a one-line justification.\n\n"
        "Valid phases: RECON, ENUMERATION, EXPLOITATION, PRIVILEGE_ESCALATION, "
        "LATERAL_MOVEMENT, POST_EXPLOITATION, EXFILTRATION, CLOSEOUT"
    ),
    "next_step_analysis": (
        "You are Ariaska, a cybersecurity AI coprocessor for authorized "
        "penetration testing. Given the current engagement state, suggest the "
        "next command to execute. Respond in this exact format:\n"
        "ACTION: <exact command>\n"
        "REASON: <one-line tactical justification>\n"
        "PHASE_FIT: <0.0-1.0 confidence this fits the current phase>"
    ),
    "tool_output_parsing": (
        "You are Ariaska, a cybersecurity AI coprocessor. Parse the given "
        "tool output and extract structured findings. Respond in JSON format:\n"
        '{"discoveries": [{"type": "<port|service|version|credential|vuln|shell>", '
        '"value": "<finding>", "confidence": <0.0-1.0>}], '
        '"phase_advancement": "<stay|advance>", "reasoning": "<brief>"}'
    ),
    "state_summarization": (
        "You are Ariaska, a cybersecurity AI coprocessor. Summarize the "
        "current engagement state concisely for agent decision-making. "
        "Include: phase, key discoveries, open attack vectors, blockers."
    ),
    "retry_or_pivot": (
        "You are Ariaska, a cybersecurity AI coprocessor. A command has "
        "failed or returned unhelpful output. Decide whether to RETRY with "
        "modifications, PIVOT to a different approach, or ESCALATE to a "
        "different attack vector. Respond:\n"
        "DECISION: RETRY|PIVOT|ESCALATE\n"
        "ACTION: <next command if RETRY/PIVOT>\n"
        "REASON: <justification>"
    ),
    "postmortem_analysis": (
        "You are Ariaska, a cybersecurity AI coprocessor. Analyze this "
        "engagement episode and provide a structured postmortem. Identify "
        "what worked, what failed, and specific improvements."
    ),
    "command_validation": (
        "You are Ariaska, a cybersecurity AI coprocessor. Validate whether "
        "the proposed command is appropriate for the current phase and state. "
        "Respond:\n"
        "VALID: true|false\n"
        "REASON: <justification>\n"
        "ALTERNATIVE: <better command if invalid>"
    ),
    "schema_obedience": (
        "You are Ariaska, a cybersecurity AI coprocessor. Respond to the "
        "query in the exact JSON schema specified. Do not deviate from the "
        "schema structure."
    ),
    "tactical_reasoning": (
        "You are Ariaska, a cybersecurity AI coprocessor for authorized "
        "penetration testing. Provide tactical reasoning about the current "
        "engagement situation. Be concise, specific, and actionable."
    ),
}


# ── Data Extraction ───────────────────────────────────────────────────────────

def load_trace_steps(trace_dir: Path) -> list[dict]:
    """Load step records from a trace run directory."""
    steps_file = trace_dir / "steps.jsonl"
    if not steps_file.exists():
        return []
    steps = []
    try:
        with open(steps_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    steps.append(json.loads(line))
    except (json.JSONDecodeError, OSError):
        pass
    return steps


def load_trace_events(event_file: Path) -> list[dict]:
    """Load event records from an events JSONL file."""
    events = []
    try:
        with open(event_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    except (json.JSONDecodeError, OSError):
        pass
    return events


def load_postmortem(pm_file: Path) -> dict | None:
    """Load a postmortem JSON file."""
    try:
        with open(pm_file) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def build_state_context(step: dict, history: list[dict]) -> str:
    """Build a state context string from step data and history."""
    parts = []
    phase = step.get("phase", step.get("phase_before", "RECON")).upper()
    parts.append(f"Current Phase: {phase}")

    # Agent info
    agent = step.get("agent", "")
    if agent:
        parts.append(f"Active Agent: {agent}")

    # Discovery context from history
    discoveries = set()
    for h in history[-10:]:
        if "new_discoveries" in h and isinstance(h["new_discoveries"], dict):
            for dtype, vals in h["new_discoveries"].items():
                if isinstance(vals, list):
                    for v in vals:
                        discoveries.add(f"{dtype}: {v}")
                elif vals:
                    discoveries.add(f"{dtype}: {vals}")

    if discoveries:
        parts.append(f"Discoveries so far: {', '.join(sorted(discoveries)[:15])}")

    # Recent commands from history
    recent_cmds = []
    for h in history[-5:]:
        cmd = h.get("chosen_action", h.get("command", ""))
        if cmd:
            recent_cmds.append(cmd)
    if recent_cmds:
        parts.append(f"Recent commands: {'; '.join(recent_cmds)}")

    # Target
    target = step.get("target_ip", "")
    if target:
        parts.append(f"Target: {target}")

    # Step number
    step_num = step.get("step", 0)
    parts.append(f"Step: {step_num}")

    return "\n".join(parts)


# ── Sample Generators by Task Family ──────────────────────────────────────────

def gen_phase_classification(steps: list[dict]) -> list[TrainingSample]:
    """Generate phase classification samples from step data."""
    samples = []
    history: list[dict] = []
    for step in steps:
        phase = step.get("phase", step.get("phase_before", "")).upper()
        if phase not in PHASES:
            history.append(step)
            continue
        reward = step.get("reward", 0)
        if reward < 0:
            history.append(step)
            continue
        ctx = build_state_context(step, history)
        user_prompt = f"Classify the current attack phase:\n\n{ctx}"
        # Build justification from available data
        agent = step.get("agent", "unknown")
        cmd = step.get("chosen_action", step.get("command", ""))
        response = f"{phase}\nJustification: Agent {agent} executing '{cmd}' is consistent with {phase} phase objectives."

        samples.append(TrainingSample(
            task_family="phase_classification",
            system_prompt=SYSTEM_PROMPTS["phase_classification"],
            user_prompt=user_prompt,
            assistant_response=response,
            metadata={"phase": phase, "reward": reward, "step": step.get("step", 0)},
        ))
        history.append(step)
    return samples


def gen_next_step(steps: list[dict]) -> list[TrainingSample]:
    """Generate next-step analysis samples from successful steps."""
    samples = []
    history: list[dict] = []
    for step in steps:
        reward = step.get("reward", 0)
        cmd = step.get("chosen_action", step.get("command", ""))
        if not cmd or reward < MIN_REWARD_THRESHOLD:
            history.append(step)
            continue
        phase = step.get("phase", step.get("phase_before", "RECON")).upper()
        ctx = build_state_context(step, history)
        user_prompt = f"Current engagement state:\n{ctx}\n\nWhat should be the next action?"

        # Compute phase_fit from reward (higher reward = better fit)
        phase_fit = min(1.0, reward / 20.0)
        mentor = step.get("mentor_response", "")
        reason = ""
        if mentor and "REASON:" in mentor:
            reason = mentor.split("REASON:")[-1].strip()[:200]
        if not reason:
            reason = f"Targets {phase.lower()} objectives with {cmd.split()[0] if cmd else 'command'}"

        response = f"ACTION: {cmd}\nREASON: {reason}\nPHASE_FIT: {phase_fit:.2f}"
        samples.append(TrainingSample(
            task_family="next_step_analysis",
            system_prompt=SYSTEM_PROMPTS["next_step_analysis"],
            user_prompt=user_prompt,
            assistant_response=response,
            metadata={"phase": phase, "reward": reward, "command": cmd},
        ))
        history.append(step)
    return samples


def gen_retry_or_pivot(steps: list[dict]) -> list[TrainingSample]:
    """Generate retry/pivot samples from step sequences with failures."""
    samples = []
    for i in range(1, len(steps)):
        prev = steps[i - 1]
        curr = steps[i]
        prev_reward = prev.get("reward", 0)
        curr_reward = curr.get("reward", 0)
        prev_cmd = prev.get("chosen_action", prev.get("command", ""))
        curr_cmd = curr.get("chosen_action", curr.get("command", ""))
        if not prev_cmd or not curr_cmd:
            continue
        # Failure then success pattern
        if prev_reward <= 0 and curr_reward >= MIN_REWARD_THRESHOLD:
            prev_tool = prev_cmd.split()[0] if prev_cmd else ""
            curr_tool = curr_cmd.split()[0] if curr_cmd else ""
            is_retry = prev_tool == curr_tool
            decision = "RETRY" if is_retry else "PIVOT"
            ctx = build_state_context(prev, steps[:i])
            user_prompt = (
                f"The following command produced no useful results:\n"
                f"Command: {prev_cmd}\n"
                f"Reward: {prev_reward}\n\n"
                f"Engagement state:\n{ctx}\n\n"
                f"Should we retry, pivot, or escalate?"
            )
            reason = f"Previous approach {'needed parameter adjustment' if is_retry else 'was ineffective, switching'}"
            response = f"DECISION: {decision}\nACTION: {curr_cmd}\nREASON: {reason}"
            samples.append(TrainingSample(
                task_family="retry_or_pivot",
                system_prompt=SYSTEM_PROMPTS["retry_or_pivot"],
                user_prompt=user_prompt,
                assistant_response=response,
                metadata={"prev_reward": prev_reward, "curr_reward": curr_reward, "decision": decision},
            ))
    return samples


def gen_command_validation(steps: list[dict]) -> list[TrainingSample]:
    """Generate command validation samples from step data."""
    samples = []
    history: list[dict] = []
    for step in steps:
        cmd = step.get("chosen_action", step.get("command", ""))
        reward = step.get("reward", 0)
        phase = step.get("phase", step.get("phase_before", "RECON")).upper()
        if not cmd:
            history.append(step)
            continue
        ctx = build_state_context(step, history)
        user_prompt = (
            f"Validate this command for the current engagement state:\n"
            f"Command: {cmd}\n"
            f"Phase: {phase}\n\n"
            f"State:\n{ctx}"
        )
        valid = reward > 0
        reason = f"Command {'aligns with' if valid else 'does not fit'} {phase} phase objectives"
        response = f"VALID: {'true' if valid else 'false'}\nREASON: {reason}"
        if not valid and len(history) > 1:
            # Suggest alternative from a successful recent command
            for h in reversed(history[-5:]):
                alt = h.get("chosen_action", h.get("command", ""))
                if alt and h.get("reward", 0) > 0:
                    response += f"\nALTERNATIVE: {alt}"
                    break
        samples.append(TrainingSample(
            task_family="command_validation",
            system_prompt=SYSTEM_PROMPTS["command_validation"],
            user_prompt=user_prompt,
            assistant_response=response,
            metadata={"valid": valid, "phase": phase, "reward": reward},
        ))
        history.append(step)
    return samples


def gen_tactical_reasoning(steps: list[dict]) -> list[TrainingSample]:
    """Generate tactical reasoning samples from high-reward steps."""
    samples = []
    history: list[dict] = []
    for step in steps:
        reward = step.get("reward", 0)
        if reward < 5.0:
            history.append(step)
            continue
        cmd = step.get("chosen_action", step.get("command", ""))
        phase = step.get("phase", step.get("phase_before", "RECON")).upper()
        agent = step.get("agent", "")
        mentor = step.get("mentor_response", "")
        ctx = build_state_context(step, history)
        user_prompt = (
            f"Provide tactical analysis for this engagement state:\n{ctx}\n\n"
            f"The {agent} is considering actions in the {phase} phase."
        )
        # Build reasoning from available data
        reasoning_parts = []
        if mentor and not mentor.startswith("ACTION:"):
            reasoning_parts.append(mentor[:300])
        reasoning_parts.append(
            f"Current phase {phase} calls for {_phase_to_objective(phase)}. "
            f"The command '{cmd}' achieved reward {reward:.1f}, indicating "
            f"{'significant progress' if reward >= 20 else 'moderate progress' if reward >= 5 else 'some progress'}."
        )
        response = " ".join(reasoning_parts)
        samples.append(TrainingSample(
            task_family="tactical_reasoning",
            system_prompt=SYSTEM_PROMPTS["tactical_reasoning"],
            user_prompt=user_prompt,
            assistant_response=response,
            metadata={"phase": phase, "reward": reward, "agent": agent},
        ))
        history.append(step)
    return samples


def gen_state_summarization(events: list[dict]) -> list[TrainingSample]:
    """Generate state summarization from event sequences."""
    samples = []
    step_events = [e for e in events if e.get("kind") == "step"]
    for i in range(5, len(step_events), 5):
        window = step_events[max(0, i - 5):i]
        if not window:
            continue
        last = window[-1]
        phase = last.get("phase_after", last.get("phase_before", "RECON"))
        discoveries = {}
        total_reward = 0
        commands = []
        for w in window:
            total_reward += w.get("step_reward_total", 0)
            nd = w.get("new_discoveries", {})
            if isinstance(nd, dict):
                for k, v in nd.items():
                    if k not in discoveries:
                        discoveries[k] = []
                    if isinstance(v, list):
                        discoveries[k].extend(v)
                    elif v:
                        discoveries[k].append(str(v))
            for ar in w.get("agent_records", []):
                c = ar.get("command", "")
                if c:
                    commands.append(f"{ar.get('agent_name', '?')}: {c}")

        user_prompt = (
            f"Summarize the current engagement state after {i} steps.\n\n"
            f"Last 5 steps commands:\n" + "\n".join(commands[-5:]) + "\n\n"
            f"Total reward in window: {total_reward:.1f}\n"
            f"Current phase: {phase}"
        )
        disc_str = "; ".join(f"{k}: {', '.join(set(str(x) for x in v[:5]))}" for k, v in discoveries.items() if v)
        response = (
            f"Phase: {phase}\n"
            f"Discoveries: {disc_str or 'None in this window'}\n"
            f"Progress: {'Good' if total_reward > 10 else 'Moderate' if total_reward > 0 else 'Stalled'} "
            f"(reward: {total_reward:.1f})\n"
            f"Recommendation: {'Continue current approach' if total_reward > 5 else 'Consider pivoting strategy'}"
        )
        samples.append(TrainingSample(
            task_family="state_summarization",
            system_prompt=SYSTEM_PROMPTS["state_summarization"],
            user_prompt=user_prompt,
            assistant_response=response,
            metadata={"phase": phase, "total_reward": total_reward, "step": i},
        ))
    return samples


def gen_postmortem_samples(pm_data: dict) -> list[TrainingSample]:
    """Generate postmortem analysis samples."""
    if not pm_data or pm_data.get("model_used") == "offline":
        return []
    outcomes = pm_data.get("key_outcomes", {})
    wins = outcomes.get("wins", [])
    fails = outcomes.get("fails", [])
    summary = outcomes.get("summary", "")
    if not summary:
        return []

    user_prompt = (
        f"Analyze this penetration testing engagement:\n\n"
        f"Run ID: {pm_data.get('run_id', 'unknown')}\n"
        f"Summary: {summary}\n"
        f"Successes: {'; '.join(wins[:5]) if wins else 'None'}\n"
        f"Failures: {'; '.join(fails[:5]) if fails else 'None'}"
    )
    # Build analysis from skill cards
    skills = pm_data.get("skill_cards", [])
    analysis_parts = [f"Engagement Analysis:\n{summary}\n"]
    if wins:
        analysis_parts.append(f"Key wins: {'; '.join(wins[:3])}")
    if fails:
        analysis_parts.append(f"Key failures: {'; '.join(fails[:3])}")
    if skills:
        for s in skills[:3]:
            if s.get("if_condition") and s.get("then_action"):
                analysis_parts.append(f"Pattern: IF {s['if_condition']} THEN {s['then_action']}")

    experiments = pm_data.get("next_experiments", [])
    if experiments:
        analysis_parts.append(f"Recommended next steps: {'; '.join(e.get('title', '') for e in experiments[:3])}")

    response = "\n".join(analysis_parts)
    return [TrainingSample(
        task_family="postmortem_analysis",
        system_prompt=SYSTEM_PROMPTS["postmortem_analysis"],
        user_prompt=user_prompt,
        assistant_response=response,
        metadata={"run_id": pm_data.get("run_id", ""), "validation": pm_data.get("validation_passed", False)},
    )]


def gen_schema_obedience(steps: list[dict]) -> list[TrainingSample]:
    """Generate schema obedience samples — model must output valid JSON."""
    samples = []
    history: list[dict] = []
    for step in steps:
        reward = step.get("reward", 0)
        if reward < 1.0:
            history.append(step)
            continue
        cmd = step.get("chosen_action", step.get("command", ""))
        phase = step.get("phase", step.get("phase_before", "RECON")).upper()
        if not cmd:
            history.append(step)
            continue
        user_prompt = (
            f"Respond in this exact JSON schema:\n"
            f'{{"command": "<str>", "phase": "<str>", "confidence": <float>, '
            f'"reasoning": "<str>", "alternatives": ["<str>"]}}\n\n'
            f"Given phase: {phase}, recent discoveries, recommend a command."
        )
        alternatives = []
        for h in reversed(history[-3:]):
            alt = h.get("chosen_action", h.get("command", ""))
            if alt and alt != cmd:
                alternatives.append(alt)
        response = json.dumps({
            "command": cmd,
            "phase": phase,
            "confidence": min(1.0, reward / 20.0),
            "reasoning": f"Selected for {phase} phase based on engagement state",
            "alternatives": alternatives[:2]
        }, indent=2)
        samples.append(TrainingSample(
            task_family="schema_obedience",
            system_prompt=SYSTEM_PROMPTS["schema_obedience"],
            user_prompt=user_prompt,
            assistant_response=response,
            metadata={"phase": phase, "reward": reward},
        ))
        history.append(step)
    return samples


# ── Helpers ───────────────────────────────────────────────────────────────────

def _phase_to_objective(phase: str) -> str:
    mapping = {
        "RECON": "identifying open ports, services, and initial attack surface",
        "ENUMERATION": "deep enumeration of services, versions, and vulnerabilities",
        "EXPLOITATION": "exploiting identified vulnerabilities to gain access",
        "PRIVILEGE_ESCALATION": "escalating privileges from initial foothold",
        "LATERAL_MOVEMENT": "moving laterally through the network",
        "POST_EXPLOITATION": "collecting data, flags, and maintaining access",
        "EXFILTRATION": "extracting critical data and capturing flags",
        "CLOSEOUT": "final verification and engagement cleanup",
    }
    return mapping.get(phase, "progressing the engagement")


def dedup_samples(samples: list[TrainingSample]) -> list[TrainingSample]:
    """Deduplicate samples by hashing user_prompt + assistant_response."""
    seen = set()
    deduped = []
    for s in samples:
        h = hashlib.md5((s.user_prompt + s.assistant_response).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            deduped.append(s)
    return deduped


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def main():
    random.seed(SEED)
    all_samples: dict[str, list[TrainingSample]] = {k: [] for k in SYSTEM_PROMPTS}

    # ── Process trace run directories ──────────────────────────────────────
    run_dirs = sorted(TRACE_DIR.glob("run_*"))
    print(f"Found {len(run_dirs)} trace run directories")

    processed = 0
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        steps = load_trace_steps(run_dir)
        if len(steps) < 3:
            continue

        # Filter to meaningful agent steps (not offline filler)
        meaningful_steps = [s for s in steps if s.get("reward", 0) != 0 or s.get("mentor_call")]

        if meaningful_steps:
            all_samples["phase_classification"].extend(gen_phase_classification(meaningful_steps))
            all_samples["next_step_analysis"].extend(gen_next_step(meaningful_steps))
            all_samples["retry_or_pivot"].extend(gen_retry_or_pivot(meaningful_steps))
            all_samples["command_validation"].extend(gen_command_validation(meaningful_steps))
            all_samples["tactical_reasoning"].extend(gen_tactical_reasoning(meaningful_steps))
            all_samples["schema_obedience"].extend(gen_schema_obedience(meaningful_steps))
            processed += 1

    print(f"Processed {processed} trace run directories")

    # ── Process events_*.jsonl files (richer data) ─────────────────────────
    event_files = sorted(TRACE_DIR.glob("events_*.jsonl"))
    print(f"Found {len(event_files)} event trace files")

    for ef in event_files:
        events = load_trace_events(ef)
        if len(events) < 5:
            continue
        all_samples["state_summarization"].extend(gen_state_summarization(events))

        # Also extract step-level data from events with agent_records
        step_events = [e for e in events if e.get("kind") == "step" and e.get("agent_records")]
        for se in step_events:
            for ar in se.get("agent_records", []):
                pseudo_step = {
                    "agent": ar.get("agent_name", ""),
                    "phase": se.get("phase_before", "RECON"),
                    "chosen_action": ar.get("command", ""),
                    "command": ar.get("command", ""),
                    "reward": ar.get("reward", 0),
                    "mentor_call": ar.get("mentor_called", False),
                    "mentor_response": ar.get("mentor_response", ""),
                    "step": se.get("step_num", 0),
                    "target_ip": se.get("target_ip", ""),
                }
                if pseudo_step["reward"] >= MIN_REWARD_THRESHOLD:
                    ns = gen_next_step([pseudo_step])
                    all_samples["next_step_analysis"].extend(ns)

    # ── Process postmortems ────────────────────────────────────────────────
    pm_files = sorted(POSTMORTEM_DIR.glob("postmortem_*.json"))
    print(f"Found {len(pm_files)} postmortem files")
    for pf in pm_files:
        pm = load_postmortem(pf)
        if pm:
            all_samples["postmortem_analysis"].extend(gen_postmortem_samples(pm))

    # ── Dedup + Sample + Write ─────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    stats = {}

    for family, samples in all_samples.items():
        deduped = dedup_samples(samples)
        if len(deduped) > MAX_SAMPLES_PER_FAMILY:
            random.shuffle(deduped)
            deduped = deduped[:MAX_SAMPLES_PER_FAMILY]
        output_file = OUTPUT_DIR / f"{family}.jsonl"
        with open(output_file, "w") as f:
            for s in deduped:
                f.write(json.dumps(s.to_chat_format()) + "\n")
        stats[family] = len(deduped)
        total += len(deduped)
        print(f"  {family}: {len(deduped)} samples -> {output_file}")

    # ── Also write combined training file ──────────────────────────────────
    combined = []
    for family, samples in all_samples.items():
        deduped = dedup_samples(samples)
        if len(deduped) > MAX_SAMPLES_PER_FAMILY:
            random.shuffle(deduped)
            deduped = deduped[:MAX_SAMPLES_PER_FAMILY]
        combined.extend(deduped)
    random.shuffle(combined)

    # 90/10 train/eval split
    split_idx = int(len(combined) * 0.9)
    train_data = combined[:split_idx]
    eval_data = combined[split_idx:]

    train_file = OUTPUT_DIR / "train.jsonl"
    eval_file = OUTPUT_DIR / "eval.jsonl"
    with open(train_file, "w") as f:
        for s in train_data:
            f.write(json.dumps(s.to_chat_format()) + "\n")
    with open(eval_file, "w") as f:
        for s in eval_data:
            f.write(json.dumps(s.to_chat_format()) + "\n")

    print(f"\n{'='*60}")
    print(f"Total unique samples: {total}")
    print(f"Train split: {len(train_data)} -> {train_file}")
    print(f"Eval split: {len(eval_data)} -> {eval_file}")
    print(f"Family breakdown: {json.dumps(stats, indent=2)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
