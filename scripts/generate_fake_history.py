#!/usr/bin/env python3
"""
Generate Pre-Seeded Fake Training History for LLM Distillation
================================================================
Creates realistic engagement result JSONs in artifacts/ that simulate
a progression of Ariaska training runs — from early failures through
partial successes to full root pwns.

These synthetic histories provide:
  • Training data for local LLM fine-tuning / mentor distillation
  • Demonstration data for the Training History browser
  • Consistent "memory" baseline across environments

Engagement profiles are modeled after real HTB boxes with realistic:
  • Kill chain phase progression
  • Discovery counts and types
  • Decision source distributions (PPO, playbook, mentor, etc.)
  • Token costs and timing
  • PPO training metrics (policy loss, value loss, entropy)
  • Reward curves that reflect learning progression

Usage:
    python scripts/generate_fake_history.py
    python scripts/generate_fake_history.py --count 50
"""
from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ═══════════════════════════════════════════════════════════════════════
#  Synthetic Target Profiles
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TargetProfile:
    """A synthetic HTB-style target with known exploit path."""
    name: str
    ip: str
    difficulty: str             # easy | medium | hard
    os_type: str                # linux | windows
    services: List[str]         # e.g. ["ssh:22", "http:80"]
    exploit_path: str           # Brief exploit narrative
    typical_user_step: int      # Step at which user flag typically found
    typical_root_step: int      # Step at which root flag typically found
    typical_reward: float       # Expected total reward on success
    user_flag: str
    root_flag: str


TARGETS = [
    TargetProfile(
        name="Lame", ip="10.129.1.12", difficulty="easy", os_type="linux",
        services=["ftp:21", "ssh:22", "smb:139", "smb:445"],
        exploit_path="vsftpd 2.3.4 backdoor → Samba usermap_script → root shell",
        typical_user_step=18, typical_root_step=25, typical_reward=2650.0,
        user_flag="a3d1f85c9e2b4c8d7f6e5a4b3c2d1e0f", root_flag="b4e2f96d0a3c5d7e8f1a2b3c4d5e6f7a",
    ),
    TargetProfile(
        name="Bashed", ip="10.129.2.35", difficulty="easy", os_type="linux",
        services=["http:80"],
        exploit_path="phpbash webshell → sudo -l → scriptmanager privesc → root cron",
        typical_user_step=22, typical_root_step=35, typical_reward=2580.0,
        user_flag="c5f3a07e1b4d6e8f9a0b1c2d3e4f5a6b", root_flag="d6a4b18f2c5e7f0a1b2c3d4e5f6a7b8c",
    ),
    TargetProfile(
        name="Nibbles", ip="10.129.3.48", difficulty="easy", os_type="linux",
        services=["ssh:22", "http:80"],
        exploit_path="Nibbleblog 4.0.3 RCE → file upload → sudo monitor.sh → root",
        typical_user_step=30, typical_root_step=42, typical_reward=2720.0,
        user_flag="e7b5c29a3d6f8a1b2c3d4e5f6a7b8c9d", root_flag="f8c6d3ab4e7a9b2c3d4e5f6a7b8c9d0e",
    ),
    TargetProfile(
        name="Optimum", ip="10.129.4.51", difficulty="easy", os_type="windows",
        services=["http:80"],
        exploit_path="HFS 2.3 RCE → CVE-2014-6287 → MS16-032 kernel privesc → SYSTEM",
        typical_user_step=20, typical_root_step=38, typical_reward=2490.0,
        user_flag="a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6", root_flag="b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7",
    ),
    TargetProfile(
        name="Shocker", ip="10.129.5.67", difficulty="easy", os_type="linux",
        services=["ssh:22", "http:80", "http:2222"],
        exploit_path="Shellshock CGI → user shell → sudo perl → root",
        typical_user_step=25, typical_root_step=32, typical_reward=2610.0,
        user_flag="c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8", root_flag="d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9",
    ),
    TargetProfile(
        name="Beep", ip="10.129.6.83", difficulty="easy", os_type="linux",
        services=["ssh:22", "http:80", "https:443", "smtp:25", "pop3:110", "imap:143"],
        exploit_path="Elastix LFI → /etc/amportal.conf creds → SSH as root",
        typical_user_step=28, typical_root_step=35, typical_reward=2550.0,
        user_flag="e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0", root_flag="f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1",
    ),
    TargetProfile(
        name="Cronos", ip="10.129.7.14", difficulty="medium", os_type="linux",
        services=["ssh:22", "dns:53", "http:80"],
        exploit_path="DNS zone transfer → admin.cronos.htb → SQLi login bypass → command injection → laravel cron privesc",
        typical_user_step=35, typical_root_step=55, typical_reward=2780.0,
        user_flag="a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2", root_flag="b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3",
    ),
    TargetProfile(
        name="Popcorn", ip="10.129.8.29", difficulty="medium", os_type="linux",
        services=["ssh:22", "http:80"],
        exploit_path="Torrent Hoster upload bypass → PHP webshell → PAM MOTD CVE-2010-0832 → root",
        typical_user_step=40, typical_root_step=65, typical_reward=2830.0,
        user_flag="c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4", root_flag="d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5",
    ),
    TargetProfile(
        name="SolidState", ip="10.129.9.42", difficulty="medium", os_type="linux",
        services=["ssh:22", "smtp:25", "pop3:110", "imap:143", "http:80", "james-admin:4555"],
        exploit_path="Apache James 2.3.2 default creds → user mailbox enum → restricted shell bypass → cron privesc",
        typical_user_step=38, typical_root_step=58, typical_reward=2690.0,
        user_flag="e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6", root_flag="f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7",
    ),
    TargetProfile(
        name="Bastard", ip="10.129.10.55", difficulty="medium", os_type="windows",
        services=["http:80", "rpc:135", "rpc:49154"],
        exploit_path="Drupal 7 Drupalgeddon2 RCE → token impersonation → SYSTEM",
        typical_user_step=30, typical_root_step=52, typical_reward=2750.0,
        user_flag="a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8", root_flag="b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9",
    ),
    TargetProfile(
        name="Nineveh", ip="10.129.11.68", difficulty="medium", os_type="linux",
        services=["http:80", "https:443"],
        exploit_path="Hydra brute force → phpLiteAdmin 1.9 RCE → phpmyadmin LFI chaining → chkrootkit privesc",
        typical_user_step=55, typical_root_step=80, typical_reward=2870.0,
        user_flag="c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0", root_flag="d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1",
    ),
    TargetProfile(
        name="Brainfuck", ip="10.129.12.81", difficulty="hard", os_type="linux",
        services=["ssh:22", "smtp:25", "pop3:110", "imap:143", "https:443"],
        exploit_path="WordPress plugin vuln → SMTP cred leak → encrypted forum → RSA key recovery → lxd privesc",
        typical_user_step=65, typical_root_step=120, typical_reward=2920.0,
        user_flag="e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2", root_flag="f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3",
    ),
]

# ═══════════════════════════════════════════════════════════════════════
#  Phase Progression Map
# ═══════════════════════════════════════════════════════════════════════

PHASES_ORDERED = [
    "RECON", "ENUMERATION", "EXPLOITATION", "PRIVILEGE_ESCALATION",
    "LATERAL_MOVEMENT", "POST_EXPLOITATION", "EXFILTRATION", "CLOSEOUT",
]

PHASE_BASE_REWARDS = {
    "RECON": 0.0, "ENUMERATION": 5.0, "EXPLOITATION": 15.0,
    "PRIVILEGE_ESCALATION": 30.0, "LATERAL_MOVEMENT": 45.0,
    "POST_EXPLOITATION": 60.0, "EXFILTRATION": 75.0, "CLOSEOUT": 90.0,
}


# ═══════════════════════════════════════════════════════════════════════
#  Engagement Result Generator
# ═══════════════════════════════════════════════════════════════════════

def generate_engagement(
    target: TargetProfile,
    run_id: int,
    base_timestamp: datetime,
    maturity: float,  # 0.0 (novice) → 1.0 (expert) — controls how far agent gets
    rng: random.Random,
) -> Dict[str, Any]:
    """Generate a single realistic engagement result."""

    # ── Determine outcome based on maturity ─────────────────────────
    # Low maturity: stuck in RECON/ENUM. High maturity: full pwn.
    maturity_noise = rng.gauss(0, 0.08)
    effective_mat = max(0.0, min(1.0, maturity + maturity_noise))

    if effective_mat < 0.15:
        highest_phase_idx = 0  # RECON
    elif effective_mat < 0.30:
        highest_phase_idx = 1  # ENUMERATION
    elif effective_mat < 0.50:
        highest_phase_idx = 2  # EXPLOITATION
    elif effective_mat < 0.65:
        highest_phase_idx = 3  # PRIVILEGE_ESCALATION
    elif effective_mat < 0.80:
        highest_phase_idx = rng.choice([4, 5])  # LATERAL or POST
    elif effective_mat < 0.90:
        highest_phase_idx = 6  # EXFILTRATION
    else:
        highest_phase_idx = 7  # CLOSEOUT

    highest_phase = PHASES_ORDERED[highest_phase_idx]
    user_flag_captured = highest_phase_idx >= 3 and rng.random() < (0.3 + 0.7 * effective_mat)
    root_flag_captured = highest_phase_idx >= 6 and rng.random() < (0.2 + 0.8 * effective_mat)

    # If we reached CLOSEOUT with high maturity, almost always root
    if highest_phase_idx == 7 and effective_mat > 0.92:
        user_flag_captured = True
        root_flag_captured = True

    # ── Steps & timing ──────────────────────────────────────────────
    base_steps = target.typical_root_step if root_flag_captured else target.typical_user_step
    step_factor = 1.0 + (1.0 - effective_mat) * 1.5  # Novice takes more steps
    total_steps = int(base_steps * step_factor * rng.uniform(0.8, 1.3))
    total_steps = max(15, min(500, total_steps))

    duration_per_step = rng.uniform(3.0, 12.0)  # seconds
    duration_s = total_steps * duration_per_step

    # ── Reward calculation ──────────────────────────────────────────
    phase_reward = PHASE_BASE_REWARDS[highest_phase]
    discovery_reward = rng.uniform(20, 80) * effective_mat
    flag_reward = 0.0
    if user_flag_captured:
        flag_reward += 50.0
    if root_flag_captured:
        flag_reward += 50.0

    # Accumulated step rewards
    step_rewards = phase_reward * total_steps * 0.15
    penalty = rng.uniform(5, 30) * (1.0 - effective_mat)  # Anti-repeat etc
    total_reward = round(step_rewards + discovery_reward + flag_reward - penalty, 2)
    total_reward = max(0.0, total_reward)

    # Scale to realistic range matching real artifacts
    if root_flag_captured:
        total_reward = round(rng.uniform(2400, 2900), 2)
    elif user_flag_captured:
        total_reward = round(rng.uniform(800, 1800), 2)
    elif highest_phase_idx >= 2:
        total_reward = round(rng.uniform(200, 800), 2)
    else:
        total_reward = round(rng.uniform(10, 200), 2)

    # ── Discovery counts ────────────────────────────────────────────
    n_ports = len(target.services)
    n_services = n_ports
    discovered_ports = min(n_ports, int(n_ports * min(1.0, effective_mat * 2) + 0.5))
    discovered_services = min(n_services, int(n_services * min(1.0, effective_mat * 1.8) + 0.5))
    discovered_creds = 1 if user_flag_captured else (1 if effective_mat > 0.4 and rng.random() < 0.3 else 0)
    discovered_shells = (2 if root_flag_captured else 1) if highest_phase_idx >= 2 else 0
    total_discoveries = discovered_ports + discovered_services + discovered_creds + discovered_shells

    # ── Decision sources ────────────────────────────────────────────
    n_decisions = total_steps + rng.randint(5, 25)  # Some agents make extra decisions
    ppo_pct = 0.35 + 0.25 * effective_mat  # PPO gets stronger with maturity
    playbook_pct = 0.25 - 0.15 * effective_mat
    anti_repeat_pct = 0.10 + 0.05 * (1 - effective_mat)
    codex_pct = 0.05 + 0.03 * effective_mat
    registry_pct = 1.0 - ppo_pct - playbook_pct - anti_repeat_pct - codex_pct

    decision_sources = {
        "ppo": int(n_decisions * ppo_pct),
        "playbook": int(n_decisions * playbook_pct),
        "registry": max(0, int(n_decisions * registry_pct)),
        "anti_repeat": int(n_decisions * anti_repeat_pct),
        "codex_meta": int(n_decisions * codex_pct),
    }

    # ── Token cost ──────────────────────────────────────────────────
    base_cost = 0.15 + total_steps * 0.008
    cost_usd = round(base_cost * rng.uniform(0.7, 1.4), 4)

    # ── PPO episode data ────────────────────────────────────────────
    episode_data = _generate_episode_data(
        target, total_steps, total_reward, highest_phase,
        decision_sources, effective_mat, rng,
    )

    # ── Assemble result ─────────────────────────────────────────────
    timestamp = base_timestamp + timedelta(
        hours=run_id * rng.uniform(2, 8),
        minutes=rng.randint(0, 59),
    )

    result: Dict[str, Any] = {
        "timestamp": timestamp.isoformat(),
        "max_steps": 500,
        "seed": rng.randint(1, 99999),
        "target_ip": target.ip,
        "target_name": target.name,
        "mode": "continuous",
        "difficulty": target.difficulty,
        "os_type": target.os_type,
        "duration_s": round(duration_s, 2),
        "total_reward": total_reward,
        "highest_phase": highest_phase,
        "total_steps": total_steps,
        "total_discoveries": total_discoveries,
        "total_cost_usd": cost_usd,
        "user_flag_captured": user_flag_captured,
        "root_flag_captured": root_flag_captured,
        "user_flag_value": target.user_flag if user_flag_captured else "",
        "root_flag_value": target.root_flag if root_flag_captured else "",
        "decision_sources": decision_sources,
        "discovery_summary": {
            "ports": discovered_ports,
            "services": discovered_services,
            "credentials": discovered_creds,
            "shells": discovered_shells,
        },
        "episode_metrics": {
            "total_steps": total_steps,
            "highest_phase": highest_phase,
            "total_discoveries": total_discoveries,
            "unique_commands": int(total_steps * rng.uniform(0.6, 0.85)),
            "diversity_ratio": round(rng.uniform(0.55, 0.85), 3),
        },
        "ppo_metrics": {
            "policy_loss": round(rng.uniform(-0.15, 0.45), 4),
            "value_loss": round(rng.uniform(0.5, 4.0), 4),
            "entropy": round(rng.uniform(2.2, 3.8), 4),
            "kl_divergence": round(rng.uniform(0.005, 0.025), 5),
            "clip_fraction": round(rng.uniform(0.05, 0.25), 4),
            "ppo_updates": int(total_steps / rng.uniform(8, 16)),
        },
        "episode_data": episode_data,
        "agent_participation": {
            "ScoutAgent": {"steps": int(total_steps * 0.25), "discoveries": discovered_ports},
            "RedAgent": {"steps": int(total_steps * 0.35), "discoveries": discovered_shells + discovered_creds},
            "BlueAgent": {"steps": int(total_steps * 0.10), "discoveries": 0},
            "ShadowAgent": {"steps": int(total_steps * 0.15), "discoveries": 0},
            "OrionAgent": {"steps": int(total_steps * 0.15), "discoveries": 0},
        },
        # Distillation metadata — useful for LLM fine-tuning
        "distillation_metadata": {
            "exploit_path": target.exploit_path,
            "services_discovered": target.services[:discovered_services],
            "maturity_signal": round(effective_mat, 3),
            "success": root_flag_captured,
            "partial_success": user_flag_captured and not root_flag_captured,
            "learning_phase": (
                "early" if effective_mat < 0.3
                else "mid" if effective_mat < 0.7
                else "late"
            ),
        },
    }

    return result


def _generate_episode_data(
    target: TargetProfile,
    total_steps: int,
    total_reward: float,
    highest_phase: str,
    decision_sources: Dict[str, int],
    maturity: float,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Generate per-episode training data matching real artifact format."""
    return [{
        "episode": 1,
        "reward": total_reward,
        "highest_phase": highest_phase,
        "steps": total_steps,
        "ppo_updates": max(1, int(total_steps / rng.uniform(8, 16))),
        "policy_loss": round(rng.uniform(-0.15, 0.45), 4),
        "value_loss": round(rng.uniform(0.5, 4.0), 4),
        "entropy": round(rng.uniform(2.2, 3.8), 4),
        "sources": decision_sources,
        "unique_commands": int(total_steps * rng.uniform(0.6, 0.85)),
        "unique_templates": int(total_steps * rng.uniform(0.5, 0.78)),
        "command_diversity": round(rng.uniform(0.55, 0.85), 4),
        "total_discoveries": int(total_steps * rng.uniform(0.05, 0.15)),
        "step_at_first_exploit": (
            int(target.typical_user_step * rng.uniform(0.7, 1.5))
            if highest_phase not in ("RECON", "ENUMERATION") else 0
        ),
    }]


# ═══════════════════════════════════════════════════════════════════════
#  Training Batch Generator (multi-episode runs)
# ═══════════════════════════════════════════════════════════════════════

def generate_training_batch(
    target: TargetProfile,
    episodes: int,
    batch_id: str,
    base_timestamp: datetime,
    maturity_start: float,
    maturity_end: float,
    rng: random.Random,
) -> Dict[str, Any]:
    """Generate a multi-episode training batch result (like r76b_3ep_results.json)."""

    episode_data = []
    rewards = []
    all_sources: Dict[str, int] = {"ppo": 0, "playbook": 0, "registry": 0, "anti_repeat": 0, "codex_meta": 0}
    total_steps_all = 0

    for ep in range(1, episodes + 1):
        # Maturity increases across episodes
        mat = maturity_start + (maturity_end - maturity_start) * (ep / episodes)
        mat = max(0.0, min(1.0, mat + rng.gauss(0, 0.05)))

        steps = rng.randint(18, 40)
        total_steps_all += steps

        # Determine phase reached
        if mat > 0.85:
            phase = "CLOSEOUT"
        elif mat > 0.70:
            phase = "EXFILTRATION"
        elif mat > 0.55:
            phase = "PRIVILEGE_ESCALATION"
        elif mat > 0.35:
            phase = "EXPLOITATION"
        elif mat > 0.15:
            phase = "ENUMERATION"
        else:
            phase = "RECON"

        # Reward
        if phase == "CLOSEOUT":
            reward = round(rng.uniform(2600, 2900), 2)
        elif phase in ("EXFILTRATION", "POST_EXPLOITATION"):
            reward = round(rng.uniform(1800, 2600), 2)
        elif phase in ("PRIVILEGE_ESCALATION", "LATERAL_MOVEMENT"):
            reward = round(rng.uniform(800, 1800), 2)
        elif phase in ("EXPLOITATION",):
            reward = round(rng.uniform(200, 800), 2)
        else:
            reward = round(rng.uniform(20, 200), 2)

        rewards.append(reward)

        # Sources
        ep_sources = {
            "ppo": int(steps * rng.uniform(0.4, 0.65)),
            "playbook": int(steps * rng.uniform(0.08, 0.20)),
            "registry": int(steps * rng.uniform(0.0, 0.05)),
            "anti_repeat": int(steps * rng.uniform(0.05, 0.18)),
            "codex_meta": int(steps * rng.uniform(0.02, 0.08)),
        }
        for k in all_sources:
            all_sources[k] += ep_sources[k]

        episode_data.append({
            "episode": ep,
            "reward": reward,
            "highest_phase": phase,
            "steps": steps,
            "ppo_updates": max(1, int(steps / rng.uniform(6, 12))),
            "policy_loss": round(rng.uniform(-0.12, 0.45), 4),
            "value_loss": round(rng.uniform(0.4, 4.5), 4),
            "entropy": round(rng.uniform(2.0, 3.8), 4),
            "sources": ep_sources,
            "unique_commands": int(steps * rng.uniform(0.6, 0.90)),
            "unique_templates": int(steps * rng.uniform(0.5, 0.82)),
            "command_diversity": round(rng.uniform(0.55, 0.90), 4),
            "total_discoveries": rng.randint(2, 8),
            "step_at_first_exploit": (
                rng.randint(5, 15) if phase not in ("RECON", "ENUMERATION") else 0
            ),
        })

    # Determine final phase
    final_phases = [ep["highest_phase"] for ep in episode_data]
    phase_distribution = {}
    for p in final_phases:
        phase_distribution[p] = phase_distribution.get(p, 0) + 1

    avg_reward = round(sum(rewards) / len(rewards), 2)
    exfil_count = sum(1 for p in final_phases if PHASES_ORDERED.index(p) >= 6)
    closeout_count = sum(1 for p in final_phases if p == "CLOSEOUT")

    duration_s = total_steps_all * rng.uniform(3, 10)

    return {
        "timestamp": base_timestamp.isoformat(),
        "episodes": episodes,
        "max_steps": 40,
        "seed": rng.randint(1, 9999),
        "target_ip": target.ip,
        "mode": "live",
        "duration_s": round(duration_s, 2),
        "avg_reward": avg_reward,
        "last10_avg": avg_reward,
        "max_reward": max(rewards),
        "min_reward": min(rewards),
        "exfil_pct": round(exfil_count / episodes * 100, 1),
        "closeout_pct": round(closeout_count / episodes * 100, 1),
        "phase_distribution": phase_distribution,
        "decision_sources": all_sources,
        "episode_data": episode_data,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Main: Generate Full History
# ═══════════════════════════════════════════════════════════════════════

def main(count: int = 30) -> None:
    """Generate a realistic progression of training history."""
    artifacts_dir = _PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)  # Deterministic for reproducibility

    # Base timestamp: start ~60 days ago to avoid collisions with real data
    base_time = datetime(2025, 12, 1, 8, 0, 0)

    generated = 0
    print(f"Generating {count} fake engagement histories in artifacts/...")

    # ── Phase 1: Early struggles (maturity 0.05-0.25) ───────────────
    early_targets = rng.sample(TARGETS[:6], min(4, len(TARGETS[:6])))
    for i, target in enumerate(early_targets):
        maturity = 0.05 + i * 0.05
        ts = base_time + timedelta(days=i, hours=rng.randint(0, 8))

        result = generate_engagement(target, i, ts, maturity, rng)
        filename = f"engagement_{ts.strftime('%Y%m%d_%H%M%S')}_results.json"
        filepath = artifacts_dir / filename

        # Don't overwrite real results
        if not filepath.exists():
            filepath.write_text(json.dumps(result, indent=2, default=str))
            generated += 1
            status = "🏴 PWNED" if result["root_flag_captured"] else (
                "🏴 user" if result["user_flag_captured"] else "INCOMPLETE"
            )
            print(f"  [{generated:3d}] {filename:<48} {target.name:<12} {result['highest_phase']:<22} {status}")

        if generated >= count:
            break

    # ── Phase 2: Learning progression (maturity 0.25-0.60) ──────────
    mid_targets = rng.sample(TARGETS[:8], min(6, len(TARGETS[:8])))
    for i, target in enumerate(mid_targets):
        if generated >= count:
            break
        maturity = 0.25 + i * 0.06
        ts = base_time + timedelta(days=5 + i, hours=rng.randint(0, 12))

        result = generate_engagement(target, generated, ts, maturity, rng)
        filename = f"engagement_{ts.strftime('%Y%m%d_%H%M%S')}_results.json"
        filepath = artifacts_dir / filename

        if not filepath.exists():
            filepath.write_text(json.dumps(result, indent=2, default=str))
            generated += 1
            status = "🏴 PWNED" if result["root_flag_captured"] else (
                "🏴 user" if result["user_flag_captured"] else "INCOMPLETE"
            )
            print(f"  [{generated:3d}] {filename:<48} {target.name:<12} {result['highest_phase']:<22} {status}")

    # ── Phase 3: Improving success (maturity 0.60-0.85) ─────────────
    improving_targets = rng.sample(TARGETS[:10], min(6, len(TARGETS[:10])))
    for i, target in enumerate(improving_targets):
        if generated >= count:
            break
        maturity = 0.60 + i * 0.04
        ts = base_time + timedelta(days=12 + i, hours=rng.randint(0, 16))

        result = generate_engagement(target, generated, ts, maturity, rng)
        filename = f"engagement_{ts.strftime('%Y%m%d_%H%M%S')}_results.json"
        filepath = artifacts_dir / filename

        if not filepath.exists():
            filepath.write_text(json.dumps(result, indent=2, default=str))
            generated += 1
            status = "🏴 PWNED" if result["root_flag_captured"] else (
                "🏴 user" if result["user_flag_captured"] else "INCOMPLETE"
            )
            print(f"  [{generated:3d}] {filename:<48} {target.name:<12} {result['highest_phase']:<22} {status}")

    # ── Phase 4: Expert runs — consistent pwns (maturity 0.85-0.98) ─
    expert_targets = rng.sample(TARGETS, min(8, len(TARGETS)))
    for i, target in enumerate(expert_targets):
        if generated >= count:
            break
        maturity = 0.85 + i * 0.015
        ts = base_time + timedelta(days=20 + i, hours=rng.randint(0, 20))

        result = generate_engagement(target, generated, ts, maturity, rng)
        filename = f"engagement_{ts.strftime('%Y%m%d_%H%M%S')}_results.json"
        filepath = artifacts_dir / filename

        if not filepath.exists():
            filepath.write_text(json.dumps(result, indent=2, default=str))
            generated += 1
            status = "🏴 PWNED" if result["root_flag_captured"] else (
                "🏴 user" if result["user_flag_captured"] else "INCOMPLETE"
            )
            print(f"  [{generated:3d}] {filename:<48} {target.name:<12} {result['highest_phase']:<22} {status}")

    # ── Phase 5: Training batches (multi-episode) ───────────────────
    batch_targets = rng.sample(TARGETS[:6], min(4, len(TARGETS[:6])))
    for i, target in enumerate(batch_targets):
        if generated >= count:
            break
        ts = base_time + timedelta(days=25 + i * 2)
        batch_result = generate_training_batch(
            target, episodes=rng.choice([3, 5, 10]),
            batch_id=f"synth_{i}", base_timestamp=ts,
            maturity_start=0.5 + i * 0.1, maturity_end=0.8 + i * 0.05,
            rng=rng,
        )
        filename = f"synth_batch{i+1}_{ts.strftime('%Y%m%d')}_results.json"
        filepath = artifacts_dir / filename

        if not filepath.exists():
            filepath.write_text(json.dumps(batch_result, indent=2, default=str))
            generated += 1
            print(f"  [{generated:3d}] {filename:<48} {target.name:<12} {batch_result['episodes']} eps, avg_reward={batch_result['avg_reward']:.0f}")

    print(f"\n✓ Generated {generated} fake history files in artifacts/")
    print(f"  Total files in artifacts/: {len(list(artifacts_dir.glob('*_results.json')))}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate fake training history")
    parser.add_argument("--count", type=int, default=30, help="Number of fake runs to generate")
    args = parser.parse_args()
    main(count=args.count)
