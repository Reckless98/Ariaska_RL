#!/usr/bin/env python3
"""
core/models/state_encoder.py — ARIASKA Rich State Encoder v3.0
🧠 90+ Meaningful Features | 🎯 Shared Across All Agents | 📊 Normalized & Information-Dense

Replaces the legacy 19-dim encoding (492 dead zeros) with a rich 512-dim
vector where ~90 dimensions carry meaningful signal about the environment,
attack progress, agent state, and temporal dynamics.

Architecture:
    Section 1: Phase Information         [0-11]    12 dims
    Section 2: State Flags               [12-26]   15 dims
    Section 3: Port Presence             [27-54]   28 dims  (MS2-expanded)
    Section 4: Service Type Presence     [55-72]   18 dims  (MS2-expanded)
    Section 5: Numeric Features          [73-84]   12 dims
    Section 6: Action History            [85-94]   10 dims
    Section 7: LLM/Mentor Features       [95-99]    5 dims
    Section 8: Temporal Features         [100-104]  5 dims
    Section 9: Agent Role Encoding       [105-109]  5 dims  (Phase 5.2+)
    Section 10: Target Profile           [110-114]  5 dims  (Phase 5.2+)
    Section 11: Discovery Breakdown      [115-122]  8 dims  (Phase 5.2+)
    ──────────────────────────────────────────────────────────
    Total meaningful dims: ~123 / 512
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional


# ── Constants ────────────────────────────────────────────────────────────
STATE_DIM = 512

PHASES = [
    "recon", "enumeration", "exploit", "privesc",
    "lateral_movement", "post_exploitation", "exfiltrate", "closeout",
]
PHASE_INDEX = {p: i for i, p in enumerate(PHASES)}

# Top 28 ports most relevant to pentesting (Metasploitable 2, HTB, etc.)
COMMON_PORTS = [
    21, 22, 23, 25, 53, 80, 110, 111, 135, 139,
    143, 443, 445, 512, 993, 995, 1099, 1433, 1524,
    2049, 3306, 3389, 3632, 5432, 5900, 6667, 8080, 8180,
]
PORT_SET = set(COMMON_PORTS)

# Common service types to encode individually (18 types, MS2-expanded)
SERVICE_TYPES = [
    "ssh", "http", "https", "ftp", "smb", "telnet",
    "mysql", "postgresql", "rdp", "dns", "smtp", "vnc",
    "irc", "nfs", "java_rmi", "tomcat", "rexec", "distcc",
]

# State flag keys (from CyberEnvironment.get_global_state())
STATE_FLAG_KEYS = [
    "ports_discovered", "services_enumerated",
    "ssh_service_found", "http_service_found",
    "smb_service_found", "ftp_service_found", "mysql_service_found",
    "vulnerability_found", "credentials_known",
    "shell_obtained", "linux_shell_obtained",
    "root_shell_obtained", "admin_credentials_known",
]

PRIVILEGE_MAP = {"none": 0.0, "user": 0.5, "root": 1.0}

# Agent role encoding (5 roles → 5-dim one-hot)
AGENT_ROLE_INDEX = {
    "recon": 0, "offensive": 1, "defensive": 2, "strategic": 3, "stealth": 4,
}

# Target profile indicators for state-aware encoding
TARGET_PROFILES = {
    "metasploitable2": {"ms2": True, "linux": True, "difficulty": 0.3},
    "metasploitable3": {"ms2": False, "linux": True, "difficulty": 0.6},
    "generic": {"ms2": False, "linux": False, "difficulty": 0.5},
}


def encode_state(
    state: Dict[str, Any],
    device: torch.device,
    *,
    action_history: Optional[List[int]] = None,
    llm_confidence: float = 0.0,
    gpt_calls_remaining: int = 30,
    gpt_calls_max: int = 30,
    mentor_active: bool = False,
    current_step: int = 0,
    max_steps: int = 100,
    steps_in_phase: int = 0,
    phase_transitions: int = 0,
    agent_role: str = "",
    target_profile: str = "",
) -> torch.Tensor:
    """Encode environment state into a rich 512-dimensional feature vector.

    Args:
        state: Environment state dict from ``env.get_global_state()``.
        device: Target torch device.
        action_history: List of recent action indices (newest last).
        llm_confidence: Current LLM confidence score (0-1).
        gpt_calls_remaining: Remaining GPT budget this episode.
        gpt_calls_max: Max GPT calls per episode.
        mentor_active: Whether mentor guidance is currently active.
        current_step: Current step in episode.
        max_steps: Max steps per episode.
        steps_in_phase: Steps spent in the current phase.
        phase_transitions: Number of phase transitions so far.

    Returns:
        torch.Tensor of shape (512,) on *device*.
    """
    vec = np.zeros(STATE_DIM, dtype=np.float32)
    idx = 0  # running write cursor

    # ─── Section 1: Phase Information (12 dims) ──────────────────────
    phase = state.get("phase", "recon")
    # One-hot phase (7 dims)
    phase_idx = PHASE_INDEX.get(phase, 0)
    vec[idx + phase_idx] = 1.0
    idx += len(PHASES)  # 7

    # Normalised phase index (1 dim)
    vec[idx] = phase_idx / max(len(PHASES) - 1, 1)
    idx += 1  # 8

    # Per-phase progress (4 dims for first 4 phases, normalised)
    phase_progress = state.get("phase_progress", {})
    for p in ["recon", "enumeration", "exploit", "privesc"]:
        pp = float(phase_progress.get(p, 0))
        vec[idx] = min(pp / 10.0, 1.0)  # cap at 1.0
        idx += 1  # 12

    # ─── Section 2: State Flags (15 dims) ────────────────────────────
    state_flags = state.get("state_flags", {})
    for key in STATE_FLAG_KEYS:
        vec[idx] = 1.0 if state_flags.get(key, False) else 0.0
        idx += 1  # 25

    # data_exfiltrated + done (2 dims)
    vec[idx] = 1.0 if state.get("data_exfiltrated", False) else 0.0
    idx += 1
    vec[idx] = 1.0 if state.get("done", False) else 0.0
    idx += 1  # 27

    # ─── Section 3: Port Presence (28 dims, MS2-expanded) ────────────
    open_ports = state.get("open_ports", [])
    open_port_set = set()
    for p in open_ports:
        try:
            open_port_set.add(int(p))
        except (ValueError, TypeError):
            pass

    for port in COMMON_PORTS:
        vec[idx] = 1.0 if port in open_port_set else 0.0
        idx += 1  # 55 after loop (28 ports)

    # ─── Section 4: Service Type Presence (18 dims, MS2-expanded) ────
    services = state.get("services", [])
    services_lower = {str(s).lower() for s in services}

    for svc in SERVICE_TYPES:
        found = any(svc in s for s in services_lower)
        vec[idx] = 1.0 if found else 0.0
        idx += 1  # 73 after loop (18 services)

    # ─── Section 5: Numeric Features (12 dims) [73-84] ──────────────
    # Privilege ordinal
    priv = state.get("privilege_level", "none")
    vec[idx] = PRIVILEGE_MAP.get(priv, 0.0)
    idx += 1

    # Normalised counts
    vec[idx] = min(len(open_ports) / 20.0, 1.0)
    idx += 1
    vec[idx] = min(len(services) / 10.0, 1.0)
    idx += 1

    vulns = state.get("discovered_vulnerabilities", [])
    vec[idx] = min(len(vulns) / 10.0, 1.0)
    idx += 1

    exploited = state.get("exploited_vulnerabilities", [])
    vec[idx] = min(len(exploited) / 5.0, 1.0)
    idx += 1

    # Detection / stealth metrics (already ~0-1 range)
    vec[idx] = min(float(state.get("detection_risk", 0.0)), 1.0)
    idx += 1
    vec[idx] = min(float(state.get("stealth_metric", 10.0)) / 10.0, 1.0)
    idx += 1
    vec[idx] = min(float(state.get("blue_team_alert", 0.0)), 1.0)
    idx += 1

    # Difficulty
    vec[idx] = min(float(state.get("difficulty", 1)) / 10.0, 1.0)
    idx += 1

    # Discovery richness
    discovered_info = state.get("discovered_info", set())
    if isinstance(discovered_info, set):
        vec[idx] = min(len(discovered_info) / 20.0, 1.0)
    else:
        vec[idx] = 0.0
    idx += 1

    # Honeypot awareness
    honeypots = state.get("honeypots", [])
    vec[idx] = min(len(honeypots) / 5.0, 1.0)
    idx += 1

    # Credentials found (binary, but we might also get a count later)
    vec[idx] = 1.0 if state.get("credentials_found", False) else 0.0
    idx += 1  # 85

    # ─── Section 6: Action History (10 dims) [85-94] ────────────────
    history = action_history or []

    # Last action normalised
    if history:
        vec[idx] = float(history[-1]) / 5.0
    idx += 1

    # Total actions normalised
    vec[idx] = min(len(history) / 100.0, 1.0)
    idx += 1

    # Last 5 actions one-hot-ish (each normalised to 0-1, 5 dims)
    last5 = history[-5:] if len(history) >= 5 else history
    for i in range(5):
        if i < len(last5):
            vec[idx] = float(last5[i]) / max(5.0, 1.0)
        idx += 1

    # Action diversity in last 10 (1 dim)
    last10 = history[-10:] if history else []
    if last10:
        vec[idx] = len(set(last10)) / min(len(last10), 5.0)
    idx += 1

    # Repeated action ratio in last 10 (1 dim)
    if len(last10) >= 2:
        repeats = sum(1 for i in range(1, len(last10)) if last10[i] == last10[i - 1])
        vec[idx] = repeats / (len(last10) - 1)
    idx += 1  # 95

    # ─── Section 7: LLM / Mentor Features (5 dims) [95-99] ──────────
    vec[idx] = float(state.get("llm_last_reward", 0.0))
    idx += 1
    vec[idx] = 1.0 if state.get("chain_updated", False) else 0.0
    idx += 1
    vec[idx] = float(llm_confidence)
    idx += 1
    vec[idx] = float(gpt_calls_remaining) / max(gpt_calls_max, 1)
    idx += 1
    vec[idx] = 1.0 if mentor_active else 0.0
    idx += 1  # 100

    # ─── Section 8: Temporal Features (5 dims) [100-104] ─────────────
    vec[idx] = float(current_step) / max(max_steps, 1)
    idx += 1
    vec[idx] = min(float(steps_in_phase) / 50.0, 1.0)
    idx += 1
    vec[idx] = min(float(phase_transitions) / 5.0, 1.0)
    idx += 1
    # Urgency signal — ramps up as episode progresses
    vec[idx] = (float(current_step) / max(max_steps, 1)) ** 2
    idx += 1
    # Phase momentum (higher = advancing faster)
    if current_step > 0:
        vec[idx] = min(float(phase_transitions) / (float(current_step) / 20.0 + 1.0), 1.0)
    idx += 1  # 105

    # ─── Section 9: Agent Role Encoding (5 dims) [105-109] ──────────
    role_idx = AGENT_ROLE_INDEX.get(agent_role, -1)
    if role_idx >= 0:
        vec[idx + role_idx] = 1.0
    idx += len(AGENT_ROLE_INDEX)  # 110

    # ─── Section 10: Target Profile (5 dims) [110-114] ──────────────
    profile = TARGET_PROFILES.get(target_profile, {})
    vec[idx] = 1.0 if profile.get("ms2", False) else 0.0
    idx += 1
    vec[idx] = float(profile.get("difficulty", 0.0))
    idx += 1
    vec[idx] = 1.0 if profile.get("linux", False) else 0.0
    idx += 1
    # Services richness: ratio of discovered services to ports
    n_services = len(services)
    n_ports = len(open_ports)
    vec[idx] = min(float(n_services) / max(float(n_ports), 1.0), 1.0)
    idx += 1
    # Port density: ratio of open ports to total common ports
    vec[idx] = min(float(n_ports) / float(len(COMMON_PORTS)), 1.0)
    idx += 1  # 115

    # ─── Section 11: Discovery Breakdown (8 dims) [115-122] ─────────
    discovery_board = state.get("discovery_board", {})
    for disc_key in ["ports", "services", "credentials", "shells",
                     "vulns", "web_paths", "users", "flags_set"]:
        disc_set = discovery_board.get(disc_key, set())
        if isinstance(disc_set, (set, list)):
            vec[idx] = min(len(disc_set) / 10.0, 1.0)
        idx += 1  # 123 after loop

    # ─── Remaining dims [123-511] are zero-padded ────────────────────
    # ~123 meaningful dims / 512 total

    return torch.tensor(vec, dtype=torch.float32, device=device)


def encode_state_batch(
    states: List[Dict[str, Any]],
    device: torch.device,
    **kwargs,
) -> torch.Tensor:
    """Encode a batch of states into a (B, 512) tensor."""
    tensors = [encode_state(s, device, **kwargs) for s in states]
    return torch.stack(tensors, dim=0)
