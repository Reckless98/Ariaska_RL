#!/usr/bin/env python3
"""
core/models/state_encoder.py — ARIASKA Rich State Encoder v3.1
🧠 140+ Meaningful Features | 🎯 Shared Across All Agents | 📊 Normalized & Information-Dense

Replaces the legacy 19-dim encoding (492 dead zeros) with a rich 512-dim
vector where ~140 dimensions carry meaningful signal about the environment,
attack progress, agent state, temporal dynamics, and reasoning context.

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
    Section 12: Reasoning Context        [123-142] 20 dims  (Phase 6.9.6)
    Section 13: Tactical Depth           [143-167] 25 dims  (Phase 12.1)
    ──────────────────────────────────────────────────────────
    Total meaningful dims: ~168 / 512  (33% utilisation)
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
    # Phase 6.9.6: Reasoning context
    exploit_graph_score: float = 0.0,
    failed_commands_ratio: float = 0.0,
    unique_tools_used: int = 0,
    commands_since_discovery: int = 0,
    skill_match_confidence: float = 0.0,
    playbook_progress: float = 0.0,
    decision_source_ppo_ratio: float = 0.0,
    anti_repeat_ratio: float = 0.0,
    reward_trend: float = 0.0,
    highest_reward_step: float = 0.0,
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

    # ─── Section 12: Reasoning Context (20 dims) [123-142] ──────────
    # Phase 6.9.6: Encode attack reasoning signals for smarter PPO

    # Exploit graph progress (0-1) — how far along known exploit paths
    vec[idx] = min(float(exploit_graph_score), 1.0)
    idx += 1

    # Command failure ratio this episode (0-1)
    vec[idx] = min(float(failed_commands_ratio), 1.0)
    idx += 1

    # Unique tools diversity (normalized)
    vec[idx] = min(float(unique_tools_used) / 20.0, 1.0)
    idx += 1

    # Stagnation signal — steps since last discovery (higher = stuck)
    vec[idx] = min(float(commands_since_discovery) / 20.0, 1.0)
    idx += 1

    # Skill library match confidence (0-1)
    vec[idx] = min(float(skill_match_confidence), 1.0)
    idx += 1

    # Playbook completion progress (0-1)
    vec[idx] = min(float(playbook_progress), 1.0)
    idx += 1

    # Decision source ratios — PPO self-reliance signal
    vec[idx] = min(float(decision_source_ppo_ratio), 1.0)
    idx += 1

    # Anti-repeat pressure (high = too many blocked, need new strategy)
    vec[idx] = min(float(anti_repeat_ratio), 1.0)
    idx += 1

    # Reward trend (normalized: -1 to 1, positive = improving)
    vec[idx] = max(min(float(reward_trend), 1.0), -1.0) * 0.5 + 0.5
    idx += 1

    # Highest step reward so far (normalized)
    vec[idx] = min(float(highest_reward_step) / 50.0, 1.0)
    idx += 1

    # Phase-specific attack surface signals (10 dims)
    # Encode which attack surfaces are available based on discoveries
    db = state.get("discovery_board", {})
    _ports = db.get("ports", set())
    _services = db.get("services", set())
    _creds = db.get("credentials", set())
    _shells = db.get("shells", set())

    # Web attack surface available
    web_ports = {80, 443, 8080, 8180, 8443, 8484}
    vec[idx] = 1.0 if (set(_ports) if isinstance(_ports, set) else set()) & web_ports else 0.0
    idx += 1

    # Database attack surface
    db_ports = {3306, 5432, 1433, 27017}
    vec[idx] = 1.0 if (set(_ports) if isinstance(_ports, set) else set()) & db_ports else 0.0
    idx += 1

    # SMB/file share surface
    smb_ports = {139, 445, 2049}
    vec[idx] = 1.0 if (set(_ports) if isinstance(_ports, set) else set()) & smb_ports else 0.0
    idx += 1

    # Remote access surface (SSH, RDP, VNC)
    remote_ports = {22, 23, 3389, 5900}
    vec[idx] = 1.0 if (set(_ports) if isinstance(_ports, set) else set()) & remote_ports else 0.0
    idx += 1

    # Credentials available for lateral movement
    vec[idx] = min(len(_creds) / 5.0, 1.0) if isinstance(_creds, (set, list)) else 0.0
    idx += 1

    # Shell access level
    vec[idx] = 1.0 if isinstance(_shells, (set, list)) and len(_shells) > 0 else 0.0
    idx += 1

    # Multi-step chain feasibility (have creds AND shell? can pivot)
    have_creds = isinstance(_creds, (set, list)) and len(_creds) > 0
    have_shell = isinstance(_shells, (set, list)) and len(_shells) > 0
    vec[idx] = 1.0 if have_creds and have_shell else (0.5 if have_creds or have_shell else 0.0)
    idx += 1

    # Exploitation readiness — have both services enumerated and vulns found?
    have_services = isinstance(_services, (set, list)) and len(_services) > 0
    have_vulns = len(vulns) > 0 if isinstance(vulns, (list, set)) else False
    vec[idx] = 1.0 if have_services and have_vulns else (0.5 if have_services else 0.0)
    idx += 1

    # Difficulty context (from difficulty preset, encoded in state)
    diff_val = state.get("difficulty_level", 0)
    vec[idx] = min(float(diff_val) / 3.0, 1.0) if diff_val else 0.0
    idx += 1

    # Episode progress relative to phase (are we behind schedule?)
    expected_phase_by_step = min(float(current_step) / max(max_steps, 1) * len(PHASES), len(PHASES) - 1)
    actual_phase = float(phase_idx)
    vec[idx] = max(min((actual_phase - expected_phase_by_step) / len(PHASES), 1.0), -1.0) * 0.5 + 0.5
    idx += 1  # 143 after section 12

    # ─── Section 13: Tactical Depth Features (25 dims) [143-167] ────
    # Phase 12.1: Higher-resolution signals derived from existing state
    # Fills previously zero-padded dims to improve PPO state utilisation

    # Kill chain completion staircase (7 dims) — which phases already passed
    # Different from Section 1's current-phase one-hot: this encodes the
    # cumulative history of progress (a "staircase" encoding)
    for kc_idx in range(7):  # 7 main phases (excluding closeout)
        vec[idx] = 1.0 if phase_idx > kc_idx else 0.0
        idx += 1  # 150 after loop

    # Port cluster fractional density (6 dims)
    # Unlike Section 12's binary "any port open in cluster", this encodes
    # how many ports in each cluster are open (0.0 to 1.0)
    _port_clusters = [
        {80, 443, 8080, 8180, 8443, 8484},         # web
        {3306, 5432, 1433, 27017},                   # database
        {25, 110, 143, 993, 995},                    # email/messaging
        {22, 23, 3389, 5900, 512, 513, 514},         # remote access
        {1524, 6667, 6697},                           # backdoor
        {53, 111, 135, 139, 445, 1099, 2049, 3632},  # infrastructure
    ]
    for cluster in _port_clusters:
        hits = len(open_port_set & cluster)
        vec[idx] = min(float(hits) / max(len(cluster), 1), 1.0)
        idx += 1  # 156 after loop

    # Service enumeration depth (3 dims)
    n_services = len(services_lower)
    n_ports = len(open_port_set)
    # Breadth: how many services identified (normalised)
    vec[idx] = min(float(n_services) / 15.0, 1.0)
    idx += 1
    # Coverage: service-to-port ratio (higher = better enumeration)
    vec[idx] = min(float(n_services) / max(float(n_ports), 1.0), 1.0)
    idx += 1
    # High-value service saturation (pentest-critical services found)
    _hv_services = ["ssh", "http", "smb", "ftp", "mysql", "postgresql"]
    _hv_count = sum(1 for hv in _hv_services if any(hv in s for s in services_lower))
    vec[idx] = min(float(_hv_count) / 3.0, 1.0)
    idx += 1  # 159

    # Exploitation depth (4 dims)
    n_vulns = len(vulns) if isinstance(vulns, (list, set)) else 0
    n_exploited = len(exploited) if isinstance(exploited, (list, set)) else 0
    # Vuln-to-exploit conversion rate
    vec[idx] = min(float(n_exploited) / max(float(n_vulns), 1.0), 1.0)
    idx += 1
    # Exploitation count (normalised)
    vec[idx] = min(float(n_exploited) / 5.0, 1.0)
    idx += 1
    # Privilege escalation progress (ordinal)
    vec[idx] = PRIVILEGE_MAP.get(priv, 0.0)
    idx += 1
    # Root achievement flag (binary shortcut for PPO)
    vec[idx] = 1.0 if state_flags.get("root_shell_obtained", False) else 0.0
    idx += 1  # 163

    # Discovery dynamics (5 dims)
    _db = state.get("discovery_board", {})
    _n_creds = len(_db.get("credentials", set()) or set()) if isinstance(_db.get("credentials"), (set, list)) else 0
    _n_shells = len(_db.get("shells", set()) or set()) if isinstance(_db.get("shells"), (set, list)) else 0
    _n_flags = len(_db.get("flags_set", set()) or set()) if isinstance(_db.get("flags_set"), (set, list)) else 0
    _n_webpaths = len(_db.get("web_paths", set()) or set()) if isinstance(_db.get("web_paths"), (set, list)) else 0
    _total_disc = n_ports + n_services + _n_creds + _n_shells + n_vulns + _n_flags + _n_webpaths
    # Total discovery saturation
    vec[idx] = min(float(_total_disc) / 30.0, 1.0)
    idx += 1
    # Discovery rate (discoveries per step, normalised)
    vec[idx] = min(float(_total_disc) / max(float(current_step + 1), 1.0), 1.0)
    idx += 1
    # Resource efficiency: discoveries per GPT call used
    _gpt_used = max(gpt_calls_max - gpt_calls_remaining, 1)
    vec[idx] = min(float(_total_disc) / float(_gpt_used), 1.0)
    idx += 1
    # Flag progress (normalised to 2-flag target)
    vec[idx] = min(float(_n_flags) / 2.0, 1.0)
    idx += 1
    # Acceleration: are we discovering faster than average?
    _expected_disc_at_step = float(current_step + 1) * 0.5  # ~0.5 disc/step baseline
    vec[idx] = max(min((float(_total_disc) - _expected_disc_at_step) / 10.0, 1.0), 0.0)
    idx += 1  # 168 after section 13

    # ─── Remaining dims [168-511] are zero-padded ────────────────────
    # ~168 meaningful dims / 512 total  (33% utilisation, up from 28%)

    return torch.tensor(vec, dtype=torch.float32, device=device)


def encode_state_batch(
    states: List[Dict[str, Any]],
    device: torch.device,
    **kwargs,
) -> torch.Tensor:
    """Encode a batch of states into a (B, 512) tensor."""
    tensors = [encode_state(s, device, **kwargs) for s in states]
    return torch.stack(tensors, dim=0)
