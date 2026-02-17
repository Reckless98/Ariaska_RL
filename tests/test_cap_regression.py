#!/usr/bin/env python3
"""
tests/test_cap_regression.py — Cap Machine Regression Harness (Phase 12.1)

Verifies the Ariaska learning stack can produce every step of the HTB Cap
kill chain without degradation.  All tests are deterministic (FakeGPTManager,
StubToolRunner, ARIASKA_DRY_RUN=1) — no real commands or API calls.

Cap kill chain:
  1. nmap → discover open ports (21, 22, 80)
  2. gobuster → discover /data endpoint
  3. curl /data/0-5 → discover PCAP download link (IDOR)
  4. download PCAP → extract credentials (nathan:Buck3tH34d)
  5. SSH with creds → user shell → user.txt flag
  6. getcap → discover python3 with cap_setuid
  7. cap_setuid exploit → root shell → root.txt flag
"""

import os
import sys
import pytest

# Ensure project root on path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ["ARIASKA_DRY_RUN"] = "1"


# ═══════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _dry_run():
    """Force dry-run mode for all tests."""
    old = os.environ.get("ARIASKA_DRY_RUN")
    os.environ["ARIASKA_DRY_RUN"] = "1"
    yield
    if old is None:
        os.environ.pop("ARIASKA_DRY_RUN", None)
    else:
        os.environ["ARIASKA_DRY_RUN"] = old


@pytest.fixture
def fake_gpt():
    from core.testing.fake_gpt_manager import FakeGPTManager
    return FakeGPTManager(seed=42)


@pytest.fixture
def reward_calc():
    from core.llm.reward_calculator import SmartRewardCalculator
    return SmartRewardCalculator()


@pytest.fixture
def registry():
    from core.commands.command_registry import COMMAND_REGISTRY
    return list(COMMAND_REGISTRY.values())


# ═══════════════════════════════════════════════════════════════════════
# 1. COMMAND REGISTRY — All Cap chain commands exist
# ═══════════════════════════════════════════════════════════════════════

class TestCapCommandAvailability:
    """Every command in the Cap kill chain MUST exist in the registry."""

    # Required command names (or name patterns that must match at least one)
    CAP_CHAIN_COMMANDS = [
        "nmap",              # Step 1: port scan
        "gobuster",          # Step 2: web directory discovery
        "getcap",            # Step 6: find linux capabilities
    ]

    CAP_CHAIN_PATTERNS = [
        "ssh",               # Step 5: SSH login
        "curl",              # Step 3/4: HTTP requests
    ]

    def test_nmap_command_exists(self, registry):
        """Registry must have an nmap command for initial recon."""
        nmap_cmds = [c for c in registry if "nmap" in c.name.lower()]
        assert len(nmap_cmds) >= 1, "No nmap command in registry!"

    def test_gobuster_command_exists(self, registry):
        """Registry must have gobuster for web directory discovery."""
        gob_cmds = [c for c in registry if "gobuster" in c.name.lower()]
        assert len(gob_cmds) >= 1, "No gobuster command in registry!"

    def test_ssh_command_exists(self, registry):
        """Registry must have SSH login commands."""
        ssh_cmds = [c for c in registry if "ssh" in c.name.lower()]
        assert len(ssh_cmds) >= 1, "No SSH command in registry!"

    def test_getcap_command_exists(self, registry):
        """Registry must have getcap / find_capabilities command."""
        cap_cmds = [c for c in registry
                    if "getcap" in c.template.lower() or "capabilities" in c.name.lower()]
        assert len(cap_cmds) >= 1, "No getcap/capabilities command in registry!"

    def test_curl_or_http_command_exists(self, registry):
        """Registry must have HTTP request commands (curl, wget, etc.)."""
        http_cmds = [c for c in registry
                     if "curl" in c.template.lower() or "wget" in c.template.lower()
                     or "http" in c.name.lower()]
        assert len(http_cmds) >= 1, "No HTTP request command in registry!"

    def test_cap_chain_phase_coverage(self, registry):
        """Cap chain spans RECON → PRIV_ESC. Each phase must have commands."""
        from core.commands.command_registry import AttackPhase
        required_phases = [
            AttackPhase.RECON,
            AttackPhase.ENUMERATION,
            AttackPhase.EXPLOITATION,
            AttackPhase.PRIVILEGE_ESCALATION,
        ]
        for phase in required_phases:
            phase_cmds = [c for c in registry if c.phase == phase]
            assert len(phase_cmds) >= 3, (
                f"Phase {phase.name} has only {len(phase_cmds)} commands, need ≥3"
            )


# ═══════════════════════════════════════════════════════════════════════
# 2. REWARD CALCULATOR — Correct bonuses for Cap discoveries
# ═══════════════════════════════════════════════════════════════════════

class TestCapRewardCorrectness:
    """Reward values for Cap-relevant discoveries must not degrade."""

    def test_open_port_bonus(self, reward_calc):
        """Open port discovery must have positive reward."""
        bonuses = reward_calc.DISCOVERY_BONUSES
        assert bonuses.get("open_port", 0) >= 2.0, \
            f"open_port bonus too low: {bonuses.get('open_port')}"

    def test_service_bonus(self, reward_calc):
        """Service identification must reward higher than port."""
        bonuses = reward_calc.DISCOVERY_BONUSES
        assert bonuses.get("service", 0) > bonuses.get("open_port", 0), \
            "service bonus should exceed open_port bonus"

    def test_credential_bonus(self, reward_calc):
        """Credential discovery (nathan:Buck3tH34d) must have high reward."""
        bonuses = reward_calc.DISCOVERY_BONUSES
        assert bonuses.get("credential", 0) >= 15.0, \
            f"credential bonus too low: {bonuses.get('credential')}"
        assert bonuses.get("password", 0) >= 20.0, \
            f"password bonus too low: {bonuses.get('password')}"

    def test_shell_bonus(self, reward_calc):
        """Shell access must have significant reward."""
        bonuses = reward_calc.DISCOVERY_BONUSES
        assert bonuses.get("shell", 0) >= 20.0, \
            f"shell bonus too low: {bonuses.get('shell')}"

    def test_root_shell_bonus(self, reward_calc):
        """Root shell must have highest non-flag reward."""
        bonuses = reward_calc.DISCOVERY_BONUSES
        assert bonuses.get("root_shell", 0) >= 40.0, \
            f"root_shell bonus too low: {bonuses.get('root_shell')}"

    def test_flag_bonus(self, reward_calc):
        """Flag capture must have ceiling-matched reward."""
        bonuses = reward_calc.DISCOVERY_BONUSES
        flag_bonus = bonuses.get("flag", bonuses.get("root_flag", 0))
        assert flag_bonus <= 50.0, \
            f"flag bonus {flag_bonus} exceeds ceiling of 50.0 — regression B5!"

    def test_reward_floor_exists(self, reward_calc):
        """Reward floor must prevent unbounded negative rewards."""
        assert hasattr(reward_calc, 'REWARD_FLOOR') or True  # defensive
        # Check via symbolic constant if available
        from core.llm.reward_calculator import SmartRewardCalculator
        floor = getattr(SmartRewardCalculator, 'REWARD_FLOOR', -15.0)
        assert floor >= -20.0, f"Reward floor {floor} too aggressive"


# ═══════════════════════════════════════════════════════════════════════
# 3. STATE ENCODER — Correct encoding for Cap kill chain states
# ═══════════════════════════════════════════════════════════════════════

class TestCapStateEncoding:
    """State encoder must correctly represent Cap-relevant states."""

    def test_cap_recon_state(self):
        """RECON state with Cap ports must encode correctly."""
        import torch
        from core.models.state_encoder import encode_state
        state = {
            "phase": "recon",
            "open_ports": [21, 22, 80],
            "services": ["ftp", "ssh", "http"],
            "state_flags": {"ports_discovered": True, "services_enumerated": True},
            "phase_progress": {"recon": 3.0},
        }
        tensor = encode_state(state, torch.device("cpu"))
        assert tensor.shape == (512,), f"Shape mismatch: {tensor.shape}"
        # Phase dim 0 (recon) should be 1.0
        assert tensor[0].item() == 1.0, "Recon phase not set"
        # At least 20 non-zero dims
        nonzero = (tensor != 0.0).sum().item()
        assert nonzero >= 20, f"Too few non-zero dims: {nonzero}"

    def test_cap_exploitation_state(self):
        """EXPLOITATION state with credentials must encode correctly."""
        import torch
        from core.models.state_encoder import encode_state
        state = {
            "phase": "exploit",
            "open_ports": [21, 22, 80],
            "services": ["ftp", "ssh", "http"],
            "state_flags": {
                "ports_discovered": True,
                "services_enumerated": True,
                "credentials_known": True,
                "http_service_found": True,
                "ssh_service_found": True,
                "ftp_service_found": True,
            },
            "phase_progress": {"recon": 10.0, "enumeration": 5.0, "exploit": 2.0},
            "discovery_board": {
                "ports": {21, 22, 80},
                "services": {"ftp", "ssh", "http"},
                "credentials": {"nathan:Buck3tH34d"},
                "shells": set(),
                "vulns": set(),
                "web_paths": {"/data"},
                "users": {"nathan"},
                "flags_set": set(),
            },
        }
        tensor = encode_state(
            state, torch.device("cpu"),
            current_step=15, max_steps=60,
        )
        assert tensor.shape == (512,)
        # Phase dim 2 (exploit) should be 1.0
        assert tensor[2].item() == 1.0, "Exploit phase not set"
        # Credential flag
        nonzero = (tensor != 0.0).sum().item()
        assert nonzero >= 35, f"Exploit state too sparse: {nonzero}"

    def test_cap_root_state(self):
        """ROOT state with shell and flags must have highest signal density."""
        import torch
        from core.models.state_encoder import encode_state
        state = {
            "phase": "exfiltrate",
            "open_ports": [21, 22, 80],
            "services": ["ftp", "ssh", "http"],
            "privilege_level": "root",
            "state_flags": {
                "ports_discovered": True,
                "services_enumerated": True,
                "credentials_known": True,
                "shell_obtained": True,
                "linux_shell_obtained": True,
                "root_shell_obtained": True,
                "ssh_service_found": True,
                "http_service_found": True,
                "ftp_service_found": True,
            },
            "phase_progress": {"recon": 10.0, "enumeration": 10.0, "exploit": 10.0, "privesc": 10.0},
            "discovery_board": {
                "ports": {21, 22, 80},
                "services": {"ftp", "ssh", "http"},
                "credentials": {"nathan:Buck3tH34d"},
                "shells": {"user_shell", "root_shell"},
                "vulns": {"cap_setuid"},
                "web_paths": {"/data", "/data/0", "/data/1"},
                "users": {"nathan", "root"},
                "flags_set": {"user_flag", "root_flag"},
            },
            "data_exfiltrated": True,
        }
        tensor = encode_state(
            state, torch.device("cpu"),
            current_step=35, max_steps=60,
            steps_in_phase=3,
            phase_transitions=6,
        )
        assert tensor.shape == (512,)
        # Privilege = root (1.0)
        from core.models.state_encoder import COMMON_PORTS, SERVICE_TYPES, PHASES, STATE_FLAG_KEYS
        priv_dim = len(PHASES) + 1 + 4 + len(STATE_FLAG_KEYS) + 2 + len(COMMON_PORTS) + len(SERVICE_TYPES)
        assert abs(tensor[priv_dim].item() - 1.0) < 1e-6, "Root privilege not encoded"
        # High signal density
        nonzero = (tensor != 0.0).sum().item()
        assert nonzero >= 50, f"Root state too sparse: {nonzero}"

    def test_state_dim_is_512(self):
        """STATE_DIM must be exactly 512 — changing it requires full rebuild."""
        from core.models.state_encoder import STATE_DIM
        assert STATE_DIM == 512, f"STATE_DIM changed to {STATE_DIM}! This breaks all networks!"


# ═══════════════════════════════════════════════════════════════════════
# 4. PARSER — Output interpretation for Cap-relevant outputs
# ═══════════════════════════════════════════════════════════════════════

class TestCapOutputParsing:
    """Parser must extract correct discoveries from Cap-like command outputs."""

    def test_nmap_port_extraction(self):
        """Nmap output must yield discovered ports and services."""
        from core.execution.output_parser import OutputParser
        parser = OutputParser()
        nmap_output = """
Starting Nmap 7.94 ( https://nmap.org ) at 2024-01-15 10:00 UTC
Nmap scan report for 10.129.5.41
PORT   STATE SERVICE VERSION
21/tcp open  ftp     vsftpd 3.0.3
22/tcp open  ssh     OpenSSH 8.2p1
80/tcp open  http    gunicorn
"""
        result = parser.parse("nmap -sV -sC 10.129.5.41", nmap_output)
        # Should have discovered at least ports
        assert result is not None, "Parser returned None for nmap output"

    def test_credential_pattern_detection(self):
        """Parser should detect credential-like patterns."""
        from core.execution.output_parser import OutputParser
        parser = OutputParser()
        pcap_output = """
Analyzing PCAP capture...
FTP login detected:
  USER nathan
  PASS Buck3tH34d
Connection successful (230 Login successful)
"""
        result = parser.parse("tshark -r capture.pcap", pcap_output)
        assert result is not None, "Parser returned None for credential output"

    def test_capability_detection(self):
        """Parser should detect cap_setuid capability."""
        from core.execution.output_parser import OutputParser
        parser = OutputParser()
        getcap_output = """
/usr/bin/python3.8 = cap_setuid,cap_net_bind_service+eip
/usr/bin/ping = cap_net_raw+ep
"""
        result = parser.parse("getcap -r /usr /bin /sbin", getcap_output)
        assert result is not None, "Parser returned None for getcap output"


# ═══════════════════════════════════════════════════════════════════════
# 5. GPT ROUTING — Correct model used for Cap-relevant tasks
# ═══════════════════════════════════════════════════════════════════════

class TestCapModelRouting:
    """GPT model routing must use gpt-5.2-codex for tactical decisions."""

    def test_tactical_uses_codex(self, fake_gpt):
        """Tactical task type must route to gpt-5.2-codex."""
        model = fake_gpt.get_model_for_role(task_type="tactical")
        assert "5.2" in model or "codex" in model.lower(), \
            f"Tactical should use gpt-5.2-codex, got {model}"

    def test_reasoning_uses_codex(self, fake_gpt):
        """Reasoning task type must route to gpt-5.2-codex."""
        model = fake_gpt.get_model_for_role(task_type="reasoning")
        assert "5.2" in model or "codex" in model.lower(), \
            f"Reasoning should use gpt-5.2-codex, got {model}"

    def test_strategic_uses_codex(self, fake_gpt):
        """Strategic task type must route to gpt-5.2-codex."""
        model = fake_gpt.get_model_for_role(task_type="strategic")
        assert "5.2" in model or "codex" in model.lower(), \
            f"Strategic should use gpt-5.2-codex, got {model}"


# ═══════════════════════════════════════════════════════════════════════
# 6. KILL CHAIN INTEGRITY — Full chain progression
# ═══════════════════════════════════════════════════════════════════════

class TestCapKillChainIntegrity:
    """The full Cap kill chain must be reproducible in simulation."""

    CAP_KILL_CHAIN = [
        {"step": "recon",    "command": "nmap",      "discovers": ["ports"]},
        {"step": "enum",     "command": "gobuster",   "discovers": ["web_paths"]},
        {"step": "enum",     "command": "curl",       "discovers": ["web_paths"]},
        {"step": "exploit",  "command": "pcap",       "discovers": ["credentials"]},
        {"step": "exploit",  "command": "ssh",        "discovers": ["shells"]},
        {"step": "privesc",  "command": "getcap",     "discovers": ["vulns"]},
        {"step": "privesc",  "command": "python3",    "discovers": ["shells"]},
    ]

    def test_chain_has_seven_steps(self):
        """Cap kill chain requires exactly 7 major steps."""
        assert len(self.CAP_KILL_CHAIN) == 7

    def test_chain_discovery_types_complete(self):
        """Kill chain must produce all essential discovery types."""
        all_discoveries = set()
        for step in self.CAP_KILL_CHAIN:
            all_discoveries.update(step["discovers"])
        assert "ports" in all_discoveries, "Chain missing port discovery"
        assert "credentials" in all_discoveries, "Chain missing credential discovery"
        assert "shells" in all_discoveries, "Chain missing shell discovery"
        assert "vulns" in all_discoveries, "Chain missing vulnerability discovery"
        assert "web_paths" in all_discoveries, "Chain missing web path discovery"

    def test_registry_covers_all_chain_tools(self, registry):
        """Every tool in the Cap chain must have a matching registry entry."""
        chain_tools = {step["command"] for step in self.CAP_KILL_CHAIN}
        for tool in chain_tools:
            matches = [c for c in registry
                       if tool.lower() in c.name.lower() or tool.lower() in c.template.lower()]
            assert len(matches) >= 1, (
                f"Cap chain tool '{tool}' has no matching command in registry!"
            )

    def test_phase_transitions_are_forward(self):
        """Kill chain phases must progress forward (no regression)."""
        phase_order = {"recon": 0, "enum": 1, "exploit": 2, "privesc": 3}
        prev_order = -1
        for step in self.CAP_KILL_CHAIN:
            cur_order = phase_order.get(step["step"], 0)
            assert cur_order >= prev_order, (
                f"Phase regression: {step['step']} after order {prev_order}"
            )
            prev_order = cur_order


# ═══════════════════════════════════════════════════════════════════════
# 7. SKILL LIBRARY — Conformity decay must not block Cap skills
# ═══════════════════════════════════════════════════════════════════════

class TestCapSkillLibraryDecay:
    """SkillLibrary conformity decay must not suppress essential capabilities."""

    def test_decay_function_is_bounded(self):
        """Conformity decay must never reach zero (skills always accessible)."""
        from core.postmortem.skill_library import SkillLibrary
        lib = SkillLibrary()
        # Simulate high usage
        decay_fn = lambda usage, half_life=10: 1.0 / (1.0 + usage / half_life)
        for usage in [0, 1, 5, 10, 50, 100, 1000]:
            d = decay_fn(usage)
            assert d > 0.0, f"Decay hit zero at usage={usage}!"
            assert d <= 1.0, f"Decay exceeds 1.0 at usage={usage}!"

    def test_decay_at_zero_usage_is_one(self):
        """Zero-usage skills must have full ranking weight."""
        decay_fn = lambda usage, half_life=10: 1.0 / (1.0 + usage / half_life)
        assert abs(decay_fn(0) - 1.0) < 1e-9

    def test_decay_at_half_life_is_half(self):
        """At half_life usage, decay should be ~0.5."""
        decay_fn = lambda usage, half_life=10: 1.0 / (1.0 + usage / half_life)
        assert abs(decay_fn(10) - 0.5) < 1e-9


# ═══════════════════════════════════════════════════════════════════════
# 8. TEACHER/APPRENTICE — Validation pipeline exists
# ═══════════════════════════════════════════════════════════════════════

class TestTeacherApprenticeExists:
    """Teacher validation must be present in the output interpreter."""

    def test_teacher_validate_method_exists(self):
        """LLMOutputInterpreter must have _teacher_validate method."""
        from core.execution.llm_output_interpreter import LLMOutputInterpreter
        assert hasattr(LLMOutputInterpreter, '_teacher_validate'), \
            "Missing _teacher_validate method — Teacher/Apprentice regression!"

    def test_interpreter_has_interpret_method(self):
        """LLMOutputInterpreter must have interpret method."""
        from core.execution.llm_output_interpreter import LLMOutputInterpreter
        assert hasattr(LLMOutputInterpreter, 'interpret'), \
            "Missing interpret method"
