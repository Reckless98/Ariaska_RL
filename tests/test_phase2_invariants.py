"""
Phase 2 Comprehensive Tests for Ariaska_RL

Tests all Phase 2A/2B/2C fixes:
1. Red double env.step() bug fix
2. Smart agent activation schedule
3. Triple advise_phase() removal
4. Blue GPT call cap
5. Token budget per episode
6. Sandboxed executor validation
7. Output parser correctness
8. Adaptive phase thresholds
9. Integration: SmartOrchestrator full step

Run with: .venv/bin/python -m pytest tests/test_phase2_invariants.py -v
"""

import os
import re
import inspect
import pytest


# ===========================================================================
# Phase 2A: Correctness & Efficiency
# ===========================================================================

class TestRedAgentNoEnvStep:
    """Red's simulate_step() must NOT call env.step()."""

    def test_red_simulate_step_does_not_call_env_step(self):
        """
        Phase 2A Fix #1: Red's simulate_step must NOT call self.env.step().
        The orchestrator is the single stepper.
        """
        from core.agents.red_agent import RedAgent

        source = inspect.getsource(RedAgent.simulate_step)

        # There should be NO 'self.env.step(' in the simulate_step body
        # (except in comments or strings — we check actual calls)
        lines = source.split('\n')
        env_step_calls = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Skip comments and strings
            if stripped.startswith('#'):
                continue
            if stripped.startswith(('"""', "'''")):
                continue
            if 'self.env.step(' in stripped or 'env.step(' in stripped:
                env_step_calls.append((i, stripped))

        assert len(env_step_calls) == 0, (
            f"Red's simulate_step must NOT call env.step(). "
            f"Found {len(env_step_calls)} call(s): {env_step_calls}"
        )

    def test_red_simulate_step_returns_result_dict(self):
        """Red's simulate_step returns a dict with expected keys."""
        from core.multiagent.agent_manager import AgentManager

        am = AgentManager(verbosity="silent")
        red = am.red_agent

        result = red.simulate_step(episode=1, step=1, shared_context={"phase": "recon"})

        assert isinstance(result, dict), "simulate_step must return a dict"
        # Should have at minimum: command and phase
        assert "command" in result or "action" in result, (
            "Result must contain 'command' or 'action'"
        )


class TestSmartAgentActivation:
    """SmartOrchestrator should gate agents per phase."""

    def test_should_activate_method_exists(self):
        """_should_activate must exist on SmartOrchestrator."""
        from core.orchestration.smart_orchestrator import SmartOrchestrator

        assert hasattr(SmartOrchestrator, '_should_activate'), (
            "SmartOrchestrator must have _should_activate method"
        )

    def test_activation_schedule_recon(self):
        """In RECON phase, ScoutAgent and RedAgent run every step, OrionAgent every 3."""
        from core.orchestration.smart_orchestrator import SmartOrchestrator

        schedule = SmartOrchestrator.AGENT_ACTIVATION_SCHEDULE.get("RECON", {})
        assert schedule.get("ScoutAgent") == 1, "Scout should activate every step in RECON"
        assert schedule.get("RedAgent") == 1, "Red should activate every step in RECON"
        assert schedule.get("OrionAgent") == 3, "Orion should activate every 3 steps in RECON"

    def test_activation_schedule_exploitation(self):
        """In EXPLOITATION, RedAgent and BlueAgent lead (every step), ScoutAgent every 3."""
        from core.orchestration.smart_orchestrator import SmartOrchestrator

        schedule = SmartOrchestrator.AGENT_ACTIVATION_SCHEDULE.get("EXPLOITATION", {})
        assert schedule.get("RedAgent") == 1, "Red should activate every step in EXPLOITATION"
        assert schedule.get("BlueAgent") == 1, "Blue should activate every step in EXPLOITATION"
        assert schedule.get("ScoutAgent") == 3, "Scout should activate every 3 steps in EXPLOITATION"

    def test_should_activate_logic(self):
        """_should_activate correctly gates based on step and phase."""
        from core.orchestration.smart_orchestrator import SmartOrchestrator
        from core.environment.cyber_environment import CyberEnvironment
        from core.gpt_manager import GPTManager

        env = CyberEnvironment(defer_reset=True)
        gpt = GPTManager()
        orch = SmartOrchestrator(env=env, gpt_manager=gpt, verbosity="silent")

        # In RECON, ScoutAgent (freq=1) should activate every step
        # Note: _should_activate uses (step+1) % freq == 0
        assert orch._should_activate("ScoutAgent", 0, "RECON") is True
        assert orch._should_activate("ScoutAgent", 1, "RECON") is True
        assert orch._should_activate("ScoutAgent", 5, "RECON") is True

        # In RECON, OrionAgent (freq=3) should activate every 3 steps
        # (step+1) % 3 == 0: step 2 (3%3=0), step 5 (6%3=0), step 8 (9%3=0)
        assert orch._should_activate("OrionAgent", 2, "RECON") is True
        assert orch._should_activate("OrionAgent", 0, "RECON") is False  # (0+1)%3=1≠0
        assert orch._should_activate("OrionAgent", 5, "RECON") is True   # (5+1)%3=0
        assert orch._should_activate("OrionAgent", 3, "RECON") is False  # (3+1)%3=1≠0


class TestTripleAdvisePhaseRemoval:
    """Red's simulate_step should not call scout.advise_phase()."""

    def test_red_simulate_step_no_advise_phase(self):
        """simulate_step must not call advise_phase directly."""
        from core.agents.red_agent import RedAgent

        source = inspect.getsource(RedAgent.simulate_step)
        lines = source.split('\n')
        advise_calls = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('#'):
                continue
            if 'advise_phase(' in stripped:
                advise_calls.append((i, stripped))

        assert len(advise_calls) == 0, (
            f"Red's simulate_step should NOT call advise_phase(). "
            f"Found {len(advise_calls)} call(s): {advise_calls}"
        )

    def test_blue_simulate_step_no_advise_phase(self):
        """Blue's simulate_step must not call advise_phase directly."""
        from core.agents.blue_agent import BlueAgent

        source = inspect.getsource(BlueAgent.simulate_step)
        lines = source.split('\n')
        advise_calls = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('#'):
                continue
            if 'advise_phase(' in stripped:
                advise_calls.append((i, stripped))

        assert len(advise_calls) == 0, (
            f"Blue's simulate_step should NOT call advise_phase(). "
            f"Found {len(advise_calls)} call(s): {advise_calls}"
        )


class TestBlueGPTCallCap:
    """Blue GPT calls should be capped per episode."""

    def test_blue_has_gpt_call_counter(self):
        """BlueAgent must have gpt_calls_this_episode attribute."""
        from core.agents.blue_agent import BlueAgent

        blue = BlueAgent(verbosity="silent")
        assert hasattr(blue, 'gpt_calls_this_episode'), (
            "BlueAgent must track GPT calls per episode"
        )
        assert hasattr(blue, 'gpt_call_limit'), (
            "BlueAgent must have a GPT call limit"
        )

    def test_blue_gpt_call_limit_exists(self):
        """Blue must have a reasonable GPT call limit."""
        from core.agents.blue_agent import BlueAgent

        blue = BlueAgent(verbosity="silent")
        assert blue.gpt_call_limit > 0, "GPT call limit must be positive"
        assert blue.gpt_call_limit <= 20, "GPT call limit should be reasonable (<= 20)"


# ===========================================================================
# Phase 2B: Metasploitable Live Execution Layer
# ===========================================================================

class TestSandboxedExecutor:
    """Test the sandboxed command executor."""

    def test_executor_import(self):
        """SandboxedExecutor can be imported."""
        from core.execution.sandboxed_executor import SandboxedExecutor
        assert SandboxedExecutor is not None

    def test_executor_simulated_mode(self):
        """Simulated mode returns result without real execution."""
        from core.execution.sandboxed_executor import SandboxedExecutor

        executor = SandboxedExecutor(mode="simulated")
        result = executor.execute("nmap -sV 10.10.10.10")

        assert not result.blocked, "Valid command should not be blocked"
        assert result.mode == "simulated"
        assert "SIMULATED" in result.stdout

    def test_executor_blocks_dangerous_commands(self):
        """Dangerous commands (rm, shutdown, etc.) are blocked."""
        from core.execution.sandboxed_executor import SandboxedExecutor

        executor = SandboxedExecutor(mode="live", allowed_targets=["10.10.10.10"])

        # rm should be blocked
        result = executor.execute("rm -rf /")
        assert result.blocked, "rm -rf should be blocked"

        # shutdown should be blocked
        result = executor.execute("shutdown -h now")
        assert result.blocked, "shutdown should be blocked"

    def test_executor_blocks_out_of_scope(self):
        """Commands targeting IPs outside scope are blocked in live mode."""
        from core.execution.sandboxed_executor import SandboxedExecutor

        executor = SandboxedExecutor(
            mode="live",
            allowed_targets={"192.168.56.101"}
        )

        # In-scope should pass
        result = executor.execute("nmap 192.168.56.101")
        assert not result.blocked, "In-scope target should not be blocked"

        # Out-of-scope should be blocked
        result = executor.execute("nmap 192.168.56.200")
        assert result.blocked, "Out-of-scope target should be blocked"

    def test_executor_blocks_localhost(self):
        """Localhost/self targeting is always blocked."""
        from core.execution.sandboxed_executor import SandboxedExecutor

        executor = SandboxedExecutor(mode="live", allowed_targets={"10.10.10.10"})

        result = executor.execute("nmap 127.0.0.1")
        assert result.blocked, "localhost should be blocked"

    def test_executor_dry_run(self):
        """Dry run mode validates but does not execute."""
        from core.execution.sandboxed_executor import SandboxedExecutor

        executor = SandboxedExecutor(mode="dry_run")
        result = executor.execute("nmap 10.10.10.10")

        assert not result.blocked
        assert result.mode == "dry_run"
        assert "DRY_RUN" in result.stdout

    def test_executor_rate_limiting(self):
        """Rate limiter prevents command flooding."""
        from core.execution.sandboxed_executor import SandboxedExecutor

        executor = SandboxedExecutor(mode="simulated", rate_limit_per_minute=3)

        # First 3 should pass
        for _ in range(3):
            result = executor.execute("echo test")
            assert not result.blocked

        # 4th should be rate-limited
        result = executor.execute("echo test")
        assert result.blocked, "Should be rate-limited after 3 commands"
        assert "Rate limit" in result.block_reason

    def test_executor_stats(self):
        """Stats tracking works correctly."""
        from core.execution.sandboxed_executor import SandboxedExecutor

        executor = SandboxedExecutor(mode="simulated")
        executor.execute("nmap 10.10.10.10")
        executor.execute("rm -rf /")  # blocked

        stats = executor.get_stats()
        assert stats["total_executions"] == 2
        assert stats["total_blocked"] == 1


class TestOutputParser:
    """Test the real output parser."""

    def test_parser_import(self):
        """OutputParser can be imported."""
        from core.execution.output_parser import OutputParser
        assert OutputParser is not None

    def test_parse_nmap_output(self):
        """Parser correctly extracts ports and services from nmap output."""
        from core.execution.output_parser import OutputParser

        parser = OutputParser()
        nmap_output = """
Starting Nmap 7.94 ( https://nmap.org ) at 2026-02-05 10:00 UTC
Nmap scan report for 192.168.56.101
Host is up (0.001s latency).

PORT     STATE SERVICE     VERSION
21/tcp   open  ftp         vsftpd 2.3.4
22/tcp   open  ssh         OpenSSH 4.7p1 Debian 8ubuntu1
80/tcp   open  http        Apache httpd 2.2.8 ((Ubuntu) DAV/2)
445/tcp  open  netbios-ssn Samba smbd 3.X - 4.X

Nmap done: 1 IP address (1 host up) scanned in 2.34 seconds
"""
        result = parser.parse("nmap -sV 192.168.56.101", nmap_output)

        assert result.success, "Nmap output with open ports should be success"
        assert 21 in result.open_ports, "Should find port 21"
        assert 22 in result.open_ports, "Should find port 22"
        assert 80 in result.open_ports, "Should find port 80"
        assert 445 in result.open_ports, "Should find port 445"
        assert len(result.open_ports) == 4
        assert "ftp" in result.services.get(21, "")
        assert "ssh" in result.services.get(22, "")

    def test_parse_hydra_output(self):
        """Parser correctly extracts credentials from hydra output."""
        from core.execution.output_parser import OutputParser

        parser = OutputParser()
        hydra_output = """
Hydra v9.5 starting
[DATA] attacking ssh://192.168.56.101:22/
[22][ssh] host: 192.168.56.101 login: admin password: admin123
[22][ssh] host: 192.168.56.101 login: root password: toor
1 of 1 target successfully completed, 2 valid passwords found
"""
        result = parser.parse("hydra -l admin -P passwords.txt ssh://192.168.56.101", hydra_output)

        assert result.success
        assert len(result.credentials) == 2
        assert result.credentials[0]["username"] == "admin"
        assert result.credentials[0]["password"] == "admin123"

    def test_parse_gobuster_output(self):
        """Parser correctly extracts web paths from gobuster output."""
        from core.execution.output_parser import OutputParser

        parser = OutputParser()
        gobuster_output = """
/admin (Status: 200, Size: 3456)
/login (Status: 200, Size: 1234)
/backup (Status: 403, Size: 0)
/api (Status: 200, Size: 567)
"""
        result = parser.parse("gobuster dir -u http://target -w wordlist.txt", gobuster_output)

        assert result.success
        assert "/admin" in result.web_paths
        assert "/login" in result.web_paths

    def test_parse_metasploit_session(self):
        """Parser correctly detects Metasploit sessions."""
        from core.execution.output_parser import OutputParser

        parser = OutputParser()
        msf_output = """
[*] Started reverse TCP handler on 192.168.56.1:4444
[*] Sending exploit...
[*] Meterpreter session 1 opened (192.168.56.1:4444 -> 192.168.56.101:45678)
"""
        result = parser.parse("msfconsole -x 'use exploit/...'", msf_output)

        assert result.success
        assert len(result.sessions) == 1
        assert result.sessions[0]["type"] == "Meterpreter"
        assert result.sessions[0]["id"] == 1

    def test_parse_enum4linux_users(self):
        """Parser correctly extracts users from enum4linux output."""
        from core.execution.output_parser import OutputParser

        parser = OutputParser()
        enum_output = """
[+] Target: 192.168.56.101
[+] RID cycling...
user:[Administrator] rid:[0x1f4]
user:[Guest] rid:[0x1f5]
user:[backup_svc] rid:[0x3e8]
"""
        result = parser.parse("enum4linux 192.168.56.101", enum_output)

        assert result.success
        assert "Administrator" in result.users
        assert "backup_svc" in result.users

    def test_parse_empty_output(self):
        """Empty output returns non-success result."""
        from core.execution.output_parser import OutputParser

        parser = OutputParser()
        result = parser.parse("nmap 10.10.10.10", "")

        assert not result.success
        assert result.error

    def test_discovery_count(self):
        """Discovery count correctly sums all findings."""
        from core.execution.output_parser import ParsedOutput

        result = ParsedOutput(
            command="test",
            open_ports=[22, 80, 443],
            services={22: "ssh", 80: "http"},
            users=["admin", "root"],
            os_info="Linux",
        )
        assert result.discovery_count == 8  # 3 ports + 2 services + 2 users + 1 OS


class TestAdaptivePhaseThresholds:
    """Test configurable phase transition thresholds."""

    def test_default_thresholds(self):
        """Default thresholds are set correctly."""
        from core.environment.cyber_environment import CyberEnvironment

        env = CyberEnvironment(defer_reset=True)
        assert env.phase_transitions["recon"]["threshold"] == 10
        assert env.phase_transitions["enumeration"]["threshold"] == 8

    def test_set_target_profile_metasploitable2(self):
        """Metasploitable 2 profile sets appropriate thresholds."""
        from core.environment.cyber_environment import CyberEnvironment

        env = CyberEnvironment(defer_reset=True)
        env.set_target_profile("metasploitable2")

        assert env.phase_transitions["recon"]["threshold"] == 5
        assert env.phase_transitions["enumeration"]["threshold"] == 4
        assert env.phase_transitions["exploit"]["threshold"] == 2

    def test_set_custom_thresholds(self):
        """Custom thresholds can be set."""
        from core.environment.cyber_environment import CyberEnvironment

        env = CyberEnvironment(defer_reset=True)
        env.set_phase_thresholds({"recon": 3, "exploit": 1})

        assert env.phase_transitions["recon"]["threshold"] == 3
        assert env.phase_transitions["exploit"]["threshold"] == 1
        # Unchanged phase should keep default
        assert env.phase_transitions["enumeration"]["threshold"] == 8

    def test_invalid_profile_raises(self):
        """Invalid profile name raises ValueError."""
        from core.environment.cyber_environment import CyberEnvironment

        env = CyberEnvironment(defer_reset=True)
        with pytest.raises(ValueError):
            env.set_target_profile("nonexistent_profile")

    def test_target_profiles_exist(self):
        """All expected profiles are defined."""
        from core.environment.cyber_environment import CyberEnvironment

        env = CyberEnvironment(defer_reset=True)
        expected = {"default", "metasploitable2", "metasploitable3", "htb_easy", "htb_hard"}
        assert expected.issubset(set(env.TARGET_PROFILES.keys()))


# ===========================================================================
# Phase 2C: Integration Tests
# ===========================================================================

class TestSmartOrchestratorIntegration:
    """Integration tests for the SmartOrchestrator with Phase 2 fixes."""

    def test_orchestrator_initializes(self):
        """SmartOrchestrator can be initialized without errors."""
        from core.orchestration.smart_orchestrator import SmartOrchestrator
        from core.environment.cyber_environment import CyberEnvironment
        from core.gpt_manager import GPTManager

        env = CyberEnvironment(defer_reset=True)
        gpt = GPTManager()
        orch = SmartOrchestrator(env=env, gpt_manager=gpt, verbosity="silent")

        assert len(orch.agents) > 0, "Should have agents initialized"
        assert len(orch.coaches) > 0, "Should have coaches initialized"

    def test_orchestrator_has_activation_schedule(self):
        """SmartOrchestrator has phase-based activation schedule."""
        from core.orchestration.smart_orchestrator import SmartOrchestrator

        assert hasattr(SmartOrchestrator, 'AGENT_ACTIVATION_SCHEDULE')
        assert "RECON" in SmartOrchestrator.AGENT_ACTIVATION_SCHEDULE
        assert "EXPLOITATION" in SmartOrchestrator.AGENT_ACTIVATION_SCHEDULE

    def test_orchestrator_agent_order(self):
        """Phase-specific agent ordering works."""
        from core.orchestration.smart_orchestrator import SmartOrchestrator
        from core.environment.cyber_environment import CyberEnvironment
        from core.gpt_manager import GPTManager

        env = CyberEnvironment(defer_reset=True)
        gpt = GPTManager()
        orch = SmartOrchestrator(env=env, gpt_manager=gpt, verbosity="silent")

        recon_order = orch.get_optimal_agent_order("RECON")
        exploit_order = orch.get_optimal_agent_order("EXPLOITATION")

        # In RECON, Scout should be first
        if "ScoutAgent" in recon_order:
            assert recon_order[0] == "ScoutAgent", "Scout should lead in RECON"

        # In EXPLOITATION, Red should be first
        if "RedAgent" in exploit_order:
            assert exploit_order[0] == "RedAgent", "Red should lead in EXPLOITATION"


class TestPhase0InvariantsStillHold:
    """Verify Phase 0 fixes are still intact after Phase 2 changes."""

    def test_env_still_shared(self):
        """Red and Blue still share the same environment."""
        from core.multiagent.agent_manager import AgentManager

        am = AgentManager(verbosity="silent")
        assert am.red_agent.env is am.blue_agent.env

    def test_blue_still_has_react_to_action(self):
        """BlueAgent still has react_to_action method."""
        from core.agents.blue_agent import BlueAgent

        assert hasattr(BlueAgent, 'react_to_action')

    def test_blue_simulate_step_still_no_env_step(self):
        """Blue's simulate_step still does NOT call env.step."""
        from core.agents.blue_agent import BlueAgent
        import ast

        source = inspect.getsource(BlueAgent.simulate_step)
        # Use AST to find actual function calls, not comments/docstrings
        # Simpler approach: filter out docstrings and comments properly
        lines = source.split('\n')
        in_docstring = False
        env_step_calls = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Track triple-quoted docstrings
            if '"""' in stripped or "'''" in stripped:
                count = stripped.count('"""') + stripped.count("'''")
                if count == 1:
                    in_docstring = not in_docstring
                continue
            if in_docstring:
                continue
            if stripped.startswith('#'):
                continue
            # Check for actual env.step( calls (not in comments/strings)
            if 'self.env.step(' in stripped:
                env_step_calls.append((i, stripped))

        assert len(env_step_calls) == 0, (
            f"Blue's simulate_step still must NOT call env.step(). "
            f"Found: {env_step_calls}"
        )

    def test_alert_scale_0_to_100(self):
        """Blue team alert still uses 0-100 scale."""
        from core.environment.cyber_environment import CyberEnvironment

        env = CyberEnvironment(defer_reset=True)
        state = env.reset()
        alert = state.get("blue_team_alert", 0)
        assert 0 <= alert <= 100, f"Alert {alert} outside 0-100 range"


class TestDiscoveryToStateFlagBridge:
    """Phase 2A: Discoveries from parsed output must advance AttackContext state flags."""

    def test_credential_discovery_sets_flag(self):
        """Credential pattern in hydra output should set credentials_known."""
        from core.orchestration.smart_orchestrator import SmartOrchestrator
        from core.environment.cyber_environment import CyberEnvironment
        from core.gpt_manager import GPTManager

        env = CyberEnvironment(defer_reset=True)
        gpt = GPTManager()
        orch = SmartOrchestrator(env=env, gpt_manager=gpt, verbosity="silent")
        orch.init_attack("10.10.10.10")

        # Parse hydra-like output
        hydra_output = "[22][ssh] host: 10.10.10.10 login: admin password: admin123"
        discoveries = orch._parse_output_for_discoveries(hydra_output)

        assert "credential" in discoveries, "Should detect credentials in hydra output"

    def test_shell_discovery_sets_flag(self):
        """Shell pattern in output should set shell_obtained."""
        from core.orchestration.smart_orchestrator import SmartOrchestrator
        from core.environment.cyber_environment import CyberEnvironment
        from core.gpt_manager import GPTManager

        env = CyberEnvironment(defer_reset=True)
        gpt = GPTManager()
        orch = SmartOrchestrator(env=env, gpt_manager=gpt, verbosity="silent")
        orch.init_attack("10.10.10.10")

        # Parse meterpreter-like output
        msf_output = "Meterpreter session 1 opened (10.10.14.2:4444 -> 10.10.10.10:8080)\nmeterpreter >"
        discoveries = orch._parse_output_for_discoveries(msf_output)

        assert discoveries.get("shell") is True, "Should detect shell in meterpreter output"

    def test_service_discovery_triggers_enumeration(self):
        """Service discovery should set state flags that enable phase advancement."""
        from core.llm.smart_mentor import AttackContext
        from core.commands.command_registry import AttackPhase

        ctx = AttackContext(target="10.10.10.10")
        assert ctx.current_phase == AttackPhase.RECON

        # Adding HTTP service should advance to ENUMERATION via state flag
        ctx.add_service("http", 80)
        assert ctx.state_flags.get("http_service_found") is True
        assert ctx.current_phase == AttackPhase.ENUMERATION

    def test_credentials_advance_to_exploitation(self):
        """Setting credentials_known should advance to EXPLOITATION phase."""
        from core.llm.smart_mentor import AttackContext
        from core.commands.command_registry import AttackPhase

        ctx = AttackContext(target="10.10.10.10")
        ctx.add_service("ssh", 22)  # Move to ENUMERATION first
        assert ctx.current_phase == AttackPhase.ENUMERATION

        ctx.set_state_flag("credentials_known")
        assert ctx.current_phase == AttackPhase.EXPLOITATION

    def test_shell_advances_to_privesc(self):
        """Setting shell_obtained should advance to PRIVILEGE_ESCALATION."""
        from core.llm.smart_mentor import AttackContext
        from core.commands.command_registry import AttackPhase

        ctx = AttackContext(target="10.10.10.10")
        ctx.set_state_flag("credentials_known")  # EXPLOITATION
        ctx.set_state_flag("shell_obtained")       # Should → PRIVESC
        assert ctx.current_phase == AttackPhase.PRIVILEGE_ESCALATION

    def test_simulated_output_has_credential_patterns(self):
        """Simulated outputs for hydra/crackmapexec include parseable credential patterns."""
        from core.orchestration.smart_orchestrator import SmartOrchestrator
        from core.environment.cyber_environment import CyberEnvironment
        from core.gpt_manager import GPTManager

        env = CyberEnvironment(defer_reset=True)
        gpt = GPTManager()
        orch = SmartOrchestrator(env=env, gpt_manager=gpt, verbosity="silent")
        orch.init_attack("10.10.10.10")

        # hydra output should contain login/password patterns
        hydra_out = orch._generate_simulated_output("hydra -l admin -P /tmp/pass.txt ssh://10.10.10.10")
        discoveries = orch._parse_output_for_discoveries(hydra_out)
        assert "credential" in discoveries, f"hydra sim output should yield credentials. Output: {hydra_out[:100]}"

    def test_simulated_output_has_shell_patterns(self):
        """Simulated outputs for metasploit include shell detection patterns."""
        from core.orchestration.smart_orchestrator import SmartOrchestrator
        from core.environment.cyber_environment import CyberEnvironment
        from core.gpt_manager import GPTManager

        env = CyberEnvironment(defer_reset=True)
        gpt = GPTManager()
        orch = SmartOrchestrator(env=env, gpt_manager=gpt, verbosity="silent")
        orch.init_attack("10.10.10.10")

        msf_out = orch._generate_simulated_output("msfconsole -q -x 'use exploit/unix/ftp/vsftpd_234_backdoor'")
        discoveries = orch._parse_output_for_discoveries(msf_out)
        assert discoveries.get("shell") is True, f"msfconsole sim output should yield shell. Output: {msf_out[:100]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
