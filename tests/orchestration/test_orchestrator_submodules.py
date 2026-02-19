"""Tests for orchestration submodules (Phase 41 — A2).

Covers: output_parser, simulated_output, state_builder, episode_runner.
"""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


# ── Imports ────────────────────────────────────────────────────────

class TestImports:
    """Verify all submodules import cleanly."""

    def test_import_output_parser(self) -> None:
        from core.orchestration.output_parser import parse_all, DiscoveryResult
        assert callable(parse_all)
        assert DiscoveryResult is not None

    def test_import_simulated_output(self) -> None:
        from core.orchestration.simulated_output import (
            generate_simulated_output,
            get_command_category,
        )
        assert callable(generate_simulated_output)
        assert callable(get_command_category)

    def test_import_state_builder(self) -> None:
        from core.orchestration.state_builder import (
            build_default_state,
            build_state_from_board,
            compute_discoveries_delta,
            StateSnapshot,
        )
        assert callable(build_default_state)
        assert callable(build_state_from_board)
        assert callable(compute_discoveries_delta)
        assert StateSnapshot is not None

    def test_import_episode_runner(self) -> None:
        from core.orchestration.episode_runner import (
            StepRecord,
            EpisodeMetrics,
            StuckDetector,
            EpisodeTracker,
        )
        assert StepRecord is not None
        assert EpisodeMetrics is not None
        assert StuckDetector is not None
        assert EpisodeTracker is not None


# ── OutputParser ──────────────────────────────────────────────────

class TestOutputParser:
    """Test DiscoveryResult and parse_* functions."""

    def test_parse_ports_nmap(self) -> None:
        from core.orchestration.output_parser import parse_ports
        output = "22/tcp   open  ssh\n80/tcp   open  http\n443/tcp  open  https"
        ports = parse_ports(output)
        assert 22 in ports
        assert 80 in ports
        assert 443 in ports

    def test_parse_services(self) -> None:
        from core.orchestration.output_parser import parse_services
        output = "22/tcp open  ssh  OpenSSH 7.4\n80/tcp open  http Apache/2.4"
        services = parse_services(output)
        assert len(services) >= 1

    def test_parse_credentials(self) -> None:
        from core.orchestration.output_parser import parse_credentials
        output = "[80][http-get] host: 10.10.10.1   login: admin   password: secret123"
        creds = parse_credentials(output)
        assert len(creds) >= 1
        assert any("admin" in c for c in creds)

    def test_parse_users(self) -> None:
        from core.orchestration.output_parser import parse_users
        output = "uid=0(root) gid=0(root)\nuid=1000(user) gid=1000(user)"
        users = parse_users(output)
        # "root" is in the skip set, so only "user" should be returned
        assert "user" in users
        assert "root" not in users

    def test_parse_web_paths(self) -> None:
        from core.orchestration.output_parser import parse_web_paths
        output = "200      123l    /admin\nStatus: 200  /login\n404      10l     /test"
        paths = parse_web_paths(output)
        assert "/admin" in paths
        assert "/login" in paths

    def test_parse_cves(self) -> None:
        from core.orchestration.output_parser import parse_cves
        output = "CVE-2021-44228 Log4Shell\nCVE-2017-0144 EternalBlue"
        cves = parse_cves(output)
        assert "CVE-2021-44228" in cves
        assert "CVE-2017-0144" in cves

    def test_parse_flags(self) -> None:
        from core.orchestration.output_parser import parse_flags
        output = "user.txt: abc123def456\nHTB{s0me_fl4g_here}"
        flags = parse_flags(output)
        assert len(flags) >= 1

    def test_parse_hashes(self) -> None:
        from core.orchestration.output_parser import parse_hashes
        output = "Administrator:500:aad3b435b51404eeaad3b435b51404ee:e02bc503339d51f71d913c245d35b50b"
        hashes = parse_hashes(output)
        assert len(hashes) >= 1

    def test_detect_shell(self) -> None:
        from core.orchestration.output_parser import detect_shell
        assert detect_shell("$ id\nuid=0(root)")
        assert detect_shell("meterpreter > sysinfo")
        assert not detect_shell("No route to host")

    def test_parse_all(self) -> None:
        from core.orchestration.output_parser import parse_all
        output = "22/tcp open ssh\n80/tcp open http\nCVE-2021-44228"
        result = parse_all(output)
        assert 22 in result.ports
        assert 80 in result.ports
        assert "CVE-2021-44228" in result.cves

    def test_discovery_result_merge(self) -> None:
        from core.orchestration.output_parser import DiscoveryResult
        result = DiscoveryResult(ports={22, 80}, services={"ssh", "http"})
        board: dict = {"ports": set(), "services": set(), "credentials": set()}
        result.merge_into_board(board)
        assert 22 in board["ports"]
        assert "ssh" in board["services"]

    def test_discovery_result_to_dict(self) -> None:
        from core.orchestration.output_parser import DiscoveryResult
        result = DiscoveryResult(ports={22}, cves={"CVE-2021-1234"})
        d = result.to_dict()
        assert 22 in d["ports"]
        assert "CVE-2021-1234" in d["cves"]


# ── SimulatedOutput ───────────────────────────────────────────────

class TestSimulatedOutput:
    """Test simulated output generation."""

    def test_get_command_category(self) -> None:
        from core.orchestration.simulated_output import get_command_category
        assert get_command_category("nmap -sV 10.10.10.1") == "recon"
        assert get_command_category("gobuster dir -u http://x") == "web"
        assert get_command_category("hydra -l admin ssh://x") == "brute"

    def test_generate_simulated_nmap(self) -> None:
        from core.orchestration.simulated_output import generate_simulated_output
        output = generate_simulated_output("nmap -sV 10.10.10.1")
        assert output  # non-empty
        assert isinstance(output, str)

    def test_generate_simulated_gobuster(self) -> None:
        from core.orchestration.simulated_output import generate_simulated_output
        output = generate_simulated_output("gobuster dir -u http://10.1.1.1/")
        assert isinstance(output, str)
        assert output

    def test_generate_simulated_unknown(self) -> None:
        from core.orchestration.simulated_output import generate_simulated_output
        output = generate_simulated_output("totally_unknown_tool --help")
        assert isinstance(output, str)

    def test_should_succeed(self) -> None:
        import random
        from core.orchestration.simulated_output import should_succeed
        # With seeded rng, result is deterministic
        rng = random.Random(42)
        results = [should_succeed("recon", rng=rng) for _ in range(5)]
        assert all(isinstance(r, bool) for r in results)


# ── StateBuilder ──────────────────────────────────────────────────

class TestStateBuilder:
    """Test state building utilities."""

    def test_build_default_state(self) -> None:
        from core.orchestration.state_builder import build_default_state
        state = build_default_state("10.10.10.1")
        assert state["target_ip"] == "10.10.10.1"
        assert state["phase"] == "recon"
        assert state["open_ports"] == []

    def test_state_snapshot_to_dict(self) -> None:
        from core.orchestration.state_builder import StateSnapshot
        snap = StateSnapshot(phase="EXPLOITATION", target_ip="1.2.3.4", open_ports=[22, 80])
        d = snap.to_dict()
        assert d["phase"] == "EXPLOITATION"
        assert 22 in d["open_ports"]

    def test_build_state_from_board(self) -> None:
        from core.orchestration.state_builder import build_state_from_board
        board = {
            "ports": {22, 80, 443},
            "services": {"ssh", "http"},
            "credentials": {("admin", "password123")},
            "shells": set(),
            "vulns": {"CVE-2021-1234"},
            "users": {"root", "user"},
            "web_paths": {"/admin"},
            "flags_set": set(),
        }
        state = build_state_from_board(board, phase="ENUMERATION", target_ip="10.10.10.1")
        assert state["phase"] == "enumeration"
        assert 22 in state["open_ports"]
        assert state["credentials_found"] == 1
        assert state["ports_discovered"] is True
        assert state["shell_obtained"] is False

    def test_build_state_with_shell(self) -> None:
        from core.orchestration.state_builder import build_state_from_board
        board = {
            "ports": {22},
            "services": {"ssh"},
            "credentials": set(),
            "shells": {"root@target"},
            "vulns": set(),
            "users": set(),
            "web_paths": set(),
            "flags_set": set(),
        }
        state = build_state_from_board(board, phase="EXPLOITATION")
        assert state["shell_obtained"] is True
        assert state["root_shell"] is True

    def test_compute_delta_new_ports(self) -> None:
        from core.orchestration.state_builder import compute_discoveries_delta
        prev = {"ports": {22}, "services": {"ssh"}}
        curr = {"ports": {22, 80, 443}, "services": {"ssh", "http"}}
        delta = compute_discoveries_delta(curr, prev)
        assert 80 in delta["ports"]
        assert 443 in delta["ports"]
        assert "http" in delta["services"]

    def test_compute_delta_no_change(self) -> None:
        from core.orchestration.state_builder import compute_discoveries_delta
        state = {"ports": {22}, "services": {"ssh"}}
        delta = compute_discoveries_delta(state, state)
        assert "ports" not in delta
        assert "services" not in delta

    def test_compute_delta_scalar_change(self) -> None:
        from core.orchestration.state_builder import compute_discoveries_delta
        prev = {"phase": "RECON", "detection_risk": 0.1}
        curr = {"phase": "ENUMERATION", "detection_risk": 0.3}
        delta = compute_discoveries_delta(curr, prev)
        assert delta["phase"] == "ENUMERATION"
        assert delta["detection_risk"] == 0.3


# ── EpisodeRunner ─────────────────────────────────────────────────

class TestEpisodeRunner:

    def test_step_record_to_dict(self) -> None:
        from core.orchestration.episode_runner import StepRecord
        rec = StepRecord(step=1, agent_id="RedAgent", command="nmap -sV x",
                         reward=2.5, discoveries=3, phase="RECON", source="ppo")
        d = rec.to_dict()
        assert d["step"] == 1
        assert d["agent_id"] == "RedAgent"
        assert d["source"] == "ppo"

    def test_episode_metrics_to_dict(self) -> None:
        from core.orchestration.episode_runner import EpisodeMetrics
        m = EpisodeMetrics(total_reward=42.5, total_steps=10, unique_commands=8)
        d = m.to_dict()
        assert d["total_reward"] == 42.5
        assert d["diversity_ratio"] == 0.0

    def test_stuck_detector_not_stuck_initially(self) -> None:
        from core.orchestration.episode_runner import StuckDetector
        det = StuckDetector(window=5)
        assert not det.is_stuck()

    def test_stuck_detector_zero_reward_streak(self) -> None:
        from core.orchestration.episode_runner import StuckDetector
        det = StuckDetector(window=5)
        for i in range(10):
            det.record(f"cmd_{i}", 0.0)
        assert det.is_stuck()

    def test_stuck_detector_repeat_commands(self) -> None:
        from core.orchestration.episode_runner import StuckDetector
        det = StuckDetector(window=5, threshold=0.5)
        for _ in range(10):
            det.record("nmap -sV 10.10.10.1", 0.0)
        assert det.is_repeat_stuck()

    def test_stuck_detector_reset(self) -> None:
        from core.orchestration.episode_runner import StuckDetector
        det = StuckDetector(window=5)
        for _ in range(10):
            det.record("x", 0.0)
        assert det.is_stuck()
        det.reset()
        assert not det.is_stuck()

    def test_episode_tracker_lifecycle(self) -> None:
        from core.orchestration.episode_runner import EpisodeTracker, StepRecord
        tracker = EpisodeTracker()
        tracker.start_episode()
        tracker.record_step(StepRecord(
            step=0, agent_id="ScoutAgent", command="nmap -sV 10.10.10.1",
            reward=2.5, discoveries=2, phase="RECON", success=True,
        ))
        tracker.record_step(StepRecord(
            step=1, agent_id="RedAgent", command="msfconsole -x exploit",
            reward=15.0, discoveries=1, phase="EXPLOITATION", success=True,
        ))
        metrics = tracker.get_metrics()
        assert metrics.total_steps == 2
        assert metrics.total_reward == 17.5
        assert metrics.total_discoveries == 3
        assert metrics.unique_commands == 2
        assert "RECON" in metrics.phases_reached
        assert "EXPLOITATION" in metrics.phases_reached
        assert metrics.step_at_first_exploit == 1

    def test_episode_tracker_diversity(self) -> None:
        from core.orchestration.episode_runner import EpisodeTracker, StepRecord
        tracker = EpisodeTracker()
        tracker.start_episode()
        for i in range(10):
            tracker.record_step(StepRecord(
                step=i, agent_id="RedAgent",
                command=f"cmd_{i}" if i < 8 else "cmd_0",
                reward=1.0, discoveries=0, phase="RECON",
            ))
        metrics = tracker.get_metrics()
        # 8 unique out of 10 steps
        assert metrics.unique_commands == 8
        assert 0.7 < metrics.diversity_ratio <= 0.85

    def test_episode_tracker_history(self) -> None:
        from core.orchestration.episode_runner import EpisodeTracker, StepRecord
        tracker = EpisodeTracker()
        tracker.start_episode()
        tracker.record_step(StepRecord(
            step=0, agent_id="BlueAgent", command="check alerts",
            reward=0.0, discoveries=0, phase="RECON",
        ))
        hist = tracker.get_step_history()
        assert len(hist) == 1
        assert hist[0]["agent_id"] == "BlueAgent"


# ── backward compat ───────────────────────────────────────────────

class TestBackwardCompat:
    """Ensure backward-compat import path works."""

    def test_import_smart_orchestrator_from_package(self) -> None:
        from core.orchestration import SmartOrchestrator
        assert SmartOrchestrator is not None
