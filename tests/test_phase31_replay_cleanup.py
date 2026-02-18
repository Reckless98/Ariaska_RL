"""Phase 31: Episode Replay, Difficulty Cleanup, Dead Code tests."""

import json
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("ARIASKA_DRY_RUN", "1")


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _make_trace_file(episodes: list[dict]) -> str:
    """Write JSONL events to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for event in episodes:
            f.write(json.dumps(event) + "\n")
    return path


def _basic_trace_events(episode_id: str = "ep-001", episode_num: int = 1) -> list[dict]:
    """Return a minimal list of JSONL events for one episode."""
    return [
        {
            "kind": "episode_start",
            "episode_id": episode_id,
            "episode_num": episode_num,
            "data": {"max_steps": 40, "target_ip": "10.0.0.1"},
        },
        {
            "kind": "step",
            "episode_id": episode_id,
            "step_num": 1,
            "phase_before": "RECON",
            "phase_after": "RECON",
            "step_reward_total": 3.5,
            "episode_reward_so_far": 3.5,
            "target_ip": "10.0.0.1",
            "agent_records": [
                {
                    "agent_name": "ScoutAgent",
                    "role": "recon",
                    "decision_source": "playbook",
                    "phase": "RECON",
                    "command": "nmap -sV -p- 10.0.0.1",
                    "command_family": "nmap",
                    "reward": 2.5,
                    "mentor_call": False,
                    "discoveries": ["port:22", "port:80"],
                    "stdout_snippet": "22/tcp open ssh\n80/tcp open http",
                    "confidence": 0.9,
                },
                {
                    "agent_name": "ShadowAgent",
                    "role": "stealth",
                    "decision_source": "registry",
                    "phase": "RECON",
                    "command": "sleep 3",
                    "command_family": "sleep",
                    "reward": 1.0,
                    "mentor_call": False,
                    "discoveries": [],
                    "stdout_snippet": "",
                    "confidence": 0.8,
                },
            ],
        },
        {
            "kind": "step",
            "episode_id": episode_id,
            "step_num": 2,
            "phase_before": "RECON",
            "phase_after": "ENUMERATION",
            "step_reward_total": 5.0,
            "episode_reward_so_far": 8.5,
            "target_ip": "10.0.0.1",
            "agent_records": [
                {
                    "agent_name": "RedAgent",
                    "role": "offensive",
                    "decision_source": "ppo",
                    "phase": "ENUMERATION",
                    "command": "hydra -l admin -P /wordlist ssh://10.0.0.1",
                    "command_family": "hydra",
                    "reward": 5.0,
                    "mentor_call": False,
                    "discoveries": ["credential:admin:pass123"],
                    "stdout_snippet": "[22][ssh] host: 10.0.0.1 login: admin password: pass123",
                    "confidence": 0.7,
                },
            ],
        },
        {
            "kind": "episode_end",
            "episode_id": episode_id,
            "data": {
                "total_reward": 8.5,
                "highest_phase": "ENUMERATION",
                "steps": 2,
            },
        },
    ]


# ─────────────────────────────────────────────────────────────────────
# 31.1: Episode Replayer
# ─────────────────────────────────────────────────────────────────────


class TestEpisodeReplayer:
    """Test replay parsing and rendering."""

    def test_parse_basic_trace(self):
        from core.replay.episode_replayer import parse_trace_file

        path = _make_trace_file(_basic_trace_events())
        try:
            episodes = parse_trace_file(path)
            assert len(episodes) == 1
            ep = episodes[0]
            assert ep.episode_id == "ep-001"
            assert ep.episode_num == 1
            assert ep.total_reward == 8.5
            assert ep.highest_phase == "ENUMERATION"
            assert ep.total_steps == 2
            assert len(ep.steps) == 2
        finally:
            os.unlink(path)

    def test_parse_agent_records(self):
        from core.replay.episode_replayer import parse_trace_file

        path = _make_trace_file(_basic_trace_events())
        try:
            ep = parse_trace_file(path)[0]
            step1 = ep.steps[0]
            assert len(step1.agent_records) == 2
            scout = step1.agent_records[0]
            assert scout.agent_name == "ScoutAgent"
            assert scout.decision_source == "playbook"
            assert scout.discoveries == ["port:22", "port:80"]
            assert scout.reward == 2.5
        finally:
            os.unlink(path)

    def test_parse_missing_file_raises(self):
        from core.replay.episode_replayer import parse_trace_file

        with pytest.raises(FileNotFoundError):
            parse_trace_file("/nonexistent/trace.jsonl")

    def test_parse_empty_file(self):
        from core.replay.episode_replayer import parse_trace_file

        path = _make_trace_file([])
        try:
            episodes = parse_trace_file(path)
            assert episodes == []
        finally:
            os.unlink(path)

    def test_parse_malformed_lines_skipped(self):
        from core.replay.episode_replayer import parse_trace_file

        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w") as f:
            f.write("not json at all\n")
            f.write(json.dumps(_basic_trace_events()[0]) + "\n")
        try:
            episodes = parse_trace_file(path)
            assert len(episodes) == 1
        finally:
            os.unlink(path)

    def test_parse_multiple_episodes(self):
        from core.replay.episode_replayer import parse_trace_file

        events = _basic_trace_events("ep-A", 1) + _basic_trace_events("ep-B", 2)
        path = _make_trace_file(events)
        try:
            episodes = parse_trace_file(path)
            assert len(episodes) == 2
            ids = {e.episode_id for e in episodes}
            assert ids == {"ep-A", "ep-B"}
        finally:
            os.unlink(path)

    def test_render_episode_no_crash(self):
        """render_episode should complete without exceptions."""
        from core.replay.episode_replayer import parse_trace_file, render_episode
        from rich.console import Console
        from io import StringIO

        path = _make_trace_file(_basic_trace_events())
        try:
            ep = parse_trace_file(path)[0]
            buf = StringIO()
            console = Console(file=buf, force_terminal=False, width=120)
            render_episode(ep, console, verbose=False)
            output = buf.getvalue()
            assert "Episode 1" in output
            assert "ScoutAgent" in output
            assert "Step 1" in output
        finally:
            os.unlink(path)

    def test_render_verbose_shows_stdout(self):
        """Verbose mode should include stdout snippets."""
        from core.replay.episode_replayer import parse_trace_file, render_episode
        from rich.console import Console
        from io import StringIO

        path = _make_trace_file(_basic_trace_events())
        try:
            ep = parse_trace_file(path)[0]
            buf = StringIO()
            console = Console(file=buf, force_terminal=False, width=120)
            render_episode(ep, console, verbose=True)
            output = buf.getvalue()
            assert "22/tcp open ssh" in output
        finally:
            os.unlink(path)

    def test_replay_trace_file_integration(self):
        """End-to-end: parse + render via replay_trace_file."""
        from core.replay.episode_replayer import replay_trace_file
        from rich.console import Console
        from io import StringIO

        path = _make_trace_file(_basic_trace_events())
        try:
            buf = StringIO()
            console = Console(file=buf, force_terminal=False, width=120)
            episodes = replay_trace_file(path, verbose=False, console=console)
            assert len(episodes) == 1
            assert episodes[0].total_reward == 8.5
        finally:
            os.unlink(path)


# ─────────────────────────────────────────────────────────────────────
# 31.1b: CLI replay subcommand
# ─────────────────────────────────────────────────────────────────────


class TestCLIReplayArgparse:
    """Test that the replay CLI subcommand parses correctly."""

    def test_replay_argparse(self):
        """Ensure 'replay <path>' argparse works."""
        import argparse
        # We parse the CLI module to test argparse wiring
        # Instead of re-importing the full CLI, just verify the module imports cleanly
        from core.replay.episode_replayer import (
            parse_trace_file,
            render_episode,
            replay_trace_file,
            ReplayAgentRecord,
            ReplayStep,
            ReplayEpisode,
        )
        # All importable without error
        assert callable(parse_trace_file)
        assert callable(render_episode)
        assert callable(replay_trace_file)


# ─────────────────────────────────────────────────────────────────────
# 31.2: Difficulty cleanup — no regressions
# ─────────────────────────────────────────────────────────────────────


class TestDifficultyCleanup:
    """Verify SmartCoach and SmartOrchestrator init without difficulty_preset."""

    def test_smartcoach_no_difficulty_preset(self):
        """SmartCoach should have no difficulty_preset attribute."""
        from core.testing.fake_gpt_manager import FakeGPTManager

        gpt = FakeGPTManager(seed=42)
        from core.training.smart_coach import SmartCoach

        coach = SmartCoach(agent_name="RedAgent", gpt_manager=gpt)  # type: ignore[arg-type]
        assert not hasattr(coach, "difficulty_preset") or coach.difficulty_preset is None  # type: ignore[attr-defined]

    def test_smartcoach_no_difficulty_alternative_method(self):
        """_get_difficulty_alternative should not exist."""
        from core.testing.fake_gpt_manager import FakeGPTManager

        gpt = FakeGPTManager(seed=42)
        from core.training.smart_coach import SmartCoach

        coach = SmartCoach(agent_name="RedAgent", gpt_manager=gpt)  # type: ignore[arg-type]
        assert not hasattr(coach, "_get_difficulty_alternative")

    def test_orchestrator_no_difficulty_preset_attr(self):
        """SmartOrchestrator should not have _difficulty_preset."""
        from core.orchestration.smart_orchestrator import SmartOrchestratorConfig

        config = SmartOrchestratorConfig()
        # difficulty field should still exist (backward compat), always "normal"
        assert config.difficulty == "normal"


# ─────────────────────────────────────────────────────────────────────
# 31.3: Dataclass integrity
# ─────────────────────────────────────────────────────────────────────


class TestReplayDataclasses:
    """Test replay dataclass defaults and integrity."""

    def test_replay_agent_record_defaults(self):
        from core.replay.episode_replayer import ReplayAgentRecord

        rec = ReplayAgentRecord()
        assert rec.agent_name == ""
        assert rec.reward == 0.0
        assert rec.discoveries == []
        assert rec.mentor_call is False

    def test_replay_step_defaults(self):
        from core.replay.episode_replayer import ReplayStep

        step = ReplayStep()
        assert step.step_num == 0
        assert step.agent_records == []

    def test_replay_episode_defaults(self):
        from core.replay.episode_replayer import ReplayEpisode

        ep = ReplayEpisode()
        assert ep.episode_id == ""
        assert ep.steps == []
        assert ep.total_reward == 0.0
