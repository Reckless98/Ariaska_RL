"""Tests for scripts.unify_training_data — conversion CLI tool."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")

from scripts.unified_data_schema import UNIFIED_SCHEMA_VERSION, validate_record
from scripts.unify_training_data import (
    CMD_OUT,
    GPU_OUT,
    KNOW_OUT,
    SCEN_OUT,
    TRAJ_OUT,
    UNIFIED_DATA_DIR,
    _normalize_phase,
    _read_jsonl_safe,
    _write_jsonl,
    convert_all,
    convert_gpu_traces,
    convert_knowledge,
    convert_learned_commands,
    convert_scenarios,
    convert_single_gpu_run,
    convert_trajectories,
    main,
    validate,
)


# ── Phase normalization ───────────────────────────────────────


class TestNormalizePhase:
    def test_canonical_phases(self) -> None:
        assert _normalize_phase("RECON") == "RECON"
        assert _normalize_phase("recon") == "RECON"
        assert _normalize_phase("reconnaissance") == "RECON"

    def test_killchain_mapping(self) -> None:
        assert _normalize_phase("foothold") == "EXPLOITATION"
        assert _normalize_phase("privesc") == "PRIVILEGE_ESCALATION"
        assert _normalize_phase("lateral") == "LATERAL_MOVEMENT"
        assert _normalize_phase("exfil") == "EXFILTRATION"

    def test_hyphenated_forms(self) -> None:
        assert _normalize_phase("privilege-escalation") == "PRIVILEGE_ESCALATION"
        assert _normalize_phase("lateral-movement") == "LATERAL_MOVEMENT"
        assert _normalize_phase("post-exploitation") == "POST_EXPLOITATION"

    def test_unknown_passthrough(self) -> None:
        assert _normalize_phase("custom_phase") == "CUSTOM_PHASE"


# ── JSONL I/O helpers ─────────────────────────────────────────


class TestJsonlIO:
    def test_write_and_read(self, tmp_path: Path) -> None:
        out = tmp_path / "test.jsonl"
        records = [{"a": 1}, {"b": 2}, {"c": 3}]
        n = _write_jsonl(out, records)
        assert n == 3
        assert out.exists()

        loaded = _read_jsonl_safe(out)
        assert len(loaded) == 3
        assert loaded[0]["a"] == 1

    def test_read_skips_malformed(self, tmp_path: Path) -> None:
        out = tmp_path / "test.jsonl"
        out.write_text('{"ok":true}\nBAD LINE\n{"ok":true}\n')
        loaded = _read_jsonl_safe(out)
        assert len(loaded) == 2

    def test_read_nonexistent(self, tmp_path: Path) -> None:
        out = tmp_path / "nope.jsonl"
        assert _read_jsonl_safe(out) == []

    def test_write_creates_parents(self, tmp_path: Path) -> None:
        out = tmp_path / "a" / "b" / "c" / "test.jsonl"
        _write_jsonl(out, [{"x": 1}])
        assert out.exists()


# ── Knowledge conversion ──────────────────────────────────────


class TestConvertKnowledge:
    def test_basic_conversion(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Create fake knowledge dir
        know_dir = tmp_path / "knowledge_candidates_v2"
        know_dir.mkdir()
        jsonl = know_dir / "test.jsonl"
        entry = {
            "candidate_id": "k1",
            "title": "Test entry",
            "taxonomy": {"killchain_step": "recon"},
            "governance": {"quality": {"evidence_coverage": 0.8}},
            "source": {"ingested_at": "2025-01-01T00:00:00Z"},
        }
        jsonl.write_text(json.dumps(entry) + "\n")

        # Patch dirs
        out_dir = tmp_path / "unified"
        monkeypatch.setattr("scripts.unify_training_data.KNOWLEDGE_DIR", know_dir)
        monkeypatch.setattr("scripts.unify_training_data.KNOW_OUT", out_dir / "knowledge")
        monkeypatch.setattr("scripts.unify_training_data.UNIFIED_DATA_DIR", out_dir)

        counts = convert_knowledge()
        assert "RECON" in counts
        assert counts["RECON"] >= 1

        # Validate output
        recon_file = out_dir / "knowledge" / "recon.jsonl"
        assert recon_file.exists()
        records = _read_jsonl_safe(recon_file)
        assert len(records) >= 1
        ok, errs = validate_record(records[0])
        assert ok, f"Validation errors: {errs}"
        assert records[0]["kind"] == "knowledge"
        assert records[0]["source"] == "knowledge_v2"

    def test_missing_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("scripts.unify_training_data.KNOWLEDGE_DIR", tmp_path / "nope")
        assert convert_knowledge() == {}


# ── Trajectory conversion ─────────────────────────────────────


class TestConvertTrajectories:
    def test_basic_conversion(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        traj_dir = tmp_path / "teacher_trajectories"
        traj_dir.mkdir()
        jsonl = traj_dir / "traj_001.jsonl"
        entries = [
            {"kind": "episode_start", "episode_id": "ep1", "difficulty": "easy",
             "timestamp": "2025-01-01T00:00:00Z"},
            {"kind": "step", "episode_id": "ep1", "phase": "RECON", "command": "nmap",
             "total_reward": 5.0, "difficulty": "easy", "timestamp": "2025-01-01T00:00:01Z"},
            {"kind": "episode_end", "episode_id": "ep1", "highest_phase": "EXPLOITATION",
             "total_reward": 25.0, "difficulty": "easy", "timestamp": "2025-01-01T00:00:02Z"},
        ]
        jsonl.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        out_dir = tmp_path / "unified"
        monkeypatch.setattr("scripts.unify_training_data.TRAJECTORIES_DIR", traj_dir)
        monkeypatch.setattr("scripts.unify_training_data.TRAJ_OUT", out_dir / "trajectories")
        monkeypatch.setattr("scripts.unify_training_data.UNIFIED_DATA_DIR", out_dir)

        counts = convert_trajectories()
        assert "easy" in counts
        assert counts["easy"] == 3

    def test_missing_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("scripts.unify_training_data.TRAJECTORIES_DIR", tmp_path / "nope")
        assert convert_trajectories() == {}


# ── GPU trace conversion ──────────────────────────────────────


class TestConvertGpuTraces:
    def test_basic_conversion(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()
        jsonl = trace_dir / "h200_distill_20260301T100000Z.jsonl"
        entries = [
            {"kind": "episode_start", "timestamp": 1709290800, "episode": 0},
            {"kind": "step", "timestamp": 1709290801, "phase": "RECON", "command": "nmap"},
            {"kind": "episode_end", "timestamp": 1709290802, "max_phase": "RECON"},
        ]
        jsonl.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        out_dir = tmp_path / "unified"
        monkeypatch.setattr("scripts.unify_training_data.TRACES_DIR", trace_dir)
        monkeypatch.setattr("scripts.unify_training_data.GPU_OUT", out_dir / "gpu_runs")
        monkeypatch.setattr("scripts.unify_training_data.UNIFIED_DATA_DIR", out_dir)

        counts = convert_gpu_traces()
        assert "20260301T100000Z" in counts
        assert counts["20260301T100000Z"] == 3

        # Check output
        out_file = out_dir / "gpu_runs" / "20260301T100000Z.jsonl"
        assert out_file.exists()
        records = _read_jsonl_safe(out_file)
        ok, errs = validate_record(records[0])
        assert ok, f"Validation errors: {errs}"

    def test_already_unified_passthrough(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Records already in unified format should pass through unchanged."""
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()
        jsonl = trace_dir / "h200_distill_run1.jsonl"
        rec = {
            "schema_version": "3.0.0",
            "kind": "step",
            "source": "gpu_distill",
            "data": {"x": 1},
            "timestamp": "2025-01-01T00:00:00Z",
        }
        jsonl.write_text(json.dumps(rec) + "\n")

        out_dir = tmp_path / "unified"
        monkeypatch.setattr("scripts.unify_training_data.TRACES_DIR", trace_dir)
        monkeypatch.setattr("scripts.unify_training_data.GPU_OUT", out_dir / "gpu_runs")
        monkeypatch.setattr("scripts.unify_training_data.UNIFIED_DATA_DIR", out_dir)

        counts = convert_gpu_traces()
        assert counts.get("run1", 0) == 1

    def test_convert_single_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        trace_file = tmp_path / "h200_distill_run42.jsonl"
        trace_file.write_text(json.dumps({"kind": "step", "timestamp": 1000, "phase": "RECON"}) + "\n")

        out_dir = tmp_path / "unified"
        monkeypatch.setattr("scripts.unify_training_data.GPU_OUT", out_dir / "gpu_runs")
        monkeypatch.setattr("scripts.unify_training_data.UNIFIED_DATA_DIR", out_dir)

        result = convert_single_gpu_run(trace_file, run_id="run42")
        assert result is not None
        assert result.exists()
        records = _read_jsonl_safe(result)
        assert len(records) == 1


# ── Learned commands conversion ───────────────────────────────


class TestConvertLearnedCommands:
    def test_basic_conversion(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cmd_file = tmp_path / "learned_commands.json"
        data = {
            "version": "1.0",
            "total_commands": 2,
            "commands": {
                "nmap_scan": {
                    "template_name": "nmap_scan",
                    "phase": "RECON",
                    "avg_reward": 3.5,
                    "success_count": 10,
                    "total_attempts": 15,
                    "last_used": "2025-01-01T00:00:00Z",
                },
                "hydra_ssh": {
                    "template_name": "hydra_ssh",
                    "phase": "EXPLOITATION",
                    "avg_reward": 8.0,
                    "success_count": 5,
                    "total_attempts": 8,
                    "last_used": "2025-01-02T00:00:00Z",
                },
            },
        }
        cmd_file.write_text(json.dumps(data))

        out_dir = tmp_path / "unified"
        monkeypatch.setattr("scripts.unify_training_data.LEARNED_COMMANDS_PATH", cmd_file)
        monkeypatch.setattr("scripts.unify_training_data.CMD_OUT", out_dir / "commands")
        monkeypatch.setattr("scripts.unify_training_data.UNIFIED_DATA_DIR", out_dir)

        counts = convert_learned_commands()
        assert counts["learned_commands"] == 2

        out_file = out_dir / "commands" / "all.jsonl"
        assert out_file.exists()
        records = _read_jsonl_safe(out_file)
        assert len(records) == 2
        # RECON should sort before EXPLOITATION
        assert records[0]["phase"] == "RECON"

    def test_missing_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("scripts.unify_training_data.LEARNED_COMMANDS_PATH", tmp_path / "nope.json")
        assert convert_learned_commands() == {}


# ── Scenario conversion ───────────────────────────────────────


class TestConvertScenarios:
    def test_basic_conversion(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        scen_dir = tmp_path / "scenarios"
        scen_dir.mkdir()
        s1 = {"scenario_id": "s1", "difficulty": "easy", "target": "10.0.0.1"}
        s2 = {"scenario_id": "s2", "difficulty": "hard", "target": "10.0.0.2"}
        (scen_dir / "s1.json").write_text(json.dumps(s1))
        (scen_dir / "s2.json").write_text(json.dumps(s2))

        out_dir = tmp_path / "unified"
        monkeypatch.setattr("scripts.unify_training_data.SCENARIOS_DIR", scen_dir)
        monkeypatch.setattr("scripts.unify_training_data.SCEN_OUT", out_dir / "scenarios")
        monkeypatch.setattr("scripts.unify_training_data.UNIFIED_DATA_DIR", out_dir)

        counts = convert_scenarios()
        assert counts["scenarios"] == 2

    def test_missing_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("scripts.unify_training_data.SCENARIOS_DIR", tmp_path / "nope")
        assert convert_scenarios() == {}


# ── convert_all ────────────────────────────────────────────────


class TestConvertAll:
    def test_end_to_end(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Full pipeline with minimal test data."""
        # Set up source dirs
        know_dir = tmp_path / "knowledge"
        know_dir.mkdir()
        (know_dir / "test.jsonl").write_text(
            json.dumps({"taxonomy": {"killchain_step": "recon"}, "source": {"ingested_at": "now"}}) + "\n"
        )

        traj_dir = tmp_path / "trajectories"
        traj_dir.mkdir()
        (traj_dir / "t1.jsonl").write_text(
            json.dumps({"kind": "step", "difficulty": "easy", "timestamp": "now"}) + "\n"
        )

        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()
        (trace_dir / "h200_distill_test.jsonl").write_text(
            json.dumps({"kind": "step", "timestamp": 1000, "phase": "RECON"}) + "\n"
        )

        cmd_file = tmp_path / "commands.json"
        cmd_file.write_text(json.dumps({
            "version": "1.0", "total_commands": 1,
            "commands": {"x": {"phase": "RECON", "avg_reward": 1.0, "success_count": 1}},
        }))

        scen_dir = tmp_path / "scenarios"
        scen_dir.mkdir()
        (scen_dir / "s1.json").write_text(json.dumps({"scenario_id": "s1", "difficulty": "easy"}))

        out_dir = tmp_path / "unified"

        monkeypatch.setattr("scripts.unify_training_data.KNOWLEDGE_DIR", know_dir)
        monkeypatch.setattr("scripts.unify_training_data.TRAJECTORIES_DIR", traj_dir)
        monkeypatch.setattr("scripts.unify_training_data.TRACES_DIR", trace_dir)
        monkeypatch.setattr("scripts.unify_training_data.LEARNED_COMMANDS_PATH", cmd_file)
        monkeypatch.setattr("scripts.unify_training_data.SCENARIOS_DIR", scen_dir)
        monkeypatch.setattr("scripts.unify_training_data.UNIFIED_DATA_DIR", out_dir)
        monkeypatch.setattr("scripts.unify_training_data.KNOW_OUT", out_dir / "knowledge")
        monkeypatch.setattr("scripts.unify_training_data.TRAJ_OUT", out_dir / "trajectories")
        monkeypatch.setattr("scripts.unify_training_data.GPU_OUT", out_dir / "gpu_runs")
        monkeypatch.setattr("scripts.unify_training_data.CMD_OUT", out_dir / "commands")
        monkeypatch.setattr("scripts.unify_training_data.SCEN_OUT", out_dir / "scenarios")
        monkeypatch.setattr("scripts.unify_training_data.IDX_OUT", out_dir / "indices")

        convert_all()

        # Manifest should exist
        manifest = out_dir / "manifest.json"
        assert manifest.exists()
        m = json.loads(manifest.read_text())
        assert m["total_records"] >= 5
        assert m["schema_version"] == UNIFIED_SCHEMA_VERSION


# ── CLI ────────────────────────────────────────────────────────


class TestCLI:
    def test_stats_no_data(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("scripts.unify_training_data.UNIFIED_DATA_DIR", tmp_path / "empty")
        ret = main(["stats"])
        assert ret == 0

    def test_validate_no_data(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("scripts.unify_training_data.UNIFIED_DATA_DIR", tmp_path / "empty")
        ret = main(["validate"])
        assert ret == 1  # no data = failure
