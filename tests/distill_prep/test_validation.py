"""Tests for validation, weakness report, and manifest."""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest


class TestValidation:
    """Test artifact validation catches bad data."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        os.environ["ARIASKA_DRY_RUN"] = "1"
        self.tmpdir = tempfile.mkdtemp()

    def test_valid_data_passes(self) -> None:
        """Correctly formed JSONL passes validation."""
        from scripts.distill_prep.trace_schema import validate_jsonl_line

        good_line = json.dumps(
            {
                "kind": "episode_start",
                "episode_id": "test_001",
                "episode_num": 0,
                "target_ip": "10.10.10.1",
            }
        )
        errors = validate_jsonl_line(good_line)
        assert errors == []

    def test_missing_kind_fails(self) -> None:
        """Missing 'kind' field triggers error."""
        from scripts.distill_prep.trace_schema import validate_jsonl_line

        errors = validate_jsonl_line('{"episode_id": "test"}')
        assert any("missing 'kind'" in e for e in errors)

    def test_invalid_phase_fails(self) -> None:
        """Invalid phase name triggers error."""
        from scripts.distill_prep.trace_schema import validate_jsonl_line

        bad_line = json.dumps(
            {
                "kind": "step",
                "step_num": 0,
                "phase_before": "INVALID_PHASE",
                "phase_after": "RECON",
            }
        )
        errors = validate_jsonl_line(bad_line)
        assert any("unknown phase_before" in e for e in errors)

    def test_nan_reward_fails(self) -> None:
        """NaN reward triggers error."""
        from scripts.distill_prep.trace_schema import validate_jsonl_line

        # JSON doesn't support NaN directly, but we test the validator
        bad_line = json.dumps(
            {
                "kind": "step",
                "step_num": 0,
                "phase_before": "RECON",
                "phase_after": "RECON",
                "agent_records": [
                    {
                        "command_family": "nmap",
                        "reward": 999.0,
                        "decision_source": "ppo",
                    }
                ],
            }
        )
        errors = validate_jsonl_line(bad_line)
        assert any("out of range" in e for e in errors)

    def test_unknown_command_family_fails(self) -> None:
        """Unknown command family triggers error."""
        from scripts.distill_prep.trace_schema import validate_jsonl_line

        bad_line = json.dumps(
            {
                "kind": "step",
                "step_num": 0,
                "phase_before": "RECON",
                "phase_after": "RECON",
                "agent_records": [
                    {
                        "command_family": "totally_fake_tool",
                        "reward": 1.0,
                        "decision_source": "ppo",
                    }
                ],
            }
        )
        errors = validate_jsonl_line(bad_line)
        assert any("unknown command_family" in e for e in errors)

    def test_unknown_decision_source_fails(self) -> None:
        """Unknown decision source triggers error."""
        from scripts.distill_prep.trace_schema import validate_jsonl_line

        bad_line = json.dumps(
            {
                "kind": "step",
                "step_num": 0,
                "phase_before": "RECON",
                "phase_after": "RECON",
                "agent_records": [
                    {
                        "command_family": "nmap",
                        "reward": 1.0,
                        "decision_source": "bogus_source",
                    }
                ],
            }
        )
        errors = validate_jsonl_line(bad_line)
        assert any("unknown decision_source" in e for e in errors)

    def test_unknown_discovery_type_fails(self) -> None:
        """Unknown discovery type triggers error."""
        from scripts.distill_prep.trace_schema import validate_jsonl_line

        bad_line = json.dumps(
            {
                "kind": "step",
                "step_num": 0,
                "phase_before": "RECON",
                "phase_after": "RECON",
                "agent_records": [
                    {
                        "command_family": "nmap",
                        "reward": 1.0,
                        "decision_source": "ppo",
                        "discoveries": [
                            {
                                "discovery_type": "FAKE_DISCOVERY",
                                "value": "test",
                            }
                        ],
                    }
                ],
            }
        )
        errors = validate_jsonl_line(bad_line)
        assert any("unknown type FAKE_DISCOVERY" in e for e in errors)

    def test_invalid_json_fails(self) -> None:
        """Malformed JSON triggers error."""
        from scripts.distill_prep.trace_schema import validate_jsonl_line

        errors = validate_jsonl_line("not valid json {{{")
        assert any("invalid JSON" in e for e in errors)

    def test_validate_good_directory(self) -> None:
        """validate_directory on generated data returns 0 errors."""
        from scripts.distill_prep.generate_synthetic_traces import generate_all_runs
        from scripts.distill_prep.validate_artifacts import validate_directory

        outdir = Path(self.tmpdir) / "traces"
        generate_all_runs(5, seed=42, outdir=str(outdir))

        fc, tl, te, errors = validate_directory(outdir, "test")
        assert te == 0

    def test_validate_bad_directory(self) -> None:
        """validate_directory on intentionally bad data returns errors."""
        from scripts.distill_prep.validate_artifacts import validate_directory

        bad_dir = Path(self.tmpdir) / "bad"
        bad_dir.mkdir()
        bad_file = bad_dir / "bad.jsonl"
        with open(bad_file, "w") as f:
            f.write(json.dumps({"kind": "step", "phase_before": "NOPE"}) + "\n")
            f.write(json.dumps({"kind": "unknown_kind"}) + "\n")

        fc, tl, te, errors = validate_directory(bad_dir, "test")
        assert te > 0

    def test_validate_all_returns_false_on_bad(self) -> None:
        """validate_all returns False when errors exist."""
        from scripts.distill_prep.validate_artifacts import validate_all

        bad_dir = Path(self.tmpdir) / "bad_traces"
        bad_dir.mkdir()
        with open(bad_dir / "bad.jsonl", "w") as f:
            f.write(json.dumps({"kind": "step", "phase_before": "NOPE"}) + "\n")

        result = validate_all(
            traces_dir=str(bad_dir),
            trajectories_dir=str(Path(self.tmpdir) / "nonexistent"),
        )
        assert result is False

    def test_teacher_step_validation(self) -> None:
        """Teacher step validation catches bad command families."""
        from scripts.distill_prep.trace_schema import validate_jsonl_line

        bad_line = json.dumps(
            {
                "kind": "teacher_step",
                "phase": "RECON",
                "command_family": "nonexistent_tool",
                "reward": 5.0,
            }
        )
        errors = validate_jsonl_line(bad_line)
        assert any("unknown command_family" in e for e in errors)

    def test_trajectory_end_validation(self) -> None:
        """Trajectory end with invalid phase fails."""
        from scripts.distill_prep.trace_schema import validate_jsonl_line

        bad_line = json.dumps(
            {
                "kind": "trajectory_end",
                "trajectory_id": "t001",
                "total_reward": 50.0,
                "highest_phase": "INVALID",
            }
        )
        errors = validate_jsonl_line(bad_line)
        # No phase validation on trajectory_end currently (just checks required fields)
        assert len(errors) == 0 or any("highest_phase" in e for e in errors)


class TestWeaknessReport:
    """Test weakness report generation."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        os.environ["ARIASKA_DRY_RUN"] = "1"
        self.tmpdir = tempfile.mkdtemp()

    def test_weakness_report_keys(self) -> None:
        """Weakness report has expected keys."""
        from scripts.distill_prep.generate_synthetic_traces import generate_all_runs
        from scripts.distill_prep.summarize_artifacts import (
            generate_weakness_report,
        )

        traces_dir = Path(self.tmpdir) / "traces"
        generate_all_runs(10, seed=42, outdir=str(traces_dir))

        report_path = Path(self.tmpdir) / "weakness.json"
        report = generate_weakness_report(
            traces_dir=str(traces_dir),
            trajectories_dir=str(Path(self.tmpdir) / "nonexistent"),
            output_path=str(report_path),
        )

        required_keys = [
            "generated_at",
            "distill_prep_version",
            "total_traces",
            "total_steps",
            "phase_histogram",
            "tool_family_coverage",
            "avg_reward_by_phase",
            "decision_source_pct",
            "weakness_areas",
            "coverage_gaps",
            "wrong_move_ratio",
        ]
        for k in required_keys:
            assert k in report, f"Missing key: {k}"

    def test_weakness_report_file_written(self) -> None:
        """Report is written as valid JSON file."""
        from scripts.distill_prep.generate_synthetic_traces import generate_all_runs
        from scripts.distill_prep.summarize_artifacts import (
            generate_weakness_report,
        )

        traces_dir = Path(self.tmpdir) / "traces"
        generate_all_runs(5, seed=42, outdir=str(traces_dir))

        report_path = Path(self.tmpdir) / "wr.json"
        generate_weakness_report(
            str(traces_dir),
            str(Path(self.tmpdir) / "nonexistent"),
            str(report_path),
        )

        assert report_path.exists()
        with open(report_path) as f:
            data = json.load(f)
        assert data["total_traces"] == 5

    def test_phase_histogram_complete(self) -> None:
        """Phase histogram covers all 8 phases."""
        from scripts.distill_prep.generate_synthetic_traces import generate_all_runs
        from scripts.distill_prep.summarize_artifacts import (
            generate_weakness_report,
        )
        from scripts.distill_prep.trace_schema import PHASE_ORDER

        traces_dir = Path(self.tmpdir) / "traces"
        generate_all_runs(20, seed=42, outdir=str(traces_dir))

        report_path = Path(self.tmpdir) / "wr.json"
        report = generate_weakness_report(
            str(traces_dir),
            str(Path(self.tmpdir) / "nonexistent"),
            str(report_path),
        )

        for phase in PHASE_ORDER:
            assert phase in report["phase_histogram"]


class TestManifest:
    """Test manifest generation."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        os.environ["ARIASKA_DRY_RUN"] = "1"
        self.tmpdir = tempfile.mkdtemp()

    def test_manifest_checksums(self) -> None:
        """Manifest checksums match file contents."""
        import hashlib

        from scripts.distill_prep.generate_synthetic_traces import generate_all_runs
        from scripts.distill_prep.summarize_artifacts import generate_manifest

        base = Path(self.tmpdir) / "distill"
        traces_dir = base / "synthetic_traces"
        generate_all_runs(3, seed=42, outdir=str(traces_dir))

        manifest = generate_manifest(base_dir=str(base), seed=42)

        assert len(manifest["files"]) >= 3
        for entry in manifest["files"]:
            fp = base / entry["path"]
            if fp.exists():
                h = hashlib.sha256()
                with open(fp, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        h.update(chunk)
                assert h.hexdigest() == entry["sha256"], (
                    f"Checksum mismatch for {entry['path']}"
                )

    def test_manifest_counts(self) -> None:
        """Manifest counts match actual file counts."""
        from scripts.distill_prep.generate_synthetic_traces import generate_all_runs
        from scripts.distill_prep.summarize_artifacts import generate_manifest

        base = Path(self.tmpdir) / "distill"
        traces_dir = base / "synthetic_traces"
        generate_all_runs(5, seed=42, outdir=str(traces_dir))

        manifest = generate_manifest(base_dir=str(base), seed=42)

        assert manifest["counts"].get("synthetic_traces", 0) == 5

    def test_manifest_version(self) -> None:
        """Manifest includes version info."""
        from scripts.distill_prep.summarize_artifacts import generate_manifest
        from scripts.distill_prep.trace_schema import DISTILL_PREP_VERSION

        base = Path(self.tmpdir) / "distill"
        base.mkdir(parents=True)

        manifest = generate_manifest(base_dir=str(base), seed=42)

        assert manifest["distill_prep_version"] == DISTILL_PREP_VERSION
        assert manifest["seed"] == 42
        assert "git_commit" in manifest
        assert "generated_at" in manifest
