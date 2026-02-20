"""Tests for synthetic trace generation."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest


class TestTraceGeneration:
    """Test synthetic trace generation produces valid, deterministic output."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        os.environ["ARIASKA_DRY_RUN"] = "1"
        self.tmpdir = tempfile.mkdtemp()

    def test_generate_single_run(self) -> None:
        """A single run produces valid JSONL with episode_start, steps, episode_end."""
        import random

        from scripts.distill_prep.generate_synthetic_traces import generate_one_run

        rng = random.Random(42)
        lines = generate_one_run(0, "easy", rng, "10.10.10.1")

        assert len(lines) >= 3  # start + >=1 step + end
        assert lines[0]["kind"] == "episode_start"
        assert lines[-1]["kind"] == "episode_end"

        # All middle lines are steps
        for line in lines[1:-1]:
            assert line["kind"] == "step"

    def test_deterministic_output(self) -> None:
        """Same seed produces identical output."""
        from scripts.distill_prep.generate_synthetic_traces import generate_all_runs

        out1 = Path(self.tmpdir) / "run1"
        out2 = Path(self.tmpdir) / "run2"

        generate_all_runs(5, seed=123, outdir=str(out1))
        generate_all_runs(5, seed=123, outdir=str(out2))

        files1 = sorted(out1.glob("*.jsonl"))
        files2 = sorted(out2.glob("*.jsonl"))

        assert len(files1) == len(files2) == 5

        for f1, f2 in zip(files1, files2):
            content1 = f1.read_text()
            content2 = f2.read_text()
            assert content1 == content2, f"Mismatch between {f1} and {f2}"

    def test_all_runs_validate(self) -> None:
        """Generated traces pass schema validation."""
        from scripts.distill_prep.generate_synthetic_traces import generate_all_runs
        from scripts.distill_prep.validate_artifacts import validate_directory

        outdir = Path(self.tmpdir) / "traces"
        generate_all_runs(10, seed=42, outdir=str(outdir))

        files_checked, total_lines, error_count, errors = validate_directory(
            outdir, "test"
        )
        assert files_checked == 10
        assert total_lines > 0
        assert error_count == 0, f"Validation errors: {errors[:5]}"

    def test_phase_progression(self) -> None:
        """Traces show sensible phase progression (never skip backwards)."""
        import random

        from scripts.distill_prep.generate_synthetic_traces import generate_one_run
        from scripts.distill_prep.trace_schema import PHASE_ORDER

        rng = random.Random(99)
        lines = generate_one_run(0, "medium", rng, "10.10.10.1")

        for line in lines:
            if line["kind"] == "step":
                before_idx = PHASE_ORDER.index(line["phase_before"])
                after_idx = PHASE_ORDER.index(line["phase_after"])
                assert after_idx >= before_idx, (
                    f"Phase went backwards: {line['phase_before']} -> {line['phase_after']}"
                )

    def test_anti_repeat_injected(self) -> None:
        """At least one step should be marked as wrong move (anti-repeat)."""
        import random

        from scripts.distill_prep.generate_synthetic_traces import generate_one_run

        rng = random.Random(42)
        lines = generate_one_run(0, "medium", rng, "10.10.10.1")

        wrong_moves = [
            line
            for line in lines
            if line.get("kind") == "step"
            and any(r.get("is_wrong_move") for r in line.get("agent_records", []))
        ]
        assert len(wrong_moves) >= 1, "No wrong moves injected"

    def test_valid_command_families(self) -> None:
        """All command families in generated traces are valid."""
        import random

        from scripts.distill_prep.generate_synthetic_traces import generate_one_run
        from scripts.distill_prep.trace_schema import COMMAND_FAMILIES

        rng = random.Random(42)
        lines = generate_one_run(0, "hard", rng, "10.10.10.1")

        for line in lines:
            if line.get("kind") == "step":
                for rec in line.get("agent_records", []):
                    family = rec.get("command_family", "")
                    assert family in COMMAND_FAMILIES, f"Unknown family: {family}"

    def test_reward_ranges(self) -> None:
        """All rewards are within valid range."""
        import random

        from scripts.distill_prep.generate_synthetic_traces import generate_one_run
        from scripts.distill_prep.trace_schema import REWARD_MAX, REWARD_MIN

        rng = random.Random(42)
        lines = generate_one_run(0, "medium", rng, "10.10.10.1")

        for line in lines:
            if line.get("kind") == "step":
                for rec in line.get("agent_records", []):
                    reward = rec.get("reward", 0.0)
                    assert REWARD_MIN <= reward <= REWARD_MAX, (
                        f"Reward out of range: {reward}"
                    )

    def test_discoveries_valid_types(self) -> None:
        """All discovery types are in the valid set."""
        import random

        from scripts.distill_prep.generate_synthetic_traces import generate_one_run
        from scripts.distill_prep.trace_schema import VALID_DISCOVERY_TYPES

        rng = random.Random(42)
        lines = generate_one_run(0, "easy", rng, "10.10.10.1")

        for line in lines:
            if line.get("kind") == "step":
                for rec in line.get("agent_records", []):
                    for disc in rec.get("discoveries", []):
                        dt = disc.get("discovery_type", "")
                        assert dt in VALID_DISCOVERY_TYPES, f"Unknown type: {dt}"

    def test_scenario_profiles_written(self) -> None:
        """Scenario profiles are written as valid JSON."""
        from scripts.distill_prep.generate_synthetic_traces import (
            write_scenario_profiles,
        )

        outdir = Path(self.tmpdir) / "scenarios"
        paths = write_scenario_profiles(str(outdir), seed=42)

        assert len(paths) >= 3  # at least 3 difficulties × 3 each
        for p in paths:
            with open(p, "r") as f:
                data = json.load(f)
            assert "scenario_id" in data
            assert "services" in data
            assert data["difficulty"] in ("easy", "medium", "hard")

    def test_step_counts_in_range(self) -> None:
        """Step counts fall within difficulty profile ranges."""
        import random

        from scripts.distill_prep.generate_synthetic_traces import (
            DIFFICULTY_PROFILES,
            generate_one_run,
        )

        for difficulty in ("easy", "medium", "hard"):
            rng = random.Random(42)
            lines = generate_one_run(0, difficulty, rng, "10.10.10.1")
            step_count = sum(1 for l in lines if l.get("kind") == "step")
            lo, hi = DIFFICULTY_PROFILES[difficulty]["step_range"]
            assert lo <= step_count <= hi, (
                f"{difficulty}: {step_count} steps not in [{lo}, {hi}]"
            )

    def test_distill_prep_version_present(self) -> None:
        """Every line has distill_prep_version field."""
        import random

        from scripts.distill_prep.generate_synthetic_traces import generate_one_run
        from scripts.distill_prep.trace_schema import DISTILL_PREP_VERSION

        rng = random.Random(42)
        lines = generate_one_run(0, "easy", rng, "10.10.10.1")

        for line in lines:
            assert line.get("distill_prep_version") == DISTILL_PREP_VERSION
