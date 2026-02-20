"""Tests for teacher trajectory generation."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest


class TestTrajectoryGeneration:
    """Test teacher trajectory generation."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        os.environ["ARIASKA_DRY_RUN"] = "1"
        self.tmpdir = tempfile.mkdtemp()

    def test_generate_trajectories(self) -> None:
        """Basic generation produces files."""
        from scripts.distill_prep.generate_teacher_trajectories import (
            generate_teacher_trajectories,
        )

        outdir = Path(self.tmpdir) / "trajectories"
        paths = generate_teacher_trajectories(10, seed=42, outdir=str(outdir))

        assert len(paths) == 10
        for p in paths:
            assert p.exists()
            assert p.suffix == ".jsonl"

    def test_trajectory_structure(self) -> None:
        """Each trajectory has start, steps, and end markers."""
        from scripts.distill_prep.generate_teacher_trajectories import (
            generate_teacher_trajectories,
        )

        outdir = Path(self.tmpdir) / "trajectories"
        paths = generate_teacher_trajectories(5, seed=42, outdir=str(outdir))

        for p in paths:
            lines = []
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        lines.append(json.loads(line))

            assert len(lines) >= 3  # start + at least 1 step + end
            assert lines[0]["kind"] == "trajectory_start"
            assert lines[-1]["kind"] == "trajectory_end"

            for mid in lines[1:-1]:
                assert mid["kind"] == "teacher_step"

    def test_wrong_moves_present(self) -> None:
        """Trajectories contain wrong moves (at least 20% per chain)."""
        from scripts.distill_prep.generate_teacher_trajectories import (
            generate_teacher_trajectories,
        )

        outdir = Path(self.tmpdir) / "trajectories"
        paths = generate_teacher_trajectories(20, seed=42, outdir=str(outdir))

        total_steps = 0
        total_wrong = 0
        for p in paths:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line.strip())
                    if obj.get("kind") == "teacher_step":
                        total_steps += 1
                        if obj.get("is_wrong_move", False):
                            total_wrong += 1

        assert total_steps > 0
        # Overall wrong move ratio should be meaningful (>10%)
        ratio = total_wrong / total_steps
        assert ratio >= 0.10, f"Wrong move ratio too low: {ratio:.1%}"

    def test_deterministic_output(self) -> None:
        """Same seed produces identical output."""
        from scripts.distill_prep.generate_teacher_trajectories import (
            generate_teacher_trajectories,
        )

        out1 = Path(self.tmpdir) / "t1"
        out2 = Path(self.tmpdir) / "t2"

        generate_teacher_trajectories(5, seed=77, outdir=str(out1))
        generate_teacher_trajectories(5, seed=77, outdir=str(out2))

        files1 = sorted(out1.glob("*.jsonl"))
        files2 = sorted(out2.glob("*.jsonl"))

        assert len(files1) == len(files2) == 5
        for f1, f2 in zip(files1, files2):
            assert f1.read_text() == f2.read_text()

    def test_valid_command_families(self) -> None:
        """All command families are in the schema's allowed list."""
        from scripts.distill_prep.generate_teacher_trajectories import (
            generate_teacher_trajectories,
        )
        from scripts.distill_prep.trace_schema import COMMAND_FAMILIES

        outdir = Path(self.tmpdir) / "trajectories"
        paths = generate_teacher_trajectories(10, seed=42, outdir=str(outdir))

        for p in paths:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line.strip())
                    if obj.get("kind") == "teacher_step":
                        family = obj.get("command_family", "")
                        assert family in COMMAND_FAMILIES, (
                            f"Unknown family: {family}"
                        )

    def test_reward_ranges(self) -> None:
        """All rewards within valid range."""
        from scripts.distill_prep.generate_teacher_trajectories import (
            generate_teacher_trajectories,
        )
        from scripts.distill_prep.trace_schema import REWARD_MAX, REWARD_MIN

        outdir = Path(self.tmpdir) / "trajectories"
        paths = generate_teacher_trajectories(10, seed=42, outdir=str(outdir))

        for p in paths:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line.strip())
                    if obj.get("kind") == "teacher_step":
                        reward = obj.get("reward", 0.0)
                        assert REWARD_MIN <= reward <= REWARD_MAX, (
                            f"Reward out of range: {reward}"
                        )

    def test_valid_phases(self) -> None:
        """All phases in trajectories are valid."""
        from scripts.distill_prep.generate_teacher_trajectories import (
            generate_teacher_trajectories,
        )
        from scripts.distill_prep.trace_schema import VALID_PHASES

        outdir = Path(self.tmpdir) / "trajectories"
        paths = generate_teacher_trajectories(10, seed=42, outdir=str(outdir))

        for p in paths:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line.strip())
                    if obj.get("kind") == "teacher_step":
                        assert obj["phase"] in VALID_PHASES, (
                            f"Unknown phase: {obj['phase']}"
                        )

    def test_trajectory_validates_against_schema(self) -> None:
        """Generated trajectories pass full schema validation."""
        from scripts.distill_prep.generate_teacher_trajectories import (
            generate_teacher_trajectories,
        )
        from scripts.distill_prep.validate_artifacts import validate_directory

        outdir = Path(self.tmpdir) / "trajectories"
        generate_teacher_trajectories(10, seed=42, outdir=str(outdir))

        fc, tl, te, errors = validate_directory(outdir, "test")
        assert fc == 10
        assert tl > 0
        assert te == 0, f"Validation errors: {errors[:5]}"

    def test_trajectory_total_reward_matches(self) -> None:
        """Trajectory end total_reward is consistent."""
        from scripts.distill_prep.generate_teacher_trajectories import (
            generate_teacher_trajectories,
        )

        outdir = Path(self.tmpdir) / "trajectories"
        paths = generate_teacher_trajectories(5, seed=42, outdir=str(outdir))

        for p in paths:
            lines = []
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    lines.append(json.loads(line.strip()))

            step_total = sum(
                l.get("reward", 0.0)
                for l in lines
                if l.get("kind") == "teacher_step"
            )
            end_total = lines[-1].get("total_reward", 0.0)
            # Allow rounding tolerance
            assert abs(step_total - end_total) < 0.1, (
                f"Reward mismatch: steps={step_total} end={end_total}"
            )
