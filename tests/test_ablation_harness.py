"""Tests for C12: AblationHarness — systematic module evaluation.

Validates:
 - ABLATION_MODULES registry completeness
 - AblationMetrics serialization
 - AblationResult delta computation
 - AblationHarness construction and config
 - Feature flag toggling during ablation
 - Report generation (Rich table)
"""
from __future__ import annotations

import json
import os
import pytest

os.environ["ARIASKA_DRY_RUN"] = "1"

from scripts.ablation_harness import (
    ABLATION_MODULES,
    DEFAULT_MODULES,
    AblationHarness,
    AblationMetrics,
    AblationResult,
)


# ───────── AblationMetrics ──────────────────────────────────────────

class TestAblationMetrics:
    def test_defaults(self) -> None:
        m = AblationMetrics(module="test", description="d")
        assert m.episodes == 0
        assert m.total_reward == 0.0
        assert m.diversity_ratio == 0.0

    def test_to_dict(self) -> None:
        m = AblationMetrics(
            module="cognition_node",
            description="CognitionNode",
            episodes=3,
            total_discoveries=10,
        )
        d = m.to_dict()
        assert d["module"] == "cognition_node"
        assert d["episodes"] == 3
        assert d["total_discoveries"] == 10
        assert isinstance(d["diversity_ratio"], float)


# ───────── AblationResult ───────────────────────────────────────────

class TestAblationResult:
    def test_compute_deltas(self) -> None:
        baseline = AblationMetrics(
            module="baseline", description="all ON",
            unique_commands=20, unique_templates=15,
            total_discoveries=10, diversity_ratio=0.5,
            total_reward=100.0,
        )
        ablation = AblationMetrics(
            module="cognition_node", description="...",
            unique_commands=18, unique_templates=12,
            total_discoveries=8, diversity_ratio=0.45,
            total_reward=90.0,
        )
        result = AblationResult(baseline=baseline)
        result.ablations["cognition_node"] = ablation
        result.compute_deltas()

        d = result.deltas["cognition_node"]
        assert d["unique_commands"] == -2
        assert d["total_discoveries"] == -2
        assert d["diversity_ratio"] == pytest.approx(-0.05, abs=1e-4)
        assert d["total_reward"] == pytest.approx(-10.0)

    def test_to_dict(self) -> None:
        baseline = AblationMetrics(module="baseline", description="all")
        result = AblationResult(baseline=baseline)
        d = result.to_dict()
        assert "baseline" in d
        assert "ablations" in d
        assert "deltas" in d
        assert "config" in d

    def test_multiple_ablations(self) -> None:
        baseline = AblationMetrics(
            module="baseline", description="all",
            unique_commands=20, total_discoveries=10,
        )
        result = AblationResult(baseline=baseline)
        for mod in ["cognition_node", "sac_shadow"]:
            result.ablations[mod] = AblationMetrics(
                module=mod, description="...",
                unique_commands=15, total_discoveries=7,
            )
        result.compute_deltas()
        assert len(result.deltas) == 2
        for d in result.deltas.values():
            assert d["unique_commands"] == -5
            assert d["total_discoveries"] == -3


# ───────── ABLATION_MODULES registry ────────────────────────────────

class TestModuleRegistry:
    def test_default_modules_subset_of_all(self) -> None:
        for m in DEFAULT_MODULES:
            assert m in ABLATION_MODULES, f"{m} not in ABLATION_MODULES"

    def test_core_modules_present(self) -> None:
        required = ["cognition_node", "sac_shadow", "neuromodulators"]
        for r in required:
            assert r in ABLATION_MODULES

    def test_descriptions_non_empty(self) -> None:
        for name, desc in ABLATION_MODULES.items():
            assert len(desc) > 0, f"{name} has empty description"


# ───────── AblationHarness construction ─────────────────────────────

class TestHarnessConstruction:
    def test_defaults(self) -> None:
        h = AblationHarness()
        assert h.episodes == 3
        assert h.max_steps == 50
        assert h.seed == 42
        assert len(h.modules) == len(DEFAULT_MODULES)

    def test_custom_modules(self) -> None:
        h = AblationHarness(modules=["cognition_node", "sac_shadow"])
        assert h.modules == ["cognition_node", "sac_shadow"]

    def test_custom_episodes(self) -> None:
        h = AblationHarness(episodes_per_condition=10, max_steps_per_episode=100)
        assert h.episodes == 10
        assert h.max_steps == 100


# ───────── Feature flag toggling ────────────────────────────────────

class TestFlagToggling:
    def setup_method(self) -> None:
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()

    def teardown_method(self) -> None:
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()

    def test_flag_can_be_disabled(self) -> None:
        from core.feature_flags import get_feature_flags, set_feature_flag
        ff = get_feature_flags()
        assert ff.cognition_node is True
        set_feature_flag("cognition_node", False)
        assert get_feature_flags().cognition_node is False

    def test_flag_resets_after(self) -> None:
        from core.feature_flags import (
            get_feature_flags,
            reset_feature_flags,
            set_feature_flag,
        )
        set_feature_flag("cognition_node", False)
        reset_feature_flags()
        assert get_feature_flags().cognition_node is True

    def test_unknown_flag_raises(self) -> None:
        from core.feature_flags import set_feature_flag
        with pytest.raises(ValueError, match="Unknown"):
            set_feature_flag("nonexistent_flag_xyz", False)


# ───────── Report generation ────────────────────────────────────────

class TestReport:
    def test_print_report_no_crash(self) -> None:
        """print_report should not crash with minimal data."""
        baseline = AblationMetrics(module="baseline", description="all")
        result = AblationResult(baseline=baseline)
        result.ablations["cognition_node"] = AblationMetrics(
            module="cognition_node", description="test",
        )
        result.compute_deltas()

        h = AblationHarness()
        # Should complete without error
        h.print_report(result)

    def test_result_serializable(self) -> None:
        """Full result should be JSON-serializable."""
        baseline = AblationMetrics(module="baseline", description="all")
        result = AblationResult(
            baseline=baseline,
            config={"episodes": 3, "steps": 50},
        )
        result.ablations["sac_shadow"] = AblationMetrics(
            module="sac_shadow", description="...",
        )
        result.compute_deltas()
        j = json.dumps(result.to_dict())
        assert len(j) > 0
        parsed = json.loads(j)
        assert "baseline" in parsed
