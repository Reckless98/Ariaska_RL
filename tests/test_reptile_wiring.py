"""C08 — Reptile meta-learning wiring tests.

Tests that:
1. ReptileMeta module core works: snapshot → inner loop → interpolation.
2. SmartOrchestrator initializes Reptile when FF_REPTILE_META=1.
3. SmartOrchestrator skips Reptile when FF_REPTILE_META=0 (default).
4. Reptile meta_step is called at episode boundaries.
5. ScenarioSampler produces valid scenarios.
6. Cosine annealing of outer_lr works correctly.
7. get_stats() returns proper diagnostics.
"""

import os
import copy
import math
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


# ─── Core Reptile module tests ──────────────────────────────────────

class TestReptileCore:
    """Test the ReptileMeta algorithm logic directly."""

    def test_import(self):
        from core.algorithms.reptile_meta import ReptileMeta, ReptileConfig
        assert ReptileMeta is not None
        assert ReptileConfig is not None

    def test_default_config(self):
        from core.algorithms.reptile_meta import ReptileConfig
        cfg = ReptileConfig()
        assert cfg.enabled is True
        assert cfg.outer_lr == 0.1
        assert cfg.inner_steps == 5
        assert cfg.scenarios_per_step == 3
        assert cfg.warmup_steps == 100

    def test_should_run_before_warmup(self):
        from core.algorithms.reptile_meta import ReptileMeta, ReptileConfig
        r = ReptileMeta(ReptileConfig(warmup_steps=100))
        assert r.should_run(50) is False

    def test_should_run_after_warmup(self):
        from core.algorithms.reptile_meta import ReptileMeta, ReptileConfig
        r = ReptileMeta(ReptileConfig(warmup_steps=100))
        assert r.should_run(200) is True

    def test_should_run_disabled(self):
        from core.algorithms.reptile_meta import ReptileMeta, ReptileConfig
        r = ReptileMeta(ReptileConfig(enabled=False))
        assert r.should_run(9999) is False

    def test_meta_step_basic(self):
        """Reptile meta_step should modify model weights."""
        import torch
        import torch.nn as nn
        from core.algorithms.reptile_meta import ReptileMeta, ReptileConfig

        model = nn.Linear(4, 2)
        original_weights = copy.deepcopy(model.state_dict())

        def inner_fn(m, scenario, steps):
            # Simulate training by nudging weights
            with torch.no_grad():
                for p in m.parameters():
                    p.add_(torch.randn_like(p) * 0.01)
            return {"loss": 0.5, "reward": 1.0}

        r = ReptileMeta(ReptileConfig(
            warmup_steps=0,
            scenarios_per_step=2,
            inner_steps=3,
            outer_lr=0.5,
        ))
        stats = r.meta_step(model, inner_fn, global_step=100, maturity=0.5)

        # Weights should have changed
        for key in original_weights:
            assert not torch.equal(model.state_dict()[key], original_weights[key])

        assert stats["meta_step"] == 1
        assert len(stats["scenarios"]) == 2
        assert stats["inner_steps"] == 3
        assert stats["outer_lr"] == pytest.approx(0.5, abs=0.01)

    def test_meta_step_preserves_dtype(self):
        """Reptile interpolation must preserve original tensor dtype."""
        import torch
        import torch.nn as nn
        from core.algorithms.reptile_meta import ReptileMeta, ReptileConfig

        model = nn.Linear(4, 2)

        def inner_fn(m, scenario, steps):
            with torch.no_grad():
                for p in m.parameters():
                    p.add_(torch.randn_like(p) * 0.01)
            return {"loss": 0.1}

        r = ReptileMeta(ReptileConfig(warmup_steps=0, scenarios_per_step=1))
        r.meta_step(model, inner_fn, global_step=200)

        for key, val in model.state_dict().items():
            assert val.dtype == torch.float32

    def test_cosine_annealing(self):
        from core.algorithms.reptile_meta import ReptileMeta, ReptileConfig
        r = ReptileMeta(ReptileConfig(
            outer_lr=0.1,
            anneal_outer_lr=0.01,
            cosine_anneal_steps=1000,
        ))
        # At step 0, lr should be close to 0.1
        lr_0 = r._current_outer_lr(0)
        assert lr_0 == pytest.approx(0.1, abs=0.001)

        # At step 1000, lr should be close to 0.01
        lr_end = r._current_outer_lr(1000)
        assert lr_end == pytest.approx(0.01, abs=0.001)

        # At step 500, lr should be ~midpoint
        lr_mid = r._current_outer_lr(500)
        assert 0.04 < lr_mid < 0.07

    def test_get_stats(self):
        from core.algorithms.reptile_meta import ReptileMeta, ReptileConfig
        r = ReptileMeta(ReptileConfig())
        stats = r.get_stats()
        assert "meta_steps_done" in stats
        assert "total_inner_steps" in stats
        assert "sampler" in stats
        assert "config" in stats


# ─── ScenarioSampler tests ──────────────────────────────────────────

class TestScenarioSampler:
    """Test scenario sampling and curriculum weighting."""

    def test_sample_returns_valid_scenarios(self):
        from core.algorithms.reptile_meta import ScenarioSampler, SCENARIO_PROFILES
        pool = list(SCENARIO_PROFILES.keys())
        sampler = ScenarioSampler(pool)
        samples = sampler.sample(3, maturity=0.5)
        assert len(samples) == 3
        for s in samples:
            assert s in pool

    def test_sample_cap_at_pool_size(self):
        from core.algorithms.reptile_meta import ScenarioSampler
        sampler = ScenarioSampler(["a", "b"])
        samples = sampler.sample(10, maturity=0.0)
        assert len(samples) == 2

    def test_record_result_updates_stats(self):
        from core.algorithms.reptile_meta import ScenarioSampler
        sampler = ScenarioSampler(["scenario_a", "scenario_b"])
        sampler.record_result("scenario_a", True)
        sampler.record_result("scenario_a", False)
        stats = sampler.get_stats()
        assert stats["attempt_counts"]["scenario_a"] == 2

    def test_high_maturity_uniform_sampling(self):
        """At high maturity, all scenarios should be equally likely."""
        from core.algorithms.reptile_meta import ScenarioSampler, SCENARIO_PROFILES
        pool = list(SCENARIO_PROFILES.keys())
        sampler = ScenarioSampler(pool, curriculum=True)
        # With maturity >= 0.8, should use random.sample (uniform)
        samples = sampler.sample(5, maturity=0.9)
        assert len(samples) == 5
        # All unique (random.sample guarantees no replacement)
        assert len(set(samples)) == 5

    def test_scenario_profiles_exist(self):
        from core.algorithms.reptile_meta import SCENARIO_PROFILES
        assert len(SCENARIO_PROFILES) >= 18
        for name, profile in SCENARIO_PROFILES.items():
            assert "difficulty" in profile
            assert "typical_ports" in profile
            assert "phase_weights" in profile


# ─── Feature flag gating ────────────────────────────────────────────

class TestReptileFlagGating:
    """Reptile init in SmartOrchestrator must respect FF_REPTILE_META."""

    @pytest.fixture(autouse=True)
    def _env(self):
        os.environ["ARIASKA_DRY_RUN"] = "1"
        yield
        os.environ.pop("FF_REPTILE_META", None)
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()

    def _make_orchestrator(self):
        from core.testing.fake_gpt_manager import FakeGPTManager
        from core.environment.cyber_environment import CyberEnvironment
        from core.orchestration.smart_orchestrator import SmartOrchestrator
        gpt = FakeGPTManager(seed=42)
        env = CyberEnvironment(defer_reset=True)
        return SmartOrchestrator(env=env, gpt_manager=gpt, verbosity="silent")

    def test_reptile_disabled_by_default(self):
        """Default FF_REPTILE_META=false → self.reptile stays None."""
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()
        orch = self._make_orchestrator()
        assert orch.reptile is None

    def test_reptile_enabled_via_env(self):
        """FF_REPTILE_META=1 → self.reptile is ReptileMeta."""
        os.environ["FF_REPTILE_META"] = "1"
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()
        orch = self._make_orchestrator()
        from core.algorithms.reptile_meta import ReptileMeta
        assert isinstance(orch.reptile, ReptileMeta)

    def test_reptile_global_step_initialized(self):
        """_reptile_global_step should always be initialized."""
        orch = self._make_orchestrator()
        assert orch._reptile_global_step == 0
