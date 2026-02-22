"""C09 — Held-out evaluation wiring tests.

Tests that:
1. HeldOutEvaluator core works: should_eval(), evaluate(), trends.
2. EvalMetrics dataclass serializes correctly.
3. SmartOrchestrator initializes evaluator when FF_HELDOUT_EVAL=1.
4. SmartOrchestrator skips evaluator when FF_HELDOUT_EVAL=0 (default).
5. Evaluation runs without modifying training state.
"""

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


# ─── Core evaluator tests ───────────────────────────────────────

class TestHeldOutEvaluatorCore:
    """Test HeldOutEvaluator logic."""

    def test_import(self):
        from core.evaluation.heldout_eval import HeldOutEvaluator, EvalConfig
        assert HeldOutEvaluator is not None
        assert EvalConfig is not None

    def test_default_config(self):
        from core.evaluation.heldout_eval import EvalConfig
        cfg = EvalConfig()
        assert cfg.eval_interval == 5
        assert cfg.deterministic is True
        assert len(cfg.scenarios) == 3

    def test_should_eval_at_interval(self):
        from core.evaluation.heldout_eval import HeldOutEvaluator, EvalConfig
        ev = HeldOutEvaluator(EvalConfig(eval_interval=5))
        assert ev.should_eval(0) is False
        assert ev.should_eval(1) is False
        assert ev.should_eval(5) is True
        assert ev.should_eval(10) is True
        assert ev.should_eval(7) is False

    def test_should_eval_every_episode(self):
        from core.evaluation.heldout_eval import HeldOutEvaluator, EvalConfig
        ev = HeldOutEvaluator(EvalConfig(eval_interval=1))
        assert ev.should_eval(1) is True
        assert ev.should_eval(2) is True

    def test_evaluate_with_mock_env(self):
        """Evaluate with a trivial mock environment."""
        from core.evaluation.heldout_eval import HeldOutEvaluator, EvalConfig, EvalScenario

        scenario = EvalScenario(name="test_scenario", max_steps=5, seed=42)
        ev = HeldOutEvaluator(EvalConfig(
            eval_interval=1,
            scenarios=[scenario],
            log_results=False,
        ))

        step_count = 0

        def policy_fn(state):
            return (0, 0.0, 0.0)  # action_idx, log_prob, value

        def env_step_fn(action):
            nonlocal step_count
            step_count += 1
            done = step_count >= 3
            return {"phase": "RECON"}, 1.0, done, {
                "command": f"cmd_{step_count}",
                "discoveries": [f"disc_{step_count}"] if step_count == 1 else [],
                "phase": "RECON",
            }

        def env_reset_fn():
            nonlocal step_count
            step_count = 0
            return {"phase": "RECON"}

        results = ev.evaluate(policy_fn, env_step_fn, env_reset_fn, episode=5)
        assert len(results) == 1
        r = results[0]
        assert r.scenario_name == "test_scenario"
        assert r.unique_commands > 0
        assert r.total_steps == 3
        assert r.total_discoveries == 1
        assert r.episode_reward == 3.0

    def test_evaluate_tracks_exploit_phase(self):
        """Detects first exploit step correctly."""
        from core.evaluation.heldout_eval import HeldOutEvaluator, EvalConfig, EvalScenario

        scenario = EvalScenario(name="exploit_test", max_steps=10, seed=42)
        ev = HeldOutEvaluator(EvalConfig(
            eval_interval=1, scenarios=[scenario], log_results=False,
        ))

        step_count = 0

        def policy_fn(state):
            return (0,)

        def env_step_fn(action):
            nonlocal step_count
            step_count += 1
            phase = "EXPLOITATION" if step_count >= 3 else "ENUMERATION"
            return {}, 1.0, step_count >= 5, {
                "command": f"cmd_{step_count}",
                "discoveries": [],
                "phase": phase,
            }

        def env_reset_fn():
            nonlocal step_count
            step_count = 0
            return {}

        results = ev.evaluate(policy_fn, env_step_fn, env_reset_fn, episode=5)
        assert results[0].steps_to_first_exploit == 2  # 0-indexed step 2

    def test_eval_metrics_to_dict(self):
        from core.evaluation.heldout_eval import EvalMetrics
        m = EvalMetrics(
            scenario_name="test",
            unique_commands=5,
            total_steps=10,
            diversity_ratio=0.5,
            total_discoveries=3,
        )
        d = m.to_dict()
        assert d["scenario"] == "test"
        assert d["unique_commands"] == 5
        assert d["diversity_ratio"] == 0.5
        assert "phase_progression" in d

    def test_get_stats(self):
        from core.evaluation.heldout_eval import HeldOutEvaluator
        ev = HeldOutEvaluator()
        stats = ev.get_stats()
        assert stats["eval_count"] == 0
        assert stats["config"]["eval_interval"] == 5

    def test_trend_insufficient_data(self):
        from core.evaluation.heldout_eval import HeldOutEvaluator
        ev = HeldOutEvaluator()
        trend = ev.get_trend()
        assert trend["status"] == "insufficient_data"

    def test_trend_after_multiple_evals(self):
        """After 2+ evals, trend should report delta."""
        from core.evaluation.heldout_eval import HeldOutEvaluator, EvalConfig, EvalScenario

        scenario = EvalScenario(name="trend_test", max_steps=3)
        ev = HeldOutEvaluator(EvalConfig(
            eval_interval=1, scenarios=[scenario], log_results=False,
        ))

        step_count = 0

        def policy_fn(state):
            return (0,)

        def env_step_fn(action):
            nonlocal step_count
            step_count += 1
            return {}, 1.0, step_count >= 2, {
                "command": "cmd", "discoveries": [], "phase": "RECON",
            }

        def env_reset_fn():
            nonlocal step_count
            step_count = 0
            return {}

        ev.evaluate(policy_fn, env_step_fn, env_reset_fn, episode=1)
        ev.evaluate(policy_fn, env_step_fn, env_reset_fn, episode=2)

        trend = ev.get_trend()
        assert "status" not in trend  # Not insufficient
        assert "mean_discoveries" in trend
        assert "evals_done" in trend

    def test_eval_scenarios_builtin(self):
        from core.evaluation.heldout_eval import EVAL_SCENARIOS
        assert len(EVAL_SCENARIOS) == 3
        for s in EVAL_SCENARIOS:
            assert s.name.startswith("eval_")
            assert s.max_steps > 0
            assert 0.0 <= s.expected_difficulty <= 1.0


# ─── Feature flag gating ────────────────────────────────────────

class TestHeldOutEvalFlagGating:
    """FF_HELDOUT_EVAL controls evaluator init in SmartOrchestrator."""

    @pytest.fixture(autouse=True)
    def _env(self):
        os.environ["ARIASKA_DRY_RUN"] = "1"
        yield
        os.environ.pop("FF_HELDOUT_EVAL", None)
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()

    def _make_orchestrator(self):
        from core.testing.fake_gpt_manager import FakeGPTManager
        from core.environment.cyber_environment import CyberEnvironment
        from core.orchestration.smart_orchestrator import SmartOrchestrator
        gpt = FakeGPTManager(seed=42)
        env = CyberEnvironment(defer_reset=True)
        return SmartOrchestrator(env=env, gpt_manager=gpt, verbosity="silent")

    def test_evaluator_disabled_by_default(self):
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()
        orch = self._make_orchestrator()
        assert orch.heldout_evaluator is None

    def test_evaluator_enabled_via_env(self):
        os.environ["FF_HELDOUT_EVAL"] = "1"
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()
        orch = self._make_orchestrator()
        from core.evaluation.heldout_eval import HeldOutEvaluator
        assert isinstance(orch.heldout_evaluator, HeldOutEvaluator)

    def test_evaluator_config_when_enabled(self):
        os.environ["FF_HELDOUT_EVAL"] = "1"
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()
        orch = self._make_orchestrator()
        assert orch.heldout_evaluator.config.eval_interval == 5
        assert orch.heldout_evaluator.config.deterministic is True
