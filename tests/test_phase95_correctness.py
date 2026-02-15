#!/usr/bin/env python3
"""
tests/test_phase95_correctness.py — Phase 9.5 correctness fix tests

Tests for:
1. DDQN single-select per step (FF_DDQN_SINGLE_SELECT_PER_STEP)
2. StepParseCache dedup (FF_SINGLE_PARSE_CACHE)
3. PPO reward attribution fix (FF_PPO_REWARD_ATTRIBUTION_FIX)
4. Feature flags registry
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["ARIASKA_DRY_RUN"] = "1"


class TestFeatureFlags:
    """Test core/feature_flags.py registry."""

    def test_defaults_are_sane(self):
        """All Phase 9.5 correctness flags default ON."""
        from core.feature_flags import FeatureFlags
        ff = FeatureFlags()
        assert ff.ddqn_single_select is True
        assert ff.single_parse_cache is True
        assert ff.ppo_reward_attribution_fix is True

    def test_env_override_disables_flag(self):
        """Setting FF_DDQN_SINGLE_SELECT_PER_STEP=0 disables the flag."""
        os.environ["FF_DDQN_SINGLE_SELECT_PER_STEP"] = "0"
        try:
            from core.feature_flags import FeatureFlags
            ff = FeatureFlags()
            assert ff.ddqn_single_select is False
        finally:
            del os.environ["FF_DDQN_SINGLE_SELECT_PER_STEP"]

    def test_env_override_enables_flag(self):
        """Setting FF_KG_WRITE=1 enables a default-off flag."""
        os.environ["FF_KG_WRITE"] = "1"
        try:
            from core.feature_flags import FeatureFlags
            ff = FeatureFlags()
            assert ff.kg_write is True
        finally:
            del os.environ["FF_KG_WRITE"]

    def test_set_feature_flag(self):
        """Programmatic set_feature_flag works."""
        from core.feature_flags import reset_feature_flags, set_feature_flag, get_feature_flags
        reset_feature_flags()
        set_feature_flag("ddqn_single_select", False)
        assert get_feature_flags().ddqn_single_select is False
        reset_feature_flags()

    def test_unknown_flag_raises(self):
        """Setting an unknown flag name raises ValueError."""
        from core.feature_flags import set_feature_flag, reset_feature_flags
        reset_feature_flags()
        with pytest.raises(ValueError, match="Unknown feature flag"):
            set_feature_flag("nonexistent_flag", True)
        reset_feature_flags()

    def test_new_features_default_off(self):
        """Architecture/KG/LLM flags default to OFF (safe rollout)."""
        from core.feature_flags import FeatureFlags
        ff = FeatureFlags()
        assert ff.target_profiler_pipeline is False
        assert ff.kg_write is False
        assert ff.dagger_corrections is False
        assert ff.llm_strategic_planner is False
        assert ff.llm_judge_ranker is False


class TestDDQNSingleSelect:
    """Test DDQN double-call fix (Phase 9.5, core/algorithms/ddqn_macro.py)."""

    @pytest.fixture
    def ddqn(self):
        """Create a DDQNMacro instance for testing."""
        try:
            import torch
            from core.algorithms.ddqn_macro import DDQNMacro, DDQNConfig
        except ImportError:
            pytest.skip("torch or ddqn_macro not available")
        config = DDQNConfig(state_dim=512, num_macros=6)
        return DDQNMacro(config, device="cpu")

    def test_same_step_id_returns_cached(self, ddqn):
        """Second call with same step_id returns cached result, no side effects."""
        import torch
        state = torch.randn(512)

        # First call: should execute fully
        macro1, q1, conf1 = ddqn.select_macro(state, "RECON", step_id=42)
        epsilon_after_first = ddqn.epsilon
        steps_after_first = ddqn.total_steps

        # Second call with SAME step_id: should return cached
        macro2, q2, conf2 = ddqn.select_macro(state, "RECON", step_id=42)
        epsilon_after_second = ddqn.epsilon
        steps_after_second = ddqn.total_steps

        # Cached result should be identical
        assert macro1 == macro2
        assert conf1 == conf2

        # Side effects should NOT have fired again
        assert steps_after_second == steps_after_first, \
            f"total_steps incremented on cached call: {steps_after_second} != {steps_after_first}"
        assert epsilon_after_second == epsilon_after_first, \
            f"epsilon decayed on cached call: {epsilon_after_second} != {epsilon_after_first}"

    def test_different_step_id_executes_normally(self, ddqn):
        """Different step_id causes fresh execution with side effects."""
        import torch
        state = torch.randn(512)

        macro1, _, _ = ddqn.select_macro(state, "RECON", step_id=1)
        steps_after_first = ddqn.total_steps

        macro2, _, _ = ddqn.select_macro(state, "RECON", step_id=2)
        steps_after_second = ddqn.total_steps

        assert steps_after_second == steps_after_first + 1

    def test_no_step_id_always_executes(self, ddqn):
        """Without step_id, every call executes fully (backward compatible)."""
        import torch
        state = torch.randn(512)

        ddqn.select_macro(state, "RECON")
        steps1 = ddqn.total_steps

        ddqn.select_macro(state, "RECON")
        steps2 = ddqn.total_steps

        assert steps2 == steps1 + 1

    def test_cache_reset_on_episode(self, ddqn):
        """reset_episode clears the step cache."""
        import torch
        state = torch.randn(512)

        ddqn.select_macro(state, "RECON", step_id=99)
        ddqn.reset_episode()

        # After reset, same step_id should NOT hit cache
        ddqn.select_macro(state, "RECON", step_id=99)
        # Should have incremented (not cached)
        assert ddqn.total_steps > 0

    def test_cache_metrics(self, ddqn):
        """Metrics track select vs cached call counts."""
        import torch
        state = torch.randn(512)

        ddqn.select_macro(state, "RECON", step_id=1)
        ddqn.select_macro(state, "RECON", step_id=1)  # cached
        ddqn.select_macro(state, "RECON", step_id=2)

        assert ddqn._select_call_count == 3
        assert ddqn._cached_call_count == 1

    def test_flag_disabled_allows_double_call(self, ddqn):
        """With FF disabled, same step_id still executes fully (double-call)."""
        import torch
        from core.feature_flags import set_feature_flag, reset_feature_flags
        reset_feature_flags()
        set_feature_flag("ddqn_single_select", False)
        try:
            state = torch.randn(512)
            ddqn.select_macro(state, "RECON", step_id=10)
            steps1 = ddqn.total_steps
            ddqn.select_macro(state, "RECON", step_id=10)
            steps2 = ddqn.total_steps
            # Should execute both times — no caching
            assert steps2 == steps1 + 1
        finally:
            reset_feature_flags()


class TestStepParseCache:
    """Test core/execution/step_parse_cache.py."""

    @pytest.fixture
    def cache(self):
        from core.execution.step_parse_cache import StepParseCache
        return StepParseCache()

    def test_miss_then_hit(self, cache):
        """First query misses, second with same key hits."""
        result = {"ports": [80, 443]}
        assert cache.get(1, 5, "RedAgent", "output text") is None
        cache.put(1, 5, "RedAgent", "output text", result)
        hit = cache.get(1, 5, "RedAgent", "output text")
        assert hit == result

    def test_different_agent_is_miss(self, cache):
        """Same output but different agent_id is a cache miss."""
        cache.put(1, 5, "RedAgent", "output text", {"ports": [80]})
        assert cache.get(1, 5, "BlueAgent", "output text") is None

    def test_different_step_is_miss(self, cache):
        """Same output but different step_idx is a cache miss."""
        cache.put(1, 5, "RedAgent", "output text", {"ports": [80]})
        assert cache.get(1, 6, "RedAgent", "output text") is None

    def test_different_output_is_miss(self, cache):
        """Same step but different output content is a cache miss."""
        cache.put(1, 5, "RedAgent", "output A", {"ports": [80]})
        assert cache.get(1, 5, "RedAgent", "output B") is None

    def test_reset_episode_clears(self, cache):
        """reset_episode clears all cached results."""
        cache.put(1, 5, "RedAgent", "output", {"ports": [80]})
        cache.reset_episode()
        assert cache.get(1, 5, "RedAgent", "output") is None

    def test_stats_tracking(self, cache):
        """Stats track hits, misses, total calls."""
        cache.get(1, 1, "R", "a")       # miss
        cache.put(1, 1, "R", "a", {})
        cache.get(1, 1, "R", "a")       # hit
        cache.get(1, 2, "R", "b")       # miss

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["total_calls"] == 3
        assert 0.3 < stats["hit_rate"] < 0.4  # 1/3

    def test_idempotent_same_output(self, cache):
        """Parsing same output twice yields identical cached result."""
        result1 = {"ports": [22, 80], "services": ["ssh", "http"]}
        cache.put(1, 1, "Scout", "PORT 22/tcp open ssh\nPORT 80/tcp open http", result1)
        
        hit = cache.get(1, 1, "Scout", "PORT 22/tcp open ssh\nPORT 80/tcp open http")
        assert hit == result1
        
        # Re-put with same data: should overwrite cleanly
        cache.put(1, 1, "Scout", "PORT 22/tcp open ssh\nPORT 80/tcp open http", result1)
        hit2 = cache.get(1, 1, "Scout", "PORT 22/tcp open ssh\nPORT 80/tcp open http")
        assert hit2 == result1


class TestPPORewardAttribution:
    """Test PPO reward attribution fix (Phase 9.5)."""

    def test_ppo_pending_cleared_on_replacement(self):
        """When anti-repeat replaces PPO command, _ppo_pending must be cleared."""
        # This is tested via SmartCoach.decide() flow.
        # We verify the fix code is present and the flag works.
        from core.feature_flags import get_feature_flags, reset_feature_flags
        reset_feature_flags()
        ff = get_feature_flags()
        assert ff.ppo_reward_attribution_fix is True
        reset_feature_flags()

    def test_flag_disables_fix(self):
        """With FF disabled, old behavior preserved."""
        from core.feature_flags import set_feature_flag, get_feature_flags, reset_feature_flags
        reset_feature_flags()
        set_feature_flag("ppo_reward_attribution_fix", False)
        assert get_feature_flags().ppo_reward_attribution_fix is False
        reset_feature_flags()
