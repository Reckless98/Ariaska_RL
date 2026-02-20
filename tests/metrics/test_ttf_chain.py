"""Phase 42 Stage 5: TTFTracker + ChainScorer unit tests."""

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestTTFTracker:
    """Tests for the TTFTracker module."""

    def _make_tracker(self):
        from core.metrics.ttf_metrics import TTFTracker
        return TTFTracker()

    def test_init(self):
        """TTFTracker initializes empty."""
        t = self._make_tracker()
        assert len(t) == 0

    def test_record_first(self):
        """First record returns True."""
        t = self._make_tracker()
        assert t.record("port", step=5, value="22/tcp") is True
        assert len(t) == 1

    def test_record_duplicate(self):
        """Duplicate record returns False."""
        t = self._make_tracker()
        t.record("port", step=5)
        assert t.record("port", step=10) is False
        assert len(t) == 1

    def test_get_ttf(self):
        """get_ttf returns step number."""
        t = self._make_tracker()
        t.record("shell", step=42)
        assert t.get_ttf("shell") == 42
        assert t.get_ttf("root") is None

    def test_episode_summary(self):
        """episode_summary returns correct structure."""
        t = self._make_tracker()
        t.record("port", step=2, value="80")
        t.record("service", step=5, value="http")
        s = t.episode_summary()
        assert s["achieved_count"] == 2
        assert "port" in s["ttf_by_type"]
        assert "service" in s["ttf_by_type"]

    def test_reset_preserves_history(self):
        """reset clears milestones but saves to history."""
        t = self._make_tracker()
        t.record("port", step=2)
        t.reset()
        assert len(t) == 0
        trend = t.get_trend("port", last_n=10)
        assert len(trend) == 1
        assert trend[0] == 2

    def test_trend(self):
        """get_trend returns TTF values across episodes."""
        t = self._make_tracker()
        # Episode 1
        t.record("port", step=3)
        t.reset()
        # Episode 2
        t.record("port", step=7)
        t.reset()
        trend = t.get_trend("port", last_n=10)
        assert trend == [3, 7]

    def test_case_insensitive(self):
        """Type names are normalized to lowercase."""
        t = self._make_tracker()
        t.record("PORT", step=5)
        assert t.get_ttf("port") == 5


class TestChainScorer:
    """Tests for the ChainScorer module."""

    def _make_scorer(self):
        from core.metrics.chain_scorer import ChainScorer
        return ChainScorer()

    def test_init(self):
        """ChainScorer initializes with zero chain."""
        s = self._make_scorer()
        assert s.get_chain_length() == 0
        assert s.get_momentum() == 0.0

    def test_productive_steps(self):
        """Productive steps build chain."""
        s = self._make_scorer()
        s.record_step(productive=True)
        s.record_step(productive=True)
        assert s.get_chain_length() == 2

    def test_momentum_above_minimum(self):
        """Momentum bonus activates above min_chain."""
        s = self._make_scorer()
        s.record_step(productive=True)
        assert s.get_momentum() == 0.0  # chain=1, min=2
        s.record_step(productive=True)
        assert s.get_momentum() > 0.0  # chain=2, min=2

    def test_failure_decays_chain(self):
        """Non-productive step decays chain."""
        s = self._make_scorer()
        for _ in range(5):
            s.record_step(productive=True)
        s.record_step(productive=False)
        assert s.get_chain_length() < 5

    def test_momentum_capped(self):
        """Momentum doesn't exceed max_bonus."""
        from core.metrics.chain_scorer import ChainScorer, ChainConfig
        cfg = ChainConfig(max_bonus=3.0)
        s = ChainScorer(config=cfg)
        for _ in range(100):
            s.record_step(productive=True)
        assert s.get_momentum() <= 3.0

    def test_summary(self):
        """summary returns correct structure."""
        s = self._make_scorer()
        s.record_step(productive=True)
        s.record_step(productive=True)
        s.record_step(productive=False)
        summary = s.summary()
        assert "chain_length" in summary
        assert "max_chain" in summary
        assert summary["max_chain"] == 2
        assert summary["total_steps"] == 3

    def test_reset(self):
        """reset clears chain."""
        s = self._make_scorer()
        s.record_step(productive=True)
        s.record_step(productive=True)
        s.reset()
        assert s.get_chain_length() == 0
