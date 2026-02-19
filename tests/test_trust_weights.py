"""
tests/test_trust_weights.py — Phase 39.2: TrustWeightEngine tests

Tests per-source trust tracking, cosine annealing, prior computation,
evidence recording, snapshot/diagnostics, and reset.
"""

import os
import math
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")

from core.ops.trust_weights import (
    ANNEAL_INITIAL_WEIGHT,
    ANNEAL_MIN_WEIGHT,
    ANNEAL_TOTAL_STEPS,
    INITIAL_TRUST,
    MAX_TRUST,
    MIN_TRUST,
    TRUST_PENALTY_CONTRADICTION,
    TRUST_PENALTY_FAILED,
    TRUST_PENALTY_LOW_NOVELTY,
    TRUST_REWARD_VALIDATED,
    TrustInfluenceResult,
    TrustSnapshot,
    TrustWeightEngine,
)


class TestConstantsValid:
    """Sanity checks on module constants."""

    def test_trust_bounds(self):
        assert 0.0 < MIN_TRUST < INITIAL_TRUST < MAX_TRUST <= 1.0

    def test_penalties_positive(self):
        assert TRUST_PENALTY_FAILED > 0.0
        assert TRUST_PENALTY_CONTRADICTION > 0.0
        assert TRUST_PENALTY_LOW_NOVELTY > 0.0

    def test_reward_positive(self):
        assert TRUST_REWARD_VALIDATED > 0.0

    def test_anneal_bounds(self):
        assert 0.0 < ANNEAL_MIN_WEIGHT < ANNEAL_INITIAL_WEIGHT <= 1.0
        assert ANNEAL_TOTAL_STEPS > 0

    def test_contradiction_worse_than_failure(self):
        assert TRUST_PENALTY_CONTRADICTION > TRUST_PENALTY_FAILED


class TestTrustSnapshot:
    """Tests for TrustSnapshot dataclass."""

    def test_defaults(self):
        s = TrustSnapshot(source="gpt", trust=0.75)
        assert s.predictions_total == 0
        assert s.predictions_validated == 0

    def test_fields(self):
        s = TrustSnapshot(
            source="mentor",
            trust=0.6,
            predictions_total=10,
            predictions_validated=7,
            predictions_failed=3,
            contradictions=1,
        )
        assert s.source == "mentor"
        assert s.predictions_validated == 7


class TestTrustInfluenceResult:
    """Tests for TrustInfluenceResult dataclass."""

    def test_defaults(self):
        r = TrustInfluenceResult()
        assert r.prior_vector == []
        assert r.prior_magnitude == 0.0
        assert not r.changed_action


class TestTrustWeightEngineInit:
    """Tests for TrustWeightEngine initialization."""

    def test_init_defaults(self):
        e = TrustWeightEngine()
        assert e._action_dim == 5
        assert e._global_step == 0

    def test_custom_action_dim(self):
        e = TrustWeightEngine(action_dim=10)
        assert e._action_dim == 10

    def test_custom_anneal_steps(self):
        e = TrustWeightEngine(anneal_total_steps=500)
        assert e._anneal_total == 500


class TestSourceRegistration:
    """Tests for source registration + trust init."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine = TrustWeightEngine()

    def test_register_source(self):
        self.engine.register_source("gpt")
        assert self.engine.get_trust("gpt") == INITIAL_TRUST

    def test_register_multiple(self):
        self.engine.register_source("gpt")
        self.engine.register_source("heuristic")
        assert self.engine.get_trust("gpt") == INITIAL_TRUST
        assert self.engine.get_trust("heuristic") == INITIAL_TRUST

    def test_double_register_idempotent(self):
        self.engine.register_source("gpt")
        self.engine.record_validated("gpt", step=1)
        t = self.engine.get_trust("gpt")
        self.engine.register_source("gpt")  # Should not reset
        assert self.engine.get_trust("gpt") == t

    def test_auto_register_on_get_trust(self):
        t = self.engine.get_trust("new_source")
        assert t == INITIAL_TRUST


class TestTrustAdjustment:
    """Tests for record_validated / record_failed / etc."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine = TrustWeightEngine()
        self.engine.register_source("gpt")

    def test_validated_increases_trust(self):
        old = self.engine.get_trust("gpt")
        new = self.engine.record_validated("gpt", step=1)
        assert new > old
        assert new == old + TRUST_REWARD_VALIDATED

    def test_failed_decreases_trust(self):
        old = self.engine.get_trust("gpt")
        new = self.engine.record_failed("gpt", step=1)
        assert new < old
        assert new == old - TRUST_PENALTY_FAILED

    def test_contradiction_decreases_more(self):
        old = self.engine.get_trust("gpt")
        new = self.engine.record_contradiction("gpt", step=1)
        assert new < old
        assert new == old - TRUST_PENALTY_CONTRADICTION

    def test_low_novelty_slight_decrease(self):
        old = self.engine.get_trust("gpt")
        new = self.engine.record_low_novelty("gpt", step=1)
        assert new < old
        assert new == old - TRUST_PENALTY_LOW_NOVELTY

    def test_trust_never_exceeds_max(self):
        for i in range(100):
            self.engine.record_validated("gpt", step=i)
        assert self.engine.get_trust("gpt") <= MAX_TRUST

    def test_trust_never_below_min(self):
        for i in range(100):
            self.engine.record_failed("gpt", step=i)
        assert self.engine.get_trust("gpt") >= MIN_TRUST

    def test_mixed_sequence(self):
        self.engine.record_validated("gpt", step=1)  # +0.05
        self.engine.record_validated("gpt", step=2)  # +0.05
        self.engine.record_failed("gpt", step=3)     # -0.08
        expected = INITIAL_TRUST + 2 * TRUST_REWARD_VALIDATED - TRUST_PENALTY_FAILED
        assert abs(self.engine.get_trust("gpt") - expected) < 1e-9


class TestCosineAnnealing:
    """Tests for compute_anneal_weight with cosine schedule."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine = TrustWeightEngine()

    def test_anneal_at_step_zero(self):
        w = self.engine.compute_anneal_weight(step=0)
        assert abs(w - ANNEAL_INITIAL_WEIGHT) < 0.01

    def test_anneal_at_end(self):
        w = self.engine.compute_anneal_weight(step=ANNEAL_TOTAL_STEPS)
        assert abs(w - ANNEAL_MIN_WEIGHT) < 0.01

    def test_anneal_midpoint(self):
        w = self.engine.compute_anneal_weight(step=ANNEAL_TOTAL_STEPS // 2)
        midpoint = (ANNEAL_INITIAL_WEIGHT + ANNEAL_MIN_WEIGHT) / 2
        assert abs(w - midpoint) < 0.05

    def test_anneal_monotonic_decrease(self):
        weights = [
            self.engine.compute_anneal_weight(step=s)
            for s in range(0, ANNEAL_TOTAL_STEPS + 1, 100)
        ]
        for i in range(1, len(weights)):
            assert weights[i] <= weights[i - 1] + 0.01

    def test_anneal_never_below_min(self):
        w = self.engine.compute_anneal_weight(step=ANNEAL_TOTAL_STEPS * 10)
        assert w >= ANNEAL_MIN_WEIGHT


class TestComputePrior:
    """Tests for trust-weighted prior computation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine = TrustWeightEngine(action_dim=5)
        self.engine.register_source("gpt")

    def test_prior_basic(self):
        prefs = [0.8, 0.1, 0.05, 0.03, 0.02]
        result = self.engine.compute_prior("gpt", prefs, current_step=0)
        assert isinstance(result, TrustInfluenceResult)
        assert len(result.prior_vector) == 5
        assert result.trust_score == INITIAL_TRUST
        assert result.source == "gpt"

    def test_prior_sums_correctly(self):
        prefs = [1.0, 0.0, 0.0, 0.0, 0.0]
        result = self.engine.compute_prior("gpt", prefs, current_step=0)
        # Magnitude = trust * anneal
        expected_mag = INITIAL_TRUST * ANNEAL_INITIAL_WEIGHT
        assert abs(result.prior_magnitude - expected_mag) < 0.02
        # First element should be magnitude (since it's 100% of preference)
        assert abs(result.prior_vector[0] - expected_mag) < 0.02

    def test_prior_uniform(self):
        prefs = [0.2, 0.2, 0.2, 0.2, 0.2]
        result = self.engine.compute_prior("gpt", prefs, current_step=0)
        for i in range(5):
            assert abs(result.prior_vector[i] - result.prior_vector[0]) < 0.001

    def test_prior_zero_prefs(self):
        prefs = [0.0, 0.0, 0.0, 0.0, 0.0]
        result = self.engine.compute_prior("gpt", prefs, current_step=0)
        # Should fall back to uniform
        assert len(result.prior_vector) == 5
        for v in result.prior_vector:
            assert v > 0

    def test_prior_short_prefs_padded(self):
        prefs = [0.5, 0.5]
        result = self.engine.compute_prior("gpt", prefs, current_step=0)
        assert len(result.prior_vector) == 5

    def test_prior_decreases_with_step(self):
        prefs = [1.0, 0.0, 0.0, 0.0, 0.0]
        r0 = self.engine.compute_prior("gpt", prefs, current_step=0)
        r_end = self.engine.compute_prior("gpt", prefs, current_step=ANNEAL_TOTAL_STEPS)
        assert r0.prior_magnitude > r_end.prior_magnitude


class TestSnapshot:
    """Tests for get_snapshot / get_all_snapshots."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine = TrustWeightEngine()
        self.engine.register_source("alpha")
        self.engine.register_source("beta")

    def test_single_snapshot(self):
        self.engine.record_validated("alpha", step=1)
        snap = self.engine.get_snapshot("alpha")
        assert snap.source == "alpha"
        assert snap.predictions_validated == 1
        assert snap.trust > INITIAL_TRUST

    def test_all_snapshots(self):
        snaps = self.engine.get_all_snapshots()
        assert "alpha" in snaps
        assert "beta" in snaps

    def test_snapshot_auto_register(self):
        snap = self.engine.get_snapshot("unknown_source")
        assert snap.source == "unknown_source"
        assert snap.trust == INITIAL_TRUST


class TestDiagnostics:
    """Tests for get_diagnostics()."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine = TrustWeightEngine()
        self.engine.register_source("gpt")

    def test_diagnostics_structure(self):
        diag = self.engine.get_diagnostics()
        assert "sources" in diag
        assert "global_step" in diag
        assert "anneal_weight" in diag
        assert "total_influences" in diag
        assert "gpt" in diag["sources"]

    def test_diagnostics_reflects_state(self):
        self.engine.record_failed("gpt", step=5)
        self.engine.set_global_step(50)
        diag = self.engine.get_diagnostics()
        assert diag["global_step"] == 50
        assert diag["sources"]["gpt"]["failed"] == 1


class TestReset:
    """Tests for reset behavior."""

    def test_reset_clears_all(self):
        engine = TrustWeightEngine()
        engine.register_source("gpt")
        engine.record_validated("gpt", step=1)
        engine.set_global_step(100)
        engine.reset()
        assert engine._global_step == 0
        assert len(engine._sources) == 0
        # After reset, auto-reg gives initial trust
        assert engine.get_trust("gpt") == INITIAL_TRUST

    def test_influence_log_cleared(self):
        engine = TrustWeightEngine()
        engine.register_source("x")
        engine.compute_prior("x", [0.5, 0.5, 0, 0, 0])
        assert len(engine._influence_log) > 0
        engine.reset()
        assert len(engine._influence_log) == 0


class TestSetGlobalStep:
    """Tests for set_global_step."""

    def test_step_update(self):
        e = TrustWeightEngine()
        e.set_global_step(42)
        assert e._global_step == 42

    def test_anneal_uses_global_step(self):
        e = TrustWeightEngine()
        e.set_global_step(ANNEAL_TOTAL_STEPS // 2)
        w = e.compute_anneal_weight()
        midpoint = (ANNEAL_INITIAL_WEIGHT + ANNEAL_MIN_WEIGHT) / 2
        assert abs(w - midpoint) < 0.05


class TestInfluenceLog:
    """Tests for influence logging."""

    def test_log_records_on_compute_prior(self):
        e = TrustWeightEngine()
        e.register_source("s")
        e.compute_prior("s", [1.0, 0, 0, 0, 0])
        log = e.get_influence_log()
        assert len(log) == 1
        assert log[0]["source"] == "s"

    def test_log_bounded_at_500(self):
        e = TrustWeightEngine()
        e.register_source("s")
        for i in range(600):
            e.compute_prior("s", [0.2, 0.2, 0.2, 0.2, 0.2], current_step=i)
        assert len(e._influence_log) <= 500
