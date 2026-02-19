"""Tests for B2: DAgger wiring."""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestDAggerConfig:
    def test_defaults(self):
        from core.training.dagger import DAggerConfig
        cfg = DAggerConfig()
        assert cfg.capacity == 4000
        assert cfg.mix_ratio == 0.5
        assert cfg.dedup is True

    def test_custom(self):
        from core.training.dagger import DAggerConfig
        cfg = DAggerConfig(capacity=100, mix_ratio=0.3)
        assert cfg.capacity == 100


class TestDAggerBuffer:
    def _make_buffer(self, **kwargs):
        from core.training.dagger import DAggerBuffer, DAggerConfig
        cfg = DAggerConfig(**kwargs)
        return DAggerBuffer(cfg)

    def test_store_and_len(self):
        buf = self._make_buffer(capacity=10)
        ok = buf.store(
            state_hash="h1", state_vector=[0.0] * 10,
            mentor_action_idx=2, mentor_command="nmap -sV",
            policy_action_idx=1, policy_command="nmap -sS",
            mentor_confidence=0.9, phase="RECON",
            episode=1, step=5,
        )
        assert ok is True
        assert len(buf) == 1

    def test_dedup(self):
        buf = self._make_buffer(capacity=10, dedup=True)
        for _ in range(3):
            buf.store(
                state_hash="h1", state_vector=[0.0] * 10,
                mentor_action_idx=2, mentor_command="nmap -sV",
                policy_action_idx=1, policy_command="nmap -sS",
                mentor_confidence=0.9, phase="RECON",
                episode=1, step=5,
            )
        assert len(buf) == 1
        stats = buf.get_stats()
        assert stats["duplicates_skipped"] == 2

    def test_no_dedup(self):
        buf = self._make_buffer(capacity=10, dedup=False)
        for i in range(3):
            buf.store(
                state_hash="h1", state_vector=[0.0] * 10,
                mentor_action_idx=2, mentor_command="nmap",
                policy_action_idx=1, policy_command="nmap -sS",
                mentor_confidence=0.9, phase="RECON",
                episode=1, step=i,
            )
        assert len(buf) == 3

    def test_sample_insufficient(self):
        buf = self._make_buffer(min_samples_for_train=10)
        buf.store(
            state_hash="h1", state_vector=[0.0] * 10,
            mentor_action_idx=0, mentor_command="nmap",
            policy_action_idx=1, policy_command="nmap -sS",
            mentor_confidence=0.8, phase="RECON",
            episode=1, step=0,
        )
        assert buf.sample(4) == []

    def test_sample_sufficient(self):
        buf = self._make_buffer(min_samples_for_train=3, dedup=False)
        for i in range(5):
            buf.store(
                state_hash=f"h{i}", state_vector=[float(i)] * 10,
                mentor_action_idx=i % 3, mentor_command=f"cmd{i}",
                policy_action_idx=0, policy_command="nmap",
                mentor_confidence=0.7, phase="RECON",
                episode=1, step=i,
            )
        batch = buf.sample(3)
        assert len(batch) == 3

    def test_decay_weights(self):
        buf = self._make_buffer(min_samples_for_train=1, dedup=False)
        buf.store(
            state_hash="h1", state_vector=[0.0] * 10,
            mentor_action_idx=0, mentor_command="nmap",
            policy_action_idx=1, policy_command="nmap -sS",
            mentor_confidence=0.9, phase="RECON",
            episode=1, step=0,
        )
        buf.decay_weights()
        from core.training.dagger import DAggerConfig
        batch = buf.sample(1)
        assert len(batch) == 1
        assert batch[0].weight < 1.0

    def test_can_train(self):
        buf = self._make_buffer(min_samples_for_train=2, dedup=False)
        assert not buf.can_train()
        for i in range(2):
            buf.store(
                state_hash=f"h{i}", state_vector=[0.0] * 10,
                mentor_action_idx=0, mentor_command="cmd",
                policy_action_idx=1, policy_command="cmd2",
                mentor_confidence=0.8, phase="RECON",
                episode=1, step=i,
            )
        assert buf.can_train()

    def test_clear(self):
        buf = self._make_buffer(min_samples_for_train=1, dedup=False)
        buf.store(
            state_hash="h1", state_vector=[0.0],
            mentor_action_idx=0, mentor_command="x",
            policy_action_idx=1, policy_command="y",
            mentor_confidence=0.5, phase="RECON",
            episode=1, step=0,
        )
        buf.clear()
        assert len(buf) == 0


class TestDAggerMixin:
    def test_mixin(self):
        from core.training.dagger import DAggerMixin

        class FakeCoach(DAggerMixin):
            pass

        coach = FakeCoach()
        coach.init_dagger()
        ok = coach.record_dagger_sample(
            state_hash="h1", state_vector=[0.0] * 10,
            mentor_action_idx=2, mentor_command="nmap",
            policy_action_idx=0, policy_command="nmap -sS",
            mentor_confidence=0.85, phase="RECON",
            episode=1, step=3,
        )
        assert ok
        buf = coach.get_dagger_buffer()
        assert buf is not None
        assert len(buf) == 1

    def test_mixin_no_init(self):
        from core.training.dagger import DAggerMixin

        class FakeCoach(DAggerMixin):
            pass

        coach = FakeCoach()
        # Don't call init_dagger
        ok = coach.record_dagger_sample(
            state_hash="h1", state_vector=[0.0],
            mentor_action_idx=0, mentor_command="cmd",
            policy_action_idx=1, policy_command="cmd2",
            mentor_confidence=0.5, phase="RECON",
            episode=1, step=0,
        )
        assert ok is False
        assert coach.get_dagger_buffer() is None
