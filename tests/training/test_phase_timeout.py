"""Tests for C2: Phase timeout escalation."""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestPhaseTimeoutManager:
    def test_import(self):
        from core.training.phase_timeout import PhaseTimeoutManager
        ptm = PhaseTimeoutManager()
        assert ptm is not None

    def test_disabled(self):
        from core.training.phase_timeout import PhaseTimeoutManager, PhaseTimeoutConfig, TimeoutAction
        ptm = PhaseTimeoutManager(config=PhaseTimeoutConfig(enabled=False))
        assert ptm.check_timeout("RECON", 999) == TimeoutAction.CONTINUE

    def test_continue_under_limit(self):
        from core.training.phase_timeout import PhaseTimeoutManager, TimeoutAction
        ptm = PhaseTimeoutManager()
        assert ptm.check_timeout("RECON", 5) == TimeoutAction.CONTINUE

    def test_warning_at_threshold(self):
        from core.training.phase_timeout import PhaseTimeoutManager, TimeoutAction
        ptm = PhaseTimeoutManager()
        result = ptm.check_timeout("RECON", 15)  # 75% of 20
        assert result == TimeoutAction.WARNING

    def test_force_advance_at_limit(self):
        from core.training.phase_timeout import PhaseTimeoutManager, TimeoutAction
        ptm = PhaseTimeoutManager()
        assert ptm.check_timeout("RECON", 20) == TimeoutAction.FORCE_ADVANCE

    def test_mentor_consult_strategy(self):
        from core.training.phase_timeout import PhaseTimeoutManager, PhaseTimeoutConfig, TimeoutAction
        ptm = PhaseTimeoutManager(config=PhaseTimeoutConfig(escalation_strategy="mentor_consult"))
        assert ptm.check_timeout("RECON", 20) == TimeoutAction.MENTOR_CONSULT

    def test_escalation_suggestion(self):
        from core.training.phase_timeout import PhaseTimeoutManager
        ptm = PhaseTimeoutManager()
        suggestion = ptm.get_escalation_suggestion("RECON", 20)
        assert "ENUMERATION" in suggestion

    def test_reset(self):
        from core.training.phase_timeout import PhaseTimeoutManager, TimeoutAction
        ptm = PhaseTimeoutManager()
        ptm.record_step("RECON")
        ptm.reset()
        assert ptm.check_timeout("RECON") == TimeoutAction.CONTINUE

    def test_per_phase_limits(self):
        from core.training.phase_timeout import PhaseTimeoutManager, TimeoutAction
        ptm = PhaseTimeoutManager()
        assert ptm.check_timeout("CLOSEOUT", 5) == TimeoutAction.FORCE_ADVANCE
        assert ptm.check_timeout("EXPLOITATION", 5) == TimeoutAction.CONTINUE
