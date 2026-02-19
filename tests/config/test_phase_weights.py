"""Tests for A5: Configurable phase weights."""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestPhaseWeights:
    def test_import(self):
        from core.config.phase_weights import get_weights, PhaseWeights
        assert callable(get_weights)

    def test_default_weights(self):
        from core.config.phase_weights import get_weights
        w = get_weights("default")
        assert w.recon == 1.0
        assert w.exploitation == 1.5

    def test_htb_easy(self):
        from core.config.phase_weights import get_weights
        w = get_weights("htb_easy")
        assert w.recon == 0.8
        assert w.exploitation == 1.8

    def test_unknown_profile_fallback(self):
        from core.config.phase_weights import get_weights
        w = get_weights("nonexistent_profile")
        assert w.recon == 1.0  # default

    def test_get_method(self):
        from core.config.phase_weights import PhaseWeights
        w = PhaseWeights()
        assert w.get("RECON") == 1.0
        assert w.get("EXPLOITATION") == 1.5
        assert w.get("unknown_phase") == 1.0

    def test_env_var_loading(self):
        from core.config.phase_weights import get_weights
        os.environ["ARIASKA_TARGET_PROFILE"] = "htb_medium"
        try:
            w = get_weights()
            assert w.exploitation == 2.0
        finally:
            os.environ.pop("ARIASKA_TARGET_PROFILE", None)

    def test_all_profiles_exist(self):
        from core.config.phase_weights import PROFILE_WEIGHTS
        assert "default" in PROFILE_WEIGHTS
        assert "htb_easy" in PROFILE_WEIGHTS
        assert "metasploitable2" in PROFILE_WEIGHTS
