#!/usr/bin/env python3
"""
tests/conftest.py — Global test fixtures for Ariaska_RL

Ensures tests run in a clean feature-flag environment regardless of
ambient env vars (e.g., when `source scripts/activate_htb_flags.sh` was run).
"""

import os
import pytest


@pytest.fixture(autouse=True)
def _clean_feature_flags(monkeypatch):
    """
    Clear all FF_* environment variables before each test,
    then reset the feature flag singleton to rebuild from clean env.

    Tests that need specific flags ON should set them via monkeypatch.setenv()
    AFTER this fixture runs.
    """
    # Remove every FF_ prefixed env var
    for key in list(os.environ.keys()):
        if key.startswith("FF_"):
            monkeypatch.delenv(key, raising=False)
    # Reset the global singleton so it reads clean env
    from core.feature_flags import reset_feature_flags
    reset_feature_flags()
    yield
    # Post-test cleanup: reset again to avoid cross-test pollution
    reset_feature_flags()
