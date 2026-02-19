"""Tests for B10: Self-Play."""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestSelfPlayManager:
    def test_import(self):
        from core.training.self_play import SelfPlayManager
        sp = SelfPlayManager()
        assert sp is not None

    def test_disabled(self):
        from core.training.self_play import SelfPlayManager, SelfPlayConfig
        sp = SelfPlayManager(config=SelfPlayConfig(enabled=False))
        assert sp.should_run_self_play(5) is False

    def test_frequency(self):
        from core.training.self_play import SelfPlayManager, SelfPlayConfig
        sp = SelfPlayManager(config=SelfPlayConfig(enabled=True, frequency=5))
        assert sp.should_run_self_play(0) is False
        assert sp.should_run_self_play(5) is True
        assert sp.should_run_self_play(6) is False
        assert sp.should_run_self_play(10) is True

    def test_red_wins(self):
        from core.training.self_play import SelfPlayManager
        sp = SelfPlayManager()
        red_r, blue_r = sp.compute_adversarial_rewards("nmap", False, True)
        assert red_r > 0
        assert blue_r < 0
        assert sp.red_wins == 1

    def test_blue_wins(self):
        from core.training.self_play import SelfPlayManager
        sp = SelfPlayManager()
        red_r, blue_r = sp.compute_adversarial_rewards("nmap", True, False)
        assert red_r < 0
        assert blue_r > 0
        assert sp.blue_wins == 1

    def test_draw(self):
        from core.training.self_play import SelfPlayManager
        sp = SelfPlayManager()
        red_r, blue_r = sp.compute_adversarial_rewards("nmap", True, True)
        assert red_r > 0 and blue_r > 0

    def test_elo_tracking(self):
        from core.training.self_play import EloTracker
        elo = EloTracker()
        elo.update("red", "blue")
        ratings = elo.get_ratings()
        assert ratings["red"] > ratings["blue"]

    def test_stats(self):
        from core.training.self_play import SelfPlayManager
        sp = SelfPlayManager()
        sp.compute_adversarial_rewards("x", False, True)
        stats = sp.get_stats()
        assert stats["total_rounds"] == 1
        assert stats["red_wins"] == 1
