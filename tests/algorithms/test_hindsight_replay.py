"""Tests for B1: Hindsight Experience Replay."""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestHindsightReplay:
    def test_import(self):
        from core.algorithms.hindsight_replay import HindsightReplay, HERConfig
        her = HindsightReplay()
        assert her is not None

    def test_relabel_empty(self):
        from core.algorithms.hindsight_replay import HindsightReplay
        her = HindsightReplay()
        assert her.relabel_episode([], "RECON", "EXFILTRATION") == []

    def test_relabel_valid(self):
        from core.algorithms.hindsight_replay import HindsightReplay
        her = HindsightReplay()
        transitions = [
            {"phase": "RECON", "reward": 0.0, "state": [0]*10},
            {"phase": "ENUMERATION", "reward": 1.0, "state": [1]*10},
        ]
        relabeled = her.relabel_episode(transitions, "ENUMERATION", "EXFILTRATION")
        assert len(relabeled) == 2
        assert all(t["is_her"] for t in relabeled)
        assert relabeled[0]["relabeled_goal"] == "ENUMERATION"

    def test_relabel_reward_success(self):
        from core.algorithms.hindsight_replay import HindsightReplay, HERConfig
        her = HindsightReplay(config=HERConfig(relabel_reward_success=5.0))
        transitions = [{"phase": "ENUMERATION", "reward": 0.0}]
        relabeled = her.relabel_episode(transitions, "ENUMERATION", "CLOSEOUT")
        assert relabeled[0]["reward"] == 5.0

    def test_process_episode_disabled(self):
        from core.algorithms.hindsight_replay import HindsightReplay, HERConfig
        her = HindsightReplay(config=HERConfig(enabled=False))
        assert her.process_episode([], "CLOSEOUT", "RECON") == 0

    def test_process_episode_already_succeeded(self):
        from core.algorithms.hindsight_replay import HindsightReplay
        her = HindsightReplay()
        transitions = [{"phase": "CLOSEOUT", "reward": 10.0}]
        assert her.process_episode(transitions, "CLOSEOUT", "CLOSEOUT") == 0

    def test_process_episode_generates(self):
        from core.algorithms.hindsight_replay import HindsightReplay
        her = HindsightReplay()
        transitions = [
            {"phase": "RECON", "reward": 0.0},
            {"phase": "ENUMERATION", "reward": 1.0},
        ]
        count = her.process_episode(transitions, "CLOSEOUT", "ENUMERATION")
        assert count > 0
        assert her.total_relabeled == count

    def test_phase_order(self):
        from core.algorithms.hindsight_replay import PHASE_ORDER
        assert PHASE_ORDER[0] == "RECON"
        assert PHASE_ORDER[-1] == "CLOSEOUT"
        assert len(PHASE_ORDER) == 8
