#!/usr/bin/env python3
"""
tests/test_p34ext_features.py — P34-EXT Feature Tests

Tests for:
  - LearningMetrics: snapshot recording, window aggregation, milestones
  - MicroChain ablation toggle (MC_NANO_ABLATION)
  - MicroChain Stage 3 JSON schema validation + retry
  - Discovery board truncation
"""

import json
import os
import sys
import tempfile

import pytest

# ── Setup test environment ────────────────────────────────────────────────

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


# ═════════════════════════════════════════════════════════════════════════════
# LEARNING METRICS TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestLearningMetrics:
    """Test the LearningMetrics collector module."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from core.analytics.learning_metrics import LearningMetrics, MilestoneTracker, ModelMix
        self.LearningMetrics = LearningMetrics
        self.MilestoneTracker = MilestoneTracker
        self.ModelMix = ModelMix

    def _make_board(self, ports=None, services=None, creds=None, shells=None,
                    web_paths=None, users=None, flags_set=None):
        """Build a discovery board dict for testing."""
        return {
            "ports": set(ports or []),
            "services": set(services or []),
            "credentials": set(creds or []),
            "shells": set(shells or []),
            "web_paths": set(web_paths or []),
            "users": set(users or []),
            "flags_set": set(flags_set or []),
        }

    def test_basic_construction(self):
        lm = self.LearningMetrics()
        assert lm.total_commands == 0
        assert lm.stagnation_steps == 0
        assert lm.unique_template_count == 0

    def test_record_step_increments_counters(self):
        lm = self.LearningMetrics()
        lm.reset_episode(0)

        board = self._make_board(ports=["22", "80"])
        snap = lm.record_step(0, board, template_name="nmap_scan", phase="RECON")

        assert lm.total_commands == 1
        assert lm.unique_template_count == 1
        assert snap.total_ports == 2
        assert snap.new_ports == 0  # First step, no prior to compare

    def test_discovery_deltas(self):
        lm = self.LearningMetrics()
        lm.reset_episode(0)

        # Step 0: initial state
        board0 = self._make_board(ports=["22"])
        lm.record_step(0, board0)

        # Step 1: new port discovered
        board1 = self._make_board(ports=["22", "80"])
        snap1 = lm.record_step(1, board1)
        assert snap1.new_ports == 1
        assert snap1.total_ports == 2

    def test_stagnation_tracking(self):
        lm = self.LearningMetrics()
        lm.reset_episode(0)

        board = self._make_board(ports=["22"])
        lm.record_step(0, board)
        lm.record_step(1, board)  # No change
        lm.record_step(2, board)  # No change

        assert lm.stagnation_steps == 3  # Steps 0, 1, 2 all had no delta (step 0 has no prior board)

    def test_stagnation_resets_on_discovery(self):
        lm = self.LearningMetrics()
        lm.reset_episode(0)

        board = self._make_board(ports=["22"])
        lm.record_step(0, board)
        lm.record_step(1, board)  # No change, stag=2 (step 0 was also stagnant)
        assert lm.stagnation_steps == 2

        board2 = self._make_board(ports=["22", "80"])
        lm.record_step(2, board2)  # New port, stag=0
        assert lm.stagnation_steps == 0

    def test_anti_repeat_counting(self):
        lm = self.LearningMetrics()
        lm.reset_episode(0)

        board = self._make_board(ports=["22"])
        lm.record_step(0, board, anti_repeat_blocked=True)
        lm.record_step(1, board, anti_repeat_blocked=False)
        lm.record_step(2, board, anti_repeat_blocked=True)

        assert lm.anti_repeat_total == 2

    def test_phase_change_tracking(self):
        lm = self.LearningMetrics()
        lm.reset_episode(0)

        board = self._make_board(ports=["22"])
        lm.record_step(0, board, phase="RECON")
        lm.record_step(1, board, phase="ENUMERATION")
        lm.record_step(2, board, phase="ENUMERATION")  # Same phase
        lm.record_step(3, board, phase="EXPLOITATION")

        assert lm.phase_changes == 2  # RECON→ENUM, ENUM→EXPLOIT

    def test_milestone_tracking(self):
        mt = self.MilestoneTracker()
        assert mt.record(5, "port") == "first_port"
        assert mt.record(10, "port") is None  # Already recorded
        assert mt.record(7, "service") == "first_service"
        assert mt.first_port == 5
        assert mt.first_service == 7
        assert mt.first_creds == -1  # Not yet

    def test_model_mix_tracking(self):
        mm = self.ModelMix()
        mm.record_call("codex", tokens=100, cost=0.01)
        mm.record_call("nano", tokens=50, cost=0.001)
        mm.record_call("mini", tokens=0, cost=0.0, cached=True)

        assert mm.calls["codex"] == 1
        assert mm.calls["nano"] == 1
        assert mm.calls["mini"] == 0  # cached, not counted
        assert mm.cache_hits == 1
        assert mm.cache_hit_rate == pytest.approx(1 / 3, abs=0.01)
        assert mm.total_cost == pytest.approx(0.011, abs=0.001)

    def test_window_metrics(self):
        lm = self.LearningMetrics(window_size=3)
        lm.reset_episode(0)

        board = self._make_board(ports=["22"])
        lm.record_step(0, board, phase="RECON")

        board2 = self._make_board(ports=["22", "80"])
        lm.record_step(1, board2, phase="RECON")

        board3 = self._make_board(ports=["22", "80", "443"])
        lm.record_step(2, board3, phase="ENUMERATION")

        wm = lm.get_window_metrics()
        assert wm.window_size == 3
        assert wm.discoveries_delta >= 2  # At least 2 new ports across steps 1+2

    def test_should_print_dashboard(self):
        lm = self.LearningMetrics(print_every=5)
        assert not lm.should_print_dashboard(0)
        assert not lm.should_print_dashboard(3)
        assert lm.should_print_dashboard(5)
        assert lm.should_print_dashboard(10)
        assert not lm.should_print_dashboard(7)

    def test_episode_summary(self):
        lm = self.LearningMetrics()
        lm.reset_episode(42)

        board = self._make_board(ports=["22"])
        lm.record_step(0, board, template_name="nmap_scan", phase="RECON")
        board2 = self._make_board(ports=["22", "80"], services=["ssh"])
        lm.record_step(1, board2, template_name="nmap_version", phase="RECON")

        summary = lm.get_episode_summary()
        assert summary["episode"] == 42
        assert summary["total_steps"] == 2
        assert summary["unique_templates"] == 2
        assert "milestones" in summary
        assert "model_mix" in summary

    def test_jsonl_logging(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lm = self.LearningMetrics(log_dir=tmpdir)
            lm.reset_episode(0)

            board = self._make_board(ports=["22"])
            lm.record_step(0, board, phase="RECON")
            lm.get_episode_summary()
            lm.close()

            log_path = os.path.join(tmpdir, "learning_metrics.jsonl")
            assert os.path.exists(log_path)
            with open(log_path) as f:
                lines = f.readlines()
            assert len(lines) >= 2  # At least step + episode summary
            # Verify JSON validity
            for line in lines:
                data = json.loads(line)
                assert "type" in data

    def test_reset_episode_clears_state(self):
        lm = self.LearningMetrics()
        lm.reset_episode(0)
        board = self._make_board(ports=["22"])
        lm.record_step(0, board, anti_repeat_blocked=True, phase="RECON")
        assert lm.total_commands == 1
        assert lm.anti_repeat_total == 1

        lm.reset_episode(1)
        assert lm.total_commands == 0
        assert lm.anti_repeat_total == 0
        assert lm.stagnation_steps == 0

    def test_flag_milestones_via_discovery_board(self):
        lm = self.LearningMetrics()
        lm.reset_episode(0)

        board = self._make_board(ports=["22"], flags_set=["user_flag_captured"])
        lm.record_step(0, board)
        assert lm.milestones.user_flag == 0

        board2 = self._make_board(ports=["22"], flags_set=["user_flag_captured", "root_flag_captured"])
        lm.record_step(1, board2)
        assert lm.milestones.root_flag == 1


# ═════════════════════════════════════════════════════════════════════════════
# MICRO-CHAIN ABLATION TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestMicroChainAblation:
    """Test the MicroChain heuristic classify and ablation toggle."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from core.testing.fake_gpt_manager import FakeGPTManager
        self.gpt = FakeGPTManager(seed=42)

    def test_heuristic_classify_recon_gap(self):
        from core.llm.micro_chain import MicroChain
        mc = MicroChain(gpt_manager=self.gpt)
        board = {"ports": set(), "services": set(), "credentials": set(), "shells": set()}
        result = mc._heuristic_classify("RECON", board)
        assert result == "recon_gap"

    def test_heuristic_classify_enum_needed(self):
        from core.llm.micro_chain import MicroChain
        mc = MicroChain(gpt_manager=self.gpt)
        board = {
            "ports": {"22", "80", "443"},
            "services": {"ssh"},
            "credentials": set(),
            "shells": set(),
        }
        result = mc._heuristic_classify("ENUMERATION", board)
        assert result == "enum_needed"

    def test_heuristic_classify_exploit_ready(self):
        from core.llm.micro_chain import MicroChain
        mc = MicroChain(gpt_manager=self.gpt)
        board = {
            "ports": {"22", "80", "443", "3306", "8080"},
            "services": {"ssh", "http", "mysql"},
            "credentials": set(),
            "shells": set(),
        }
        result = mc._heuristic_classify("EXPLOITATION", board)
        assert result == "exploit_ready"

    def test_heuristic_classify_post_exploit(self):
        from core.llm.micro_chain import MicroChain
        mc = MicroChain(gpt_manager=self.gpt)
        board = {
            "ports": {"22", "80"},
            "services": {"ssh", "http"},
            "credentials": {"root:toor"},
            "shells": {"172.0.0.1:4444"},
        }
        result = mc._heuristic_classify("POST_EXPLOITATION", board)
        assert result == "post_exploit"

    def test_heuristic_classify_privesc_needed(self):
        from core.llm.micro_chain import MicroChain
        mc = MicroChain(gpt_manager=self.gpt)
        board = {
            "ports": {"22", "80"},
            "services": {"ssh"},
            "credentials": set(),
            "shells": {"172.0.0.1:4444"},
        }
        result = mc._heuristic_classify("PRIVILEGE_ESCALATION", board)
        assert result == "privesc_needed"

    def test_heuristic_classify_with_creds(self):
        from core.llm.micro_chain import MicroChain
        mc = MicroChain(gpt_manager=self.gpt)
        board = {
            "ports": {"22"},
            "services": {"ssh"},
            "credentials": {"admin:password"},
            "shells": set(),
        }
        result = mc._heuristic_classify("EXPLOITATION", board)
        assert result == "exploit_ready"


# ═════════════════════════════════════════════════════════════════════════════
# MICRO-CHAIN SCHEMA VALIDATION TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestMicroChainSchemaValidation:
    """Test Stage 3 JSON schema validation in _score_candidates."""

    def test_safe_json_load_list_valid(self):
        from core.llm.micro_chain import _safe_json_load_list
        result = _safe_json_load_list('[{"idx":0,"phase_fit":0.8}]')
        assert result is not None
        assert len(result) == 1

    def test_safe_json_load_list_fenced(self):
        from core.llm.micro_chain import _safe_json_load_list
        text = '```json\n[{"idx":0,"phase_fit":0.8}]\n```'
        result = _safe_json_load_list(text)
        assert result is not None

    def test_safe_json_load_list_empty(self):
        from core.llm.micro_chain import _safe_json_load_list
        assert _safe_json_load_list("") is None
        assert _safe_json_load_list(None) is None

    def test_safe_json_load_valid(self):
        from core.llm.micro_chain import _safe_json_load
        result = _safe_json_load('{"command":"nmap -sV"}')
        assert result is not None
        assert result["command"] == "nmap -sV"

    def test_safe_json_load_fenced(self):
        from core.llm.micro_chain import _safe_json_load
        text = '```json\n{"command":"nmap -sV"}\n```'
        result = _safe_json_load(text)
        assert result is not None

    def test_micro_chain_candidate_to_dict(self):
        from core.llm.micro_chain import MicroChainCandidate
        c = MicroChainCandidate(
            command="nmap -sV", template_name="nmap_version",
            score=0.75, phase_fit=0.8, evidence_support=0.7, novelty=0.6,
        )
        d = c.to_dict()
        assert d["command"] == "nmap -sV"
        assert d["score"] == 0.75


# ═════════════════════════════════════════════════════════════════════════════
# DISCOVERY BOARD TRUNCATION TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestDiscoveryBoardTruncation:
    """Test that discovery board panel truncates large sets."""

    def test_large_web_paths_truncated(self):
        """Verify that the panel builder handles 1000+ items without OOM/hang."""
        from core.observability.live_dashboard import LiveDashboard
        dash = LiveDashboard()

        # Create a board with 1000 web paths
        web_paths = [f"/path/{i}" for i in range(1000)]
        board = {
            "ports": ["22", "80"],
            "services": ["ssh", "http"],
            "credentials": [],
            "vulns": [],
            "shells": [],
            "users": [],
            "web_paths": web_paths,
            "flags_set": [],
            "phase": "ENUMERATION",
        }

        # Should not raise an error and should truncate
        panel = dash._build_discovery_board_panel(board)
        assert panel is not None

    def test_small_set_not_truncated(self):
        """Small sets should show all items."""
        from core.observability.live_dashboard import LiveDashboard
        dash = LiveDashboard()

        board = {
            "ports": ["22", "80", "443"],
            "services": ["ssh", "http"],
            "credentials": [],
            "vulns": [],
            "shells": [],
            "users": [],
            "web_paths": ["/index.html", "/login"],
            "flags_set": [],
            "phase": "RECON",
        }

        panel = dash._build_discovery_board_panel(board)
        assert panel is not None


# ═════════════════════════════════════════════════════════════════════════════
# EVIDENCE GATE STATS TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestEvidenceGateStats:
    """Test EvidenceGateStats tracking."""

    def test_evidence_gate_recording(self):
        from core.analytics.learning_metrics import LearningMetrics
        lm = LearningMetrics()
        lm.reset_episode(0)

        lm.record_evidence_gate("pass")
        lm.record_evidence_gate("pass")
        lm.record_evidence_gate("log_reject")
        lm.record_evidence_gate("enforce_reject")

        assert lm.evidence_gate.passed == 2
        assert lm.evidence_gate.log_rejected == 1
        assert lm.evidence_gate.enforce_rejected == 1
        assert lm.evidence_gate.total == 4

    def test_mentor_intervention_recording(self):
        from core.analytics.learning_metrics import LearningMetrics
        lm = LearningMetrics()
        lm.reset_episode(0)

        lm.record_mentor_intervention("stagnation", "RedAgent")
        lm.record_mentor_intervention("phase_stuck", "ScoutAgent")

        summary = lm.get_episode_summary()
        assert summary["mentor_interventions"] == 2
        assert "stagnation" in summary["mentor_reasons"]
