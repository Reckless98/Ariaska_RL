#!/usr/bin/env python3
"""
tests/test_phase101_knock.py — Phase 10.1D: Port Knocking Tests

Tests for KnockSequence, KnockInferenceEngine, timing inference,
registry templates, and telemetry.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestKnockSequence:
    """Test KnockSequence data object."""

    def test_basic_sequence(self):
        from core.tools.knock_sequence import KnockSequence
        seq = KnockSequence(
            sequence=[7000, 8000, 9000],
            protocol="tcp",
            target_port=22,
        )
        assert len(seq.sequence) == 3
        assert seq.target_port == 22

    def test_to_command_tcp(self):
        from core.tools.knock_sequence import KnockSequence
        seq = KnockSequence(
            sequence=[7000, 8000, 9000],
            delays_ms=[500],
        )
        cmd = seq.to_command("192.168.1.1")
        assert "knock" in cmd
        assert "192.168.1.1" in cmd
        assert "7000" in cmd
        assert "8000" in cmd
        assert "9000" in cmd

    def test_to_command_udp(self):
        from core.tools.knock_sequence import KnockSequence
        seq = KnockSequence(
            sequence=[100, 200, 300],
            protocol="udp",
        )
        cmd = seq.to_command("10.10.10.1")
        assert "nmap" in cmd
        assert "sU" in cmd

    def test_to_verify_command(self):
        from core.tools.knock_sequence import KnockSequence
        seq = KnockSequence(sequence=[1, 2, 3], target_port=22)
        cmd = seq.to_verify_command("10.0.0.1")
        assert "nmap" in cmd
        assert "22" in cmd
        assert "10.0.0.1" in cmd

    def test_to_dict(self):
        from core.tools.knock_sequence import KnockSequence
        seq = KnockSequence(
            sequence=[7000, 8000, 9000],
            target_port=22,
            source="knowledge",
            confidence=0.8,
        )
        d = seq.to_dict()
        assert d["sequence"] == [7000, 8000, 9000]
        assert d["target_port"] == 22
        assert d["source"] == "knowledge"
        assert d["confidence"] == 0.8


class TestKnockInference:
    """Test KnockInferenceEngine inference sources."""

    def test_infer_from_knowledge(self):
        from core.tools.knock_sequence import KnockInferenceEngine
        engine = KnockInferenceEngine()
        candidates = [{
            "title": "Port knocking setup for SSH",
            "description": "knockd configured to unlock SSH",
            "taxonomy": {"tags": ["knock", "ssh"]},
            "execution": {
                "original_command": "knock target 7000 8000 9000"
            },
        }]
        # Pass open_ports=[22] so CTF patterns for port 22 are suppressed,
        # isolating the knowledge-derived sequence
        results = engine.infer({"open_ports": [22]}, knowledge_candidates=candidates)
        assert len(results) > 0
        # Should have extracted the sequence from knowledge
        knowledge_seqs = [s for s in results if s.source == "knowledge"]
        assert len(knowledge_seqs) >= 1
        seq = knowledge_seqs[0]
        assert seq.sequence == [7000, 8000, 9000]
        assert seq.confidence >= 0.5

    def test_infer_from_ctf_patterns(self):
        from core.tools.knock_sequence import KnockInferenceEngine
        engine = KnockInferenceEngine()
        # Port 22 not open → suggest CTF patterns
        state = {"open_ports": [80, 443], "state_flags": {}}
        results = engine.infer(state)
        ssh_sequences = [s for s in results if s.target_port == 22]
        assert len(ssh_sequences) > 0

    def test_no_ctf_patterns_when_port_open(self):
        from core.tools.knock_sequence import KnockInferenceEngine
        engine = KnockInferenceEngine()
        # Port 22 already open → don't suggest knock for it
        state = {"open_ports": [22, 80], "state_flags": {}}
        results = engine.infer(state)
        ssh_sequences = [s for s in results if s.target_port == 22]
        assert len(ssh_sequences) == 0

    def test_infer_from_banner(self):
        from core.tools.knock_sequence import KnockInferenceEngine
        engine = KnockInferenceEngine()
        state = {
            "open_ports": [80],
            "service_banners": {
                "80": "Hint: port knocking sequence 1234 5678 9012"
            },
            "state_flags": {},
        }
        results = engine.infer(state)
        banner_results = [s for s in results if s.source == "symptom_banner"]
        assert len(banner_results) > 0
        # Should have extracted the ports from banner
        seq = banner_results[0]
        assert 1234 in seq.sequence

    def test_dedup_sequences(self):
        from core.tools.knock_sequence import KnockInferenceEngine
        engine = KnockInferenceEngine()
        candidates = [
            {
                "title": "Knock SSH",
                "taxonomy": {"tags": ["knock"]},
                "execution": {"original_command": "knock host 7000 8000 9000"},
            },
            {
                "title": "Knock SSH again",
                "taxonomy": {"tags": ["knock"]},
                "execution": {"original_command": "knock host 7000 8000 9000"},
            },
        ]
        results = engine.infer(
            {"open_ports": [22]}, knowledge_candidates=candidates
        )
        # Same sequence should be deduplicated
        seqs = [str(s.sequence) for s in results if s.source == "knowledge"]
        assert len(seqs) == len(set(seqs))


class TestKnockDecisionLogic:
    """Test should_propose_knock logic."""

    def setup_method(self):
        from core.feature_flags import set_feature_flag, reset_feature_flags
        reset_feature_flags()

    def test_not_proposed_when_flag_off(self):
        from core.tools.knock_sequence import KnockInferenceEngine
        from core.feature_flags import set_feature_flag
        set_feature_flag("port_knocking", False)
        engine = KnockInferenceEngine()
        assert engine.should_propose_knock({}, step=5) is False

    def test_proposed_when_flag_on(self):
        from core.tools.knock_sequence import KnockInferenceEngine
        from core.feature_flags import set_feature_flag
        set_feature_flag("port_knocking", True)
        engine = KnockInferenceEngine()
        assert engine.should_propose_knock({}, step=5, max_step=40) is True

    def test_not_proposed_late_in_episode(self):
        from core.tools.knock_sequence import KnockInferenceEngine
        from core.feature_flags import set_feature_flag
        set_feature_flag("port_knocking", True)
        engine = KnockInferenceEngine()
        # Step 35 of 40 → too late (>70%)
        assert engine.should_propose_knock({}, step=35, max_step=40) is False

    def test_not_proposed_after_max_attempts(self):
        from core.tools.knock_sequence import KnockInferenceEngine
        from core.feature_flags import set_feature_flag
        set_feature_flag("port_knocking", True)
        engine = KnockInferenceEngine()
        engine._telemetry.knock_attempts = 5
        assert engine.should_propose_knock({}, step=5) is False

    def teardown_method(self):
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()


class TestKnockRegistryTemplates:
    """Test knock templates in CommandRegistry."""

    def test_knock_sequence_template(self):
        from core.commands.command_registry import COMMAND_REGISTRY
        assert "knock_sequence" in COMMAND_REGISTRY
        cmd = COMMAND_REGISTRY["knock_sequence"]
        assert "target" in cmd.required_params
        assert "ports" in cmd.required_params
        assert cmd.required_tool == "knock"

    def test_knock_sequence_udp_template(self):
        from core.commands.command_registry import COMMAND_REGISTRY
        assert "knock_sequence_udp" in COMMAND_REGISTRY

    def test_verify_port_open_template(self):
        from core.commands.command_registry import COMMAND_REGISTRY
        assert "verify_port_open" in COMMAND_REGISTRY
        cmd = COMMAND_REGISTRY["verify_port_open"]
        assert "port" in cmd.required_params


class TestKnockTelemetry:
    """Test knock telemetry tracking."""

    def test_telemetry_to_dict(self):
        from core.tools.knock_sequence import KnockTelemetry
        tel = KnockTelemetry(
            knock_attempts=3,
            knock_success=1,
            sequences_inferred=2,
        )
        d = tel.to_dict()
        assert d["attempts"] == 3
        assert d["success"] == 1
        assert d["sequences_inferred"] == 2

    def test_inference_telemetry(self):
        from core.tools.knock_sequence import KnockInferenceEngine
        engine = KnockInferenceEngine()
        engine.infer({"open_ports": [], "state_flags": {}})
        assert engine.telemetry.sequences_inferred >= 0
