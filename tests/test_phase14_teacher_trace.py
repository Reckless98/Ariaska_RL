#!/usr/bin/env python3
"""
tests/test_phase14_teacher_trace.py — Phase 14.0: TeacherTrace + BCBuffer Tests

Contract C3.1: 9 required tests for teacher trace, BC buffer, and BCSample.
"""

import os
import sys
import time
import pytest

os.environ["ARIASKA_DRY_RUN"] = "1"


class TestTeacherTrace:
    """C3.1: TeacherTrace schema and immutability tests."""

    def test_trace_creation_valid(self):
        """TeacherTrace with all required fields creates successfully."""
        from core.reasoning.teacher_trace import TeacherTrace
        trace = TeacherTrace(
            state_id="s_001",
            state_vector=[0.1] * 512,
            teacher_action_idx=2,
            teacher_command="nmap -sV 10.0.0.1",
            teacher_template="nmap_version",
            rationale="Need to discover service versions on open ports",
            confidence=0.9,
            episode=1,
            step=3,
            agent_id="RedAgent",
            phase="RECON",
        )
        assert trace.trace_id != ""
        assert trace.teacher_action_idx == 2
        assert trace.confidence == 0.9
        assert trace.agent_id == "RedAgent"
        assert trace.phase == "RECON"

    def test_trace_rationale_truncation(self):
        """Rationale gets truncated to ≤512 chars in __post_init__."""
        from core.reasoning.teacher_trace import TeacherTrace
        long_rationale = "x" * 1000
        trace = TeacherTrace(
            state_id="s_002",
            state_vector=[0.0] * 512,
            teacher_action_idx=0,
            teacher_command="whoami",
            teacher_template="whoami",
            rationale=long_rationale,
            confidence=0.5,
            episode=1,
            step=1,
            agent_id="RedAgent",
            phase="RECON",
        )
        assert len(trace.rationale) <= 512

    def test_trace_expected_obs_truncation(self):
        """Expected observation truncated to ≤256 chars."""
        from core.reasoning.teacher_trace import TeacherTrace
        trace = TeacherTrace(
            state_id="s_003",
            state_vector=[0.0] * 512,
            teacher_action_idx=1,
            teacher_command="id",
            teacher_template="id",
            rationale="Check user",
            expected_obs="O" * 500,
            confidence=0.8,
            episode=1,
            step=2,
            agent_id="RedAgent",
            phase="EXPLOITATION",
        )
        assert len(trace.expected_obs) <= 256

    def test_trace_alt_actions_capped(self):
        """Alt actions list capped at 3."""
        from core.reasoning.teacher_trace import TeacherTrace
        trace = TeacherTrace(
            state_id="s_004",
            state_vector=[0.0] * 512,
            teacher_action_idx=0,
            teacher_command="ls",
            teacher_template="ls",
            rationale="list files",
            alt_actions=[1, 2, 3, 4, 5],
            confidence=0.7,
            episode=1,
            step=1,
            agent_id="ScoutAgent",
            phase="RECON",
        )
        assert len(trace.alt_actions) <= 3

    def test_trace_to_compact_omits_state_vector(self):
        """to_compact() returns dict without state_vector."""
        from core.reasoning.teacher_trace import TeacherTrace
        trace = TeacherTrace(
            state_id="s_005",
            state_vector=[0.5] * 512,
            teacher_action_idx=3,
            teacher_command="nmap -p- 10.0.0.1",
            teacher_template="nmap_full",
            rationale="Full port scan",
            confidence=0.95,
            episode=2,
            step=5,
            agent_id="ScoutAgent",
            phase="RECON",
        )
        compact = trace.to_compact()
        assert "state_vector" not in compact
        assert compact["teacher_command"] == "nmap -p- 10.0.0.1"

    def test_trace_compute_divergence_same_action(self):
        """Divergence = 0.0 when student matches teacher."""
        from core.reasoning.teacher_trace import TeacherTrace
        trace = TeacherTrace(
            state_id="s_006",
            state_vector=[0.0] * 512,
            teacher_action_idx=2,
            teacher_command="nmap -sV 10.0.0.1",
            teacher_template="nmap_version",
            rationale="version scan",
            student_action_idx=2,
            student_command="nmap -sV 10.0.0.1",
            confidence=0.8,
            episode=1,
            step=1,
            agent_id="RedAgent",
            phase="RECON",
        )
        assert trace.compute_divergence() == 0.0

    def test_trace_compute_divergence_different_action(self):
        """Divergence = 1.0 when student differs from teacher."""
        from core.reasoning.teacher_trace import TeacherTrace
        trace = TeacherTrace(
            state_id="s_007",
            state_vector=[0.0] * 512,
            teacher_action_idx=2,
            teacher_command="nmap -sV 10.0.0.1",
            teacher_template="nmap_version",
            rationale="version scan",
            student_action_idx=4,
            student_command="gobuster dir -u http://10.0.0.1",
            confidence=0.8,
            episode=1,
            step=1,
            agent_id="RedAgent",
            phase="RECON",
        )
        assert trace.compute_divergence() == 1.0

    def test_trace_to_bc_sample(self):
        """to_bc_sample() produces valid BCSample."""
        from core.reasoning.teacher_trace import TeacherTrace
        import torch
        trace = TeacherTrace(
            state_id="s_008",
            state_vector=[0.1] * 512,
            teacher_action_idx=1,
            teacher_command="dirb http://10.0.0.1",
            teacher_template="dirb_default",
            rationale="Web directory scan",
            confidence=0.85,
            episode=3,
            step=7,
            agent_id="RedAgent",
            phase="ENUMERATION",
        )
        sample = trace.to_bc_sample(decay_factor=0.9)
        assert sample.teacher_action == 1
        assert sample.episode == 3
        assert sample.step == 7
        assert sample.weight == pytest.approx(0.85 * 0.9, rel=1e-3)

    def test_bc_buffer_store_and_sample(self):
        """BCBuffer stores traces and samples correctly, with dedup."""
        from core.reasoning.teacher_trace import TeacherTrace, BCBuffer
        buf = BCBuffer(capacity=100)

        trace1 = TeacherTrace(
            state_id="s_100",
            state_vector=[0.1] * 512,
            teacher_action_idx=0,
            teacher_command="cmd1",
            teacher_template="t1",
            rationale="r1",
            confidence=0.9,
            episode=1, step=1, agent_id="Red", phase="RECON",
        )
        trace2 = TeacherTrace(
            state_id="s_101",
            state_vector=[0.2] * 512,
            teacher_action_idx=1,
            teacher_command="cmd2",
            teacher_template="t2",
            rationale="r2",
            confidence=0.8,
            episode=1, step=2, agent_id="Red", phase="RECON",
        )
        # Store unique traces
        assert buf.store(trace1) is True
        assert buf.store(trace2) is True
        # Duplicate should be rejected
        assert buf.store(trace1) is False

        stats = buf.get_stats()
        assert stats["size"] == 2
        assert stats["capacity"] == 100

        samples = buf.sample(2)
        assert len(samples) == 2
