#!/usr/bin/env python3
"""
tests/test_phase15_sensory.py — Phase 15.0: Sensory Buffer tests

Tests SensoryBuffer ring buffer, dedup, bounded summary.
"""

import os
import time
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestSensoryBufferEntry:
    """SensoryBufferEntry schema tests."""

    def test_signature_hash_stable(self):
        from core.neuro.sensory_buffer import SensoryBufferEntry
        e1 = SensoryBufferEntry(command="nmap -sV", output_signature="abc")
        e2 = SensoryBufferEntry(command="nmap -sV", output_signature="abc")
        assert e1.signature_hash() == e2.signature_hash()

    def test_summary_line_bounded(self):
        from core.neuro.sensory_buffer import SensoryBufferEntry
        e = SensoryBufferEntry(
            command="x" * 200,
            discoveries=["port_80", "port_443"],
            confidence=0.85,
            phase="RECON",
        )
        assert len(e.summary_line()) <= 128


class TestSensoryBuffer:
    """SensoryBuffer ring buffer tests."""

    def test_capacity_bounded(self):
        from core.neuro.sensory_buffer import SensoryBuffer, SensoryBufferEntry
        buf = SensoryBuffer(capacity=4)
        for i in range(10):
            buf.push(SensoryBufferEntry(
                command=f"cmd_{i}",
                output_signature=f"sig_{i}",
                timestamp=time.time() + i,
            ))
        assert len(buf) <= 4

    def test_dedup_within_ttl(self):
        from core.neuro.sensory_buffer import SensoryBuffer, SensoryBufferEntry
        buf = SensoryBuffer(capacity=16, ttl=60.0)
        now = time.time()
        e1 = SensoryBufferEntry(command="nmap -sV", output_signature="abc", timestamp=now)
        e2 = SensoryBufferEntry(command="nmap -sV", output_signature="abc", timestamp=now + 1)
        assert buf.push(e1) is True
        assert buf.push(e2) is False  # deduped

    def test_dedup_after_ttl(self):
        from core.neuro.sensory_buffer import SensoryBuffer, SensoryBufferEntry
        buf = SensoryBuffer(capacity=16, ttl=1.0)
        now = time.time()
        e1 = SensoryBufferEntry(command="nmap -sV", output_signature="abc", timestamp=now)
        e2 = SensoryBufferEntry(command="nmap -sV", output_signature="abc", timestamp=now + 2.0)
        assert buf.push(e1) is True
        assert buf.push(e2) is True  # TTL expired, allow re-push

    def test_summary_bounded(self):
        from core.neuro.sensory_buffer import SensoryBuffer, SensoryBufferEntry
        buf = SensoryBuffer(capacity=16)
        for i in range(16):
            buf.push(SensoryBufferEntry(
                command=f"long_command_{i}_" + "x" * 50,
                output_signature=f"sig_{i}",
                phase="RECON",
                timestamp=time.time() + i,
            ))
        summary = buf.get_summary(max_chars=512)
        assert len(summary) <= 512

    def test_stats(self):
        from core.neuro.sensory_buffer import SensoryBuffer, SensoryBufferEntry
        buf = SensoryBuffer(capacity=8)
        buf.push(SensoryBufferEntry(command="test", output_signature="s1"))
        stats = buf.get_stats()
        assert stats["size"] == 1
        assert stats["capacity"] == 8
        assert stats["total_pushed"] == 1
