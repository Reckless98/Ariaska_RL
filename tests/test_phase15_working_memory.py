#!/usr/bin/env python3
"""
tests/test_phase15_working_memory.py — Phase 15.0: Working Memory tests

Tests WorkingMemory bounded slots, TTL eviction, vector output,
and prompt fragment bounding.
"""

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestWorkingMemorySlot:
    """WorkingMemorySlot schema tests."""

    def test_expiry(self):
        from core.memory.working_memory import WorkingMemorySlot
        slot = WorkingMemorySlot(created_step=0, ttl_steps=5)
        assert not slot.is_expired(4)
        assert slot.is_expired(5)

    def test_content_hash_stable(self):
        from core.memory.working_memory import WorkingMemorySlot
        s1 = WorkingMemorySlot(key="test", content="hello")
        s2 = WorkingMemorySlot(key="test", content="hello")
        assert s1.content_hash() == s2.content_hash()


class TestWorkingMemory:
    """WorkingMemory bounded tests."""

    def test_max_slots_enforced(self):
        from core.memory.working_memory import WorkingMemory
        wm = WorkingMemory(max_slots=4)
        for i in range(10):
            wm.push(f"key_{i}", f"content_{i}", priority=float(i) / 10)
        assert len(wm) <= 4

    def test_ttl_eviction(self):
        from core.memory.working_memory import WorkingMemory
        wm = WorkingMemory(max_slots=8, default_ttl=3)
        wm.push("a", "content_a")
        assert wm.get("a") is not None
        evicted = wm.step(3)
        assert evicted == 1
        assert wm.get("a") is None

    def test_update_existing_key(self):
        from core.memory.working_memory import WorkingMemory
        wm = WorkingMemory()
        wm.push("hyp", "old hypothesis")
        wm.push("hyp", "new hypothesis")
        assert len(wm) == 1
        assert wm.get("hyp").content == "new hypothesis"

    def test_vector_dim(self):
        from core.memory.working_memory import WorkingMemory
        wm = WorkingMemory()
        wm.push("test", "content", slot_type="hypothesis", numeric_features=[0.1, 0.2])
        v = wm.to_vector()
        assert len(v) == 32
        assert v[0] > 0  # occupancy > 0

    def test_vector_empty(self):
        from core.memory.working_memory import WorkingMemory
        wm = WorkingMemory()
        v = wm.to_vector()
        assert len(v) == 32
        assert all(x == 0.0 for x in v)

    def test_prompt_fragment_bounded(self):
        from core.memory.working_memory import WorkingMemory
        wm = WorkingMemory()
        for i in range(8):
            wm.push(f"key_{i}", "x" * 100, slot_type="evidence", priority=0.5)
        frag = wm.to_prompt_fragment(max_chars=256)
        assert len(frag) <= 256

    def test_content_bounded(self):
        from core.memory.working_memory import WorkingMemory
        wm = WorkingMemory()
        wm.push("big", "x" * 1000)  # should be truncated to 256
        slot = wm.get("big")
        assert len(slot.content) <= 256

    def test_clear(self):
        from core.memory.working_memory import WorkingMemory
        wm = WorkingMemory()
        wm.push("a", "b")
        wm.clear()
        assert len(wm) == 0
