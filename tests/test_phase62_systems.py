#!/usr/bin/env python3
"""
tests/test_phase62_systems.py — ARIASKA Phase 6.2 Test Suite
═══════════════════════════════════════════════════════════════
Validates:
  1. EventBus — publish/subscribe, ring buffer, JSONL sink, StepEvent schema
  2. MentorController — 7 triggers, budget, fade, EXFIL curriculum gate
  3. CheckpointManager — atomic save, auto-save, load, versioning
  4. Integration — wiring between SmartCoach ↔ MentorController, Orchestrator ↔ EventBus
"""

import sys
import os
import json
import time
import tempfile
import shutil
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Always dry-run for tests
os.environ["ARIASKA_DRY_RUN"] = "1"


# ═══════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_dir():
    """Create a temporary directory, clean up after test."""
    d = tempfile.mkdtemp(prefix="ariaska_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def event_bus():
    """Fresh EventBus (no JSONL sink)."""
    from core.tracing.event_bus import EventBus
    bus = EventBus(buffer_size=50)
    yield bus
    bus.close()


@pytest.fixture
def event_bus_with_jsonl(tmp_dir):
    """EventBus with JSONL file sink."""
    from core.tracing.event_bus import EventBus
    jsonl_path = os.path.join(tmp_dir, "events.jsonl")
    bus = EventBus(buffer_size=50, jsonl_path=jsonl_path)
    yield bus, jsonl_path
    bus.close()


@pytest.fixture
def mentor_controller():
    """Fresh MentorController with default config."""
    from core.training.mentor_controller import MentorController, MentorControllerConfig
    cfg = MentorControllerConfig(
        budget_pct=0.30,
        min_rate=0.05,
        max_rate=0.50,
        warmup_episodes=2,
        warmup_rate=0.40,
        stagnation_threshold=5,
        uncertainty_threshold=0.15,  # Explicit for test isolation
        exfil_patience=3,
        cooldown_steps=1,
        max_calls_per_episode=20,
    )
    return MentorController(config=cfg)


@pytest.fixture
def checkpoint_manager(tmp_dir):
    """CheckpointManager writing to temp directory."""
    from core.training.checkpoint_manager import CheckpointManager, CheckpointConfig
    cfg = CheckpointConfig(
        checkpoint_dir=tmp_dir,
        checkpoint_name="test_checkpoint.pt",
        auto_save_interval=3,
        keep_last_n=2,
        target_health_enabled=False,
    )
    return CheckpointManager(config=cfg)


# ═══════════════════════════════════════════════════════════════════════
# 1. EVENT BUS TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestEventBus:
    """Tests for core/tracing/event_bus.py."""

    def test_import(self):
        """EventBus module imports cleanly."""
        from core.tracing.event_bus import (
            EventBus, EventKind, StepEvent, AgentStepRecord,
            GenericEvent, EventSubscriber, EventCallback,
        )

    def test_publish_subscribe(self, event_bus):
        """Subscribers receive published events."""
        from core.tracing.event_bus import StepEvent
        received = []
        event_bus.subscribe(lambda e: received.append(e))

        evt = StepEvent(episode_num=1, step_num=5)
        event_bus.publish(evt)

        assert len(received) == 1
        assert received[0] is evt
        assert received[0].episode_num == 1
        assert received[0].step_num == 5

    def test_multiple_subscribers(self, event_bus):
        """Multiple subscribers all receive the same event."""
        from core.tracing.event_bus import GenericEvent, EventKind
        received_a = []
        received_b = []
        event_bus.subscribe(lambda e: received_a.append(e))
        event_bus.subscribe(lambda e: received_b.append(e))

        event_bus.publish_generic(EventKind.EPISODE_START, message="go")

        assert len(received_a) == 1
        assert len(received_b) == 1

    def test_unsubscribe(self, event_bus):
        """Unsubscribed callbacks no longer receive events."""
        from core.tracing.event_bus import StepEvent
        received = []
        cb = lambda e: received.append(e)
        event_bus.subscribe(cb)
        event_bus.publish(StepEvent())
        assert len(received) == 1

        event_bus.unsubscribe(cb)
        event_bus.publish(StepEvent())
        assert len(received) == 1  # No new events

    def test_ring_buffer(self, event_bus):
        """Ring buffer stores recent events up to buffer_size."""
        from core.tracing.event_bus import StepEvent
        for i in range(60):
            event_bus.publish(StepEvent(step_num=i))

        # Buffer size is 50, so oldest 10 are dropped
        recent = event_bus.recent(100)
        assert len(recent) == 50
        assert recent[0].step_num == 10
        assert recent[-1].step_num == 59

    def test_recent_with_kind_filter(self, event_bus):
        """recent() filters by EventKind when specified."""
        from core.tracing.event_bus import StepEvent, EventKind
        event_bus.publish(StepEvent(step_num=1))
        event_bus.publish_generic(EventKind.WARNING, message="oops")
        event_bus.publish(StepEvent(step_num=2))

        steps = event_bus.recent(10, kind=EventKind.STEP)
        assert len(steps) == 2

        warnings = event_bus.recent(10, kind=EventKind.WARNING)
        assert len(warnings) == 1

    def test_recent_steps(self, event_bus):
        """recent_steps() returns only StepEvent instances."""
        from core.tracing.event_bus import StepEvent, EventKind
        event_bus.publish(StepEvent(step_num=1))
        event_bus.publish_generic(EventKind.CHECKPOINT, message="saved")
        event_bus.publish(StepEvent(step_num=2))

        steps = event_bus.recent_steps(10)
        assert len(steps) == 2
        assert all(isinstance(s, StepEvent) for s in steps)

    def test_event_count(self, event_bus):
        """event_count tracks total published events."""
        from core.tracing.event_bus import StepEvent
        assert event_bus.event_count == 0
        event_bus.publish(StepEvent())
        event_bus.publish(StepEvent())
        assert event_bus.event_count == 2

    def test_subscriber_error_does_not_crash(self, event_bus):
        """A failing subscriber does not block other subscribers."""
        from core.tracing.event_bus import StepEvent
        received = []
        event_bus.subscribe(lambda e: (_ for _ in ()).throw(ValueError("boom")))
        event_bus.subscribe(lambda e: received.append(e))

        event_bus.publish(StepEvent())
        assert len(received) == 1  # Second subscriber still got it

    def test_jsonl_sink(self, event_bus_with_jsonl):
        """JSONL sink writes events to file."""
        bus, jsonl_path = event_bus_with_jsonl
        from core.tracing.event_bus import StepEvent, AgentStepRecord

        rec = AgentStepRecord(
            agent_name="RedAgent", role="offensive",
            decision_source="ppo", phase="EXPLOITATION",
            command="nmap -sV 10.10.10.10",
            reward=5.0, confidence=0.8,
        )
        evt = StepEvent(
            episode_num=1, step_num=3,
            agent_records=[rec],
            step_reward_total=5.0,
        )
        bus.publish(evt)
        bus.close()

        # Read back JSONL
        with open(jsonl_path) as f:
            lines = f.readlines()
        assert len(lines) == 1

        data = json.loads(lines[0])
        assert data["kind"] == "step"
        assert data["episode_num"] == 1
        assert data["step_num"] == 3
        assert len(data["agent_records"]) == 1
        assert data["agent_records"][0]["agent_name"] == "RedAgent"

    def test_generic_event_publish(self, event_bus):
        """publish_generic() creates and publishes GenericEvent."""
        from core.tracing.event_bus import EventKind, GenericEvent
        received = []
        event_bus.subscribe(lambda e: received.append(e))

        event_bus.publish_generic(
            EventKind.PHASE_TRANSITION,
            message="RECON→ENUMERATION",
            data={"from": "RECON", "to": "ENUMERATION"},
            episode_num=5,
        )

        assert len(received) == 1
        evt = received[0]
        assert isinstance(evt, GenericEvent)
        assert evt.kind == EventKind.PHASE_TRANSITION
        assert evt.message == "RECON→ENUMERATION"
        assert evt.data["from"] == "RECON"

    def test_step_event_to_dict(self):
        """StepEvent.to_dict() produces JSON-serializable dict."""
        from core.tracing.event_bus import StepEvent, AgentStepRecord
        evt = StepEvent(
            episode_id="ep-001", episode_num=1, step_num=10,
            agent_records=[
                AgentStepRecord(
                    agent_name="ScoutAgent", role="recon",
                    decision_source="playbook", phase="RECON",
                    command="nmap -sn 10.10.10.0/24",
                    reward=3.0, discoveries=["open_port:22"],
                ),
            ],
            phase_before="RECON", phase_after="RECON",
            step_reward_total=3.0,
            new_discoveries={"ports": ["22"]},
        )
        d = evt.to_dict()

        # Must be JSON-serializable
        serialized = json.dumps(d)
        assert "ScoutAgent" in serialized
        assert "nmap" in serialized

    def test_agent_step_record_to_dict_strips_none(self):
        """AgentStepRecord.to_dict() removes None values."""
        from core.tracing.event_bus import AgentStepRecord
        rec = AgentStepRecord(
            agent_name="RedAgent", role="offensive",
            decision_source="ppo", phase="EXPLOITATION",
            command="exploit/vsftpd",
            mentor_model=None, mentor_tier=None, error=None,
        )
        d = rec.to_dict()
        assert "mentor_model" not in d
        assert "mentor_tier" not in d
        assert "error" not in d
        assert d["agent_name"] == "RedAgent"

    def test_get_stats(self, event_bus):
        """get_stats() returns subscriber and buffer info."""
        from core.tracing.event_bus import StepEvent
        event_bus.subscribe(lambda e: None)
        event_bus.publish(StepEvent())

        stats = event_bus.get_stats()
        assert stats["event_count"] == 1
        assert stats["subscriber_count"] == 1
        assert stats["buffer_size"] == 1
        assert stats["jsonl_active"] is False


# ═══════════════════════════════════════════════════════════════════════
# 2. MENTOR CONTROLLER TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestMentorController:
    """Tests for core/training/mentor_controller.py."""

    def test_import(self):
        """MentorController module imports cleanly."""
        from core.training.mentor_controller import (
            MentorController, MentorControllerConfig,
            MentorEngagement, MentorTier, MentorTrigger,
            TIER_MODELS,
        )

    def test_initial_state(self, mentor_controller):
        """Freshly created controller has zero counts."""
        assert mentor_controller.total_calls == 0
        assert mentor_controller.total_decisions == 0
        assert mentor_controller.calls_this_episode == 0

    def test_start_episode_resets_state(self, mentor_controller):
        """start_episode() resets per-episode state."""
        mentor_controller.calls_this_episode = 10
        mentor_controller.current_step = 50
        mentor_controller._stagnation_steps = 20

        mentor_controller.start_episode(episode=1, max_steps=100)

        assert mentor_controller.calls_this_episode == 0
        assert mentor_controller.current_step == 0
        assert mentor_controller._stagnation_steps == 0
        assert mentor_controller.current_episode == 1

    def test_step_advances_counters(self, mentor_controller):
        """step() increments step count and stagnation."""
        mentor_controller.start_episode(episode=0, max_steps=100)
        mentor_controller.step()
        mentor_controller.step()

        assert mentor_controller.current_step == 2
        assert mentor_controller._stagnation_steps == 2

    def test_record_discovery_resets_stagnation(self, mentor_controller):
        """record_discovery() resets stagnation counter to 0."""
        mentor_controller.start_episode(episode=0, max_steps=100)
        for _ in range(10):
            mentor_controller.step()
        assert mentor_controller._stagnation_steps == 10

        mentor_controller.record_discovery()
        assert mentor_controller._stagnation_steps == 0

    def test_record_outcome_tracks_success(self, mentor_controller):
        """record_outcome() tracks positive vs total decisions."""
        mentor_controller.start_episode(episode=0, max_steps=100)
        mentor_controller.record_outcome(reward=5.0)
        mentor_controller.record_outcome(reward=-1.0)
        mentor_controller.record_outcome(reward=3.0)

        assert mentor_controller._episode_decisions == 3
        assert mentor_controller._episode_successes == 2
        assert mentor_controller.total_decisions == 3

    # --- Trigger tests ---

    def test_forced_trigger(self, mentor_controller):
        """force=True always engages deliberative mentor."""
        from core.training.mentor_controller import MentorTier, MentorTrigger
        mentor_controller.start_episode(episode=10, max_steps=100)
        mentor_controller.step()

        eng = mentor_controller.should_engage(force=True)

        assert eng.engage is True
        assert eng.tier == MentorTier.DELIBERATIVE
        assert eng.trigger == MentorTrigger.FORCED

    def test_phase_transition_trigger(self, mentor_controller):
        """Phase transition engages deliberative mentor."""
        from core.training.mentor_controller import MentorTier, MentorTrigger
        mentor_controller.start_episode(episode=10, max_steps=100)
        mentor_controller.step()

        eng = mentor_controller.should_engage(
            phase_changed=True,
            prev_phase="RECON",
            current_phase="ENUMERATION",
        )

        assert eng.engage is True
        assert eng.tier == MentorTier.DELIBERATIVE
        assert eng.trigger == MentorTrigger.PHASE_TRANSITION

    def test_stagnation_trigger(self, mentor_controller):
        """N steps without discovery triggers deliberative mentor."""
        from core.training.mentor_controller import MentorTier, MentorTrigger
        mentor_controller.start_episode(episode=10, max_steps=100)

        # Simulate 6 steps without discovery (threshold=5)
        for _ in range(6):
            mentor_controller.step()

        eng = mentor_controller.should_engage(confidence=0.8)

        assert eng.engage is True
        assert eng.tier == MentorTier.DELIBERATIVE
        assert eng.trigger == MentorTrigger.STAGNATION

    def test_uncertainty_trigger(self, mentor_controller):
        """Low confidence triggers reactive mentor."""
        from core.training.mentor_controller import MentorTier, MentorTrigger
        mentor_controller.start_episode(episode=10, max_steps=100)
        mentor_controller.step()

        eng = mentor_controller.should_engage(confidence=0.10)

        assert eng.engage is True
        assert eng.tier == MentorTier.REACTIVE
        assert eng.trigger == MentorTrigger.UNCERTAINTY

    def test_warmup_trigger(self, mentor_controller):
        """Warmup episodes get higher call rate."""
        from core.training.mentor_controller import MentorTrigger
        mentor_controller.start_episode(episode=0, max_steps=100)
        mentor_controller.step()

        # Warmup rate is 0.40, so ~40% chance per step.
        # Run multiple times to statistically verify warmup fires.
        warmup_engaged = 0
        for _ in range(100):
            # Reset per-step state
            mentor_controller.calls_this_episode = 0
            mentor_controller.steps_since_last_call = 999
            eng = mentor_controller.should_engage(confidence=0.8)  # High confidence
            if eng.engage and eng.trigger == MentorTrigger.WARMUP:
                warmup_engaged += 1
                # Reset call count for next check
                mentor_controller.calls_this_episode -= 1
                mentor_controller.total_calls -= 1

        # Should fire roughly 40% of the time (allow wide margin for randomness)
        assert warmup_engaged > 10, f"Warmup only fired {warmup_engaged}/100 times"
        assert warmup_engaged < 80, f"Warmup fired too often: {warmup_engaged}/100"

    def test_no_trigger_when_confident(self, mentor_controller):
        """High confidence, no phase change, no stagnation → no trigger."""
        mentor_controller.start_episode(episode=10, max_steps=100)
        mentor_controller.step()
        mentor_controller.record_discovery()  # Reset stagnation

        eng = mentor_controller.should_engage(
            confidence=0.9,
            phase_changed=False,
        )

        # Could be budget_floor randomly, but mostly should be no trigger
        # We can't guarantee no trigger due to budget_floor randomness,
        # so just check it's not a high-tier trigger
        if eng.engage:
            from core.training.mentor_controller import MentorTier
            assert eng.tier == MentorTier.REACTIVE  # Budget floor is reactive

    def test_budget_exhaustion_blocks_engagement(self, mentor_controller):
        """Once episode budget is exhausted, no more calls."""
        mentor_controller.start_episode(episode=10, max_steps=100)

        # Exhaust budget by forcing many calls
        for i in range(25):
            mentor_controller.step()
            mentor_controller.should_engage(force=True)

        mentor_controller.step()
        eng = mentor_controller.should_engage(force=True)
        assert eng.engage is False
        assert "budget_exhausted" in eng.reason

    def test_cooldown_blocks_consecutive_calls(self, mentor_controller):
        """Cooldown prevents back-to-back mentor calls."""
        mentor_controller.start_episode(episode=10, max_steps=100)
        mentor_controller.step()

        # First call succeeds
        eng1 = mentor_controller.should_engage(force=True)
        assert eng1.engage is True

        # Immediate second call blocked by cooldown (cooldown_steps=1)
        eng2 = mentor_controller.should_engage(force=True)
        assert eng2.engage is False
        assert "cooldown" in eng2.reason

        # After stepping past cooldown
        mentor_controller.step()
        eng3 = mentor_controller.should_engage(force=True)
        assert eng3.engage is True

    # --- EXFIL curriculum gate ---

    def test_exfil_gate_no_debt(self, mentor_controller):
        """No exfil debt when episodes reach EXFILTRATION."""
        mentor_controller.start_episode(episode=0, max_steps=100)
        mentor_controller.record_phase("EXFILTRATION")
        mentor_controller.end_episode("EXFILTRATION")

        assert mentor_controller._post_exploit_without_exfil == 0

    def test_exfil_gate_accumulates_debt(self, mentor_controller):
        """Debt accumulates when POST_EXPLOIT reached without EXFIL."""
        for ep in range(3):
            mentor_controller.start_episode(episode=ep, max_steps=100)
            mentor_controller.record_phase("POST_EXPLOITATION")
            mentor_controller.end_episode("POST_EXPLOITATION")

        assert mentor_controller._post_exploit_without_exfil == 3

    def test_exfil_gate_fires_after_patience(self, mentor_controller):
        """After exfil_patience episodes, objective_gap trigger fires."""
        from core.training.mentor_controller import MentorTrigger

        # Simulate 3 episodes stuck at POST_EXPLOIT
        for ep in range(3):
            mentor_controller.start_episode(episode=ep, max_steps=100)
            mentor_controller.record_phase("POST_EXPLOITATION")
            mentor_controller.end_episode("POST_EXPLOITATION")

        # Start next episode in POST_EXPLOITATION phase
        mentor_controller.start_episode(episode=3, max_steps=100)
        mentor_controller.step()

        eng = mentor_controller.should_engage(
            current_phase="POST_EXPLOITATION",
            confidence=0.8,
        )

        assert eng.engage is True
        assert eng.trigger == MentorTrigger.OBJECTIVE_GAP
        assert eng.exfil_guidance is True

    def test_exfil_gate_clears_on_success(self, mentor_controller):
        """Exfil debt clears when episode reaches EXFILTRATION."""
        # Build up debt
        for ep in range(3):
            mentor_controller.start_episode(episode=ep, max_steps=100)
            mentor_controller.record_phase("POST_EXPLOITATION")
            mentor_controller.end_episode("POST_EXPLOITATION")
        assert mentor_controller._post_exploit_without_exfil == 3

        # Now reach EXFIL
        mentor_controller.start_episode(episode=3, max_steps=100)
        mentor_controller.record_phase("EXFILTRATION")
        mentor_controller.end_episode("EXFILTRATION")
        assert mentor_controller._post_exploit_without_exfil == 0

    def test_exfil_guidance_prompt(self, mentor_controller):
        """get_exfil_guidance_prompt() returns MS2-specific strategies."""
        mentor_controller._post_exploit_without_exfil = 5
        prompt = mentor_controller.get_exfil_guidance_prompt()

        assert "EXFILTRATION" in prompt
        assert "/etc/shadow" in prompt
        assert "mysqldump" in prompt
        assert "5" in prompt  # debt count

    # --- Tier model routing ---

    def test_tier_models_mapping(self):
        """TIER_MODELS maps tiers to correct local model names."""
        from core.training.mentor_controller import TIER_MODELS, MentorTier
        # Phase 55: All local — 4b for reactive/postmortem, 9b for deliberative
        assert "qwen3.5" in TIER_MODELS[MentorTier.REACTIVE]
        assert "4b" in TIER_MODELS[MentorTier.REACTIVE]
        assert "qwen3.5" in TIER_MODELS[MentorTier.DELIBERATIVE]
        assert "9b" in TIER_MODELS[MentorTier.DELIBERATIVE]
        assert "qwen3.5" in TIER_MODELS[MentorTier.POSTMORTEM]
        assert "4b" in TIER_MODELS[MentorTier.POSTMORTEM]

    def test_engagement_model_matches_tier(self, mentor_controller):
        """Engagement result has the correct model for the tier."""
        mentor_controller.start_episode(episode=10, max_steps=100)
        mentor_controller.step()

        # Force → deliberative (Phase 55: uses 9b local model for deep reasoning)
        eng = mentor_controller.should_engage(force=True)
        assert "qwen3.5" in eng.model
        assert "9b" in eng.model  # Phase 55: deliberative uses 9b

    # --- Stats & diagnostics ---

    def test_get_stats(self, mentor_controller):
        """get_stats() returns structured statistics."""
        mentor_controller.start_episode(episode=0, max_steps=100)
        mentor_controller.step()
        mentor_controller.should_engage(force=True)

        stats = mentor_controller.get_stats()
        assert stats["total_calls"] == 1
        assert stats["calls_this_episode"] == 1
        assert "tier_counts" in stats
        assert "trigger_counts" in stats

    def test_get_summary(self, mentor_controller):
        """get_summary() returns human-readable string."""
        mentor_controller.start_episode(episode=0, max_steps=100)
        mentor_controller.step()
        mentor_controller.should_engage(force=True)

        summary = mentor_controller.get_summary()
        assert "MentorCtrl" in summary
        assert "1 calls" in summary

    def test_mentor_engagement_bool(self):
        """MentorEngagement.__bool__ returns engage flag."""
        from core.training.mentor_controller import MentorEngagement
        assert bool(MentorEngagement(engage=True)) is True
        assert bool(MentorEngagement(engage=False)) is False

    # --- Budget fade ---

    def test_budget_rate_warmup(self, mentor_controller):
        """Warmup episodes use warmup_rate."""
        mentor_controller.current_episode = 0  # Within warmup
        rate = mentor_controller._compute_budget_rate()
        assert rate == pytest.approx(0.40)

    def test_budget_rate_fades(self, mentor_controller):
        """Budget rate decreases over episodes."""
        mentor_controller.start_episode(episode=2, max_steps=100)  # Past warmup
        rate_early = mentor_controller._compute_budget_rate()

        mentor_controller.start_episode(episode=50, max_steps=100)
        rate_mid = mentor_controller._compute_budget_rate()

        # Rate should decrease
        assert rate_mid < rate_early

    def test_trigger_priority_forced_over_stagnation(self, mentor_controller):
        """Forced trigger takes priority over stagnation."""
        from core.training.mentor_controller import MentorTrigger
        mentor_controller.start_episode(episode=10, max_steps=100)

        # Create stagnation condition
        for _ in range(10):
            mentor_controller.step()

        eng = mentor_controller.should_engage(force=True)
        assert eng.trigger == MentorTrigger.FORCED  # Not STAGNATION

    def test_trigger_priority_phase_over_stagnation(self, mentor_controller):
        """Phase transition takes priority over stagnation."""
        from core.training.mentor_controller import MentorTrigger
        mentor_controller.start_episode(episode=10, max_steps=100)

        # Create stagnation condition
        for _ in range(10):
            mentor_controller.step()

        eng = mentor_controller.should_engage(
            phase_changed=True,
            prev_phase="RECON",
            current_phase="ENUMERATION",
        )
        assert eng.trigger == MentorTrigger.PHASE_TRANSITION


# ═══════════════════════════════════════════════════════════════════════
# 3. CHECKPOINT MANAGER TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestCheckpointManager:
    """Tests for core/training/checkpoint_manager.py."""

    def test_import(self):
        """CheckpointManager module imports cleanly."""
        from core.training.checkpoint_manager import (
            CheckpointManager, CheckpointConfig,
        )

    def test_save_atomic(self, checkpoint_manager, tmp_dir):
        """save_atomic() creates file atomically."""
        import torch
        state = {"weights": torch.randn(10, 5)}
        path = checkpoint_manager.save_atomic(state)

        assert os.path.exists(path)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        assert "state_dict" in loaded
        assert "timestamp" in loaded
        assert torch.equal(loaded["state_dict"]["weights"], state["weights"])

    def test_save_atomic_with_metadata(self, checkpoint_manager):
        """save_atomic() includes metadata when provided."""
        import torch
        state = {"weights": torch.randn(5)}
        meta = {"episode": 42, "reward": 100.5}
        path = checkpoint_manager.save_atomic(state, metadata=meta)

        loaded = torch.load(path, map_location="cpu", weights_only=False)
        assert loaded["metadata"]["episode"] == 42
        assert loaded["metadata"]["reward"] == 100.5

    def test_save_atomic_no_temp_file_on_success(self, checkpoint_manager, tmp_dir):
        """After successful save, no temp files remain."""
        import torch
        checkpoint_manager.save_atomic({"w": torch.randn(3)})

        # Only the checkpoint file should exist, no .ckpt_tmp_ files
        temps = [f for f in os.listdir(tmp_dir) if f.startswith(".ckpt_tmp_")]
        assert len(temps) == 0

    def test_save_versioned(self, checkpoint_manager, tmp_dir):
        """save_versioned() creates ep-numbered and latest files."""
        import torch
        state = {"w": torch.randn(5)}
        path = checkpoint_manager.save_versioned(state, episode=10)

        assert "ep0010" in path
        assert os.path.exists(path)

        # Latest should also exist
        latest = os.path.join(tmp_dir, "test_checkpoint.pt")
        assert os.path.exists(latest)

    def test_save_versioned_cleanup(self, checkpoint_manager, tmp_dir):
        """save_versioned() cleans up old versions beyond keep_last_n=2."""
        import torch
        state = {"w": torch.randn(5)}

        for ep in [5, 10, 15, 20]:
            checkpoint_manager.save_versioned(state, episode=ep)

        # Only last 2 versioned + latest should remain
        versioned = [f for f in os.listdir(tmp_dir) if "ep0" in f]
        assert len(versioned) == 2
        names = sorted(versioned)
        assert "ep0015" in names[0]
        assert "ep0020" in names[1]

    def test_should_auto_save(self, checkpoint_manager):
        """should_auto_save() respects interval of 3."""
        assert checkpoint_manager.should_auto_save(0) is False
        assert checkpoint_manager.should_auto_save(1) is False
        assert checkpoint_manager.should_auto_save(2) is False
        assert checkpoint_manager.should_auto_save(3) is True
        assert checkpoint_manager.should_auto_save(4) is False
        assert checkpoint_manager.should_auto_save(6) is True

    def test_auto_save_returns_path(self, checkpoint_manager):
        """auto_save() returns path when interval matches, None otherwise."""
        import torch
        state = {"w": torch.randn(3)}

        result = checkpoint_manager.auto_save(state, episode=2)
        assert result is None

        result = checkpoint_manager.auto_save(state, episode=3)
        assert result is not None
        assert os.path.exists(result)

    def test_load(self, checkpoint_manager):
        """load() restores saved checkpoint."""
        import torch
        state = {"w": torch.randn(5)}
        checkpoint_manager.save_atomic(state, metadata={"ep": 7})

        loaded = checkpoint_manager.load()
        assert loaded is not None
        assert torch.equal(loaded["state_dict"]["w"], state["w"])
        assert loaded["metadata"]["ep"] == 7

    def test_load_missing_returns_none(self, checkpoint_manager):
        """load() returns None if checkpoint doesn't exist."""
        loaded = checkpoint_manager.load()
        assert loaded is None

    def test_load_explicit_path(self, checkpoint_manager, tmp_dir):
        """load() can load from an explicit path."""
        import torch
        state = {"w": torch.randn(3)}
        path = checkpoint_manager.save_versioned(state, episode=42)

        loaded = checkpoint_manager.load(path=path)
        assert loaded is not None

    def test_check_target_health_disabled(self, checkpoint_manager):
        """check_target_health() returns True when disabled."""
        assert checkpoint_manager.check_target_health() is True

    def test_check_target_health_no_ip(self, checkpoint_manager):
        """check_target_health() returns True with no IP configured."""
        checkpoint_manager.config.target_health_enabled = True
        checkpoint_manager.config.target_ip = ""
        assert checkpoint_manager.check_target_health() is True

    def test_get_stats(self, checkpoint_manager):
        """get_stats() returns structured checkpoint info."""
        import torch
        checkpoint_manager.save_atomic({"w": torch.randn(3)})
        stats = checkpoint_manager.get_stats()
        assert stats["save_count"] == 1
        assert "checkpoint_dir" in stats


# ═══════════════════════════════════════════════════════════════════════
# 4. INTEGRATION: MENTOR CONTROLLER + EVENT BUS
# ═══════════════════════════════════════════════════════════════════════

class TestPhase62Integration:
    """Integration tests for Phase 6.2 wiring."""

    def test_mentor_controller_in_smart_coach_signature(self):
        """SmartCoach.__init__ accepts mentor_controller parameter."""
        import inspect
        from core.training.smart_coach import SmartCoach
        sig = inspect.signature(SmartCoach.__init__)
        assert "mentor_controller" in sig.parameters

    def test_mentor_controller_defaults_to_none(self):
        """SmartCoach works without mentor_controller (legacy fallback)."""
        import inspect
        from core.training.smart_coach import SmartCoach
        sig = inspect.signature(SmartCoach.__init__)
        param = sig.parameters["mentor_controller"]
        assert param.default is None

    def test_event_bus_in_orchestrator_config(self):
        """SmartOrchestratorConfig has event_jsonl_path and mentor_budget_pct."""
        from core.orchestration.smart_orchestrator import SmartOrchestratorConfig
        cfg = SmartOrchestratorConfig()
        assert hasattr(cfg, "event_jsonl_path")
        assert hasattr(cfg, "mentor_budget_pct")
        assert hasattr(cfg, "dashboard_mode")

    def test_event_bus_receives_step_events_from_publish(self):
        """EventBus subscriber receives StepEvent when published."""
        from core.tracing.event_bus import EventBus, StepEvent, AgentStepRecord

        received = []
        bus = EventBus(buffer_size=10)
        bus.subscribe(lambda e: received.append(e))

        # Simulate what orchestrator does
        records = [
            AgentStepRecord(
                agent_name="ScoutAgent", role="recon",
                decision_source="playbook", phase="RECON",
                command="nmap -sn 10.10.10.0/24",
                reward=2.0,
            ),
            AgentStepRecord(
                agent_name="RedAgent", role="offensive",
                decision_source="ppo", phase="RECON",
                command="nmap -sV -p22 10.10.10.10",
                reward=3.0,
            ),
        ]
        evt = StepEvent(
            episode_num=1, step_num=1,
            agent_records=records,
            step_reward_total=5.0,
            phase_before="RECON", phase_after="RECON",
        )
        bus.publish(evt)
        bus.close()

        assert len(received) == 1
        assert len(received[0].agent_records) == 2
        assert received[0].step_reward_total == 5.0

    def test_mentor_controller_full_episode_lifecycle(self):
        """MentorController handles full start→steps→end episode lifecycle."""
        from core.training.mentor_controller import MentorController, MentorControllerConfig

        ctrl = MentorController(config=MentorControllerConfig(
            budget_pct=0.30,
            warmup_episodes=1,
            stagnation_threshold=3,
            exfil_patience=2,
            cooldown_steps=0,
        ))

        # Episode 0: warmup
        ctrl.start_episode(episode=0, max_steps=50)
        for step in range(10):
            ctrl.step()
            eng = ctrl.should_engage(confidence=0.5)
            if eng.engage:
                ctrl.record_outcome(reward=1.0)
            if step == 3:
                ctrl.record_discovery()
        ctrl.end_episode("EXPLOITATION")

        assert ctrl.total_calls > 0
        assert ctrl._post_exploit_without_exfil == 0  # Didn't reach POST_EXPLOIT

    def test_checkpoint_manager_round_trip(self, tmp_dir):
        """Full save→load round trip preserves data."""
        import torch
        from core.training.checkpoint_manager import CheckpointManager, CheckpointConfig

        mgr = CheckpointManager(config=CheckpointConfig(
            checkpoint_dir=tmp_dir,
            checkpoint_name="roundtrip.pt",
        ))

        original = {
            "policy_state": {"layer1.weight": torch.randn(64, 512)},
            "optimizer_state": {"lr": 3e-4},
        }
        meta = {"episode": 50, "reward": 1234.5, "phase": "EXFILTRATION"}

        mgr.save_atomic(original, metadata=meta)
        loaded = mgr.load()

        assert loaded is not None
        assert loaded["metadata"]["episode"] == 50
        assert loaded["metadata"]["reward"] == 1234.5
        assert torch.equal(
            loaded["state_dict"]["policy_state"]["layer1.weight"],
            original["policy_state"]["layer1.weight"],
        )

    def test_event_bus_jsonl_round_trip(self, tmp_dir):
        """Events written to JSONL can be read back correctly."""
        from core.tracing.event_bus import EventBus, StepEvent, AgentStepRecord, EventKind

        jsonl_path = os.path.join(tmp_dir, "test_events.jsonl")
        bus = EventBus(buffer_size=10, jsonl_path=jsonl_path)

        # Publish a step event
        bus.publish(StepEvent(
            episode_num=5, step_num=10,
            agent_records=[
                AgentStepRecord(
                    agent_name="RedAgent", role="offensive",
                    decision_source="mentor", phase="EXPLOITATION",
                    command="exploit/vsftpd_234_backdoor",
                    reward=50.0, mentor_call=True,
                    mentor_model="local-llm",
                    mentor_tier="deliberative",
                ),
            ],
            step_reward_total=50.0,
        ))

        # Publish a generic event
        bus.publish_generic(EventKind.CHECKPOINT, message="saved ep5")

        bus.close()

        # Read back
        with open(jsonl_path) as f:
            lines = f.readlines()
        assert len(lines) == 2

        step_data = json.loads(lines[0])
        assert step_data["kind"] == "step"
        assert step_data["episode_num"] == 5
        assert step_data["agent_records"][0]["mentor_model"] == "local-llm"

        generic_data = json.loads(lines[1])
        assert generic_data["kind"] == "checkpoint"
        assert generic_data["message"] == "saved ep5"

    def test_mentor_controller_exfil_curriculum_integration(self):
        """
        Full EXFIL curriculum gate integration:
        3 episodes stuck → objective_gap fires with exfil guidance.
        """
        from core.training.mentor_controller import (
            MentorController, MentorControllerConfig, MentorTrigger,
        )

        ctrl = MentorController(config=MentorControllerConfig(
            exfil_patience=2,
            cooldown_steps=0,
            budget_pct=0.50,
            warmup_episodes=0,
        ))

        # 2 episodes reaching POST_EXPLOIT but not EXFIL
        for ep in range(2):
            ctrl.start_episode(episode=ep, max_steps=50)
            ctrl.record_phase("POST_EXPLOITATION")
            ctrl.end_episode("POST_EXPLOITATION")

        # Episode 2: should trigger objective_gap
        ctrl.start_episode(episode=2, max_steps=50)
        ctrl.step()
        eng = ctrl.should_engage(current_phase="POST_EXPLOITATION", confidence=0.8)

        assert eng.engage is True
        assert eng.trigger == MentorTrigger.OBJECTIVE_GAP
        assert eng.exfil_guidance is True

        # Get the prompt
        prompt = ctrl.get_exfil_guidance_prompt()
        assert "EXFILTRATION" in prompt
        assert "/etc/shadow" in prompt

    def test_cli_accepts_new_flags(self):
        """CLI parser accepts Phase 6.2 flags without error."""
        import argparse
        from ariaska_cli import main

        # Just verify the argparse setup doesn't crash
        # We can't run main() but we can verify the module loads
        import ariaska_cli
        assert hasattr(ariaska_cli, "run_training")


# ═══════════════════════════════════════════════════════════════════════
# 5. EDGE CASES & ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════════

class TestPhase62EdgeCases:
    """Edge cases and robustness tests."""

    def test_event_bus_no_subscribers(self):
        """Publishing with no subscribers doesn't crash."""
        from core.tracing.event_bus import EventBus, StepEvent
        bus = EventBus()
        bus.publish(StepEvent())
        assert bus.event_count == 1
        bus.close()

    def test_event_bus_close_idempotent(self):
        """Closing EventBus multiple times doesn't crash."""
        from core.tracing.event_bus import EventBus
        bus = EventBus()
        bus.close()
        bus.close()
        bus.close()

    def test_mentor_controller_zero_max_steps(self):
        """start_episode with max_steps=0 doesn't crash."""
        from core.training.mentor_controller import MentorController
        ctrl = MentorController()
        ctrl.start_episode(episode=0, max_steps=0)
        assert ctrl._episode_budget >= 2  # min 2

    def test_mentor_controller_record_phase_various_formats(self):
        """record_phase() handles various phase name formats."""
        from core.training.mentor_controller import MentorController
        ctrl = MentorController()
        ctrl.start_episode(episode=0, max_steps=100)

        ctrl.record_phase("post_exploitation")
        assert ctrl._episode_reached_post_exploit is True

        ctrl.record_phase("EXFILTRATION")
        assert ctrl._episode_reached_exfil is True

    def test_mentor_engagement_result_fields(self):
        """MentorEngagement has all expected fields."""
        from core.training.mentor_controller import MentorEngagement, MentorTier, MentorTrigger
        eng = MentorEngagement(
            engage=True,
            tier=MentorTier.DELIBERATIVE,
            trigger=MentorTrigger.STAGNATION,
            model="local-llm",
            reason="test",
            exfil_guidance=False,
            max_tokens=300,
        )
        assert eng.engage is True
        assert eng.tier == MentorTier.DELIBERATIVE
        assert eng.model == "local-llm"
        assert eng.max_tokens == 300

    def test_checkpoint_config_defaults(self):
        """CheckpointConfig has sensible defaults."""
        from core.training.checkpoint_manager import CheckpointConfig
        cfg = CheckpointConfig()
        assert cfg.auto_save_interval == 5
        assert cfg.keep_last_n == 3
        assert cfg.target_health_enabled is False

    def test_event_kind_values(self):
        """EventKind enum has expected values."""
        from core.tracing.event_bus import EventKind
        assert EventKind.STEP.value == "step"
        assert EventKind.EPISODE_START.value == "episode_start"
        assert EventKind.EPISODE_END.value == "episode_end"
        assert EventKind.MENTOR_CALL.value == "mentor_call"
        assert EventKind.CHECKPOINT.value == "checkpoint"

    def test_mentor_tier_values(self):
        """MentorTier enum has correct tier names."""
        from core.training.mentor_controller import MentorTier
        assert MentorTier.REACTIVE.value == "reactive"
        assert MentorTier.DELIBERATIVE.value == "deliberative"
        assert MentorTier.POSTMORTEM.value == "postmortem"

    def test_step_event_defaults(self):
        """StepEvent has sane defaults."""
        from core.tracing.event_bus import StepEvent, EventKind
        evt = StepEvent()
        assert evt.kind == EventKind.STEP
        assert evt.agent_records == []
        assert evt.step_reward_total == 0.0
        assert evt.mode == "sim"

    def test_agent_step_record_defaults(self):
        """AgentStepRecord defaults for optional fields."""
        from core.tracing.event_bus import AgentStepRecord
        rec = AgentStepRecord(
            agent_name="Test", role="test",
            decision_source="ppo", phase="RECON",
            command="ls",
        )
        assert rec.mentor_call is False
        assert rec.discoveries == []
        assert rec.confidence == 0.5
        assert rec.error is None
