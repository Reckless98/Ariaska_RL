"""Tests for Phase 41 coach submodule extraction."""
from __future__ import annotations
import os, pytest
os.environ.setdefault("ARIASKA_DRY_RUN", "1")

class TestCoachImports:
    def test_import_from_old_path(self):
        from core.training.smart_coach import SmartCoach, SmartDecisionResult, SmartStepContext
        assert SmartCoach is not None
    def test_import_from_new_path(self):
        from core.training.coach import SmartCoach, SmartDecisionResult, SmartStepContext
        assert SmartCoach is not None
    def test_same_class(self):
        from core.training.smart_coach import SmartCoach as Old
        from core.training.coach import SmartCoach as New
        assert Old is New
    def test_wrapper_import(self):
        from core.training.coach import SmartCoachWrapper, create_smart_coach
        assert SmartCoachWrapper is not None

class TestSubmoduleDelegation:
    """Tests that SmartCoach instantiates and delegates to submodules."""
    @pytest.fixture(autouse=True)
    def setup(self):
        os.environ["ARIASKA_DRY_RUN"] = "1"
        from core.testing.fake_gpt_manager import FakeGPTManager
        self.gpt = FakeGPTManager(seed=42)

    def test_coach_has_anti_repeat_config(self):
        from core.training.smart_coach import SmartCoach
        from core.training.coach.anti_repeat import AntiRepeatConfig
        coach = SmartCoach("RedAgent", self.gpt)
        assert hasattr(coach, '_anti_repeat_config')
        assert isinstance(coach._anti_repeat_config, AntiRepeatConfig)

    def test_coach_has_evidence_gate(self):
        from core.training.smart_coach import SmartCoach
        from core.training.coach.evidence_gate import EvidenceGate
        coach = SmartCoach("RedAgent", self.gpt)
        assert hasattr(coach, '_evidence_gate')
        assert isinstance(coach._evidence_gate, EvidenceGate)

    def test_coach_has_metrics_tracker(self):
        from core.training.smart_coach import SmartCoach
        from core.training.coach.metrics_tracker import MetricsTracker
        coach = SmartCoach("RedAgent", self.gpt)
        assert hasattr(coach, '_metrics_tracker')
        assert isinstance(coach._metrics_tracker, MetricsTracker)

    def test_coach_has_episode_lifecycle(self):
        from core.training.smart_coach import SmartCoach
        from core.training.coach.episode_lifecycle import EpisodeLifecycle
        coach = SmartCoach("RedAgent", self.gpt)
        assert hasattr(coach, '_episode_lifecycle')
        assert isinstance(coach._episode_lifecycle, EpisodeLifecycle)

    def test_submodule_evidence_gate_matches_inline(self):
        """Verify standalone EvidenceGate produces same result as inline method."""
        from core.training.smart_coach import SmartCoach, SmartDecisionResult
        coach = SmartCoach("RedAgent", self.gpt)
        result = SmartDecisionResult(command="hydra -l admin -p pass ssh://10.0.0.1")
        board = {"ports": set(), "services": set()}
        from core.commands.command_registry import AttackPhase
        v1, r1 = coach._validate_exploit_evidence(result, board, AttackPhase.EXPLOITATION)
        v2, r2 = coach._evidence_gate.validate(result.command, "EXPLOITATION", board)
        assert v1 == v2

class TestEvidenceGate:
    def test_import(self):
        from core.training.coach.evidence_gate import EvidenceGate, EvidenceGateConfig
        gate = EvidenceGate()
        assert gate.config.mode == "enforce"
    def test_local_cmd_passes(self):
        from core.training.coach.evidence_gate import EvidenceGate
        v, _ = EvidenceGate().validate("cat /etc/passwd", "EXPLOITATION", {"ports": set(), "services": set()})
        assert v is True
    def test_ssh_wrapped_passes(self):
        from core.training.coach.evidence_gate import EvidenceGate
        v, r = EvidenceGate().validate("sshpass -p x ssh u@h", "EXPLOITATION", {"ports": {22}, "services": {"ssh"}})
        assert v is True and any("ssh_wrapped" in x for x in r)
    def test_no_ports_blocks(self):
        from core.training.coach.evidence_gate import EvidenceGate
        v, r = EvidenceGate().validate("hydra -l a -p b h", "RECON", {"ports": set(), "services": set()})
        assert v is False
    def test_hallucination_block(self):
        from core.training.coach.evidence_gate import EvidenceGate
        # Command contains 'smb' substring but SMB is not in discovered services
        v, r = EvidenceGate().validate("smbclient //10.10.10.1/share", "RECON", {"ports": {22, 80}, "services": {"ssh", "http"}})
        assert any("hallucination" in x for x in r)
    def test_off_mode(self):
        from core.training.coach.evidence_gate import EvidenceGate, EvidenceGateConfig
        v, _ = EvidenceGate(EvidenceGateConfig(mode="off")).validate("hydra x", "EXPLOITATION", {"ports": set(), "services": set()})
        assert v is True
    def test_empty_command(self):
        from core.training.coach.evidence_gate import EvidenceGate
        v, _ = EvidenceGate().validate("", "RECON", {})
        assert v is True
    def test_stats(self):
        from core.training.coach.evidence_gate import EvidenceGate
        g = EvidenceGate()
        g.validate("cat /etc/passwd", "RECON", {"ports": set(), "services": set()})
        g.validate("hydra admin", "RECON", {"ports": set(), "services": set()})
        s = g.get_stats()
        assert s["pass_count"] >= 1 and s["reject_count"] >= 1

class TestAntiRepeatGuard:
    def test_no_repeat_first(self):
        from core.training.coach.anti_repeat import AntiRepeatGuard
        assert AntiRepeatGuard().is_repeat("nmap -sV h") is False
    def test_exact_repeat(self):
        from core.training.coach.anti_repeat import AntiRepeatGuard
        g = AntiRepeatGuard(); g.record_command("nmap -sV h")
        assert g.is_repeat("nmap -sV h") is True
    def test_prefix_repeat(self):
        from core.training.coach.anti_repeat import AntiRepeatGuard
        g = AntiRepeatGuard(); g.record_command("nmap -sV h"); g.record_command("nmap -sS h")
        assert g.is_repeat("nmap -p80 h") is True
    def test_extract_prefix(self):
        from core.training.coach.anti_repeat import AntiRepeatGuard
        assert AntiRepeatGuard._extract_prefix("nmap -sV h") == "nmap"
        assert AntiRepeatGuard._extract_prefix("/usr/bin/gobuster dir") == "gobuster"
        assert AntiRepeatGuard._extract_prefix("") == ""
    def test_reset(self):
        from core.training.coach.anti_repeat import AntiRepeatGuard
        g = AntiRepeatGuard(); g.record_command("t"); g.reset()
        assert len(g._recent_commands) == 0

class TestMetricsTracker:
    def test_step_reasoning(self):
        from core.training.coach.metrics_tracker import MetricsTracker
        t = MetricsTracker(); t.log_step_reasoning(1, "nmap"); t.log_step_reasoning(2, "gobuster")
        assert len(t.get_step_reasoning()) == 2
    def test_mentor_tracking(self):
        from core.training.coach.metrics_tracker import MetricsTracker
        t = MetricsTracker(); t.log_mentor_call(); t.log_mentor_call(overrode=True)
        assert t.get_stats()["mentor_calls"] == 2 and t.get_stats()["mentor_overrides"] == 1
    def test_curriculum_empty(self):
        from core.training.coach.metrics_tracker import MetricsTracker
        assert MetricsTracker().get_curriculum_performance() == 0.5
    def test_curriculum_with_data(self):
        from core.training.coach.metrics_tracker import MetricsTracker
        t = MetricsTracker()
        for i in range(5):
            t.record_episode_performance(float(i * 10), i + 1, 0.5 + i * 0.05)
        assert 0.0 <= t.get_curriculum_performance() <= 1.0
    def test_bounded_history(self):
        from core.training.coach.metrics_tracker import MetricsTracker, MetricsTrackerConfig
        t = MetricsTracker(MetricsTrackerConfig(history_window=5, max_history_multiplier=2))
        for i in range(20):
            t.record_episode_performance(float(i), i, 0.5)
        assert len(t._episode_rewards) <= 10

class TestEpisodeLifecycle:
    def test_start(self):
        from core.training.coach.episode_lifecycle import EpisodeLifecycle
        lc = EpisodeLifecycle(); lc.start_episode(5)
        assert lc.current.episode == 5 and lc.current.step == 0
    def test_record_step(self):
        from core.training.coach.episode_lifecycle import EpisodeLifecycle
        lc = EpisodeLifecycle(); lc.start_episode(1); lc.record_step("nmap", 5.0, 2)
        assert lc.current.step == 1 and lc.current.total_reward == 5.0
    def test_diversity(self):
        from core.training.coach.episode_lifecycle import EpisodeLifecycle
        lc = EpisodeLifecycle(); lc.start_episode(1)
        lc.record_step("nmap", 1.0); lc.record_step("nmap", 1.0); lc.record_step("gobuster", 1.0)
        assert lc.current.diversity_ratio == pytest.approx(2/3, abs=0.01)
    def test_terminal_bonus_closeout(self):
        from core.training.coach.episode_lifecycle import EpisodeLifecycle
        assert EpisodeLifecycle().compute_terminal_bonus("CLOSEOUT", 20) > 90.0
    def test_terminal_bonus_exfil(self):
        from core.training.coach.episode_lifecycle import EpisodeLifecycle
        assert EpisodeLifecycle().compute_terminal_bonus("EXFILTRATION", 30) == 70.0
    def test_highest_phase_no_downgrade(self):
        from core.training.coach.episode_lifecycle import EpisodeLifecycle
        lc = EpisodeLifecycle(); lc.start_episode(1)
        lc.update_highest_phase("ENUMERATION"); lc.update_highest_phase("RECON")
        assert lc.current.highest_phase == "ENUMERATION"

class TestPipelineStages:
    def test_import(self):
        from core.training.coach.pipeline_stages import PipelineStage, PipelineResult
        assert PipelineStage.PLAYBOOK.value == "playbook"
    def test_add_trace(self):
        from core.training.coach.pipeline_stages import PipelineResult
        r = PipelineResult(); r.add_trace("source", "ppo", 0.8); r.add_trace("tc", "rej", 0.3, False)
        assert len(r.traces) == 2 and r.rejected_stages == ["tc"]
    def test_to_dict(self):
        from core.training.coach.pipeline_stages import PipelineResult
        r = PipelineResult(); r.add_trace("source", "ppo", 0.8)
        d = r.to_dict()
        assert d[0]["stage"] == "source" and d[0]["score"] == 0.8
