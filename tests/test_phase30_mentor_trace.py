"""Phase 30: MentorTrace + Episode Summary + State Encoder tests."""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("ARIASKA_DRY_RUN", "1")

import torch
from core.llm.mentor_trace import MentorTrace, EpisodeSummary
from core.models.state_encoder import encode_state, STATE_DIM


class TestMentorTrace:
    """Test MentorTrace dataclass."""

    def test_defaults(self):
        t = MentorTrace()
        assert t.confidence == 0.5
        assert t.command == ""
        assert t.reasoning == ""

    def test_truncation(self):
        t = MentorTrace(reasoning="x" * 600, expected_outcome="y" * 300)
        assert len(t.reasoning) == 512
        assert len(t.expected_outcome) == 256

    def test_to_summary_features(self):
        t = MentorTrace(
            confidence=0.9,
            phase="EXPLOITATION",
            step=15,
            stagnation_steps=5,
            discoveries_at_call=10,
            produced_discovery=True,
            mentor_was_correct=True,
            actual_reward=20.0,
        )
        features = t.to_summary_features()
        assert len(features) == 16
        assert features["mentor_confidence"] == 0.9
        assert features["mentor_phase_exploit"] == 1.0
        assert features["mentor_produced_discovery"] == 1.0

    def test_to_teacher_trace(self):
        t = MentorTrace(
            command="nmap -sV 10.0.0.1",
            template_name="nmap_version",
            reasoning="Port scan",
            confidence=0.85,
            phase="RECON",
            step=3,
            episode=1,
            agent_id="RedAgent",
            state_vector=[0.0] * 512,
        )
        tt = t.to_teacher_trace()
        assert tt.teacher_command == "nmap -sV 10.0.0.1"
        assert tt.confidence == 0.85
        assert len(tt.state_vector) == 512

    def test_to_dict(self):
        t = MentorTrace(command="id", confidence=0.7)
        d = t.to_dict()
        assert d["command"] == "id"
        assert d["confidence"] == 0.7
        assert "trace_id" in d


class TestEpisodeSummary:
    """Test episode summary aggregation."""

    def test_empty_summary(self):
        s = EpisodeSummary()
        emb = s.compute_embedding()
        assert len(emb) == 16
        assert all(v == 0.0 for v in emb)

    def test_single_trace_summary(self):
        s = EpisodeSummary()
        s.add_trace(MentorTrace(confidence=0.9, phase="RECON", step=2))
        emb = s.compute_embedding()
        assert len(emb) == 16
        assert emb[0] == pytest.approx(0.9)  # mentor_confidence

    def test_multi_trace_averages(self):
        s = EpisodeSummary()
        s.add_trace(MentorTrace(confidence=0.8, phase="RECON"))
        s.add_trace(MentorTrace(confidence=0.4, phase="EXPLOITATION"))
        emb = s.compute_embedding()
        assert emb[0] == pytest.approx(0.6, abs=0.01)  # avg confidence

    def test_mentor_accuracy(self):
        s = EpisodeSummary()
        s.add_trace(MentorTrace(produced_discovery=True))
        s.add_trace(MentorTrace(produced_discovery=False))
        assert s.mentor_accuracy == pytest.approx(0.5)

    def test_avg_confidence(self):
        s = EpisodeSummary()
        s.add_trace(MentorTrace(confidence=0.3))
        s.add_trace(MentorTrace(confidence=0.7))
        assert s.avg_confidence == pytest.approx(0.5)


class TestStateEncoderEpisodeSummary:
    """Test Section 16 of state encoder."""

    def _find_section16_start(self) -> int:
        """Find where the episode summary embedding starts dynamically."""
        state = {"phase": "recon"}
        marker = [0.99] * 16
        vec = encode_state(state, torch.device("cpu"), episode_summary_embedding=marker)
        for i in range(STATE_DIM):
            if abs(vec[i].item() - 0.99) < 0.01:
                return i
        raise RuntimeError("Could not find section 16 start")

    def test_without_embedding(self):
        """Default: no episode summary → section 16 dims are zeros."""
        state = {"phase": "recon"}
        vec = encode_state(state, torch.device("cpu"))
        assert vec.shape == (STATE_DIM,)
        start = self._find_section16_start()
        # Without embedding, these should be zero
        vec_no_emb = encode_state(state, torch.device("cpu"))
        assert vec_no_emb[start:start+16].sum().item() == pytest.approx(0.0)

    def test_with_embedding(self):
        """Providing episode_summary_embedding fills section 16."""
        emb = [0.5] * 16
        state = {"phase": "recon"}
        start = self._find_section16_start()
        vec = encode_state(state, torch.device("cpu"), episode_summary_embedding=emb)
        assert vec[start].item() == pytest.approx(0.5)
        assert vec[start + 15].item() == pytest.approx(0.5)
        # Past section 16 should still be zero
        assert vec[start + 16].item() == pytest.approx(0.0)

    def test_partial_embedding_padded(self):
        """Short embedding should fill as many dims as provided."""
        emb = [0.9, 0.1]
        state = {"phase": "recon"}
        start = self._find_section16_start()
        vec = encode_state(state, torch.device("cpu"), episode_summary_embedding=emb)
        assert vec[start].item() == pytest.approx(0.9)
        assert vec[start + 1].item() == pytest.approx(0.1)
        assert vec[start + 2].item() == pytest.approx(0.0)
