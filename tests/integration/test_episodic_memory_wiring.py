"""Phase 42 Stage 1G: EpisodicMemory → CognitionNode wiring tests."""

import os
import pytest
import torch

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestEpisodicMemoryWiring:
    """Verify EpisodicMemory vote injection in CognitionNode."""

    @pytest.fixture(autouse=True)
    def setup(self):
        os.environ["ARIASKA_DRY_RUN"] = "1"
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()

    def test_cognition_node_accepts_episodic_memory(self):
        """CognitionNode __init__ accepts episodic_memory kwarg."""
        from core.algorithms.cognition_node import CognitionNode

        node = CognitionNode(episodic_memory=None)
        assert node._episodic_memory is None

    def test_cognition_node_stores_episodic_memory(self):
        """CognitionNode stores episodic_memory reference."""
        from core.algorithms.cognition_node import CognitionNode

        class FakeEpisodicMemory:
            def retrieve(self, state, k=5):
                return [{"reward": 5.0}, {"reward": 3.0}]

        mem = FakeEpisodicMemory()
        node = CognitionNode(episodic_memory=mem)
        assert node._episodic_memory is mem

    def test_brain_win_counts_includes_episodic_memory(self):
        """_brain_win_counts tracks episodic_memory."""
        from core.algorithms.cognition_node import CognitionNode

        node = CognitionNode()
        assert "episodic_memory" in node._brain_win_counts
        assert node._brain_win_counts["episodic_memory"] == 0

    def test_episodic_vote_added_when_memory_present(self):
        """think() includes episodic memory vote when memory is present."""
        from core.algorithms.cognition_node import CognitionNode, CognitionConfig
        from core.feature_flags import set_feature_flag

        set_feature_flag("episodic_memory_vote", True)

        class FakeEpisodicMemory:
            def retrieve(self, state, k=5):
                return [{"reward": 6.0}, {"reward": 4.0}]

        class FakePPO:
            def __init__(self):
                self.device = torch.device("cpu")

            def select_action(self, state, deterministic=False, action_mask=None):
                return 1, -0.5, 2.0

        mem = FakeEpisodicMemory()
        ppo = FakePPO()
        config = CognitionConfig()
        node = CognitionNode(config=config, ppo=ppo, episodic_memory=mem)

        state = torch.zeros(512)
        action_mask = torch.ones(5)
        result = node.think(state=state, phase="RECON", action_mask=action_mask)

        # Check that episodic_memory vote was added
        vote_names = [v.brain_name for v in result.votes]
        assert "episodic_memory" in vote_names

    def test_episodic_vote_skipped_when_flag_off(self):
        """think() skips episodic vote when flag is off."""
        from core.algorithms.cognition_node import CognitionNode, CognitionConfig
        from core.feature_flags import set_feature_flag

        set_feature_flag("episodic_memory_vote", False)

        class FakeEpisodicMemory:
            def retrieve(self, state, k=5):
                return [{"reward": 6.0}]

        class FakePPO:
            def __init__(self):
                self.device = torch.device("cpu")

            def select_action(self, state, deterministic=False, action_mask=None):
                return 1, -0.5, 2.0

        mem = FakeEpisodicMemory()
        ppo = FakePPO()
        node = CognitionNode(ppo=ppo, episodic_memory=mem)

        state = torch.zeros(512)
        action_mask = torch.ones(5)
        result = node.think(state=state, phase="RECON", action_mask=action_mask)

        vote_names = [v.brain_name for v in result.votes]
        assert "episodic_memory" not in vote_names
