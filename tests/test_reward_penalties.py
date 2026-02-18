"""Phase 27.5: Reward failure penalties tests."""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("ARIASKA_DRY_RUN", "1")

from core.llm.reward_calculator import SmartRewardCalculator, AttackPhase


class TestRewardFailurePenalties:
    """Test output-based failure penalties and adjusted discovery bonuses."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.calc = SmartRewardCalculator()

    # ---- Discovery bonus adjustments ----

    def test_shell_bonus_boosted(self):
        assert self.calc.DISCOVERY_BONUSES["shell"] == 40.0

    def test_root_shell_bonus_boosted(self):
        assert self.calc.DISCOVERY_BONUSES["root_shell"] == 80.0

    def test_user_flag_bonus_boosted(self):
        assert self.calc.DISCOVERY_BONUSES["user_flag"] == 50.0

    # ---- Output-based failure penalties ----

    def test_penalty_command_not_found(self):
        bd = self.calc.calculate_reward(
            template_name="nmap_basic",
            command="nmapx -sV 10.0.0.1",
            success=False,
            raw_output="bash: nmapx: command not found",
            current_phase=AttackPhase.RECON,
            state_flags={},
        )
        # Base failure + output penalty
        assert bd.failure_penalty > 1.5
        assert "command not found" in bd.explanation.lower()

    def test_penalty_connection_refused(self):
        bd = self.calc.calculate_reward(
            template_name="ssh_login",
            command="ssh user@10.0.0.1",
            success=False,
            raw_output="ssh: connect to host 10.0.0.1 port 22: Connection refused",
            current_phase=AttackPhase.EXPLOITATION,
            state_flags={},
        )
        assert bd.failure_penalty > 0
        assert "connection refused" in bd.explanation.lower()

    def test_penalty_permission_denied_softened_in_privesc(self):
        """Permission denied during PRIVESC should get reduced penalty."""
        bd_recon = self.calc.calculate_reward(
            template_name="cat_shadow",
            command="cat /etc/shadow",
            success=False,
            raw_output="cat: /etc/shadow: Permission denied",
            current_phase=AttackPhase.RECON,
            state_flags={},
        )
        calc2 = SmartRewardCalculator()
        bd_priv = calc2.calculate_reward(
            template_name="cat_shadow",
            command="cat /etc/shadow",
            success=False,
            raw_output="cat: /etc/shadow: Permission denied",
            current_phase=AttackPhase.PRIVILEGE_ESCALATION,
            state_flags={},
        )
        # PRIVESC penalty should be lower than RECON penalty
        # (both have output penalty, but PRIVESC softens it)
        # Note: base failure_penalty also differs (exploit phase=0.5 vs other=1.0)
        # so we just check PRIVESC output penalty portion is smaller
        assert bd_priv.failure_penalty < bd_recon.failure_penalty

    def test_penalty_empty_output_no_extra(self):
        """Empty output should not trigger output-based penalties."""
        bd = self.calc.calculate_reward(
            template_name="nmap_basic",
            command="nmap -sV 10.0.0.1",
            success=True,
            raw_output="",
            current_phase=AttackPhase.RECON,
            state_flags={},
        )
        # Only possible penalty is from other sections (category-overuse etc)
        # No output-based penalty
        assert "Output(" not in bd.explanation

    def test_penalty_scale_applied(self):
        """FAILURE_PENALTY_SCALE should reduce base penalty."""
        assert SmartRewardCalculator.FAILURE_PENALTY_SCALE == 0.5
        # "command not found" base=2.0, scaled=1.0
        bd = self.calc.calculate_reward(
            template_name="nmap_basic",
            command="nmapx -sV 10.0.0.1",
            success=True,  # success=True but output still has error text
            raw_output="bash: nmapx: command not found",
            current_phase=AttackPhase.RECON,
            state_flags={},
        )
        # Output penalty = 2.0 * 0.5 = 1.0
        assert bd.failure_penalty >= 0.9  # allow float rounding

    def test_only_one_pattern_matches(self):
        """Only the first matching pattern should fire (no stacking)."""
        bd = self.calc.calculate_reward(
            template_name="ssh_login",
            command="ssh user@10.0.0.1",
            success=False,
            raw_output="Connection refused\nPermission denied\ncommand not found",
            current_phase=AttackPhase.RECON,
            state_flags={},
        )
        # Count output penalty entries
        output_penalties = [p for p in bd.explanation.split(" | ") if "Output(" in p]
        assert len(output_penalties) == 1


class TestFailurePenaltiesDict:
    """Validate FAILURE_PENALTIES structure."""

    def test_all_values_positive(self):
        for pattern, val in SmartRewardCalculator.FAILURE_PENALTIES.items():
            assert val > 0, f"Penalty for '{pattern}' must be positive"

    def test_expected_patterns_present(self):
        patterns = SmartRewardCalculator.FAILURE_PENALTIES
        assert "command not found" in patterns
        assert "Connection refused" in patterns
        assert "Permission denied" in patterns
