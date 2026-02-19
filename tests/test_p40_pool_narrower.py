"""
tests/test_p40_pool_narrower.py — Phase 40: Command Pool Narrower Tests
"""

import os
import pytest
from dataclasses import dataclass, field
from typing import Set
from enum import Enum

os.environ["ARIASKA_DRY_RUN"] = "1"


class MockPhase(Enum):
    RECON = "recon"
    EXPLOITATION = "exploitation"


@dataclass
class MockTemplate:
    name: str
    description: str = ""
    tags: Set[str] = field(default_factory=set)
    os_affinity: str = "any"
    phase: MockPhase = MockPhase.RECON


class TestCommandPoolNarrower:
    """Test CommandPoolNarrower filtering and weighting."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from core.ops.pool_narrower import CommandPoolNarrower, NarrowerConfig
        self.config = NarrowerConfig()
        self.narrower = CommandPoolNarrower(config=self.config)

    def test_init_stats(self):
        stats = self.narrower.get_stats()
        assert stats["detected_os"] is None
        assert stats["tracked_templates"] == 0
        assert stats["narrowing_ratio"] == 0.0

    def test_set_target_os(self):
        self.narrower.set_target_os("linux")
        assert self.narrower._detected_os == "linux"
        self.narrower.set_target_os("Windows")
        assert self.narrower._detected_os == "windows"

    def test_detect_os_from_output_linux(self):
        output = "OpenSSH_8.2p1 Ubuntu-4ubuntu0.5\nApache/2.4.41 (Ubuntu)\nroot:x:0:0"
        result = self.narrower.detect_os_from_output(output)
        assert result == "linux"

    def test_detect_os_from_output_windows(self):
        output = "Microsoft Windows Server 2019\nIIS/10.0\nPowershell available\nNT Authority"
        result = self.narrower.detect_os_from_output(output)
        assert result == "windows"

    def test_detect_os_from_output_ambiguous(self):
        output = "Connection established"
        result = self.narrower.detect_os_from_output(output)
        assert result is None

    def test_narrow_for_os_linux(self):
        self.narrower.set_target_os("linux")
        candidates = [
            MockTemplate(name="linpeas", os_affinity="linux"),
            MockTemplate(name="mimikatz", os_affinity="windows"),
            MockTemplate(name="nmap", os_affinity="any"),
            MockTemplate(name="ssh_brute", os_affinity="any"),
            MockTemplate(name="enum4linux", os_affinity="linux"),
        ]
        result = self.narrower.narrow_for_os(candidates)
        names = [c.name for c in result]
        assert "linpeas" in names
        assert "nmap" in names
        assert "enum4linux" in names
        assert "mimikatz" not in names

    def test_narrow_for_os_windows(self):
        self.narrower.set_target_os("windows")
        candidates = [
            MockTemplate(name="linpeas", os_affinity="linux"),
            MockTemplate(name="mimikatz", os_affinity="windows"),
            MockTemplate(name="nmap", os_affinity="any"),
            MockTemplate(name="crackmapexec", os_affinity="any"),
            MockTemplate(name="psexec", os_affinity="windows"),
            MockTemplate(name="enum4linux", os_affinity="linux"),
        ]
        result = self.narrower.narrow_for_os(candidates)
        names = [c.name for c in result]
        assert "mimikatz" in names
        assert "nmap" in names
        assert "psexec" in names
        assert "linpeas" not in names
        assert "enum4linux" not in names

    def test_narrow_for_os_disabled(self):
        from core.ops.pool_narrower import NarrowerConfig, CommandPoolNarrower
        cfg = NarrowerConfig(os_filter_enabled=False)
        n = CommandPoolNarrower(config=cfg)
        n.set_target_os("linux")
        candidates = [
            MockTemplate(name="linpeas", os_affinity="linux"),
            MockTemplate(name="mimikatz", os_affinity="windows"),
        ]
        result = n.narrow_for_os(candidates)
        assert len(result) == 2  # No filtering applied

    def test_narrow_too_few_keeps_all(self):
        """If narrowing would leave < 3 candidates, return all."""
        self.narrower.set_target_os("linux")
        candidates = [
            MockTemplate(name="linpeas", os_affinity="linux"),
            MockTemplate(name="mimikatz", os_affinity="windows"),
            MockTemplate(name="winpeas", os_affinity="windows"),
        ]
        result = self.narrower.narrow_for_os(candidates)
        # Only 1 linux match < 3, so all returned
        assert len(result) == 3

    def test_record_result_and_weight(self):
        self.narrower.record_result("nmap", True, reward=5.0, step=1)
        self.narrower.record_result("nmap", True, reward=3.0, step=2)
        self.narrower.record_result("nmap", False, reward=0.0, step=3)
        stats = self.narrower.get_stats()
        assert stats["tracked_templates"] == 1
        assert stats["avg_success_rate"] > 0.5

    def test_deprioritize_after_failures(self):
        for i in range(6):
            self.narrower.record_result("bad_cmd", False, reward=0.0, step=i)
        weight = self.narrower._compute_weight("bad_cmd", 10)
        assert weight < 0.2  # Should be heavily deprioritized

    def test_unknown_template_full_weight(self):
        weight = self.narrower._compute_weight("never_seen", 10)
        assert weight == 1.0

    def test_narrow_for_services(self):
        candidates = [
            MockTemplate(name="ssh_brute", tags={"ssh"}, description="SSH brute force"),
            MockTemplate(name="smb_enum", tags={"smb"}, description="SMB enumeration"),
            MockTemplate(name="nmap", tags={"network"}, description="Port scanner"),
        ]
        result = self.narrower.narrow_for_services(candidates, {"ssh"})
        # ssh_brute should be first (highest relevance)
        assert result[0].name == "ssh_brute"

    def test_get_weighted_candidates(self):
        candidates = [
            MockTemplate(name="recon1", phase=MockPhase.RECON),
            MockTemplate(name="exploit1", phase=MockPhase.EXPLOITATION),
        ]
        result = self.narrower.get_weighted_candidates(
            candidates, phase="RECON", step=1
        )
        assert len(result) == 2
        # First should be RECON (phase bonus)
        assert result[0][1].name == "recon1"
        assert result[0][0] > result[1][0]  # Higher weight

    def test_reset(self):
        self.narrower.record_result("nmap", True, reward=5.0, step=1)
        self.narrower.set_target_os("linux")
        self.narrower.reset()
        stats = self.narrower.get_stats()
        assert stats["tracked_templates"] == 0
        assert stats["detected_os"] is None

    def test_consecutive_failure_penalty(self):
        for i in range(5):
            self.narrower.record_result("flaky_cmd", False, reward=0.0, step=i)
        weight = self.narrower._compute_weight("flaky_cmd", 10)
        assert weight < 0.5

    def test_thread_safe_record(self):
        import threading
        errors = []
        def _record(i):
            try:
                self.narrower.record_result(f"cmd_{i % 3}", i % 2 == 0, step=i)
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=_record, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
