"""Tests for C5: LiveExecutor CTF mode."""
from __future__ import annotations

import os
import time
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestCTFModeTracker:
    """Phase 41 C5: CTF mode tracker tests."""

    def test_import(self):
        from core.execution.ctf_mode import CTFModeTracker, CTFConfig
        tracker = CTFModeTracker()
        assert tracker is not None

    def test_scan_md5_flag(self):
        from core.execution.ctf_mode import CTFModeTracker
        tracker = CTFModeTracker()
        output = "User flag: a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6"
        caps = tracker.scan_output(output, command="cat /home/user/user.txt", agent="RedAgent")
        assert len(caps) == 1
        assert caps[0].flag_type == "user"
        assert caps[0].value == "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6"

    def test_scan_root_flag(self):
        from core.execution.ctf_mode import CTFModeTracker
        tracker = CTFModeTracker()
        output = "root.txt: deadbeef12345678abcdef1234567890"
        caps = tracker.scan_output(output, command="cat /root/root.txt")
        assert len(caps) == 1
        assert caps[0].flag_type == "root"

    def test_scan_curly_brace_flag(self):
        from core.execution.ctf_mode import CTFModeTracker
        tracker = CTFModeTracker()
        output = "Congrats! flag{you_pwned_it_2025}"
        caps = tracker.scan_output(output, command="cat flag.txt")
        assert len(caps) == 1
        assert "you_pwned_it_2025" in caps[0].value

    def test_scan_thm_flag(self):
        from core.execution.ctf_mode import CTFModeTracker
        tracker = CTFModeTracker()
        output = "THM{tryhackme_rocks}"
        caps = tracker.scan_output(output, command="cat /tmp/flag")
        assert len(caps) == 1
        assert caps[0].value == "THM{tryhackme_rocks}"

    def test_no_duplicate_capture(self):
        from core.execution.ctf_mode import CTFModeTracker
        tracker = CTFModeTracker()
        output = "deadbeef12345678abcdef1234567890"
        caps1 = tracker.scan_output(output, command="cat /root/root.txt")
        caps2 = tracker.scan_output(output, command="cat /root/root.txt")
        assert len(caps1) == 1
        assert len(caps2) == 0  # Already captured

    def test_both_flags_detection(self):
        from core.execution.ctf_mode import CTFModeTracker
        tracker = CTFModeTracker()
        tracker.scan_output(
            "aaaa1111bbbb2222cccc3333dddd4444",
            command="cat /home/user/user.txt",
        )
        assert tracker.has_user_flag
        assert not tracker.has_root_flag
        assert not tracker.both_flags
        tracker.scan_output(
            "eeee5555ffff6666aaaa7777bbbb8888",
            command="cat /root/root.txt",
        )
        assert tracker.has_root_flag
        assert tracker.both_flags

    def test_empty_output(self):
        from core.execution.ctf_mode import CTFModeTracker
        tracker = CTFModeTracker()
        assert tracker.scan_output("") == []
        assert tracker.scan_output("no flags here") == []

    def test_harvest_queue(self):
        from core.execution.ctf_mode import CTFModeTracker
        tracker = CTFModeTracker()
        first = tracker.next_harvest_command()
        assert first is not None
        assert "user.txt" in first or "root.txt" in first

    def test_harvest_queue_exhaustion(self):
        from core.execution.ctf_mode import CTFModeTracker
        tracker = CTFModeTracker()
        commands = []
        while True:
            cmd = tracker.next_harvest_command()
            if cmd is None:
                break
            commands.append(cmd)
        assert len(commands) > 0
        assert tracker.next_harvest_command() is None

    def test_harvest_disabled(self):
        from core.execution.ctf_mode import CTFModeTracker, CTFConfig
        config = CTFConfig(auto_harvest=False)
        tracker = CTFModeTracker(config=config)
        assert tracker.next_harvest_command() is None

    def test_add_custom_harvest(self):
        from core.execution.ctf_mode import CTFModeTracker
        tracker = CTFModeTracker()
        tracker.add_harvest_command("cat /opt/flag.txt")
        # Exhaust default queue
        while tracker.next_harvest_command() is not None:
            pass
        # Should be empty now since we already consumed custom cmd too
        # Re-add
        tracker.add_harvest_command("cat /var/flag.txt")
        cmd = tracker.next_harvest_command()
        assert cmd == "cat /var/flag.txt"

    def test_time_budget(self):
        from core.execution.ctf_mode import CTFModeTracker, CTFConfig
        config = CTFConfig(time_limit_minutes=60)
        tracker = CTFModeTracker(config=config)
        assert tracker.elapsed_minutes < 1.0
        assert tracker.remaining_minutes > 59.0
        assert not tracker.time_expired

    def test_time_expired(self):
        from core.execution.ctf_mode import CTFModeTracker, CTFConfig
        config = CTFConfig(time_limit_minutes=0)
        tracker = CTFModeTracker(config=config)
        assert tracker.time_expired

    def test_submit_flag_stub(self):
        from core.execution.ctf_mode import CTFModeTracker
        tracker = CTFModeTracker()
        assert tracker.submit_flag("abc123") is True
        assert tracker.submit_flag("abc123") is False  # Already submitted

    def test_stats(self):
        from core.execution.ctf_mode import CTFModeTracker
        tracker = CTFModeTracker()
        stats = tracker.get_stats()
        assert stats["enabled"] is True
        assert stats["flags_captured"] == 0
        assert stats["has_user_flag"] is False
        assert "remaining_minutes" in stats

    def test_reset(self):
        from core.execution.ctf_mode import CTFModeTracker
        tracker = CTFModeTracker()
        tracker.scan_output(
            "aabbccdd11223344aabbccdd11223344",
            command="cat /root/root.txt",
        )
        assert tracker.flag_count == 1
        tracker.reset()
        assert tracker.flag_count == 0
        assert not tracker.has_root_flag

    def test_flag_count(self):
        from core.execution.ctf_mode import CTFModeTracker
        tracker = CTFModeTracker()
        tracker.scan_output("aaaa1111bbbb2222cccc3333dddd4444", command="cmd1")
        tracker.scan_output("eeee5555ffff6666aaaa7777bbbb8888", command="cmd2")
        assert tracker.flag_count == 2

    def test_captured_flags_list(self):
        from core.execution.ctf_mode import CTFModeTracker
        tracker = CTFModeTracker()
        tracker.scan_output("flag{test_flag_123}", command="cat flag.txt")
        flags = tracker.captured_flags
        assert len(flags) == 1
        assert flags[0].value == "flag{test_flag_123}"
