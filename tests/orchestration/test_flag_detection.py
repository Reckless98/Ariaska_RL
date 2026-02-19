"""Tests for C1: CTF flag detection."""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestFlagDetection:
    def test_import(self):
        from core.orchestration.flag_detector import detect_flags
        assert callable(detect_flags)

    def test_htb_flag(self):
        from core.orchestration.flag_detector import detect_flags
        flags = detect_flags("You win! HTB{th1s_1s_a_fl4g}")
        assert len(flags) == 1
        assert flags[0].flag_type == "HTB"
        assert flags[0].value == "HTB{th1s_1s_a_fl4g}"

    def test_thm_flag(self):
        from core.orchestration.flag_detector import detect_flags
        flags = detect_flags("THM{easy_machine_pwned}")
        assert len(flags) == 1
        assert flags[0].flag_type == "THM"

    def test_generic_flag(self):
        from core.orchestration.flag_detector import detect_flags
        flags = detect_flags("flag{generic_ctf_flag}")
        assert len(flags) >= 1

    def test_md5_flag(self):
        from core.orchestration.flag_detector import detect_flags
        flags = detect_flags("abcdef0123456789abcdef0123456789")
        assert len(flags) >= 1
        assert flags[0].flag_type == "MD5"

    def test_no_false_positive_zeros(self):
        from core.orchestration.flag_detector import detect_flags
        flags = detect_flags("00000000000000000000000000000000")
        assert len(flags) == 0

    def test_no_flags_normal_output(self):
        from core.orchestration.flag_detector import detect_flags
        flags = detect_flags("Starting Nmap 7.94\nHost is up")
        assert len(flags) == 0

    def test_user_flag_context(self):
        from core.orchestration.flag_detector import detect_flags
        flags = detect_flags(
            "abcdef0123456789abcdef0123456789",
            command="cat user.txt"
        )
        assert len(flags) >= 1
        assert flags[0].is_user_flag is True

    def test_root_flag_context(self):
        from core.orchestration.flag_detector import detect_flags
        flags = detect_flags(
            "deadbeef0123456789abcdef01234567",
            command="cat root.txt"
        )
        assert len(flags) >= 1
        assert flags[0].is_root_flag is True

    def test_celebration_format(self):
        from core.orchestration.flag_detector import detect_flags, format_flag_celebration
        flags = detect_flags("HTB{test}")
        celebration = format_flag_celebration(flags)
        assert "FLAG CAPTURED" in celebration

    def test_celebration_empty(self):
        from core.orchestration.flag_detector import format_flag_celebration
        assert format_flag_celebration([]) == ""

    def test_multiple_flags_in_output(self):
        from core.orchestration.flag_detector import detect_flags
        output = "User: HTB{user_flag}\nRoot: HTB{root_flag}"
        flags = detect_flags(output)
        assert len(flags) == 2
