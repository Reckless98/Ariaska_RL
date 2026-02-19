"""Tests for C3: Credential spraying automation."""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestCredentialSprayer:
    def test_import(self):
        from core.ops.credential_sprayer import CredentialSprayer
        cs = CredentialSprayer()
        assert cs is not None

    def test_register_credential(self):
        from core.ops.credential_sprayer import CredentialSprayer
        cs = CredentialSprayer()
        cs.register_credential("admin", "password123", "hydra")
        assert len(cs._credentials) == 1

    def test_register_service(self):
        from core.ops.credential_sprayer import CredentialSprayer
        cs = CredentialSprayer()
        cs.register_service("ssh", 22, "10.10.10.1")
        assert len(cs._services) == 1

    def test_spray_commands_empty(self):
        from core.ops.credential_sprayer import CredentialSprayer
        cs = CredentialSprayer()
        assert cs.get_spray_commands() == []

    def test_spray_commands_ssh(self):
        from core.ops.credential_sprayer import CredentialSprayer
        cs = CredentialSprayer()
        cs.register_credential("admin", "pass123", "found")
        cs.register_service("ssh", 22, "10.10.10.1")
        cmds = cs.get_spray_commands(max_commands=1)
        assert len(cmds) == 1
        assert "ssh" in cmds[0]
        assert "admin" in cmds[0]

    def test_no_repeat_spray(self):
        from core.ops.credential_sprayer import CredentialSprayer
        cs = CredentialSprayer()
        cs.register_credential("admin", "pass", "x")
        cs.register_service("ssh", 22, "host")
        cmds1 = cs.get_spray_commands(current_step=0)
        cmds2 = cs.get_spray_commands(current_step=10)
        assert len(cmds2) == 0  # Already tried

    def test_priority_ordering(self):
        from core.ops.credential_sprayer import CredentialSprayer
        cs = CredentialSprayer()
        cs.register_credential("admin", "pass", "x")
        cs.register_service("ftp", 21, "host")
        cs.register_service("ssh", 22, "host")
        cmds = cs.get_spray_commands(max_commands=1)
        assert "ssh" in cmds[0]

    def test_record_result(self):
        from core.ops.credential_sprayer import CredentialSprayer
        cs = CredentialSprayer()
        cs.record_result("admin:pass", "ssh", True)
        assert cs.success_rate == 1.0

    def test_disabled(self):
        from core.ops.credential_sprayer import CredentialSprayer, CredentialSprayerConfig
        cs = CredentialSprayer(config=CredentialSprayerConfig(enabled=False))
        cs.register_credential("a", "b", "c")
        cs.register_service("ssh", 22, "host")
        assert cs.get_spray_commands() == []
