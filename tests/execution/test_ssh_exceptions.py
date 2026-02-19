"""Tests for A4: SSH exception hierarchy + SSH pool hardening."""
from __future__ import annotations

import os
import time
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestSSHExceptions:
    def test_hierarchy(self):
        from core.execution.ssh_exceptions import (
            SSHPoolError, SSHConnectionError, SSHAuthError, SSHTimeoutError
        )
        assert issubclass(SSHConnectionError, SSHPoolError)
        assert issubclass(SSHAuthError, SSHPoolError)
        assert issubclass(SSHTimeoutError, SSHPoolError)
        assert issubclass(SSHPoolError, Exception)

    def test_raise_connection(self):
        from core.execution.ssh_exceptions import SSHConnectionError
        with pytest.raises(SSHConnectionError):
            raise SSHConnectionError("failed to connect")

    def test_raise_auth(self):
        from core.execution.ssh_exceptions import SSHAuthError
        with pytest.raises(SSHAuthError):
            raise SSHAuthError("auth failed")

    def test_catch_base(self):
        from core.execution.ssh_exceptions import SSHPoolError, SSHTimeoutError
        with pytest.raises(SSHPoolError):
            raise SSHTimeoutError("timed out")


class TestSSHPoolHardening:
    """Phase 41: backoff, key auth, is_alive, credential opts."""

    def test_credential_key_path(self):
        from core.execution.ssh_pool import SSHCredential
        cred = SSHCredential(
            username="root", key_path="/tmp/id_ed25519", key_type="ed25519"
        )
        assert cred.key_path == "/tmp/id_ed25519"
        assert cred.key_type == "ed25519"
        assert cred.password is None

    def test_credential_password_only(self):
        from core.execution.ssh_pool import SSHCredential
        cred = SSHCredential(username="admin", password="pass123")
        assert cred.password == "pass123"
        assert cred.key_path is None

    def test_session_backoff(self):
        from core.execution.ssh_pool import SSHSession, SSHCredential
        cred = SSHCredential(username="u", password="p")
        sess = SSHSession(credential=cred, host="10.0.0.1")
        assert not sess.in_backoff
        sess.connect_attempts = 2
        sess.set_backoff()
        assert sess.in_backoff  # delay = 2^2 = 4s, should be in backoff

    def test_session_backoff_expires(self):
        from core.execution.ssh_pool import SSHSession, SSHCredential
        cred = SSHCredential(username="u", password="p")
        sess = SSHSession(credential=cred, host="10.0.0.1")
        sess.connect_attempts = 0
        sess.set_backoff()  # delay = 2^0 = 1s
        # Manually expire the backoff
        sess._backoff_until = time.time() - 1
        assert not sess.in_backoff

    def test_add_credentials_with_key(self):
        from core.execution.ssh_pool import SSHSessionPool
        pool = SSHSessionPool()
        pool.add_credentials(
            username="root", host="10.0.0.5",
            key_path="/tmp/id_rsa", key_type="rsa",
        )
        assert pool.has_credentials("10.0.0.5")

    def test_is_alive_no_session(self):
        from core.execution.ssh_pool import SSHSessionPool
        pool = SSHSessionPool()
        assert pool.is_alive("10.0.0.99") is False

    def test_is_alive_not_connected(self):
        from core.execution.ssh_pool import SSHSessionPool
        pool = SSHSessionPool()
        pool.add_credentials(username="u", password="p", host="10.0.0.1")
        # Create session but don't connect
        pool._get_or_create_session("10.0.0.1")
        assert pool.is_alive("10.0.0.1") is False

    def test_execute_no_credentials(self):
        from core.execution.ssh_pool import SSHSessionPool
        pool = SSHSessionPool()
        stdout, stderr, code = pool.execute("10.0.0.99", "id")
        assert code == 1
        assert "No credentials" in stderr

    def test_stats_include_commands(self):
        from core.execution.ssh_pool import SSHSessionPool
        pool = SSHSessionPool()
        pool.execute("10.0.0.99", "whoami")
        stats = pool.get_stats()
        assert stats["commands_executed"] == 1
