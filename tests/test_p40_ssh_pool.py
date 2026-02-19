"""
tests/test_p40_ssh_pool.py — Phase 40: SSH Session Pool Tests
"""

import os
import threading
import pytest

os.environ["ARIASKA_DRY_RUN"] = "1"


class TestSSHSessionPool:
    """Test SSHSessionPool without real SSH connections."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from core.execution.ssh_pool import SSHSessionPool
        self.pool = SSHSessionPool(
            keepalive_interval=30,
            connect_timeout=5,
            command_timeout=10,
        )

    def test_init_stats(self):
        stats = self.pool.get_stats()
        assert stats["connections_created"] == 0
        assert stats["connections_reused"] == 0
        assert stats["commands_executed"] == 0

    def test_add_credentials(self):
        self.pool.add_credentials("user", "pass", "10.10.10.1", port=22)
        assert self.pool.has_credentials("10.10.10.1")
        assert not self.pool.has_credentials("10.10.10.2")

    def test_add_multiple_credentials(self):
        self.pool.add_credentials("user1", "pass1", "10.10.10.1")
        self.pool.add_credentials("user2", "pass2", "10.10.10.2")
        assert self.pool.has_credentials("10.10.10.1")
        assert self.pool.has_credentials("10.10.10.2")

    def test_execute_no_credentials(self):
        stdout, stderr, code = self.pool.execute("10.10.10.99", "whoami")
        assert code == 1
        assert "No credentials" in stderr

    def test_execute_with_credentials_no_paramiko(self):
        """When paramiko is unavailable, falls back to sshpass."""
        self.pool.add_credentials("testuser", "testpass", "10.10.10.1")
        stdout, stderr, code = self.pool.execute("10.10.10.1", "whoami")
        # Will fail since we can't actually connect, but should attempt
        assert self.pool.get_stats()["commands_executed"] == 1

    def test_active_sessions_initially_zero(self):
        assert self.pool.active_sessions() == 0

    def test_close_all(self):
        self.pool.add_credentials("user", "pass", "10.10.10.1")
        self.pool.close_all()
        assert self.pool.active_sessions() == 0

    def test_reset_retries(self):
        self.pool.add_credentials("user", "pass", "10.10.10.1")
        # Attempt execute to create session
        self.pool.execute("10.10.10.1", "test")
        self.pool.reset_retries("10.10.10.1")

    def test_thread_safety(self):
        """Concurrent credential additions shouldn't crash."""
        self.pool.add_credentials("user", "pass", "10.10.10.1")
        errors = []

        def _add_cred(i):
            try:
                self.pool.add_credentials(f"user{i}", f"pass{i}", f"10.10.10.{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_add_cred, args=(i,)) for i in range(2, 12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_stats_after_operations(self):
        self.pool.add_credentials("user", "pass", "10.10.10.1")
        self.pool.execute("10.10.10.1", "test")
        stats = self.pool.get_stats()
        assert stats["commands_executed"] >= 1
