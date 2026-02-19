"""Tests for C5: Execution mode configuration."""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestExecutorConfig:
    def test_import(self):
        from core.execution.executor_config import ExecutorConfig
        cfg = ExecutorConfig()
        assert cfg.mode == "simulated"

    def test_is_live(self):
        from core.execution.executor_config import ExecutorConfig
        cfg = ExecutorConfig(mode="live")
        assert cfg.is_live is True
        cfg2 = ExecutorConfig(mode="simulated")
        assert cfg2.is_live is False

    def test_from_env(self):
        from core.execution.executor_config import ExecutorConfig
        os.environ["ARIASKA_EXECUTION_MODE"] = "live"
        try:
            cfg = ExecutorConfig.from_env()
            assert cfg.mode == "live"
        finally:
            os.environ.pop("ARIASKA_EXECUTION_MODE", None)


class TestDangerousCommands:
    def test_dangerous_rm_rf(self):
        from core.execution.executor_config import is_dangerous_command
        assert is_dangerous_command("rm -rf /etc") is True

    def test_safe_rm(self):
        from core.execution.executor_config import is_dangerous_command
        assert is_dangerous_command("rm -rf /tmp/test") is False

    def test_dangerous_shutdown(self):
        from core.execution.executor_config import is_dangerous_command
        assert is_dangerous_command("shutdown -h now") is True

    def test_safe_command(self):
        from core.execution.executor_config import is_dangerous_command
        assert is_dangerous_command("nmap -sV 10.10.10.1") is False

    def test_dangerous_dd(self):
        from core.execution.executor_config import is_dangerous_command
        assert is_dangerous_command("dd if=/dev/zero of=/dev/sda") is True


class TestExecutionResult:
    def test_dataclass(self):
        from core.execution.executor_config import ExecutionResult
        er = ExecutionResult(output="test", exit_code=0, is_simulated=True)
        assert er.output == "test"
        assert er.is_simulated is True
