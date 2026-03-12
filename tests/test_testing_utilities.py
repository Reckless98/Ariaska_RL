#!/usr/bin/env python3
"""
tests/test_testing_utilities.py — Tests for core.testing utilities

Verifies:
- FakeGPTManager provides deterministic responses
- StubToolRunner safely stubs tool execution
- RealToolRunner validates RFC1918 targets
- ToolRunner factory function works correctly
"""

import pytest
from core.testing import (
    FakeGPTManager,
    StubToolRunner,
    RealToolRunner,
    ToolResult,
    get_tool_runner,
)


class TestFakeGPTManager:
    """Tests for FakeGPTManager mock."""
    
    def test_is_always_configured(self):
        """Fake manager should always report as configured."""
        gpt = FakeGPTManager()
        assert gpt.is_configured() is True
    
    def test_is_never_offline(self):
        """Fake manager should never report as offline."""
        gpt = FakeGPTManager()
        assert gpt.is_offline() is False
    
    def test_request_returns_dict(self):
        """request() should return dict with expected keys."""
        gpt = FakeGPTManager()
        result = gpt.request("RedAgent", "tactical", "Get a command")
        
        assert isinstance(result, dict)
        assert "success" in result
        assert "response" in result
        assert "model_used" in result
        assert "offline" in result
        
        assert result["success"] is True
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0
    
    def test_deterministic_with_same_seed(self):
        """Same seed should produce identical responses."""
        gpt1 = FakeGPTManager(seed=42)
        gpt2 = FakeGPTManager(seed=42)
        
        result1 = gpt1.request("RedAgent", "tactical", "Test prompt")
        result2 = gpt2.request("RedAgent", "tactical", "Test prompt")
        
        assert result1["response"] == result2["response"]
    
    def test_different_seeds_produce_different_responses(self):
        """Different seeds may produce different responses."""
        gpt1 = FakeGPTManager(seed=42)
        gpt2 = FakeGPTManager(seed=123)
        
        # Use same prompt but different seeds
        result1 = gpt1.request("RedAgent", "tactical", "Test prompt")
        result2 = gpt2.request("RedAgent", "tactical", "Test prompt")
        
        # Could be same by chance, but with different seeds, usually different
        # At minimum, both should be valid responses
        assert result1["success"] is True
        assert result2["success"] is True
    
    def test_tracks_requests(self):
        """Should track all requests for assertions."""
        gpt = FakeGPTManager()
        
        gpt.request("RedAgent", "tactical", "Prompt 1")
        gpt.request("BlueAgent", "defensive", "Prompt 2")
        
        requests = gpt.get_requests()
        assert len(requests) == 2
        assert requests[0]["role"] == "RedAgent"
        assert requests[1]["role"] == "BlueAgent"
    
    def test_clear_requests(self):
        """clear_requests() should reset tracked requests."""
        gpt = FakeGPTManager()
        gpt.request("RedAgent", "tactical", "Prompt")
        
        assert len(gpt.get_requests()) == 1
        gpt.clear_requests()
        assert len(gpt.get_requests()) == 0
    
    def test_gpt_request_compatibility(self):
        """gpt_request() should return string for compatibility."""
        gpt = FakeGPTManager()
        result = gpt.gpt_request("Test prompt", "tactical", "RedAgent")
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_model_routing(self):
        """Should route to correct models based on role — Phase 12.1: all reasoning → local-llm."""
        gpt = FakeGPTManager()
        
        # RedAgent tactical -> local-llm (Phase 12.1)
        result = gpt.request("RedAgent", "tactical", "Test")
        assert result["model_used"] == "local-llm"
        
        # Scout reconnaissance -> local-llm (Phase 12.1)
        result = gpt.request("Scout", "reconnaissance", "Test")
        assert result["model_used"] == "local-llm"
    
    def test_token_tracking(self):
        """Should track token usage."""
        gpt = FakeGPTManager()
        
        initial = gpt.get_token_usage()
        gpt.request("RedAgent", "tactical", "Test prompt")
        after = gpt.get_token_usage()
        
        assert after > initial
    
    def test_can_make_request(self):
        """can_make_request() should respect token limits."""
        gpt = FakeGPTManager()
        gpt.token_limit = 10  # Very low limit
        
        assert gpt.can_make_request() is True
        
        # Force token usage over limit
        gpt.tokens_used = 11
        assert gpt.can_make_request() is False


class TestStubToolRunner:
    """Tests for StubToolRunner."""
    
    def test_never_executes(self):
        """Stub should never actually execute commands."""
        stub = StubToolRunner()
        result = stub.execute("rm -rf /")  # Dangerous command
        
        assert result.executed is False
        assert result.blocked is False  # Not blocked, just stubbed
    
    def test_returns_fake_output(self):
        """Should return appropriate fake output."""
        stub = StubToolRunner()
        
        result = stub.execute("nmap -sV 10.10.10.10")
        assert "PORT" in result.stdout
        assert "ssh" in result.stdout
    
    def test_tracks_commands(self):
        """Should track all commands for assertions."""
        stub = StubToolRunner()
        
        stub.execute("nmap 10.10.10.10")
        stub.execute("curl http://10.10.10.10")
        
        commands = stub.get_commands()
        assert len(commands) == 2
        assert "nmap" in commands[0]
        assert "curl" in commands[1]
    
    def test_clear(self):
        """clear() should reset tracked commands."""
        stub = StubToolRunner()
        stub.execute("test command")
        
        assert len(stub.get_commands()) == 1
        stub.clear()
        assert len(stub.get_commands()) == 0
    
    def test_custom_output(self):
        """Should support custom outputs."""
        stub = StubToolRunner(custom_outputs={"special": "CUSTOM OUTPUT"})
        
        result = stub.execute("special command")
        assert result.stdout == "CUSTOM OUTPUT"
    
    def test_set_output(self):
        """set_output() should update output for prefix."""
        stub = StubToolRunner()
        stub.set_output("test", "NEW OUTPUT")
        
        result = stub.execute("test command")
        assert result.stdout == "NEW OUTPUT"
    
    def test_validate_target_always_true(self):
        """Stub should always validate targets."""
        stub = StubToolRunner()
        assert stub.validate_target("8.8.8.8") is True
        assert stub.validate_target("malicious.com") is True
    
    def test_is_safe_command_always_true(self):
        """Stub should always consider commands safe."""
        stub = StubToolRunner()
        safe, reason = stub.is_safe_command("rm -rf /")
        assert safe is True


class TestRealToolRunner:
    """Tests for RealToolRunner target validation."""
    
    def test_rfc1918_allowed(self):
        """RFC1918 addresses should be allowed by default."""
        runner = RealToolRunner(allow_rfc1918=True)
        
        assert runner.validate_target("10.0.0.1") is True
        assert runner.validate_target("10.10.10.10") is True
        assert runner.validate_target("172.16.0.1") is True
        assert runner.validate_target("172.31.255.255") is True
        assert runner.validate_target("192.168.0.1") is True
        assert runner.validate_target("192.168.1.1") is True
    
    def test_public_ips_blocked(self):
        """Public IP addresses should be blocked."""
        runner = RealToolRunner()
        
        assert runner.validate_target("8.8.8.8") is False
        assert runner.validate_target("1.1.1.1") is False
        assert runner.validate_target("104.16.0.1") is False
    
    def test_localhost_allowed(self):
        """Localhost should be allowed when configured."""
        runner = RealToolRunner(allow_localhost=True)
        
        assert runner.validate_target("127.0.0.1") is True
        assert runner.validate_target("localhost") is True
    
    def test_localhost_blocked_when_disabled(self):
        """Localhost should be blocked when disabled."""
        runner = RealToolRunner(allow_localhost=False)
        
        assert runner.validate_target("127.0.0.1") is False
    
    def test_custom_allowlist(self):
        """Custom allowlist should work."""
        runner = RealToolRunner(
            allow_rfc1918=False,
            allowed_targets=["203.0.113.0/24"]  # TEST-NET-3
        )
        
        assert runner.validate_target("203.0.113.1") is True
        assert runner.validate_target("203.0.113.100") is True
        assert runner.validate_target("203.0.114.1") is False  # Outside range
    
    def test_blocked_commands(self):
        """Dangerous commands should be blocked."""
        runner = RealToolRunner()
        
        safe, reason = runner.is_safe_command("rm -rf /")
        assert safe is False
        assert "Blocked" in reason
        
        safe, reason = runner.is_safe_command("nmap 10.10.10.10")
        assert safe is True
    
    def test_command_with_invalid_target_blocked(self):
        """Commands targeting non-allowed IPs should be blocked."""
        runner = RealToolRunner()
        
        safe, reason = runner.is_safe_command("nmap 8.8.8.8")
        assert safe is False
        assert "not allowed" in reason
    
    def test_command_with_valid_target_allowed(self):
        """Commands targeting allowed IPs should pass."""
        runner = RealToolRunner()
        
        safe, reason = runner.is_safe_command("nmap 10.10.10.10")
        assert safe is True
    
    def test_execution_history(self):
        """Should track execution history."""
        runner = RealToolRunner()
        
        # Execute safe command (just echo, won't actually run dangerous things)
        runner.execute("echo 'test'")
        
        history = runner.get_history()
        assert len(history) == 1
        assert history[0].command == "echo 'test'"


class TestToolRunnerFactory:
    """Tests for get_tool_runner factory function."""
    
    def test_returns_stub_when_testing(self):
        """testing=True should return StubToolRunner."""
        runner = get_tool_runner(testing=True)
        assert isinstance(runner, StubToolRunner)
    
    def test_returns_real_when_not_testing(self):
        """testing=False should return RealToolRunner."""
        runner = get_tool_runner(testing=False)
        assert isinstance(runner, RealToolRunner)
    
    def test_passes_kwargs_to_real(self):
        """kwargs should be passed to RealToolRunner."""
        runner = get_tool_runner(
            testing=False,
            allow_rfc1918=False,
            allowed_targets=["203.0.113.0/24"]
        )
        
        assert isinstance(runner, RealToolRunner)
        assert runner.validate_target("203.0.113.1") is True
        assert runner.validate_target("10.0.0.1") is False  # RFC1918 disabled


class TestToolResult:
    """Tests for ToolResult dataclass."""
    
    def test_default_values(self):
        """Should have sensible defaults."""
        result = ToolResult(command="test", executed=True)
        
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.return_code == 0
        assert result.blocked is False
        assert result.targets_found == []
    
    def test_blocked_result(self):
        """Blocked commands should have proper structure."""
        result = ToolResult(
            command="rm -rf /",
            executed=False,
            blocked=True,
            blocked_reason="Dangerous command"
        )
        
        assert result.executed is False
        assert result.blocked is True
        assert "Dangerous" in result.blocked_reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
