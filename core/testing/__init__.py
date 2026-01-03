"""
core/testing — Testing utilities for ARIASKA

Provides:
- FakeGPTManager: Mock GPT client for deterministic testing
- StubToolRunner: Safe tool execution stub for tests
- RealToolRunner: Production tool runner with RFC1918 validation
- ToolRunner: Abstract base class
- ToolResult: Execution result dataclass
- get_tool_runner: Factory function
"""

from .fake_gpt_manager import FakeGPTManager
from .tool_runner import (
    StubToolRunner,
    RealToolRunner,
    ToolRunner,
    ToolResult,
    get_tool_runner,
)

__all__ = [
    "FakeGPTManager",
    "StubToolRunner",
    "RealToolRunner",
    "ToolRunner",
    "ToolResult",
    "get_tool_runner",
]
