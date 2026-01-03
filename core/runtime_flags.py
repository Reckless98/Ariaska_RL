#!/usr/bin/env python3
"""
core/runtime_flags.py — Global runtime configuration for ARIASKA

This module provides a central place to set and get runtime flags that affect
component behavior across the codebase. Set once by AriaskaTrainer before
any components are instantiated.

Usage:
    # In trainer initialization:
    from core.runtime_flags import set_runtime_flags
    set_runtime_flags(offline=True, enable_llm=False, require_llm=False)
    
    # In GPTManager or other components:
    from core.runtime_flags import get_runtime_flags
    flags = get_runtime_flags()
    if flags.offline:
        # use offline behavior
"""

from dataclasses import dataclass
from typing import Optional
import threading

# Thread-safe lock for flag updates
_lock = threading.Lock()


@dataclass
class RuntimeFlags:
    """Runtime configuration flags."""
    offline: bool = False           # Force offline mode (no LLM calls)
    enable_llm: bool = True         # Whether LLM is enabled
    require_llm: bool = True        # Fail fast if LLM required but no API key
    initialized: bool = False       # Whether flags have been explicitly set


# Global singleton instance
_flags = RuntimeFlags()


def set_runtime_flags(
    offline: bool = False,
    enable_llm: bool = True,
    require_llm: bool = True
) -> None:
    """
    Set global runtime flags. Should be called once at startup before
    any components are instantiated.
    
    Args:
        offline: Force offline mode (no LLM calls, deterministic placeholders)
        enable_llm: Whether LLM calls are enabled at all
        require_llm: If True and enable_llm, fail fast if no API key
    """
    global _flags
    with _lock:
        _flags = RuntimeFlags(
            offline=offline,
            enable_llm=enable_llm,
            require_llm=require_llm,
            initialized=True
        )


def get_runtime_flags() -> RuntimeFlags:
    """
    Get current runtime flags.
    
    Returns:
        RuntimeFlags instance with current settings
    """
    with _lock:
        return _flags


def reset_runtime_flags() -> None:
    """Reset flags to defaults. Primarily for testing."""
    global _flags
    with _lock:
        _flags = RuntimeFlags()


def is_offline_mode() -> bool:
    """Quick check for offline mode."""
    return _flags.offline or not _flags.enable_llm
