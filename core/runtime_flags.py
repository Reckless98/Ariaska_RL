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
    # Phase 43: GPU + Local LLM detection
    gpu_available: bool = False     # CUDA GPU detected
    gpu_name: str = ""              # GPU model name (e.g. "NVIDIA RTX 3090")
    gpu_vram_gb: float = 0.0        # GPU VRAM in GB
    local_llm_available: bool = False  # Local LLM model file found + server reachable
    local_llm_model: str = ""       # Local model name (e.g. "qwen3-32b-instruct")


# Global singleton instance
_flags = RuntimeFlags()


def set_runtime_flags(
    offline: bool = False,
    enable_llm: bool = True,
    require_llm: bool = True,
    gpu_available: bool = False,
    gpu_name: str = "",
    gpu_vram_gb: float = 0.0,
    local_llm_available: bool = False,
    local_llm_model: str = "",
) -> None:
    """
    Set global runtime flags. Should be called once at startup before
    any components are instantiated.
    
    Args:
        offline: Force offline mode (no LLM calls, deterministic placeholders)
        enable_llm: Whether LLM calls are enabled at all
        require_llm: If True and enable_llm, fail fast if no API key
        gpu_available: Whether a CUDA GPU was detected
        gpu_name: GPU model name
        gpu_vram_gb: GPU VRAM in GB
        local_llm_available: Whether local LLM is available
        local_llm_model: Local LLM model name
    """
    global _flags
    with _lock:
        _flags = RuntimeFlags(
            offline=offline,
            enable_llm=enable_llm,
            require_llm=require_llm,
            initialized=True,
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            gpu_vram_gb=gpu_vram_gb,
            local_llm_available=local_llm_available,
            local_llm_model=local_llm_model,
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


def detect_gpu() -> dict:
    """Auto-detect GPU presence and capabilities.
    
    Returns:
        Dict with gpu_available, gpu_name, gpu_vram_gb.
    """
    result = {"gpu_available": False, "gpu_name": "", "gpu_vram_gb": 0.0}
    try:
        import torch
        if torch.cuda.is_available():
            result["gpu_available"] = True
            result["gpu_name"] = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_mem
            result["gpu_vram_gb"] = round(vram_bytes / (1024 ** 3), 1)
    except ImportError:
        pass
    return result


def detect_local_llm() -> dict:
    """Auto-detect local LLM model file availability.
    
    Checks for model files in standard locations and whether a local
    LLM server is reachable.
    
    Returns:
        Dict with local_llm_available, local_llm_model.
    """
    import os
    from pathlib import Path
    
    result = {"local_llm_available": False, "local_llm_model": ""}
    
    # Check env var first
    model_path = os.getenv("ARIASKA_LOCAL_MODEL_PATH", "")
    if model_path and Path(model_path).exists():
        result["local_llm_available"] = True
        result["local_llm_model"] = Path(model_path).stem
        return result
    
    # Check standard locations
    search_dirs = [
        Path.home() / "models",
        Path("/models"),
        Path("/root/models"),
        Path.cwd() / "models",
    ]
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for ext in ("*.gguf", "*.awq", "*.gptq"):
            for model_file in search_dir.glob(ext):
                result["local_llm_available"] = True
                result["local_llm_model"] = model_file.stem
                # Set env var for downstream consumers
                os.environ.setdefault("ARIASKA_LOCAL_MODEL_PATH", str(model_file))
                return result
    
    # Check if local LLM server is already running
    port = int(os.getenv("ARIASKA_LOCAL_LLM_PORT", "8192"))
    try:
        import urllib.request
        req = urllib.request.Request(
            f"http://localhost:{port}/v1/models",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                result["local_llm_available"] = True
                result["local_llm_model"] = "local-server"
    except Exception:
        pass
    
    return result
