#!/usr/bin/env python3
"""
core/llm/local_llm_provider.py — Phase 43: Local LLM Provider (GPU Acceleration)

Manages a local LLM server (llama-cpp-python or vLLM) for offloading
nano+mini tier requests to a GPU-accelerated local model. OpenAI-compatible
API on localhost, drop-in replacement for Venice integration slot.

Supported backends:
  - llama-cpp-python (GGUF models, already in requirements.txt)
  - vLLM (AWQ/GPTQ models, optional — faster for concurrent requests)

Model: Qwen3-32B-Instruct (Q4_K_M or Q5_K_M GGUF, ~18-22GB VRAM)

Feature-flag gated: FF_LOCAL_LLM (auto-detected based on model file + CUDA).

Author: Phase 43 — GPU Acceleration Layer
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariaska.llm.local_provider")

# ── Constants ───────────────────────────────────────────────────────────────

DEFAULT_PORT = 8192
DEFAULT_HOST = "127.0.0.1"
HEALTH_CHECK_INTERVAL = 30  # seconds
STARTUP_TIMEOUT = 120  # seconds to wait for server to start
MAX_RETRIES = 3

# Model search paths (checked in order)
_MODEL_SEARCH_PATHS = [
    Path("/models"),
    Path("/root/models"),
    Path.home() / "models",
    Path.cwd() / "models",
]

# Supported model file extensions
_MODEL_EXTENSIONS = {".gguf", ".awq", ".gptq"}


@dataclass
class LocalLLMConfig:
    """Configuration for the local LLM server."""
    model_path: str = ""
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    n_gpu_layers: int = -1  # -1 = offload all layers to GPU
    n_ctx: int = 8192       # Context window
    n_batch: int = 512      # Batch size for prompt processing
    n_threads: int = 4      # CPU threads (fallback)
    flash_attn: bool = True # Use flash attention if available
    backend: str = "llama-cpp"  # "llama-cpp" or "vllm"
    # Performance tuning
    continuous_batching: bool = True
    max_concurrent: int = 4
    # Cost tracking (local = $0.00 API cost)
    cost_per_1k_tokens: float = 0.0


@dataclass
class LocalLLMStats:
    """Runtime statistics for the local LLM server."""
    total_requests: int = 0
    total_tokens: int = 0
    total_errors: int = 0
    avg_latency_ms: float = 0.0
    server_uptime_s: float = 0.0
    _latencies: List[float] = field(default_factory=list)
    
    def record_request(self, tokens: int, latency_ms: float, error: bool = False) -> None:
        self.total_requests += 1
        self.total_tokens += tokens
        if error:
            self.total_errors += 1
        self._latencies.append(latency_ms)
        if len(self._latencies) > 100:
            self._latencies = self._latencies[-100:]
        self.avg_latency_ms = sum(self._latencies) / len(self._latencies)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_errors": self.total_errors,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "tokens_per_second": round(
                self.total_tokens / max(1, self.server_uptime_s), 1
            ) if self.server_uptime_s > 0 else 0.0,
        }


class LocalLLMProvider:
    """
    Manages a local LLM server and exposes an OpenAI-compatible client.
    
    The server runs as a subprocess (llama-cpp-python HTTP server) and
    is accessed via the standard OpenAI Python client with a custom
    base_url pointing to localhost.
    
    Usage:
        provider = LocalLLMProvider.from_env()
        if provider.is_available():
            client = provider.get_client()
            # Use client like OpenAI client: client.chat.completions.create(...)
    """
    
    def __init__(self, config: Optional[LocalLLMConfig] = None):
        self.config = config or LocalLLMConfig()
        self.stats = LocalLLMStats()
        self._client = None
        self._server_process: Optional[subprocess.Popen] = None
        self._server_started = False
        self._server_start_time: float = 0.0
        self._lock = threading.Lock()
        self._health_thread: Optional[threading.Thread] = None
        self._stop_health = threading.Event()
        
        # Auto-detect model path if not configured
        if not self.config.model_path:
            self.config.model_path = os.getenv("ARIASKA_LOCAL_MODEL_PATH", "")
        if not self.config.model_path:
            self.config.model_path = self._find_model_file()
        
        # Port from env
        self.config.port = int(os.getenv("ARIASKA_LOCAL_LLM_PORT", str(DEFAULT_PORT)))
    
    @classmethod
    def from_env(cls) -> "LocalLLMProvider":
        """Create provider from environment variables."""
        config = LocalLLMConfig(
            model_path=os.getenv("ARIASKA_LOCAL_MODEL_PATH", ""),
            host=os.getenv("ARIASKA_LOCAL_LLM_HOST", DEFAULT_HOST),
            port=int(os.getenv("ARIASKA_LOCAL_LLM_PORT", str(DEFAULT_PORT))),
            n_gpu_layers=int(os.getenv("ARIASKA_LOCAL_GPU_LAYERS", "-1")),
            n_ctx=int(os.getenv("ARIASKA_LOCAL_CTX_SIZE", "8192")),
            backend=os.getenv("ARIASKA_LOCAL_BACKEND", "llama-cpp"),
        )
        return cls(config)
    
    def _find_model_file(self) -> str:
        """Search for a GGUF/AWQ/GPTQ model file in standard locations."""
        for search_dir in _MODEL_SEARCH_PATHS:
            try:
                if not search_dir.exists():
                    continue
                for f in sorted(search_dir.iterdir()):
                    if f.suffix.lower() in _MODEL_EXTENSIONS and f.is_file():
                        logger.info(f"Found local model: {f}")
                        return str(f)
            except (PermissionError, OSError):
                continue
        return ""
    
    def has_model(self) -> bool:
        """Check if a model file is available."""
        return bool(self.config.model_path) and Path(self.config.model_path).exists()
    
    def is_server_running(self) -> bool:
        """Check if the local LLM server is reachable."""
        try:
            import urllib.request
            url = f"http://{self.config.host}:{self.config.port}/v1/models"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False
    
    def is_available(self) -> bool:
        """Check if local LLM is available (model exists OR server running)."""
        return self.has_model() or self.is_server_running()
    
    def start_server(self, wait: bool = True) -> bool:
        """
        Start the local LLM server as a subprocess.
        
        Args:
            wait: Whether to wait for the server to be ready.
            
        Returns:
            True if server started successfully.
        """
        with self._lock:
            if self.is_server_running():
                logger.info("Local LLM server already running")
                self._server_started = True
                self._server_start_time = time.time()
                return True
            
            if not self.has_model():
                logger.error(f"No model file found at: {self.config.model_path}")
                return False
            
            model_path = self.config.model_path
            
            if self.config.backend == "vllm":
                cmd = self._build_vllm_cmd(model_path)
            else:
                cmd = self._build_llamacpp_cmd(model_path)
            
            logger.info(f"Starting local LLM server: {' '.join(cmd[:5])}...")
            
            try:
                self._server_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid if hasattr(os, 'setsid') else None,
                )
            except FileNotFoundError as e:
                logger.error(f"Failed to start LLM server — command not found: {e}")
                return False
            except Exception as e:
                logger.error(f"Failed to start LLM server: {e}")
                return False
        
        if wait:
            return self._wait_for_server()
        return True
    
    def _build_llamacpp_cmd(self, model_path: str) -> List[str]:
        """Build llama-cpp-python server command."""
        cmd = [
            "python3", "-m", "llama_cpp.server",
            "--model", model_path,
            "--host", self.config.host,
            "--port", str(self.config.port),
            "--n_ctx", str(self.config.n_ctx),
            "--n_gpu_layers", str(self.config.n_gpu_layers),
        ]
        if self.config.flash_attn:
            cmd.extend(["--flash_attn", "true"])
        return cmd
    
    def _build_vllm_cmd(self, model_path: str) -> List[str]:
        """Build vLLM server command."""
        cmd = [
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--host", self.config.host,
            "--port", str(self.config.port),
            "--max-model-len", str(self.config.n_ctx),
            "--dtype", "auto",
            "--gpu-memory-utilization", "0.85",
        ]
        if self.config.continuous_batching:
            cmd.append("--enable-chunked-prefill")
        return cmd
    
    def _wait_for_server(self) -> bool:
        """Wait for the server to become ready."""
        start = time.time()
        while time.time() - start < STARTUP_TIMEOUT:
            if self.is_server_running():
                elapsed = time.time() - start
                logger.info(f"Local LLM server ready in {elapsed:.1f}s")
                self._server_started = True
                self._server_start_time = time.time()
                self._start_health_monitor()
                return True
            
            # Check if process died
            if self._server_process and self._server_process.poll() is not None:
                stderr = ""
                if self._server_process.stderr:
                    stderr = self._server_process.stderr.read().decode(errors="replace")[-500:]
                logger.error(f"LLM server process died: {stderr}")
                return False
            
            time.sleep(2)
        
        logger.error(f"LLM server failed to start within {STARTUP_TIMEOUT}s")
        self.stop_server()
        return False
    
    def stop_server(self) -> None:
        """Stop the local LLM server."""
        self._stop_health.set()
        if self._server_process:
            try:
                if hasattr(os, 'killpg'):
                    os.killpg(os.getpgid(self._server_process.pid), signal.SIGTERM)
                else:
                    self._server_process.terminate()
                self._server_process.wait(timeout=10)
            except Exception:
                if self._server_process:
                    self._server_process.kill()
            self._server_process = None
        self._server_started = False
        self._client = None
        logger.info("Local LLM server stopped")
    
    def get_client(self) -> Any:
        """
        Get an OpenAI-compatible client pointing to the local server.
        
        Returns:
            openai.OpenAI client configured for local server.
            
        Raises:
            RuntimeError if server is not running.
        """
        if self._client is not None:
            return self._client
        
        if not self.is_server_running() and not self.start_server():
            raise RuntimeError("Local LLM server is not available")
        
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key="local-no-key-needed",
                base_url=f"http://{self.config.host}:{self.config.port}/v1",
            )
            logger.info(
                f"Local LLM client initialized | "
                f"model={Path(self.config.model_path).stem if self.config.model_path else 'server'} | "
                f"port={self.config.port}"
            )
            return self._client
        except ImportError:
            raise RuntimeError("openai package required for local LLM client")
    
    def get_model_name(self) -> str:
        """Get the local model name for API calls."""
        if self.config.model_path:
            return Path(self.config.model_path).stem
        return "local-model"
    
    def _start_health_monitor(self) -> None:
        """Start background health monitoring thread."""
        self._stop_health.clear()
        self._health_thread = threading.Thread(
            target=self._health_loop, daemon=True, name="local-llm-health"
        )
        self._health_thread.start()
    
    def _health_loop(self) -> None:
        """Periodic health check for the local server."""
        while not self._stop_health.is_set():
            self._stop_health.wait(HEALTH_CHECK_INTERVAL)
            if self._stop_health.is_set():
                break
            if self._server_started and not self.is_server_running():
                logger.warning("Local LLM server health check failed — attempting restart")
                self._client = None
                self._server_started = False
                self.start_server(wait=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get runtime statistics."""
        if self._server_started:
            self.stats.server_uptime_s = time.time() - self._server_start_time
        stats = self.stats.to_dict()
        stats["server_running"] = self._server_started or self.is_server_running()
        stats["model"] = self.get_model_name()
        stats["backend"] = self.config.backend
        stats["port"] = self.config.port
        return stats
    
    def __del__(self):
        """Cleanup on garbage collection."""
        try:
            self.stop_server()
        except Exception:
            pass


# ── Singleton ───────────────────────────────────────────────────────────────

_provider: Optional[LocalLLMProvider] = None
_provider_lock = threading.Lock()


def get_local_llm_provider() -> LocalLLMProvider:
    """Get or create the singleton LocalLLMProvider."""
    global _provider
    if _provider is None:
        with _provider_lock:
            if _provider is None:
                _provider = LocalLLMProvider.from_env()
    return _provider


def reset_local_llm_provider() -> None:
    """Reset singleton — primarily for testing."""
    global _provider
    with _provider_lock:
        if _provider is not None:
            _provider.stop_server()
        _provider = None
