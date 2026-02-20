"""
tests/test_local_llm_provider.py — Phase 43: Local LLM Provider Tests

Tests for:
  - LocalLLMConfig defaults and env-var overrides
  - Model file detection in search paths
  - Server lifecycle (start/stop/health)
  - OpenAI client creation
  - Stats tracking
  - Singleton management
"""
import os
import pytest
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

# Set dry-run before any imports
os.environ["ARIASKA_DRY_RUN"] = "1"


class TestLocalLLMConfig:
    """Test LocalLLMConfig dataclass."""

    def test_default_config(self):
        from core.llm.local_llm_provider import LocalLLMConfig
        config = LocalLLMConfig()
        assert config.port == 8192
        assert config.host == "127.0.0.1"
        assert config.n_ctx == 8192
        assert config.n_gpu_layers == -1
        assert config.backend == "llama-cpp"
        assert config.cost_per_1k_tokens == 0.0

    def test_config_custom_values(self):
        from core.llm.local_llm_provider import LocalLLMConfig
        config = LocalLLMConfig(
            model_path="/models/test.gguf",
            port=9000,
            n_gpu_layers=40,
            backend="vllm",
        )
        assert config.model_path == "/models/test.gguf"
        assert config.port == 9000
        assert config.n_gpu_layers == 40
        assert config.backend == "vllm"


class TestLocalLLMStats:
    """Test LocalLLMStats tracking."""

    def test_stats_default(self):
        from core.llm.local_llm_provider import LocalLLMStats
        stats = LocalLLMStats()
        assert stats.total_requests == 0
        assert stats.total_tokens == 0
        assert stats.avg_latency_ms == 0.0

    def test_record_request(self):
        from core.llm.local_llm_provider import LocalLLMStats
        stats = LocalLLMStats()
        stats.record_request(tokens=100, latency_ms=50.0)
        assert stats.total_requests == 1
        assert stats.total_tokens == 100
        assert stats.avg_latency_ms == 50.0

    def test_record_multiple_requests(self):
        from core.llm.local_llm_provider import LocalLLMStats
        stats = LocalLLMStats()
        stats.record_request(tokens=100, latency_ms=50.0)
        stats.record_request(tokens=200, latency_ms=100.0)
        assert stats.total_requests == 2
        assert stats.total_tokens == 300
        assert stats.avg_latency_ms == 75.0  # (50 + 100) / 2

    def test_record_error(self):
        from core.llm.local_llm_provider import LocalLLMStats
        stats = LocalLLMStats()
        stats.record_request(tokens=0, latency_ms=0.0, error=True)
        assert stats.total_errors == 1

    def test_to_dict(self):
        from core.llm.local_llm_provider import LocalLLMStats
        stats = LocalLLMStats()
        stats.record_request(tokens=100, latency_ms=50.0)
        d = stats.to_dict()
        assert d["total_requests"] == 1
        assert d["total_tokens"] == 100

    def test_latency_window_cap(self):
        from core.llm.local_llm_provider import LocalLLMStats
        stats = LocalLLMStats()
        for i in range(150):
            stats.record_request(tokens=1, latency_ms=float(i))
        # Only last 100 latencies kept
        assert len(stats._latencies) == 100


class TestLocalLLMProvider:
    """Test LocalLLMProvider core functionality."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Reset singleton after each test."""
        yield
        from core.llm.local_llm_provider import reset_local_llm_provider
        reset_local_llm_provider()

    def test_no_model_has_model_false(self):
        from core.llm.local_llm_provider import LocalLLMProvider, LocalLLMConfig
        config = LocalLLMConfig(model_path="/nonexistent/model.gguf")
        provider = LocalLLMProvider(config)
        assert not provider.has_model()

    def test_model_name_from_path(self):
        from core.llm.local_llm_provider import LocalLLMProvider, LocalLLMConfig
        config = LocalLLMConfig(model_path="/models/Qwen3-32B-Instruct-Q4_K_M.gguf")
        provider = LocalLLMProvider(config)
        assert provider.get_model_name() == "Qwen3-32B-Instruct-Q4_K_M"

    def test_model_name_no_path(self):
        from core.llm.local_llm_provider import LocalLLMProvider, LocalLLMConfig
        config = LocalLLMConfig(model_path="")
        provider = LocalLLMProvider(config)
        assert provider.get_model_name() == "local-model"

    def test_server_not_running(self):
        from core.llm.local_llm_provider import LocalLLMProvider, LocalLLMConfig
        config = LocalLLMConfig(model_path="", port=59999)
        provider = LocalLLMProvider(config)
        assert not provider.is_server_running()

    def test_build_llamacpp_cmd(self):
        from core.llm.local_llm_provider import LocalLLMProvider, LocalLLMConfig
        config = LocalLLMConfig(
            model_path="/models/test.gguf",
            host="0.0.0.0",
            port=8192,
            n_ctx=4096,
            n_gpu_layers=99,
        )
        provider = LocalLLMProvider(config)
        cmd = provider._build_llamacpp_cmd("/models/test.gguf")
        assert "python3" in cmd[0]
        assert "--model" in cmd
        assert "/models/test.gguf" in cmd
        assert "--port" in cmd
        assert "8192" in cmd

    def test_build_vllm_cmd(self):
        from core.llm.local_llm_provider import LocalLLMProvider, LocalLLMConfig
        config = LocalLLMConfig(
            model_path="/models/test.awq",
            backend="vllm",
            port=8192,
        )
        provider = LocalLLMProvider(config)
        cmd = provider._build_vllm_cmd("/models/test.awq")
        assert "vllm" in " ".join(cmd)
        assert "/models/test.awq" in cmd

    def test_start_server_no_model(self):
        from core.llm.local_llm_provider import LocalLLMProvider, LocalLLMConfig
        config = LocalLLMConfig(model_path="/nonexistent/model.gguf")
        provider = LocalLLMProvider(config)
        assert not provider.start_server(wait=False)

    def test_stop_server_no_crash(self):
        from core.llm.local_llm_provider import LocalLLMProvider, LocalLLMConfig
        config = LocalLLMConfig(model_path="")
        provider = LocalLLMProvider(config)
        # Should not crash even if nothing is running
        provider.stop_server()

    def test_get_stats(self):
        from core.llm.local_llm_provider import LocalLLMProvider, LocalLLMConfig
        config = LocalLLMConfig(model_path="", port=59999)
        provider = LocalLLMProvider(config)
        stats = provider.get_stats()
        assert "server_running" in stats
        assert "model" in stats
        assert stats["backend"] == "llama-cpp"

    def test_from_env(self):
        from core.llm.local_llm_provider import LocalLLMProvider
        with patch.dict(os.environ, {
            "ARIASKA_LOCAL_MODEL_PATH": "/tmp/test.gguf",
            "ARIASKA_LOCAL_LLM_PORT": "9999",
            "ARIASKA_LOCAL_BACKEND": "vllm",
        }):
            provider = LocalLLMProvider.from_env()
            assert provider.config.model_path == "/tmp/test.gguf"
            assert provider.config.port == 9999
            assert provider.config.backend == "vllm"


class TestSingleton:
    """Test singleton pattern."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        from core.llm.local_llm_provider import reset_local_llm_provider
        reset_local_llm_provider()

    def test_get_returns_same_instance(self):
        from core.llm.local_llm_provider import get_local_llm_provider
        p1 = get_local_llm_provider()
        p2 = get_local_llm_provider()
        assert p1 is p2

    def test_reset_creates_new_instance(self):
        from core.llm.local_llm_provider import (
            get_local_llm_provider,
            reset_local_llm_provider,
        )
        p1 = get_local_llm_provider()
        reset_local_llm_provider()
        p2 = get_local_llm_provider()
        assert p1 is not p2


class TestModelSearch:
    """Test model file discovery."""

    def test_find_model_in_temp_dir(self, tmp_path):
        from core.llm.local_llm_provider import LocalLLMProvider, LocalLLMConfig, _MODEL_SEARCH_PATHS
        
        # Create a fake model file
        model_file = tmp_path / "test-model.gguf"
        model_file.write_text("fake model data")
        
        # Temporarily add tmp_path to search paths
        original = list(_MODEL_SEARCH_PATHS)
        import core.llm.local_llm_provider as mod
        mod._MODEL_SEARCH_PATHS = [tmp_path] + original
        
        try:
            config = LocalLLMConfig(model_path="")
            provider = LocalLLMProvider(config)
            # The provider should find the model
            assert provider.config.model_path.endswith(".gguf")
        finally:
            mod._MODEL_SEARCH_PATHS = original
