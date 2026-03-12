#!/usr/bin/env python3
"""
core/llm/hf_provider.py — Phase 44: HuggingFace Native LLM Provider

HuggingFace Transformers provider for Ariaska_RL. Runs transformer models
locally using the HuggingFace pipeline — no server process required.

Supports Qwen, Mistral, and Phi instruction-tuned models out of the box.
Models are lazy-loaded on first inference request to avoid startup overhead.

Author: Phase 44 — OpenAI Removal & Local LLM Migration
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.llm.llm_provider import LLMProvider, LLMResponse

logger = logging.getLogger("ariaska.llm.hf_provider")

# Optional heavy imports — guarded to allow graceful degradation
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False
    logger.warning("torch not installed — HuggingFace provider unavailable")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoModelForCausalLM = None  # type: ignore[assignment,misc]
    AutoTokenizer = None  # type: ignore[assignment,misc]
    _TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed — HuggingFace provider unavailable")


DEFAULT_MODELS: Dict[str, str] = {
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "phi-3.5": "microsoft/Phi-3.5-mini-instruct",
}


@dataclass
class HFProviderConfig:
    """Configuration for the HuggingFace provider.

    Attributes:
        model_alias: Short alias key into DEFAULT_MODELS (e.g. 'qwen-7b').
        model_id: Full HuggingFace model identifier.
        device: Target device ('auto', 'cuda', 'cpu').
        dtype: Torch dtype string ('auto', 'float16', 'bfloat16', 'float32').
        max_model_len: Maximum context length for generation.
    """

    model_alias: str
    model_id: str
    device: str = "auto"
    dtype: str = "auto"
    max_model_len: int = 8192


@dataclass
class _ProviderStats:
    """Internal mutable statistics tracker."""

    total_requests: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    errors: int = 0

    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests


class HFProvider(LLMProvider):
    """HuggingFace Transformers LLM provider.

    Loads a causal-LM model directly via ``transformers`` and runs inference
    on the local GPU (or CPU). No server process is needed — the model lives
    in-process.

    Configuration is read from environment variables on construction:
        - ``ARIASKA_HF_MODEL``  — alias into DEFAULT_MODELS (default ``'qwen-7b'``)
        - ``ARIASKA_HF_DEVICE`` — torch device (default ``'auto'``)
        - ``ARIASKA_HF_DTYPE``  — torch dtype (default ``'auto'``)

    The model and tokenizer are **lazy-loaded** on the first call to
    :meth:`chat_completion` to keep import and construction lightweight.
    """

    def __init__(self, config: HFProviderConfig | None = None) -> None:
        if config is not None:
            self._config = config
        else:
            alias = os.getenv("ARIASKA_HF_MODEL", "qwen-7b")
            model_id = DEFAULT_MODELS.get(alias, alias)
            device = os.getenv("ARIASKA_HF_DEVICE", "auto")
            dtype = os.getenv("ARIASKA_HF_DTYPE", "auto")
            self._config = HFProviderConfig(
                model_alias=alias,
                model_id=model_id,
                device=device,
                dtype=dtype,
            )

        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._stats = _ProviderStats()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_device(self) -> str:
        """Resolve ``'auto'`` to the best available device string."""
        if self._config.device != "auto":
            return self._config.device
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _resolve_dtype(self) -> Any:
        """Resolve dtype string to a ``torch.dtype``."""
        if not _TORCH_AVAILABLE:
            return None
        mapping: Dict[str, Any] = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if self._config.dtype in mapping:
            return mapping[self._config.dtype]
        # 'auto' — prefer bfloat16 on cuda, float32 on cpu
        device = self._resolve_device()
        if device == "cuda":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    def _ensure_model(self) -> None:
        """Lazy-load model and tokenizer on first use."""
        if self._model is not None:
            return

        if not _TORCH_AVAILABLE or not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "torch and transformers must be installed for HFProvider"
            )

        device = self._resolve_device()
        dtype = self._resolve_dtype()
        model_id = self._config.model_id

        logger.info(
            "Loading HF model %s on %s (dtype=%s)", model_id, device, dtype
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True,
        )
        if device == "cpu":
            self._model = self._model.to(device)

        self._model.eval()
        logger.info("HF model %s loaded successfully", model_id)

    def _apply_chat_template(
        self, messages: List[Dict[str, str]]
    ) -> str:
        """Format messages using the tokenizer's chat template if available.

        Args:
            messages: OpenAI-style message dicts.

        Returns:
            Formatted prompt string ready for tokenization.
        """
        if (
            self._tokenizer is not None
            and hasattr(self._tokenizer, "apply_chat_template")
        ):
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        # Fallback: simple concatenation
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|{role}|>\n{content}")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int = 150,
        temperature: float = 0.3,
        response_format: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> LLMResponse:
        """Run chat completion locally via HuggingFace Transformers.

        Args:
            messages: Chat messages in OpenAI format.
            model: Model name (ignored — uses configured model).
            max_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            response_format: Unused, accepted for interface compatibility.
            timeout: Unused, accepted for interface compatibility.

        Returns:
            LLMResponse with generated text and metadata.
        """
        start = time.perf_counter()
        try:
            self._ensure_model()

            prompt = self._apply_chat_template(messages)
            inputs = self._tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self._model.device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self._model.device)

            input_len = input_ids.shape[1]

            generate_kwargs: Dict[str, Any] = {
                "input_ids": input_ids,
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": self._tokenizer.eos_token_id,
            }
            if attention_mask is not None:
                generate_kwargs["attention_mask"] = attention_mask
            if temperature > 0:
                generate_kwargs["temperature"] = temperature
                generate_kwargs["top_p"] = 0.95

            with torch.no_grad():
                output_ids = self._model.generate(**generate_kwargs)

            # Decode only the new tokens
            new_tokens = output_ids[0][input_len:]
            text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

            output_len = len(new_tokens)
            total_tokens = input_len + output_len
            latency_ms = (time.perf_counter() - start) * 1000

            self._stats.total_requests += 1
            self._stats.total_tokens += total_tokens
            self._stats.total_latency_ms += latency_ms

            return LLMResponse(
                text=text.strip(),
                model=self._config.model_alias,
                provider="huggingface",
                input_tokens=input_len,
                output_tokens=output_len,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                cost_usd=0.0,
            )

        except Exception as e:
            self._stats.errors += 1
            latency_ms = (time.perf_counter() - start) * 1000
            self._stats.total_latency_ms += latency_ms
            logger.error("HuggingFace inference failed: %s", e)
            return LLMResponse(
                text="",
                model=self._config.model_alias,
                provider="huggingface",
                latency_ms=latency_ms,
                error=str(e),
            )

    def is_available(self) -> bool:
        """Check if the provider can serve requests.

        Returns:
            True if running on CPU or CUDA is available.
        """
        if not _TORCH_AVAILABLE or not _TRANSFORMERS_AVAILABLE:
            return False
        device = self._resolve_device()
        if device == "cpu":
            return True
        return torch.cuda.is_available()

    def get_model_name(self) -> str:
        """Return the configured model alias."""
        return self._config.model_alias

    def get_provider_name(self) -> str:
        """Return the provider identifier."""
        return "huggingface"

    def get_cost_per_1k_tokens(self, model: Optional[str] = None) -> float:
        """Local inference is free."""
        return 0.0

    def shutdown(self) -> None:
        """Release model, tokenizer, and GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("HFProvider shut down, resources released")

    def get_stats(self) -> Dict[str, Any]:
        """Return runtime statistics.

        Returns:
            Dict with provider info, request counts, token totals,
            average latency, and error count.
        """
        base = super().get_stats()
        base.update(
            {
                "total_requests": self._stats.total_requests,
                "total_tokens": self._stats.total_tokens,
                "avg_latency_ms": round(self._stats.avg_latency_ms, 2),
                "errors": self._stats.errors,
                "model_id": self._config.model_id,
                "device": self._resolve_device(),
                "model_loaded": self._model is not None,
            }
        )
        return base
