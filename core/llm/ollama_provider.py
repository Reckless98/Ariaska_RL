"""Phase 44: CPU/Ollama integration for local LLM fallback."""

from __future__ import annotations

import json
import logging
import time
import requests
from typing import Any, Dict, List, Optional

from core.llm.llm_provider import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

class OllamaProvider(LLMProvider):
    """Provider wrapping a local Ollama server (optimized for CPU/iGPU).
    
    Connects to the native Ollama API on http://localhost:11434.
    """

    def __init__(self, host: str = "http://localhost:11434", default_model: str = "jaahas/qwen3.5-uncensored:4b") -> None:
        self.host = host.rstrip("/")
        self.default_model = default_model
        self._stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "errors": 0,
        }

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int = 150,
        temperature: float = 0.3,
        response_format: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> LLMResponse:
        
        start = time.time()
        actual_model = model if model not in ["local-llm", "local"] else self.default_model
        
        # Ollama API translation
        payload = {
            "model": actual_model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            },
            "stream": False
        }
        
        if isinstance(response_format, dict) and response_format.get("type") == "json_object":
            payload["format"] = "json"
        elif isinstance(response_format, str) and response_format == "json_object":
            payload["format"] = "json"
        
        req_timeout = timeout or 600.0
        
        try:
            response = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=req_timeout
            )
            response.raise_for_status()
            result = response.json()
                
            text = result.get("message", {}).get("content", "")
            input_tokens = result.get("prompt_eval_count", 0)
            output_tokens = result.get("eval_count", 0)
            total_tokens = input_tokens + output_tokens
            latency = (time.time() - start) * 1000

            self._stats["total_requests"] += 1
            self._stats["total_tokens"] += total_tokens

            return LLMResponse(
                text=text.strip(),
                model=actual_model,
                provider="ollama",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_ms=latency,
                cost_usd=0.0,
            )
        except Exception as e:
            self._stats["errors"] += 1
            latency = (time.time() - start) * 1000
            logger.error(f"Ollama request failed: {e}")
            return LLMResponse(
                text="",
                model=actual_model,
                provider="ollama",
                latency_ms=latency,
                error=str(e),
            )

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=2.0)
            if response.status_code == 200:
                return True
        except Exception:
            pass
        return False

    def get_model_name(self) -> str:
        return self.default_model

    def get_provider_name(self) -> str:
        return "ollama"
        
    def get_stats(self) -> Dict[str, Any]:
        base = super().get_stats()
        base.update(self._stats)
        return base
