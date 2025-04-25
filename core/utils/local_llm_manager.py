import requests

class LocalLLMManager:
    """
    Local LLM manager for SenecaLLM or LilyLLM via Ollama or compatible API.
    """
    def __init__(self, model_name="wahidmounir/SenecaLLM_x_Qwen2.5-7B-CyberSecurity-Q8_0-GGUF", host="http://localhost:11434"):
        self.model_name = model_name
        self.host = host

    def query(self, prompt):
        # TODO: Implement actual call to Ollama or local LLM API
        # For now, just echo the prompt and model name
        try:
            # Example: call Ollama or HuggingFace pipeline here
            # Replace with actual inference code
            return f"[{self.model_name}] {prompt[:80]}"
        except Exception as e:
            return f"LLM error: {e}"

class LocalLilyLLMManager(LocalLLMManager):
    """
    LilyLLM manager. Uses same API as LocalLLMManager, but with Lily model.
    """
    def __init__(self, host="http://localhost:11434"):
        super().__init__(model_name="QuantFactory/Lily-Cybersecurity-7B-v0.2-GGUF:Q8_0", host=host)
