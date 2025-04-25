from core.utils.local_llm_manager import LocalLLMManager
from core.gpt_manager import GPTManager
from rich.console import Console
import time
import json
import os

console = Console()

class LLMRouter:
    """
    Modular LLM router for ARIASKA_RL.
    Routes tasks to SenecaLLM, LilyLLM, or GPTManager based on task_type and prompt complexity.
    """
    def __init__(self, log_path="logs/llm_usage.jsonl"):
        self.seneca = LocalLLMManager(model_name=os.environ.get("ARIASKA_SENECA_MODEL", "wahidmounir/SenecaLLM_x_Qwen2.5-7B-CyberSecurity-Q8_0-GGUF"))
        self.lily   = LocalLLMManager(model_name=os.environ.get("ARIASKA_LILY_MODEL", "QuantFactory/Lily-Cybersecurity-7B-v0.2-GGUF:Q8_0"))
        self.gpt    = GPTManager()
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.cache = {}

    def route_task(self, task_type, description, **kwargs):
        """
        Smart routing: short/simple → Lily, factual/planner → Seneca, complex/critical/stuck → GPT-4o.
        Fallback/escalate to GPT-4o if local model fails or output is invalid.
        """
        model = None
        response = ""
        tokens = 0
        t0 = time.time()
        cache_key = f"{task_type}|{description.strip()[:120]}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        try:
            # Route based on task type and prompt complexity
            prompt_len = len(description.strip())
            is_simple = prompt_len < 120 and ("yes" in description.lower() or "no" in description.lower() or "summarize" in description.lower())
            if task_type == "planner":
                model = "SenecaLLM"
                response = self.seneca.query(description)
                console.print(f"[blue]🦉 SenecaLLM (planner): {response}[/blue]")
            elif task_type == "tactical" or is_simple:
                model = "LilyLLM"
                lily_prompt = "You are a concise assistant: answer succinctly.\n" + description
                response = self.lily.query(lily_prompt)
                console.print(f"[cyan]🌸 LilyLLM (tactical): {response}[/cyan]")
            elif task_type == "dual-llm-feedback":
                model = "GPT-4o"
                response = self.gpt.dual_llm_feedback(description)
                console.print(f"[magenta]🤖 Dual-LLM Feedback: {response}[/magenta]")
            elif task_type == "strategic" or prompt_len > 300 or "stuck" in description.lower() or "plan" in description.lower():
                model = "GPT-4o"
                response = self.gpt.smart_decision(task_type, description)
                console.print(f"[magenta]🤖 GPTManager: {response}[/magenta]")
            else:
                model = "SenecaLLM"
                response = self.seneca.query(description)
                console.print(f"[blue]🦉 SenecaLLM (default): {response}[/blue]")
            # Fallback/escalation if output is empty, too long, or gibberish
            if not response or len(response) > 800 or response.lower().startswith("error") or response.count(" ") < 2:
                model = "GPT-4o"
                response = self.gpt.smart_decision(task_type, description)
                console.print(f"[magenta]🤖 Fallback to GPTManager: {response}[/magenta]")
            tokens = len(response.split())
            self.cache[cache_key] = response
        except Exception as e:
            model = "GPT-4o"
            response = self.gpt.smart_decision(task_type, description)
            console.print(f"[red]❌ LLMRouter error: {e}[/red]")
            self.cache[cache_key] = response
        self._log_llm_call(model, task_type, description, response, tokens, time.time() - t0)
        return response

    def _log_llm_call(self, model, task_type, prompt, response, tokens, elapsed):
        entry = {
            "timestamp": time.time(),
            "model": model,
            "task_type": task_type,
            "prompt": prompt[:200],
            "response": response[:200],
            "tokens": tokens,
            "elapsed": elapsed
        }
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass
