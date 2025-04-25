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
    Routes tasks to SenecaLLM, LilyLLM, or GPTManager based on task_type.
    """
    def __init__(self, log_path="logs/llm_usage.jsonl"):
        self.seneca = LocalLLMManager(model_name="wahidmounir/SenecaLLM_x_Qwen2.5-7B-CyberSecurity-Q8_0-GGUF")
        self.lily   = LocalLLMManager(model_name="QuantFactory/Lily-Cybersecurity-7B-v0.2-GGUF:Q8_0")
        self.gpt    = GPTManager()
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def route_task(self, task_type, description, **kwargs):
        """
        Select and query the appropriate LLM based on task_type.
        Fallback/escalate to GPT-4o if local model fails or output is invalid.
        """
        model = None
        response = ""
        tokens = 0
        t0 = time.time()
        try:
            if task_type == "planner":
                model = "SenecaLLM"
                response = self.seneca.query(description)
                console.print(f"[blue]🦉 SenecaLLM (planner): {response}[/blue]")
            elif task_type == "tactical":
                model = "LilyLLM"
                response = self.lily.query(description)
                console.print(f"[cyan]🌸 LilyLLM (tactical): {response}[/cyan]")
            elif task_type == "strategic":
                model = "GPT-4o"
                response = self.gpt.smart_decision(task_type, description)
                console.print(f"[magenta]🤖 GPTManager: {response}[/magenta]")
            else:
                model = "SenecaLLM"
                response = self.seneca.query(description)
                console.print(f"[blue]🦉 SenecaLLM (default): {response}[/blue]")
            # Fallback/escalation if output is empty or error
            if not response or "<error>" in response or response.strip() == "":
                model = "GPT-4o"
                response = self.gpt.smart_decision(task_type, description)
                console.print(f"[magenta]🤖 GPTManager (fallback): {response}[/magenta]")
            tokens = len(response.split())
        except Exception as e:
            model = "GPT-4o"
            response = self.gpt.smart_decision(task_type, description)
            console.print(f"[red]❌ LLMRouter error: {e}[/red]")
            tokens = len(response.split())
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
