import subprocess
from core.gpt_manager import GPTManager

class GPTCacheHandler:
    """
    Legacy cache handler for GPT queries. Now delegates all GPT calls to GPTManager for centralized orchestration,
    caching, token tracking, and fallback logic. Maintains a local cache for compatibility.
    """
    def __init__(self):
        self.cache = {}
        self.gpt_manager = GPTManager()

    def query(self, prompt, model="gpt-5.1-codex-mini", agent_id=None):
        """
        Query GPT via GPTManager with caching and fallback. Use agent_id for token tracking.
        """
        key = hash(prompt)
        if key in self.cache:
            return self.cache[key]
        try:
            response = self.gpt_manager.gpt_request(prompt, model=model, agent_id=agent_id)
        except Exception as e:
            response = f"GPT unavailable: {e}"
        self.cache[key] = response
        return response
