import subprocess

class GPTCacheHandler:
    def __init__(self):
        self.cache = {}

    def query(self, prompt, model="gpt-4o-mini"):
        key = hash(prompt)
        if key in self.cache:
            return self.cache[key]
        try:
            result = subprocess.run(
                ["sgpt", "--model", model, "--role", "aria", prompt],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
                text=True,
            )
            response = result.stdout.strip()
        except Exception:
            response = "GPT unavailable."
        self.cache[key] = response
        return response
