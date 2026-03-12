#!/usr/bin/env python3
"""Replace hardcoded GPT model strings with 'local-llm' across core Python files."""
import os
import re

patterns = [
    (r'"gpt-5\.2-codex"', '"local-llm"'),
    (r'"gpt-5-nano"', '"local-llm"'),
    (r'"gpt-5\.2-mini"', '"local-llm"'),
    (r"'gpt-5\.2-codex'", "'local-llm'"),
    (r"'gpt-5-nano'", "'local-llm'"),
    (r"'gpt-5\.2-mini'", "'local-llm'"),
]

changed_files = []
for root, dirs, files in os.walk("core"):
    for fname in files:
        if not fname.endswith(".py"):
            continue
        path = os.path.join(root, fname)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        new_content = content
        for pat, repl in patterns:
            new_content = re.sub(pat, repl, new_content)
        if new_content != content:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)
            changed_files.append(path)

for p in sorted(changed_files):
    print(p)
print(f"Total: {len(changed_files)} files changed")
