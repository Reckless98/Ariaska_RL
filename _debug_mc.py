#!/usr/bin/env python3
"""Debug micro_chain test."""
import json, os, sys
os.environ['ARIASKA_DRY_RUN'] = '1'

class _StubGPT:
    def __init__(self):
        self._calls = []
    def gpt_request(self, prompt, task_type='', agent_id='', max_tokens=100, model=None, **kw):
        self._calls.append({'task_type': task_type, 'model': model, 'prompt_start': prompt[:60]})
        msg = "  Call #%d: task_type=%r, model=%r" % (len(self._calls), task_type, model)
        print(msg)
        if model == 'local-llm':
            return 'recon_gap'
        return '{}'
    def can_make_request(self, **kw):
        return True

import core.llm.micro_chain as mc_mod
mc_mod.NANO_ABLATION = False
from core.llm.micro_chain import MicroChain
gpt = _StubGPT()
mc = MicroChain(gpt)
board = {'ports': {22, 80}, 'services': {'ssh', 'http'}, 'credentials': set(), 'vulns': set(), 'shells': set(), 'users': set(), 'web_paths': set(), 'phase': 'RECON', 'flags_set': set()}
result = mc.decide(phase='RECON', discovery_board=board, recent_commands=[], available_templates=['nmap'], agent_role='recon')
print(f'\nResult: {result}')
print(f'Total calls: {len(gpt._calls)}')
