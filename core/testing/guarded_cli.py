"""Run only the bounded, simulated trainer CLI in an isolated guarded interpreter.

The ordinary subprocess guard stays installed in both parent and child. This
helper admits one fixed Python entry point, never arbitrary shell/tool commands.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

from core.testing import offline_guard as guard


def run_trainer_cli(arguments, directory, timeout=60):
    arguments = list(arguments)
    if any(type(value) is not str for value in arguments):
        raise ValueError("CLI arguments must be plain strings")
    flags = {'--offline', '--no-require-llm', '--no-postmortem', '--dry-run', '--no-watch', '--help'}
    bounds = {'--episodes': (1, 2), '--max-steps': (1, 3), '--seed': (0, 2**31 - 1)}
    seen = set()
    i = 0
    while i < len(arguments):
        flag = arguments[i]
        if flag in seen:
            raise ValueError('Duplicate CLI option')
        seen.add(flag)
        if flag in flags:
            i += 1
            continue
        if flag not in bounds and flag != '--verbosity':
            raise ValueError('CLI option is not permitted by the offline harness')
        if i + 1 >= len(arguments):
            raise ValueError('Missing CLI option value')
        value = arguments[i + 1]
        if flag == '--verbosity':
            if value != 'quiet':
                raise ValueError('Only quiet CLI output is supported')
        elif not bounds[flag][0] <= int(value) <= bounds[flag][1]:
            raise ValueError('CLI training budget exceeds the offline harness bound')
        i += 2
    if '--help' not in seen:
        for flag, value in [('--episodes', '1'), ('--max-steps', '2'), ('--seed', '17'), ('--verbosity', 'quiet')]:
            if flag not in seen:
                arguments.extend([flag, value])
        for flag in ('--no-watch', '--no-postmortem', '--dry-run'):
            if flag not in seen:
                arguments.append(flag)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    # -I excludes inherited Python paths/startup hooks. The child installs the
    # deny guard before importing the production trainer and has no credentials.
    bootstrap = f'''
import sys
sys.path.insert(0, {str(repo)!r})
import core.testing.offline_guard as guard
import json, runpy, socket, subprocess
from pathlib import Path
sys.argv = ['core.training.ariaska_trainer'] + {arguments!r}
try:
    runpy.run_module('core.training.ariaska_trainer', run_name='__main__')
finally:
    Path('offline_guard.json').write_text(json.dumps(dict(blocked_attempts=len(guard.blocked),
        guarded=(subprocess.Popen is guard.deny and socket.socket.connect is guard.deny
                 and socket.create_connection is guard.deny))) + '\\n')
'''
    environment = {
        'PATH': os.defpath, 'HOME': str(directory), 'TMPDIR': str(directory),
        'PYTHONHASHSEED': '0', 'PYTHONIOENCODING': 'utf-8',
        'OPENAI_API_KEY': '', 'ANTHROPIC_API_KEY': '', 'HF_TOKEN': '',
        'FF_LOCAL_LLM': '0', 'ARIASKA_MENTOR_BACKEND': 'none',
        'ARIASKA_DRY_RUN': '1', 'ARIASKA_LIVE_MODE': 'false',
        'ARIASKA_TARGET_IP': '192.0.2.10', 'ARIASKA_DATA_DIR': str(directory / 'data'),
    }
    command = [sys.executable, '-I', '-c', bootstrap]
    guard.permitted_python_children.append(arguments)
    with guard._original_popen(command, cwd=directory, env=environment,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        try:
            stdout, stderr = process.communicate(timeout=min(timeout, 90))
        except BaseException:
            process.kill()
            process.communicate()
            raise
    result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    result.guard_report = json.loads((directory / 'offline_guard.json').read_text())
    if not result.guard_report['guarded']:
        raise RuntimeError('Child network/process guard was modified')
    return result
