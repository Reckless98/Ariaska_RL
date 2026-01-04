#!/usr/bin/env python3
"""Quick test for role separation and reward checking."""

import os
import sys
import uuid
import logging

# Suppress logs
logging.getLogger().setLevel(logging.WARNING)
os.environ['ARIASKA_DRY_RUN'] = '1'

from collections import Counter
from core.environment.cyber_environment import CyberEnvironment
from core.gpt_manager import GPTManager
from core.orchestration.smart_orchestrator import SmartOrchestrator, SmartOrchestratorConfig

def main():
    # Setup
    env = CyberEnvironment(defer_reset=False)
    gpt_manager = GPTManager()

    config = SmartOrchestratorConfig(
        default_target='10.10.10.10',
        max_steps_per_episode=25,
        dashboard_mode='off',
    )

    orchestrator = SmartOrchestrator(
        env=env,
        gpt_manager=gpt_manager,
        config=config,
    )

    # Run episode
    episode_id = str(uuid.uuid4())[:8]
    result = orchestrator.run_episode(episode_id=episode_id, episode_number=0)

    # Handle result (dict format)
    if isinstance(result, dict):
        total_reward = result.get('total_reward', 0)
        steps = result.get('total_steps', result.get('episode_steps', 0))
        # Extract step records from agents data
        step_records = []
        agents_data = result.get('agents', {})
        for agent_name, agent_info in agents_data.items():
            commands = agent_info.get('commands', [])
            for cmd in commands:
                step_records.append({
                    'agent': agent_name,
                    'command': cmd
                })
    else:
        total_reward = result.total_reward
        steps = result.steps
        step_records = result.step_records

    print(f'\n=== FINAL RESULT ===')
    print(f'Total Reward: {total_reward:.1f}')
    print(f'Steps: {steps}')
    print(f'Commands Collected: {len(step_records)}')

    # Count commands per agent
    agent_cmds = Counter()
    cmd_counts = Counter()
    for step in step_records:
        if isinstance(step, dict):
            agent = step.get('agent', step.get('agent_name', 'Unknown'))
            cmd = step.get('command', '').split()[0] if step.get('command') else 'None'
        else:
            agent = step.agent_name
            cmd = step.command.split()[0] if step.command else 'None'
        agent_cmds[agent] += 1
        cmd_counts[f'{agent}:{cmd}'] += 1

    print(f'\nAgent Commands:')
    for agent, count in sorted(agent_cmds.items()):
        print(f'  {agent}: {count} commands')

    print(f'\nCommand Distribution (top 15):')
    for key, count in cmd_counts.most_common(15):
        print(f'  {key}: {count}')

    # Check for ssh-audit usage by non-ShadowAgent
    print(f'\nSSH-audit usage check:')
    found = False
    for step in step_records:
        if isinstance(step, dict):
            command = step.get('command', '')
            agent = step.get('agent', step.get('agent_name', 'Unknown'))
        else:
            command = step.command
            agent = step.agent_name
        if 'ssh' in command.lower() and 'audit' in command.lower():
            print(f'  {agent}: {command[:50]}')
            found = True
    if not found:
        print('  No ssh-audit commands (good!)')

if __name__ == '__main__':
    main()
