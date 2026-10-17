"""Harmless contextual two-choice MDP for offline learning-contract validation.

Registry templates serve only as opaque action identities. This environment
never executes commands, opens sockets, invokes processes or calls models.
"""
import numpy as np


class SyntheticRoleEnvironment:
    live_mode = False
    mode = 'sim'

    def __init__(self, seed=0, horizon=8, truncate=False):
        from core.algorithms.command_action_mapper import CommandActionMapper
        self.roles = ['RedAgent', 'BlueAgent', 'ScoutAgent', 'ShadowAgent', 'OrionAgent']
        self.action_dims = {r: CommandActionMapper(r).action_dim for r in self.roles}
        self.rng = np.random.default_rng(seed)
        self.horizon = horizon
        self.truncate = truncate
        self.executions = []
        self.episode = -1

    def reset(self):
        self.episode += 1
        self.step = 0
        self.cue = int(self.rng.integers(2))
        return self.get_global_state()

    def get_global_state(self):
        flags = {'ports_discovered': True, 'services_enumerated': True,
                 'credentials_known': True, 'shell_obtained': True}
        observation = {'phase': 'recon', 'target_ip': '192.0.2.10',
                       'state_flags': flags, 'open_ports': [21 if self.cue == 0 else 22],
                       'available_action_indices': [0, 1]}
        return dict(observation, agents={r: dict(observation) for r in self.roles})

    def step_agents(self, actions):
        if set(actions) != set(self.roles):
            raise ValueError('Every role needs exactly one action')
        cue = self.cue
        self.step += 1
        boundary = self.step >= self.horizon
        outcomes = {}
        for role, action in actions.items():
            success = action == cue
            outcomes[role] = dict(reward=1.0 if success else -1.0,
                success=success, terminated=boundary and not self.truncate,
                truncated=boundary and self.truncate, output='synthetic observation')
            self.executions.append(dict(role=role, episode=self.episode,
                step=self.step-1, action=int(action), cue=cue, **outcomes[role]))
        self.cue = int(self.rng.integers(2))
        return self.get_global_state(), outcomes
