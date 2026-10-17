#!/usr/bin/env python3
"""Fixed-budget, offline recovery experiment; no cybersecurity commands execute."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import core.testing.offline_guard as guard  # Install before application imports.
import argparse, contextlib, csv, hashlib, json, logging, os, random, tempfile
import numpy as np
import torch
from core.testing.synthetic_role_env import SyntheticRoleEnvironment
from core.orchestration.smart_orchestrator import SmartOrchestrator, SmartOrchestratorConfig
from core.gpt_manager import GPTManager
from core.decision.decision_core import DecisionCore, DEFAULT_SOURCE_WEIGHTS
from core.checkpoints.unified_checkpoint import UnifiedCheckpoint


def evaluate(o, seed, episodes, horizon, kind):
    env = SyntheticRoleEnvironment(seed + 10000, horizon)
    rng = np.random.default_rng(seed + 20000)
    rows = []
    for episode in range(episodes):
        state = env.reset(); returns = {r: 0 for r in env.roles}; correct = dict(returns)
        for step in range(horizon):
            actions = {}
            for role, c in o.coaches.items():
                if kind == 'random': actions[role] = int(rng.integers(2))
                elif kind == 'registry': actions[role] = 0
                else:
                    s = c._encode_execution_state(state['agents'][role], step)
                    mask = torch.arange(c.action_mapper.action_dim) < 2
                    actions[role] = c.ppo_agent.select_action(s, training=False,
                        action_mask=mask, phase_group=0)[0]
            state, outcomes = env.step_agents(actions)
            for role, result in outcomes.items():
                returns[role] += result['reward']; correct[role] += int(result['success'])
        for role in env.roles:
            rows.append(dict(seed=seed, baseline=kind, episode=episode, role=role,
                episode_return=returns[role], accuracy=correct[role]/horizon,
                completed=int(correct[role] == horizon)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=40)
    parser.add_argument('--evaluation-episodes', type=int, default=128)
    parser.add_argument('--horizon', type=int, default=8)
    parser.add_argument('--seeds', type=int, nargs='+', default=[11,22,33])
    args = parser.parse_args(); output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if any(output.glob('seed*.pt')) or (output/'metrics.json').exists():
        raise FileExistsError('Use a new output directory; existing experiments are preserved')
    repo = Path(__file__).resolve().parents[1]
    source_hashes = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [repo/'core/training/smart_coach.py', repo/'core/algorithms/ppo_agent.py',
                  repo/'core/orchestration/smart_orchestrator.py', repo/'core/checkpoints/unified_checkpoint.py']}
    curves=[]; evaluations=[]; diagnostics=[]
    for seed in args.seeds:
        print(f'seed {seed}: fixed-budget training {args.episodes} episodes',flush=True)
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        with tempfile.TemporaryDirectory(prefix='ariaska-offline-') as root, open(os.devnull,'w') as quiet:
            original=Path.cwd();os.chdir(root);os.environ['ARIASKA_DATA_DIR']=str(Path(root)/'data')
            try:
                with contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet):
                    env=SyntheticRoleEnvironment(seed,args.horizon)
                    gm=GPTManager(enable_llm=False,offline=True)
                    o=SmartOrchestrator(env,gm,config=SmartOrchestratorConfig(
                        dashboard_enabled=False,dashboard_mode='off',dynamic_episode_length=False,
                        max_steps_per_episode=args.horizon,checkpoint_dir=str(Path(root)/'checkpoints')),verbosity='quiet')
                    logging.disable(logging.CRITICAL)
                    for c in o.coaches.values():
                        weights={s:.01 for s in DEFAULT_SOURCE_WEIGHTS};weights['ppo']=2
                        c._decision_core=DecisionCore(source_weights=weights)
                    for baseline in ['random','registry','frozen']:
                        evaluations.extend(evaluate(o,seed,args.evaluation_episodes,args.horizon,baseline))
                    for episode in range(args.episodes):
                        start=len(env.executions)
                        o.run_episode(f'seed{seed}_ep{episode}',episode,max_steps=args.horizon,target='192.0.2.10')
                        for role,c in o.coaches.items():
                            rows=[r for r in env.executions[start:] if r['role']==role]
                            curves.append(dict(seed=seed,episode=episode,role=role,
                                episode_return=sum(r['reward'] for r in rows),
                                accuracy=sum(r['success'] for r in rows)/len(rows),
                                policy_updates=c.ppo_agent.updates_done))
                    evaluations.extend(evaluate(o,seed,args.evaluation_episodes,args.horizon,'trained'))
                    path=UnifiedCheckpoint.from_coaches(o.coaches,run_id=f'seed{seed}',
                        episode=args.episodes-1).save(output/f'seed{seed}.pt')
                    diagnostics.append(dict(seed=seed,provider_requests=gm.stats['total_requests'],
                        updates={r:c.ppo_agent.updates_done for r,c in o.coaches.items()},checkpoint=Path(path).name))
            finally:os.chdir(original)
        print(f'seed {seed}: complete',flush=True)
    for name,rows in [('training_curve.csv',curves),('heldout_results.csv',evaluations)]:
        with (output/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    summary=[]
    for baseline in ['random','registry','frozen','trained']:
        means=[]
        for seed in args.seeds:
            rows=[r for r in evaluations if r['baseline']==baseline and r['seed']==seed]
            means.append(dict(seed=seed,episode_return=float(np.mean([r['episode_return'] for r in rows])),
                accuracy=float(np.mean([r['accuracy'] for r in rows])),completion=float(np.mean([r['completed'] for r in rows]))))
        summary.append(dict(baseline=baseline,seeds=means,
            mean_accuracy=float(np.mean([m['accuracy'] for m in means])),
            sd_accuracy=float(np.std([m['accuracy'] for m in means],ddof=1)) if len(means)>1 else 0,
            mean_return=float(np.mean([m['episode_return'] for m in means]))))
    result=dict(config=dict(seeds=args.seeds,episodes=args.episodes,evaluation_episodes=args.evaluation_episodes,
        horizon=args.horizon,python=sys.version,torch=torch.__version__,numpy=np.__version__,
        arbitration='PPO weight 2; every other source .01',
        evaluation='Greedy base PPO with legal-action mask and phase head 0; no mentor prior or cyber heuristic bias',
        task='Cue is port21/22; correct opaque role action is 0/1; reward +1/-1; identical held-out cues for all baselines'),
        source_hashes=source_hashes,diagnostics=diagnostics,guard_blocked_attempts=len(guard.blocked),summary=summary)
    (output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
