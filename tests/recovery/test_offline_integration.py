"""Real SmartOrchestrator → SmartCoach/DecisionCore → PPO → persistence."""
import random
from pathlib import Path
import numpy as np
import torch
import pytest
from core.testing.synthetic_role_env import SyntheticRoleEnvironment
from core.orchestration.smart_orchestrator import SmartOrchestrator,SmartOrchestratorConfig
from core.gpt_manager import GPTManager
from core.decision.decision_core import DecisionCore,DEFAULT_SOURCE_WEIGHTS
from core.checkpoints.unified_checkpoint import UnifiedCheckpoint


def make_orchestrator(tmp_path,monkeypatch,seed=17,horizon=4,truncate=False):
    monkeypatch.chdir(tmp_path);monkeypatch.setenv('ARIASKA_DATA_DIR',str(tmp_path/'data'))
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    env=SyntheticRoleEnvironment(seed,horizon,truncate)
    config=SmartOrchestratorConfig(dashboard_enabled=False,dashboard_mode='off',
        max_steps_per_episode=horizon,checkpoint_dir=str(tmp_path/'checkpoints'))
    o=SmartOrchestrator(env,GPTManager(enable_llm=False,offline=True),config=config,verbosity='quiet')
    # Predeclared PPO research ablation; arbitration remains the real scorer.
    for c in o.coaches.values():
        weights={source:.01 for source in DEFAULT_SOURCE_WEIGHTS};weights['ppo']=2
        c._decision_core=DecisionCore(source_weights=weights)
    return o


@pytest.mark.parametrize('truncate',[False,True])
def test_all_five_actual_pipeline_train_checkpoint_restore_continue(tmp_path,monkeypatch,truncate):
    o=make_orchestrator(tmp_path,monkeypatch,truncate=truncate)
    observed=[]
    for c in o.coaches.values():
        original=c.record_result
        def record(*a,_c=c,_original=original,**kw):
            result=_original(*a,**kw);t=_c.last_executed_transition
            observed.append(t)
            if t.policy_sample:
                p=t.policy_sample
                _,lp,_,_=_c.ppo_agent.evaluate_actions(p['state'].reshape(1,-1),
                    torch.tensor([p['action']]),[p['behavior_context']])
                assert torch.exp(lp-p['log_prob']).item()==pytest.approx(1,abs=2e-6)
            return result
        monkeypatch.setattr(c,'record_result',record)
    result=o.run_training(episodes=3,target_ip='192.0.2.10')
    assert result['episodes_completed']==3
    assert len(o.env.executions)==3*4*5
    assert all(m['total_steps']==4 for m in result['all_episode_metrics'])
    for row,t in zip(o.env.executions,observed):
        assert (row['role'],row['episode'],row['step'],row['action'])==(t.role,t.episode,t.step,t.action)
        assert row['reward']==t.reward
        assert row['terminated']==t.terminated and row['truncated']==t.truncated
    for c in o.coaches.values():
        assert c.ppo_agent.updates_done>=3, c.agent_name
        assert c._decision_core.decision_count>=1
        assert not c._ppo_trajectory and not len(c.ppo_agent.buffer)
    cp=UnifiedCheckpoint.load(result['checkpoint'])
    fingerprints={r:{k:v.clone() for k,v in c.ppo_agent.network.state_dict().items()} for r,c in o.coaches.items()}
    for c in o.coaches.values():
        for p in c.ppo_agent.network.parameters():p.data.zero_()
    assert cp.apply_coaches(o.coaches)==5
    for r,c in o.coaches.items():
        assert all(torch.equal(v,c.ppo_agent.network.state_dict()[k]) for k,v in fingerprints[r].items())
    o.run_episode('continued',3,max_steps=4,target='192.0.2.10')
    assert all(c.ppo_agent.updates_done>=4 for c in o.coaches.values())


def test_interruption_saves_partial_state_and_preserves_last_valid(tmp_path,monkeypatch):
    o=make_orchestrator(tmp_path,monkeypatch)
    first=o.run_training(episodes=1,target_ip='192.0.2.10');path=Path(first['checkpoint']);before=path.read_bytes()
    original=o.env.step_agents;calls=[]
    def interrupted(actions):
        if calls:raise KeyboardInterrupt()
        calls.append(1);return original(actions)
    monkeypatch.setattr(o.env,'step_agents',interrupted)
    with pytest.raises(KeyboardInterrupt):o.run_episode('interrupted',1,max_steps=4)
    assert path.read_bytes()==before
    cp=UnifiedCheckpoint.load(o._last_interruption_checkpoint)
    assert cp.metadata['interrupted'] is True and cp.apply_coaches(o.coaches)==5
    assert all(not c._ppo_trajectory and not len(c.ppo_agent.buffer) for c in o.coaches.values())


def test_seed_repeatability_and_explicit_step_cap(tmp_path,monkeypatch):
    fingerprints=[];actions=[]
    for run in range(2):
        root=tmp_path/str(run);root.mkdir()
        o=make_orchestrator(root,monkeypatch,seed=29,horizon=10)
        result=o.run_episode('bounded',0,max_steps=3,target='192.0.2.10')
        assert result['total_steps']==3 and len(o.env.executions)==15
        assert all(c.last_executed_transition.truncated for c in o.coaches.values())
        actions.append([x['action'] for x in o.env.executions])
        fingerprints.append({r:torch.cat([x.flatten() for x in c.ppo_agent.network.state_dict().values()]).clone() for r,c in o.coaches.items()})
    assert actions[0]==actions[1]
    assert all(torch.equal(fingerprints[0][r],fingerprints[1][r]) for r in fingerprints[0])
