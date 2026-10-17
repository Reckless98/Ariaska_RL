import copy
import random
from types import SimpleNamespace
import numpy as np
import torch
import pytest
from core.algorithms.ppo_agent import PPOAgent,PPOConfig
from core.algorithms.sac_agent import SACAgent,SACConfig
from core.algorithms.command_action_mapper import CommandActionMapper
from core.checkpoints.unified_checkpoint import UnifiedCheckpoint
from core.reasoning.teacher_trace import BCBuffer,TeacherTrace

ROLES=['RedAgent','BlueAgent','ScoutAgent','ShadowAgent','OrionAgent']

def coaches():
    result={}
    for i,role in enumerate(ROLES):
        m=CommandActionMapper(role)
        p=PPOAgent(PPOConfig(action_dim=m.action_dim,hidden_dims=[16,16,8]),device='cpu')
        s=torch.randn(512);a,lp,v=p.select_action(s)
        p.store_transition(s,a,lp,i+1,v,True,behavior_context=p.last_selection_context)
        p.update(finalize=True)
        sac=SACAgent(SACConfig(action_dim=m.action_dim,hidden_dims=[16,16]))
        sac.store_transition(s,a,i+1,s+1,True)
        bc=BCBuffer();bc.store(TeacherTrace(state_id=role,state_vector=[float(i)]*512,teacher_action_idx=i))
        result[role]=SimpleNamespace(ppo_agent=p,sac_agent=sac,ddqn_macro=None,agent=None,
            action_mapper=m,_p14_bc_buffer=bc,_ppo_trajectory=[],_ppo_episode_history=[],current_episode=3)
    return result


def fingerprints(cs):
    return {r:torch.cat([v.detach().cpu().flatten() for v in c.ppo_agent.network.state_dict().values()]).clone() for r,c in cs.items()}


def test_all_five_restore_parameters_optimizer_bc_replay_and_rng(tmp_path):
    torch.manual_seed(33);random.seed(33);np.random.seed(33)
    cs=coaches();before=fingerprints(cs)
    checkpoint=UnifiedCheckpoint.from_coaches(cs,run_id='fixture',episode=3)
    path=tmp_path/'roles.pt';checkpoint.save(path)
    expected=(random.random(),np.random.random(),torch.rand(1))
    for c in cs.values():
        for p in c.ppo_agent.network.parameters():p.data.zero_()
        c._p14_bc_buffer=BCBuffer();c.sac_agent.replay_buffer.buffer.clear()
    loaded=UnifiedCheckpoint.load(path)
    assert loaded.apply_coaches(cs)==5
    actual=(random.random(),np.random.random(),torch.rand(1))
    assert actual[:2]==expected[:2] and torch.equal(actual[2],expected[2])
    for role,c in cs.items():
        assert torch.equal(fingerprints(cs)[role],before[role])
        assert len(c._p14_bc_buffer)==1 and c.ppo_agent.config.bc_buffer is c._p14_bc_buffer
        assert len(c.sac_agent.replay_buffer)==1
        assert c.ppo_agent.optimizer.state and c.ppo_agent.updates_done==1
        s=torch.randn(512);a,lp,v=c.ppo_agent.select_action(s)
        c.ppo_agent.store_transition(s,a,lp,1,v,True,behavior_context=c.ppo_agent.last_selection_context)
        assert c.ppo_agent.update(finalize=True)
        assert c.ppo_agent.updates_done==2


@pytest.mark.parametrize('corrupt',['shape','map','optimizer'])
def test_late_role_rejected_before_any_mutation(corrupt):
    cs=coaches();cp=UnifiedCheckpoint.from_coaches(cs);before=fingerprints(cs)
    if corrupt=='shape':
        key=next(iter(cp.ppo_states['OrionAgent']['network_state_dict']))
        cp.ppo_states['OrionAgent']['network_state_dict'][key]=torch.zeros(1)
    elif corrupt=='map': cp.action_maps['OrionAgent'].reverse()
    else: cp.ppo_states['OrionAgent']['optimizer_state_dict']['param_groups'][0]['params'].pop()
    with pytest.raises((ValueError,RuntimeError)): cp.apply_coaches(cs)
    assert all(torch.equal(fingerprints(cs)[r],v) for r,v in before.items())


def test_load_failure_rolls_back_every_role(monkeypatch):
    cs=coaches();cp=UnifiedCheckpoint.from_coaches(cs)
    for c in cs.values():
        for p in c.ppo_agent.network.parameters():p.data.add_(.1)
    before=fingerprints(cs)
    p=cs['OrionAgent'].ppo_agent;original=p.load_from_state_dict;calls=[]
    def fail_once(state):
        if not calls:
            calls.append(1);raise RuntimeError('fixture load failure')
        return original(state)
    monkeypatch.setattr(p,'load_from_state_dict',fail_once)
    with pytest.raises(RuntimeError,match='fixture load failure'):cp.apply_coaches(cs)
    assert all(torch.equal(fingerprints(cs)[r],v) for r,v in before.items())


def test_failed_save_preserves_last_valid_checkpoint(tmp_path,monkeypatch):
    path=tmp_path/'last.pt';cp=UnifiedCheckpoint(run_id='valid');cp.save(path);before=path.read_bytes()
    def fail(*a,**kw):raise OSError('fixture full disk')
    monkeypatch.setattr(torch,'save',fail)
    with pytest.raises(OSError):cp.save(path)
    assert path.read_bytes()==before
    assert not list(tmp_path.glob('*.tmp'))


def test_legacy_state_is_not_broadcast():
    cs=coaches();cp=UnifiedCheckpoint.from_ppo_agent(cs['RedAgent'].ppo_agent)
    with pytest.raises(ValueError,match='Legacy single-policy'):cp.apply_coaches(cs)
    before=fingerprints(cs)
    assert cp.apply_ppo(cs['BlueAgent'].ppo_agent) is False
    assert torch.equal(fingerprints(cs)['BlueAgent'],before['BlueAgent'])
    assert cp.apply_ppo(cs['RedAgent'].ppo_agent) is True
