"""Outcome ownership tests use real policies and real SmartCoach result recording."""
import copy
import torch
import pytest
from core.training.smart_coach import SmartCoach, SmartStepContext, SmartDecisionResult
from core.llm.smart_mentor import AttackContext
from core.gpt_manager import GPTManager
from core.algorithms.ppo_agent import PPOAgent, PPOConfig

ROLES={'RedAgent':79,'BlueAgent':18,'ScoutAgent':37,'ShadowAgent':54,'OrionAgent':12}

@pytest.fixture(params=ROLES)
def coach(request,tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path); monkeypatch.setenv('ARIASKA_DATA_DIR',str(tmp_path/'data'))
    c=SmartCoach(request.param,GPTManager(enable_llm=False,offline=True))
    c.attack_context=AttackContext(target='192.0.2.10')
    c.ppo_agent=PPOAgent(PPOConfig(action_dim=ROLES[request.param],hidden_dims=[32,32,16],
        minibatch_size=8,rollout_size=16),device='cpu')
    c.cognition_node.ppo=c.ppo_agent
    return c


def sample_result(c,step,action):
    state=torch.full((512,),step/10)
    mask=torch.zeros(c.action_mapper.action_dim,dtype=torch.bool);mask[action]=True
    a,lp,v=c.ppo_agent.select_action(state,action_mask=mask,phase_group=step%3)
    name=c.action_mapper.action_to_name(a)
    r=SmartDecisionResult(command=f'synthetic-choice-{a}',template_name=name,source='ppo')
    r._ppo_sample=dict(state=state,action=a,log_prob=lp,value=v,command=r.command,
        template_name=name,behavior_context=copy.deepcopy(c.ppo_agent.last_selection_context))
    return r


def test_only_executed_sample_receives_reward(coach,monkeypatch):
    c=coach; first=sample_result(c,0,0)
    def candidates(*a,**kw):
        c._ppo_pending=sample_result(c,0,1)._ppo_sample
        c.episode_used_commands.add('discarded')
        c.attack_context.command_history.append('discarded')
        return first
    monkeypatch.setattr(c,'_decide_impl',candidates)
    decision=c.decide(SmartStepContext(0,0,c.agent_name,c.attack_context,{}))
    assert 'discarded' not in c.episode_used_commands
    assert c.attack_context.command_history==[]
    assert not len(c.ppo_agent.buffer)
    next_state=torch.ones(512)
    result=c.record_result(decision,True,'synthetic success',next_state=next_state)
    t=c.last_executed_transition
    assert t.action==0 and torch.equal(t.next_state,next_state)
    assert t.reward==pytest.approx(c._ppo_trajectory[0]['reward'])
    assert c._ppo_trajectory[0]['action']==0
    assert not len(c.ppo_agent.buffer)
    assert c.command_repeat_count[decision.template_name]==1
    with pytest.raises(ValueError,match='twice'):
        c.record_result(decision,True,'duplicate')
    metrics=c.end_episode_ppo(done=False)
    assert metrics and all(torch.isfinite(torch.tensor(x)) for x in metrics.values())


def test_overridden_command_is_not_ppo_experience(coach,monkeypatch):
    c=coach;r=sample_result(c,0,0);r.command='synthetic-override';r.source='anti_repeat'
    monkeypatch.setattr(c,'_decide_impl',lambda *a,**kw:r)
    d=c.decide(SmartStepContext(0,0,c.agent_name,c.attack_context,{}))
    c.record_result(d,False,'synthetic failure',done=True,next_state=torch.ones(512))
    assert c.last_executed_transition.action==-1
    assert not c._ppo_trajectory and not len(c.ppo_agent.buffer)
    assert c.last_executed_transition.policy_sample is None


def test_periodic_and_incomplete_final_updates(coach,monkeypatch):
    c=coach
    for step in range(18):
        r=sample_result(c,step,step%2)
        monkeypatch.setattr(c,'_decide_impl',lambda *a,_r=r,**kw:_r)
        d=c.decide(SmartStepContext(0,step,c.agent_name,c.attack_context,{}))
        c.record_result(d,True,'synthetic outcome',done=step==17,next_state=torch.ones(512))
    assert c.ppo_agent.updates_done==1
    assert len(c._ppo_trajectory)==2
    assert len(c._ppo_episode_history)==16
    c.end_episode_ppo(done=True)
    assert c.ppo_agent.updates_done==2
    assert not c._ppo_trajectory and not c._ppo_episode_history


def test_teacher_unknown_mapping_and_success_are_not_fabricated():
    from core.reasoning.teacher_trace import TeacherTrace,BCBuffer
    t=TeacherTrace(state_vector=[0.]*512,teacher_action_idx=-1)
    assert not BCBuffer().store(t)
    t.finalize_outcome(100,False,True,actual_success=False)
    assert t.teacher_was_correct is False
    t.finalize_outcome(-100,False,False,actual_success=True)
    assert t.teacher_was_correct is None


@pytest.mark.parametrize('answer',['accepted','unknown','fallback'])
def test_mock_mentor_student_snapshot_phase_and_bc_lifecycle(coach,monkeypatch,answer):
    from types import SimpleNamespace
    from core.llm.smart_mentor import MentorResponse
    from core.decision.decision_core import DecisionCore
    c=coach;ctx=c.attack_context
    original=c._decision_for_action(0,ctx,'mentor')
    response=MentorResponse(command=original.command,
        template_name=original.template_name if answer!='unknown' else 'unmapped-fixture',
        params=original.params,reasoning='in-memory teacher',confidence=.9,phase=ctx.current_phase,
        is_fallback=answer=='fallback')
    c.smart_mentor=SimpleNamespace(get_command=lambda *a,**k:response)
    monkeypatch.setattr(c.gpt_manager,'is_offline',lambda:False)
    c._maturity_controller=SimpleNamespace(should_call_mentor=lambda *_:True,state=SimpleNamespace(maturity=0))
    c._decision_core=DecisionCore(source_weights={'mentor':2,'ppo':.01,'registry':.01})
    step=SmartStepContext(0,0,c.agent_name,ctx,{'available_action_indices':[0,1]})
    before=len(c._p14_bc_buffer)
    d=c.decide(step,force_mentor=True)
    trace=c._pending_teacher_trace
    if answer=='fallback':
        assert trace is None and c._mentor_ledger.rejected_by_reason['local_fallback']==1
    else:
        assert trace is not None
        assert trace.phase==ctx.current_phase.name
        assert trace.student_action_idx>=0
        assert trace.student_template==c.action_mapper.action_to_name(trace.student_action_idx)
        assert len(c._p14_bc_buffer)==before
        if answer=='accepted':assert d.source=='mentor'
        else:assert trace.teacher_action_idx==-1
    c.record_result(d,False,'synthetic failure',environment_reward=1,next_state=torch.ones(512))
    assert len(c._p14_bc_buffer)==before+(answer=='accepted')
    if answer=='accepted':assert trace.teacher_was_correct is False
    assert c._pending_teacher_trace is None


def test_skipped_teacher_proposal_is_not_attached_to_next_action(coach,monkeypatch):
    from core.reasoning.teacher_trace import TeacherTrace
    c=coach;t=TeacherTrace(state_vector=[0.]*512,teacher_action_idx=0)
    c._pending_teacher_trace=t;c._p14_traces_this_episode=[t]
    result=sample_result(c,0,1)
    monkeypatch.setattr(c,'_decide_impl',lambda *a,**kw:result)
    before=len(c._p14_bc_buffer)
    d=c.decide(SmartStepContext(0,0,c.agent_name,c.attack_context,{}))
    c.record_result(d,True,'synthetic',next_state=torch.ones(512))
    assert len(c._p14_bc_buffer)==before and t.actual_reward is None
    assert c._discarded_teacher_traces==1
    c.reset_episode(1)
    assert not c._p14_traces_this_episode
