"""Final-stack regressions: no subprocesses, providers, tools or targets."""
import copy
import pickle
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from core.algorithms.ppo_agent import PPOAgent, PPOConfig, RolloutBuffer
from core.algorithms.sac_agent import SACAgent, SACConfig
from core.orchestration.smart_orchestrator import SmartOrchestrator, SmartOrchestratorConfig
from core.training.smart_coach import SmartDecisionResult
from core.gpt_manager import GPTManager
from test_executed_transitions import coach, sample_result


def test_reptile_preserves_whole_main_rollout_and_inner_likelihood(monkeypatch):
    p = PPOAgent(PPOConfig(action_dim=12, hidden_dims=[16,16,8], minibatch_size=2,
                          epochs_per_update=1), device='cpu')
    s = torch.zeros(512)
    mask = torch.arange(12) < 2
    a, lp, v = p.select_action(s, action_mask=mask, phase_group=1)
    p.buffer.add(s,a,lp,1,v,False,behavior_context=p.last_selection_context,
                 next_value=7,truncated=True,episode_id='main',step_id=3)
    main = p.buffer
    saved = copy.deepcopy(vars(main))
    selections = []
    original_update = p.update
    def update(*args, **kw):
        batch = p.buffer
        for i in range(len(batch)):
            _, new, _, _ = p.evaluate_actions(batch.states[i][None],
                torch.tensor([batch.actions[i]]),[batch.behavior_contexts[i]])
            selections.append(torch.exp(new-batch.log_probs[i]).item())
        return original_update(*args, **kw)
    monkeypatch.setattr(p,'update',update)
    class Meta:
        def meta_step(self, model, inner_train_fn, **kw):
            inner_train_fn(model,'web_server',2)
            inner_train_fn(model,'web_server',2)
            return {'done':True}
    env = SimpleNamespace(open_ports=[123],services=['fixture'],current_phase='recon',live_mode=False,
                          get_global_state=lambda:{'phase':'recon','state_flags':{}})
    o = SimpleNamespace(coaches={'OrionAgent':SimpleNamespace(ppo_agent=p,
        _get_action_mask=lambda _:mask)},env=env,reptile=Meta(),_maturity_controller=None,
        _reptile_global_step=0,_scenario_ports_to_services=SmartOrchestrator._scenario_ports_to_services)
    SmartOrchestrator._run_reptile_meta_step(o,0)
    assert p.buffer is main
    assert all(len(getattr(main,k))==1 for k in saved if isinstance(saved[k],list))
    assert main.next_values==[7] and main.episode_ids==['main'] and main.step_ids==[3]
    assert torch.equal(main.behavior_contexts[0]['action_mask'],saved['behavior_contexts'][0]['action_mask'])
    assert selections==pytest.approx([1]*4,abs=2e-6)
    main.add(s,a,lp,1,v,True,episode_id='main',step_id=4)
    assert p.update(finalize=True)


def test_simulated_generic_success_reaches_real_reward_recording(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ARIASKA_DATA_DIR',str(tmp_path/'data'))
    class Env:
        live_mode=False;mode='sim'
        def reset(self):return {'phase':'recon','target_ip':'192.0.2.10','state_flags':{}}
        def step(self,_):return self.reset(),0,False,{}
    o=SmartOrchestrator(Env(),GPTManager(enable_llm=False,offline=True),config=SmartOrchestratorConfig(
        dashboard_enabled=False,dashboard_mode='off',dynamic_episode_length=False),verbosity='quiet')
    o._sim_deterministic=True
    monkeypatch.setattr(o,'_check_repeat_stuck',lambda *_:(True,10))
    monkeypatch.setattr(o,'_parse_output_for_discoveries',lambda *a,**kw:{})
    observed=[]
    for c in o.coaches.values():
        monkeypatch.setattr(c,'_force_novel_action',lambda *a,**kw:SmartDecisionResult(
            command='synthetic generic operation',source='forced'))
        original=c.record_result
        def record(*a,_original=original,**kw):
            observed.append((kw['raw_output'],kw['success']))
            return _original(*a,**kw)
        monkeypatch.setattr(c,'record_result',record)
    o.run_episode('generic-sim',0,max_steps=1,target='192.0.2.10')
    assert observed and all(text.startswith('[SIM]') and ok for text,ok in observed)


def test_sac_normal_checkpoints_use_restricted_loader(tmp_path,monkeypatch):
    s=SACAgent(SACConfig(action_dim=2,hidden_dims=[8,8]))
    path=tmp_path/'sac.pt';s.save(path)
    calls=[];original=torch.load
    def load(*a,**kw):calls.append(kw);return original(*a,**kw)
    monkeypatch.setattr(torch,'load',load)
    s.load(path)
    assert calls[0]['weights_only'] is True


def test_sac_rejects_pickle_global_without_executing_it(tmp_path,monkeypatch):
    marker=tmp_path/'payload-ran'
    class Payload:
        def __reduce__(self):return (Path.write_text,(marker,'unsafe pickle executed'))
    path=tmp_path/'untrusted.pt';torch.save(Payload(),path)
    s=SACAgent(SACConfig(action_dim=2,hidden_dims=[8,8]))
    monkeypatch.setattr(s,'load_state_dict',lambda _:None)
    with pytest.raises(pickle.UnpicklingError):s.load(path)
    assert not marker.exists()


def test_discarded_ddqn_cannot_receive_deterministic_reward(coach):
    c=coach
    c.ddqn_macro=Mock()
    c._ddqn_pending={'state':torch.zeros(512),'macro':2,'prev_macro':1}
    c._ddqn_prev_macro=2;c._active_macro=2;c._active_macro_q=torch.ones(3);c._ddqn_confidence=.9
    c.discard_unexecuted_proposals()
    assert c._ddqn_pending is None and c._ddqn_prev_macro==1
    assert c._active_macro is None and c._active_macro_q is None and c._ddqn_confidence==0
    d=SmartDecisionResult(command='synthetic deterministic',source='forced')
    c.record_result(d,True,'synthetic outcome',next_state=torch.ones(512))
    c.ddqn_macro.store_transition.assert_not_called()
    c.ddqn_macro.record_macro_outcome.assert_not_called()


def test_outcome_preparation_failure_can_be_retried_once(coach,monkeypatch):
    c=coach;d=sample_result(c,0,0)
    c._current_decision_packet=None
    original=c._encode_execution_state
    monkeypatch.setattr(c,'_encode_execution_state',Mock(side_effect=RuntimeError('prepare failed')))
    with pytest.raises(RuntimeError,match='prepare failed'):
        c.record_result(d,True,'synthetic',next_state={})
    monkeypatch.setattr(c,'_encode_execution_state',original)
    c.record_result(d,True,'synthetic',next_state=torch.ones(512))
    assert c.command_repeat_count[d.template_name]==1 and len(c._ppo_trajectory)==1
    with pytest.raises(ValueError):c.record_result(d,True,'duplicate')


def test_reptile_failure_restores_buffer_env_optimizer_and_model(monkeypatch):
    p=PPOAgent(PPOConfig(action_dim=2,hidden_dims=[8,8,8],minibatch_size=2),device='cpu')
    main=p.buffer
    before=copy.deepcopy(p.network.state_dict())
    class Meta:
        def meta_step(self, model, inner_train_fn, **kw):return inner_train_fn(model,'web_server',2)
    env=SimpleNamespace(open_ports=[123],services=['fixture'],current_phase='original',live_mode=True,
                        get_global_state=lambda:{'phase':'recon'})
    o=SimpleNamespace(coaches={'RedAgent':SimpleNamespace(ppo_agent=p)},env=env,reptile=Meta(),
        _maturity_controller=None,_reptile_global_step=0,
        _scenario_ports_to_services=SmartOrchestrator._scenario_ports_to_services)
    def fail(*a,**k):
        next(p.network.parameters()).data.zero_()
        raise ArithmeticError('inner failure')
    monkeypatch.setattr(p,'update',fail)
    with pytest.raises(ArithmeticError,match='inner failure'):SmartOrchestrator._run_reptile_meta_step(o,0)
    assert p.buffer is main and not len(main) and p.updates_done==0
    assert (env.open_ports,env.services,env.current_phase,env.live_mode)==([123],['fixture'],'original',True)
    assert all(torch.equal(before[k],v) for k,v in p.network.state_dict().items())


def test_nonfinal_truncation_requires_its_own_bootstrap():
    b=RolloutBuffer()
    b.add(torch.zeros(512),0,0,1,0,False,truncated=True)
    b.add(torch.zeros(512),0,0,100,0,True)
    with pytest.raises(ValueError,match='requires next_value'):b.compute_returns_and_advantages(0)


def test_short_accumulation_group_has_full_gradient_and_refreshes_critic(monkeypatch):
    norms=[]
    for accumulation in (1,2):
        torch.manual_seed(19)
        p=PPOAgent(PPOConfig(action_dim=2,hidden_dims=[16,16,8],minibatch_size=8,
            epochs_per_update=1,grad_accum_steps=accumulation),device='cpu')
        s=torch.randn(512);a,lp,v=p.select_action(s)
        p.store_transition(s,a,lp,1,v,True,behavior_context=p.last_selection_context)
        old=[m.weight_u.clone() for m in p.network.critic.modules() if hasattr(m,'weight_u')]
        def step():norms.append(torch.cat([x.grad.flatten() for x in p.network.parameters() if x.grad is not None]).clone())
        monkeypatch.setattr(p.optimizer,'step',step)
        p.update(finalize=True)
        new=[m.weight_u for m in p.network.critic.modules() if hasattr(m,'weight_u')]
        assert any(not torch.equal(a,b) for a,b in zip(old,new))
    assert torch.allclose(norms[0],norms[1],rtol=1e-5,atol=1e-7)


def test_v2_autoload_restores_each_ddqn_without_broadcasting_ppo(tmp_path,monkeypatch):
    from core.checkpoints.unified_checkpoint import UnifiedCheckpoint
    monkeypatch.chdir(tmp_path)
    ddqn=Mock()
    cp=UnifiedCheckpoint(format_version=2,ppo_state={'legacy':'ambiguous'},ddqn_states={'RedAgent':{'x':1}})
    monkeypatch.setattr(UnifiedCheckpoint,'find_best',lambda:tmp_path/'legacy.pt')
    monkeypatch.setattr(UnifiedCheckpoint,'load',lambda _:cp)
    o=SimpleNamespace(coaches={'RedAgent':SimpleNamespace(ddqn_macro=ddqn)})
    assert SmartOrchestrator._auto_load_all_checkpoints(o)==1
    ddqn.load_state_dict.assert_called_once_with({'x':1})


def test_ddqn_rollback_snapshot_is_detached(monkeypatch):
    from test_role_checkpoints import coaches
    from core.algorithms.ddqn_macro import DDQNMacro,DDQNConfig
    from core.checkpoints.unified_checkpoint import UnifiedCheckpoint
    cs=coaches()
    d=DDQNMacro(DDQNConfig(hidden_dims=[16,16]),device='cpu')
    cs['BlueAgent'].ddqn_macro=d
    cp=UnifiedCheckpoint.from_coaches(cs)
    for parameter in d.online_net.parameters():parameter.data.add_(.5)
    before=copy.deepcopy(d.state_dict())
    p=cs['OrionAgent'].ppo_agent;original=p.load_from_state_dict;calls=[]
    def fail_once(state):
        if not calls:calls.append(1);raise RuntimeError('late failure')
        original(state)
    monkeypatch.setattr(p,'load_from_state_dict',fail_once)
    with pytest.raises(RuntimeError,match='late failure'):cp.apply_coaches(cs)
    assert all(torch.equal(v,d.online_net.state_dict()[k]) for k,v in before['online_net'].items())


def test_cuda_rng_contract_with_mocked_generators(monkeypatch):
    from core.checkpoints.unified_checkpoint import _capture_rng,_restore_rng
    states=[torch.tensor([1,2,3],dtype=torch.uint8),torch.tensor([4,5,6],dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:states)
    restore=Mock();monkeypatch.setattr(torch.cuda,'set_rng_state_all',restore)
    state=_capture_rng();states[0][0]=9
    assert state['cuda'][0][0]==1
    _restore_rng(state)
    assert all(torch.equal(a,b) for a,b in zip(restore.call_args.args[0],state['cuda']))
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    with pytest.raises(ValueError,match='device count'):_restore_rng(state)


def test_unified_loader_rejects_pickle_globals(tmp_path):
    from core.checkpoints.unified_checkpoint import UnifiedCheckpoint
    marker=tmp_path/'unsafe'
    class Payload:
        def __reduce__(self):return (Path.write_text,(marker,'executed'))
    path=tmp_path/'bad.pt';torch.save(Payload(),path)
    with pytest.raises(pickle.UnpicklingError):UnifiedCheckpoint.load(path)
    assert not marker.exists()


def test_v3_disk_payload_does_not_expose_broadcast_alias(tmp_path):
    from test_role_checkpoints import coaches
    from core.checkpoints.unified_checkpoint import UnifiedCheckpoint
    cp=UnifiedCheckpoint.from_coaches(coaches());path=tmp_path/'v3.pt';cp.save(path)
    restored=UnifiedCheckpoint.load(path)
    assert restored.ppo_state is None and restored.sac_state is None
    assert len(restored.ppo_states)==5 and len(restored.sac_states)==5


def test_sac_configuration_and_fixed_temperature_resume(tmp_path):
    config=SACConfig(action_dim=2,hidden_dims=[8,8],auto_alpha=False,alpha=.7,gamma=.8,batch_size=3,buffer_size=99)
    a=SACAgent(config);path=tmp_path/'sac.pt';a.save(path)
    b=SACAgent(SACConfig(action_dim=2,hidden_dims=[8,8],auto_alpha=False,alpha=.1,gamma=.99))
    b.load(path)
    assert b.alpha==.7 and b.config.gamma==.8 and b.config.batch_size==3
    assert b.replay_buffer.buffer.maxlen==99


def test_reward_preparation_rolls_back_discovery_payment_on_retry(coach,monkeypatch):
    c=coach;d=sample_result(c,0,0);c._current_decision_packet=None
    original=c.reward_calculator.calculate_reward
    calls=[];shared=set()
    def fail_once(*a,**kw):
        result=original(*a,**kw)
        if not calls:calls.append(1);raise RuntimeError('reward preparation failure')
        return result
    monkeypatch.setattr(c.reward_calculator,'calculate_reward',fail_once)
    kwargs=dict(new_discoveries={'open_port':[12345]},shared_discoveries=shared,next_state=torch.ones(512))
    with pytest.raises(RuntimeError,match='reward preparation'):c.record_result(d,True,'synthetic',**kwargs)
    assert shared==set() and c.attack_context.command_history==[]
    result=c.record_result(d,True,'synthetic',**kwargs)
    assert result.discovery_bonus>0 and c.command_repeat_count[d.template_name]==1


def test_partial_commit_fails_closed_instead_of_duplicating_experience(coach,monkeypatch):
    c=coach;d=sample_result(c,0,0);c._current_decision_packet=None
    hm=Mock();hm.store_transition.side_effect=RuntimeError('memory commit failure')
    monkeypatch.setattr(c,'_get_hybrid_memory',lambda:hm)
    with pytest.raises(RuntimeError,match='memory commit'):c.record_result(d,True,'synthetic',next_state=torch.ones(512))
    assert not getattr(d,'_outcome_recorded',False)
    assert len(c._ppo_trajectory)==1
    with pytest.raises(RuntimeError,match='restore a checkpoint'):c.record_result(d,True,'synthetic',next_state=torch.ones(512))
    assert len(c._ppo_trajectory)==1


def test_episode_summary_reads_flushed_history(coach,monkeypatch):
    c=coach;d=sample_result(c,0,0);c._current_decision_packet=None
    hm=Mock();monkeypatch.setattr(c,'_get_hybrid_memory',lambda:hm)
    c.record_result(d,True,'synthetic',done=True,next_state=torch.ones(512))
    c.end_episode_ppo(done=True)
    summary=hm.end_episode.call_args.kwargs['summary']
    assert summary['steps']==1 and summary['unique_actions']==1


def test_multi_episode_flags_and_phase_are_aggregated(tmp_path,monkeypatch):
    from test_offline_integration import make_orchestrator
    o=make_orchestrator(tmp_path,monkeypatch,horizon=1)
    results=iter([
        dict(total_steps=1,total_reward=2,highest_phase='EXPLOITATION',phase_progression=['RECON','EXPLOITATION'],
             user_flag_captured=True,user_flag_value='synthetic-user'),
        dict(total_steps=2,total_reward=3,highest_phase='CLOSEOUT',phase_progression=['RECON','CLOSEOUT'],
             root_flag_captured=True,root_flag_value='synthetic-root'),
        dict(total_steps=3,total_reward=4,highest_phase='RECON',phase_progression=['RECON'])])
    monkeypatch.setattr(o,'run_episode',lambda *a,**kw:next(results))
    result=o.run_training(episodes=3,target_ip='192.0.2.10')
    assert result['final_metrics']['highest_phase']=='CLOSEOUT'
    assert result['user_flag_captured'] and result['root_flag_captured']
    assert result['user_flag_value']=='synthetic-user' and result['root_flag_value']=='synthetic-root'
    assert result['final_metrics']['total_steps']==6 and result['final_score']==9


def test_evaluation_refuses_every_nonempty_directory_and_tracks_all_sources(tmp_path):
    from scripts.evaluate_recovery import prepare_output,experiment_provenance
    root=tmp_path/'experiment';root.mkdir();(root/'training_curve.csv').write_text('preserve me')
    with pytest.raises(FileExistsError):prepare_output(root)
    assert (root/'training_curve.csv').read_text()=='preserve me'
    empty=prepare_output(tmp_path/'new')
    with pytest.raises(FileExistsError):prepare_output(empty)
    repo=Path(__file__).resolve().parents[2]
    provenance=experiment_provenance(repo)
    for name in ['scripts/evaluate_recovery.py','core/testing/synthetic_role_env.py',
                 'core/decision/decision_core.py','core/models/state_encoder.py',
                 'core/algorithms/command_action_mapper.py','requirements-recovery.lock']:
        assert name in provenance['source_hashes']
    source=tmp_path/'source';(source/'core').mkdir(parents=True)
    (source/'core'/'task.py').write_text('first')
    first=experiment_provenance(source)['source_sha256']
    (source/'core'/'task.py').write_text('second')
    assert first!=experiment_provenance(source)['source_sha256']


def test_standalone_episode_runs_runtime_integrity_check(tmp_path,monkeypatch):
    from test_offline_integration import make_orchestrator
    from core.decision.integrity_check import IntegrityCheck
    o=make_orchestrator(tmp_path,monkeypatch,horizon=1)
    called=[];original=IntegrityCheck.check_runtime
    def check(self,*a,**kw):called.append(1);return original(self,*a,**kw)
    monkeypatch.setattr(IntegrityCheck,'check_runtime',check)
    o.run_episode('standalone',0,max_steps=1)
    assert called==[1]


def test_cognition_demo_enters_sil_loss_buffer_only():
    from core.algorithms.cognition_node import CognitionNode,CognitionConfig
    p=PPOAgent(PPOConfig(action_dim=2,hidden_dims=[8,8,8]),device='cpu')
    node=CognitionNode(config=CognitionConfig(),ppo=p)
    node._sil_buffer=[dict(state=torch.zeros(512),action=0,reward=10,log_prob=-99)]
    node._step_count=5
    assert node._check_sil(torch.zeros(512),torch.ones(2,dtype=torch.bool),'RECON')
    assert len(p.sil_buffer)==1 and len(p.buffer)==0
    node.external_transition_owner=True
    assert node._check_sil(torch.zeros(512),torch.ones(2,dtype=torch.bool),'RECON') is None


def test_restricted_loader_preserves_pre_hardening_bc_and_ledger_pickles(tmp_path,monkeypatch):
    from collections import deque
    from core.checkpoints.unified_checkpoint import UnifiedCheckpoint
    from core.reasoning.teacher_trace import BCBuffer,TeacherTrace
    from core.training.mentor_ledger import MentorLedger
    bc=BCBuffer(capacity=7)
    bc.store(TeacherTrace(state_id='legacy',state_vector=[0.]*512,teacher_action_idx=1))
    ledger=MentorLedger();ledger.record_outcome('ppo',2,False,actual_success=True)
    # Emulate the original #13 writer: pickled deque/defaultdict rather than lists/dicts.
    def old_bc_state(self):
        result=self.__dict__.copy();result.pop('_lock',None);return result
    monkeypatch.setattr(BCBuffer,'__getstate__',old_bc_state)
    monkeypatch.setattr(MentorLedger,'__getstate__',lambda self:self.__dict__.copy())
    path=tmp_path/'legacy-v3.pt'
    cp=UnifiedCheckpoint(resume_states={'RedAgent':{'_p14_bc_buffer':bc,'_mentor_ledger':ledger}})
    cp.save(path)
    loaded=UnifiedCheckpoint.load(path).resume_states['RedAgent']
    assert len(loaded['_p14_bc_buffer'])==1
    assert isinstance(loaded['_p14_bc_buffer']._samples,deque)
    assert loaded['_p14_bc_buffer']._samples.maxlen==7
    restored=loaded['_mentor_ledger']
    assert restored.policy_steps['RECON'].count==0  # known factory reconstructed
    restored.record_outcome('ppo',3,False,actual_success=True)
    assert restored.policy_steps['ALL'].count==2


@pytest.mark.parametrize('success',[False,True])
def test_simulator_display_value_remains_deepcopy_compatible(success):
    from core.orchestration.simulated_output import SimulatedOutput
    value=SimulatedOutput('[SIM] synthetic display',success)
    cloned=copy.deepcopy(value)
    assert cloned==value and cloned.success is success


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Physical CUDA device unavailable; CPU is the supported CI target')
def test_physical_cuda_rng_replays_each_visible_device():
    from core.checkpoints.unified_checkpoint import _capture_rng, _restore_rng
    original = _capture_rng()
    try:
        expected = [torch.rand(16, device=f'cuda:{index}').cpu()
                    for index in range(torch.cuda.device_count())]
        _restore_rng(original)
        actual = [torch.rand(16, device=f'cuda:{index}').cpu()
                  for index in range(torch.cuda.device_count())]
        assert all(torch.equal(left, right) for left, right in zip(expected, actual))
    finally:
        _restore_rng(original)
