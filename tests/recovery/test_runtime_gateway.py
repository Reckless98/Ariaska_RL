"""Recovery regressions: production code, in-memory providers/environment only."""
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import pytest
from core.gpt_manager import GPTManager
from core.llm.llm_provider import LLMResponse
from core.llm.gateway_chat_client import GatewayChatClient


@pytest.fixture
def gateway(monkeypatch,tmp_path):
    class Provider:
        default_timeout=2
        def __init__(self): self.calls=[]; self.replies=[]
        def get_provider_name(self): return 'fixture'
        def get_model_name(self): return 'fixture-model'
        def is_available(self): return True
        def chat_completion(self,**kw):
            self.calls.append(kw)
            if self.replies: return self.replies.pop(0)
            return LLMResponse(text='{}',model='fixture-served',provider='fixture',total_tokens=7,input_tokens=5,output_tokens=2)
    p=Provider()
    monkeypatch.setenv('ARIASKA_MENTOR_BACKEND','cloud')
    monkeypatch.setattr(GPTManager,'_detect_cloud_provider',lambda gm:gm._install_cloud_provider(p))
    monkeypatch.setattr(GPTManager,'_load_cache',lambda gm:None)
    gm=GPTManager(enable_llm=True,offline=False); gm.min_request_interval=0
    gm.cache_file=tmp_path/'cache.json'
    return gm,p


def test_real_mentor_prompts_gateway_parse_and_retry(gateway,tmp_path):
    from core.llm.smart_mentor import SmartMentor,AttackContext
    from core.commands.learned_commands import LearnedCommandStore
    from core.commands.command_registry import COMMAND_REGISTRY
    gm,p=gateway
    reply=json.dumps(dict(selected_command='nmap_top_ports',parameters={'target':'192.0.2.10','num_ports':'100'},reasoning='fixture accepted',confidence=.82))
    p.replies=[LLMResponse(text=t,model='fixture',provider='fixture',total_tokens=9) for t in ['invalid JSON',reply]]
    mentor=SmartMentor(GatewayChatClient(gm),learned_store=LearnedCommandStore(str(tmp_path/'learned.json')))
    result=mentor.get_command(AttackContext(target='192.0.2.10'),[COMMAND_REGISTRY['nmap_top_ports']])
    assert result.reasoning=='fixture accepted'
    assert result.template_name=='nmap_top_ports'
    assert result.tokens_used==18
    assert len(p.calls)==2
    assert p.calls[0]['response_format']=={'type':'json_object'}
    assert 'EXPERT DECISION RULES' in p.calls[0]['messages'][0]['content']
    assert not gm.cache


def test_cache_options_and_error_accounting(gateway):
    gm,p=gateway
    for sys,fmt in [('A',None),('B',None),('B','json_object')]:
        gm.gpt_request('same',system_prompt=sys,response_format=fmt)
    assert len(p.calls)==3
    p.replies=[LLMResponse(text='',model='fixture',provider='fixture',total_tokens=120,cost_usd=.03,error='fixture refusal')]
    assert gm.gpt_request('refusal',task_type='analysis')==''
    assert gm.get_request_record()['tokens']==120
    assert gm.get_request_record()['error']=='fixture refusal'
    assert gm.tokens_used==141
    assert gm.get_cost_summary()['cumulative_usd']==pytest.approx(.03)


def test_concurrent_request_records(gateway):
    gm,p=gateway
    def reply(**kw):
        i=int(kw["messages"][-1]["content"])
        return LLMResponse(text="{}",model=f"fixture-{i}",provider="fixture",total_tokens=i+1)
    p.chat_completion=reply
    def call(i):
        client=GatewayChatClient(gm,agent_id=str(i))
        r=client.create_sync(messages=[{'role':'user','content':str(i)}],response_format={'type':'json_object'})
        return r.usage.total_tokens,r.model
    with ThreadPoolExecutor(max_workers=4) as pool:
        result=list(pool.map(call,range(8)))
    assert result==[(i+1,f'fixture-{i}') for i in range(8)]
    assert gm.tokens_used==36


def test_discovery_episode_completes(monkeypatch,tmp_path):
    from core.orchestration.smart_orchestrator import SmartOrchestrator,SmartOrchestratorConfig
    from core.testing.fake_gpt_manager import FakeGPTManager
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ARIASKA_DATA_DIR',str(tmp_path/'data'))
    class Env:
        live_mode=False; mode='sim'
        def reset(self): return {'phase':'recon','target_ip':'192.0.2.10','state_flags':{}}
        def get_global_state(self): return self.reset()
        def step(self,a): return self.reset(),0,False,{}
    orch=SmartOrchestrator(Env(),FakeGPTManager(seed=2),config=SmartOrchestratorConfig(dashboard_enabled=False,dashboard_mode='off',dynamic_episode_length=False,max_steps_per_episode=2),verbosity='quiet')
    monkeypatch.setattr(orch,'_generate_simulated_output',lambda _: 'harmless synthetic observation only')
    monkeypatch.setattr(orch,'_parse_output_for_discoveries',lambda *a,**kw:{'open_port':[12345]})
    payments=[]
    for c in orch.coaches.values():
        original=c.reward_calculator.calculate_reward
        def calculate(*a,_original=original,**kw):
            value=_original(*a,**kw); payments.append(value.discovery_bonus); return value
        monkeypatch.setattr(c.reward_calculator,'calculate_reward',calculate)
    result=orch.run_episode('fixture',0,max_steps=2)
    assert result is not None
    assert orch._dyn_recent_discoveries
    assert orch.rnd_curiosity.get_stats()['rnd_updates'] >= 2
    assert orch._rnd_state_tensor is None  # predictor update consumed the observation
    assert 'open_port:12345' in orch._episode_rewarded_discoveries
    assert sum(payments)==pytest.approx(2.5)


def test_provider_timeouts_bound_threads_and_account_late_completion(gateway):
    import threading
    gm,p=gateway;release=threading.Event();calls=[]
    def blocked_provider(**kwargs):
        calls.append(1);release.wait(2)
        return LLMResponse(text='{}',model='late-fixture',provider='fixture',total_tokens=5,cost_usd=.01)
    p.chat_completion=blocked_provider
    try:
        for i in range(4):
            gm.gpt_request(f'timeout-{i}',timeout=.01,use_cache=False)
            assert gm.get_request_record()['error']=='provider_timeout'
        gm.gpt_request('over-capacity',timeout=.01,use_cache=False)
        assert gm.get_request_record()['error']=='provider_capacity_exhausted'
        assert len(calls)==4
    finally:
        release.set()
        gm._provider_executor.shutdown(wait=True)
    assert gm.get_cost_summary()['cumulative_usd']==pytest.approx(.04)
    assert gm.stats['tokens_used_total']==20
    assert all(r['completed_after_timeout'] for r in gm.get_recent_calls(4))
    gm.cleanup()


def test_cache_preserves_served_model_without_charging_again(gateway):
    gm,p=gateway;client=GatewayChatClient(gm)
    first=client.create_sync(messages=[{'role':'user','content':'cache metadata'}],response_format={'type':'json_object'})
    second=client.create_sync(messages=[{'role':'user','content':'cache metadata'}],response_format={'type':'json_object'})
    assert first.model==second.model=='fixture-served'
    assert second.usage.total_tokens==0
    assert len(p.calls)==1 and gm.tokens_used==7
