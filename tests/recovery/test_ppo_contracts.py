import torch
import pytest
from core.algorithms.ppo_agent import PPOAgent,PPOConfig,RolloutBuffer

@pytest.mark.parametrize('dim',[79,18,37,54,12])
@pytest.mark.parametrize('phase',[0,1,2])
def test_behavior_likelihood_is_reproducible(dim,phase,monkeypatch):
    torch.manual_seed(10)
    p=PPOAgent(PPOConfig(action_dim=dim,hidden_dims=[32,32,16],dropout_rate=.3),device='cpu')
    with torch.no_grad():
        for i, gate in enumerate(p.network.phase_gates):
            gate[-1].bias.copy_(torch.linspace(-1, 1, dim) * (i + 1))
    s=torch.randn(512); mask=torch.arange(dim)%3==0
    bias=torch.linspace(-2,1,dim); prior=torch.randn(dim)
    a,lp,v=p.select_action(s,action_mask=mask,logit_bias=bias,llm_prior=prior,prior_alpha=.4,phase_group=phase)
    _,new,_,_=p.evaluate_actions(s[None],torch.tensor([a]),[p.last_selection_context])
    assert torch.exp(new-lp).item()==pytest.approx(1,abs=2e-6)
    with torch.no_grad():
        logits, _ = p.network(s[None], phase_group=torch.tensor([phase]))
        expected = torch.distributions.Categorical(logits=(logits + bias + .4*prior).masked_fill(~mask, float('-inf')))
    assert mask[a] and lp==pytest.approx(expected.log_prob(torch.tensor([a])).item(),abs=2e-6)
    calls=[]; original=p.evaluate_actions
    def evaluate(*args,**kwargs):
        calls.append(len(args[0]));return original(*args,**kwargs)
    monkeypatch.setattr(p,'evaluate_actions',evaluate)
    p.store_transition(s,a,lp,1,v,True,behavior_context=p.last_selection_context)
    result=p.update(finalize=True)
    assert result and all(torch.isfinite(torch.tensor(v)) for v in result.values())
    assert calls and set(calls)=={1}

@pytest.mark.parametrize('dim',[79,18,37,54,12])
def test_mixed_teacher_batch_matches_role(dim):
    b=RolloutBuffer()
    for td in [torch.ones(dim)/dim,None]: b.add(torch.zeros(512),0,0,1,0,False,teacher_distribution=td)
    r,a=b.compute_returns_and_advantages(0)
    batches=list(b.get_batches(r,a,2,torch.device('cpu')))
    assert batches[0]['teacher_distributions'].shape==(2,dim)
    assert batches[0]['teacher_has_mask'].sum()==1
    td=batches[0]['teacher_distributions']; present=batches[0]['teacher_has_mask'].bool()
    assert torch.equal(td[present],torch.ones(1,dim)/dim)
    assert torch.equal(td[~present],torch.zeros(1,dim))

@pytest.mark.parametrize('terminated,truncated,expected',[(True,False,1),(False,True,10),(False,False,10)])
def test_bootstrap_and_boundary(terminated,truncated,expected):
    b=RolloutBuffer();b.add(torch.zeros(512),0,0,1,2,terminated,next_value=10,truncated=truncated,episode_id='a',step_id=0)
    r,a=b.compute_returns_and_advantages(999,gamma=.9)
    assert r.item()==pytest.approx(expected)
    assert a.item()==pytest.approx(expected-2)

def test_gae_does_not_cross_episode_boundary():
    b=RolloutBuffer()
    b.add(torch.zeros(512),0,0,1,0,False,next_value=2,truncated=True,episode_id='a',step_id=0)
    b.add(torch.zeros(512),0,0,100,0,True,next_value=0,episode_id='b',step_id=0)
    r,_=b.compute_returns_and_advantages(0,gamma=.5)
    assert r.tolist()==pytest.approx([2,100])
    with pytest.raises(ValueError,match='chronological'):
        b.add(torch.zeros(512),0,0,0,0,False,episode_id='b',step_id=0)
