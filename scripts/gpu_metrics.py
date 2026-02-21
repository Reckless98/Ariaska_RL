"""Extract training metrics from JSONL traces on GPU. Disposable utility."""
import json, sys, glob, os

trace_dir = '/root/Ariaska_RL/traces/h200_distill/'
traces_files = sorted(glob.glob(os.path.join(trace_dir, '*.jsonl')))
if not traces_files:
    print("NO TRACES FOUND"); sys.exit(1)

latest = traces_files[-1]
traces = []
with open(latest) as f:
    for line in f:
        if line.strip():
            traces.append(json.loads(line.strip()))

steps = [t for t in traces if t.get('kind')=='step']
ep_starts = [t for t in traces if t.get('kind')=='episode_start']
ep_ends = [t for t in traces if t.get('kind')=='episode_end']

print(f"TRACE_FILE={os.path.basename(latest)}")
print(f"TOTAL_RECORDS={len(traces)}")
print(f"TOTAL_STEPS={len(steps)}")
print(f"EPISODES_STARTED={len(ep_starts)}")
print(f"EPISODES_ENDED={len(ep_ends)}")

if not steps:
    sys.exit(0)

last = steps[-1]
print(f"LATEST_EP={last.get('episode',0)}")
print(f"LATEST_STEP={last.get('step',0)}")
print(f"LATEST_PHASE={last.get('phase','?')}")
print(f"LATEST_REWARD={last.get('reward',0)}")
print(f"LATEST_SOURCE={last.get('mentor_source','none')}")
print(f"LATEST_OVERRIDE={last.get('teacher_overrode',False)}")
print(f"ECHO_BANS={max(s.get('echo_banned',0) for s in steps)}")
print(f"LATEST_FAMILY={last.get('cmd_family','?')}")
print(f"ANNEAL_STAGE={last.get('anneal_stage','?')}")
print(f"CODEX_BUDGET={last.get('codex_budget_remaining',0):.2f}")

rewards = [s['reward'] for s in steps]
print(f"REWARD_SUM={sum(rewards):.1f}")
print(f"REWARD_AVG={sum(rewards)/len(rewards):.2f}")
print(f"REWARD_MIN={min(rewards):.1f}")
print(f"REWARD_MAX={max(rewards):.1f}")

overrides = sum(1 for s in steps if s.get('teacher_overrode'))
print(f"OVERRIDES={overrides}")
print(f"OVERRIDE_PCT={100*overrides/len(steps):.1f}")

families = {}; phases = {}; sources = {}
for s in steps:
    f = s.get('cmd_family','?'); families[f] = families.get(f,0)+1
    p = s.get('phase','?'); phases[p] = phases.get(p,0)+1
    src = s.get('mentor_source','none'); sources[src] = sources.get(src,0)+1

unique_cmds = len(set(s.get('command','') for s in steps))
print(f"UNIQUE_CMDS={unique_cmds}")
print(f"DIVERSITY={unique_cmds/len(steps)*100:.1f}")

ep_rewards = {}
for s in steps:
    ep = s.get('episode',0)
    if ep not in ep_rewards: ep_rewards[ep] = []
    ep_rewards[ep].append(s['reward'])

print("EP_REWARDS=" + json.dumps({str(k): round(sum(v),1) for k,v in sorted(ep_rewards.items())}))
print("EP_STEPS=" + json.dumps({str(k): len(v) for k,v in sorted(ep_rewards.items())}))
print("SOURCES=" + json.dumps(dict(sorted(sources.items(), key=lambda x:-x[1]))))
print("PHASES=" + json.dumps(dict(sorted(phases.items(), key=lambda x:-x[1]))))
top_fams = dict(sorted(families.items(), key=lambda x:-x[1])[:15])
print("FAMILIES=" + json.dumps(top_fams))

for ep_end in ep_ends:
    um = ep_end.get('update_metrics', {})
    if um:
        ep_id = ep_end.get('episode',0)
        print(f"PPO_EP{ep_id}_PLOSS={um.get('policy_loss',0):.4f}")
        print(f"PPO_EP{ep_id}_VLOSS={um.get('value_loss',0):.4f}")
        print(f"PPO_EP{ep_id}_ENTROPY={um.get('entropy',0):.4f}")
