"""Phase 41 Runtime Smoke Test — prove end-to-end wiring without NaNs/crashes.

Rewritten with introspected APIs — all constructors and methods verified.
"""
from __future__ import annotations

import os
import sys
import math
import warnings

os.environ["ARIASKA_DRY_RUN"] = "1"
os.environ["PYTHONPATH"] = os.getcwd()

# Treat warnings as errors for this test
warnings.filterwarnings("error", category=DeprecationWarning)
warnings.filterwarnings("error", category=RuntimeWarning)
# Allow PyTorch's batch_first warning
warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")

import torch


def check_no_nan(tensor_or_val, label: str) -> bool:
    if isinstance(tensor_or_val, torch.Tensor):
        if torch.isnan(tensor_or_val).any():
            print(f"  FAIL NaN detected in {label}")
            return False
        if torch.isinf(tensor_or_val).any():
            print(f"  FAIL Inf detected in {label}")
            return False
    elif isinstance(tensor_or_val, float):
        if math.isnan(tensor_or_val) or math.isinf(tensor_or_val):
            print(f"  FAIL NaN/Inf in {label}: {tensor_or_val}")
            return False
    print(f"  OK   {label}")
    return True


# ---------------------------------------------------------------------------
# B1 — PPOAgent
# ---------------------------------------------------------------------------
def test_ppo_agent():
    print("\n=== PPOAgent ===")
    from core.algorithms.ppo_agent import PPOAgent
    ppo = PPOAgent(device="cpu")  # PPOConfig defaults: state_dim=512, action_dim=5
    state = torch.randn(512)
    action, log_prob, value = ppo.select_action(state)
    ok = True
    ok &= check_no_nan(torch.tensor([float(action)]), "action")
    ok &= check_no_nan(log_prob, "log_prob")
    ok &= check_no_nan(value, "value")
    # Store 10 transitions and update
    for i in range(10):
        s = torch.randn(512)
        a, lp, v = ppo.select_action(s)
        ppo.store_transition(s, a, lp, 1.0 + i * 0.1, v, i == 9)
    stats = ppo.update(last_value=0.0)
    if stats:
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                ok &= check_no_nan(float(v), f"ppo_update.{k}")
    return ok


# ---------------------------------------------------------------------------
# B4 — EpisodicMemory
# API: store(state_tensor, action_idx, reward, phase, outcome), retrieve(query, k)
# ---------------------------------------------------------------------------
def test_episodic_memory():
    print("\n=== EpisodicMemory ===")
    from core.algorithms.episodic_memory import EpisodicMemory
    mem = EpisodicMemory()  # config=None -> defaults
    for i in range(20):
        s = torch.randn(512)
        mem.store(s, action_idx=i % 5, reward=float(i), phase="RECON", outcome="neutral")
    query = torch.randn(512)
    neighbors = mem.retrieve(query)
    ok = True
    ok &= check_no_nan(torch.tensor([float(len(neighbors))]), "neighbor_count")
    print(f"  Retrieved {len(neighbors)} neighbors")
    injection = mem.format_for_injection(neighbors)
    print(f"  format_for_injection returned: {type(injection).__name__}")
    return ok


# ---------------------------------------------------------------------------
# B6 — StateTransformerEncoder
# ---------------------------------------------------------------------------
def test_transformer_encoder():
    print("\n=== TransformerEncoder ===")
    from core.models.transformer_encoder import StateTransformerEncoder, StateWindowBuffer
    enc = StateTransformerEncoder()  # config=None -> defaults (state_dim=512, d_model=256)
    buf = StateWindowBuffer(window_size=4, state_dim=512)
    for _ in range(4):
        buf.add(torch.randn(512))
    window = buf.get_window()  # shape: [4, 512]
    out = enc(window.unsqueeze(0))  # add batch dim -> [1, 4, 512] -> [1, 512]
    ok = True
    ok &= check_no_nan(out, "transformer_output")
    print(f"  Output shape: {out.shape}")
    return ok


# ---------------------------------------------------------------------------
# B10 — ReflectiveMetaLearner
# API: reflect_on_episode(episode_data, gpt_manager), get_context_injection(last_n)
# ---------------------------------------------------------------------------
def test_reflective_meta_learner():
    print("\n=== ReflectiveMetaLearner ===")
    from core.llm.reflective_meta_learner import ReflectiveMetaLearner
    rml = ReflectiveMetaLearner()  # config=None -> defaults
    episode_data = {
        "episode": 1,
        "reward": 15.0,
        "steps": 50,
        "discoveries": ["port:22", "service:ssh"],
        "phase": "RECON",
    }
    result = rml.reflect_on_episode(episode_data, gpt_manager=None)
    ok = True
    print(f"  reflect_on_episode returned: {type(result).__name__}")
    injection = rml.get_context_injection(last_n=3)
    ok &= isinstance(injection, str)
    print(f"  get_context_injection: len={len(injection)}")
    return ok


# ---------------------------------------------------------------------------
# B5 — ContrastiveLoss (nn.Module)
# API: compute_loss(backbone_features, phase_labels, discovery_counts)
# ---------------------------------------------------------------------------
def test_contrastive_state():
    print("\n=== ContrastiveLoss ===")
    from core.algorithms.contrastive_state import ContrastiveLoss, ContrastiveConfig
    cfg = ContrastiveConfig(enabled=True, feature_dim=256)
    loss_fn = ContrastiveLoss(config=cfg)
    backbone_features = torch.randn(8, 256)  # feature_dim=256
    phase_labels = torch.randint(0, 8, (8,))
    discovery_counts = torch.randint(0, 10, (8,)).float()
    loss = loss_fn.compute_loss(backbone_features, phase_labels, discovery_counts)
    ok = True
    ok &= check_no_nan(loss, "contrastive_loss")
    print(f"  Loss value: {loss.item():.4f}")
    return ok


# ---------------------------------------------------------------------------
# B3 — NStepConfig (dataclass — config only)
# ---------------------------------------------------------------------------
def test_nstep_returns():
    print("\n=== NStepConfig ===")
    from core.algorithms.nstep_returns import NStepConfig
    cfg = NStepConfig(n=3, gamma=0.99, blend_alpha=0.3)
    ok = True
    ok &= (cfg.n == 3)
    ok &= (cfg.gamma == 0.99)
    ok &= (cfg.blend_alpha == 0.3)
    ok &= isinstance(cfg.enabled, bool)
    print(f"  NStepConfig(n={cfg.n}, gamma={cfg.gamma}, blend={cfg.blend_alpha}, enabled={cfg.enabled})")
    return ok


# ---------------------------------------------------------------------------
# B7 — HindsightReplay
# API: process_episode(transitions, target_phase, achieved_phase),
#      relabel_episode(transitions, achieved_phase, target_phase)
# ---------------------------------------------------------------------------
def test_her():
    print("\n=== HindsightReplay ===")
    from core.algorithms.hindsight_replay import HindsightReplay
    her = HindsightReplay()  # config=None -> defaults
    trajectory = []
    for i in range(5):
        trajectory.append({
            "state": torch.randn(512).tolist(),
            "action": i % 5,
            "reward": 1.0,
            "next_state": torch.randn(512).tolist(),
            "done": (i == 4),
            "phase": "EXPLOITATION",
        })
    relabeled = her.relabel_episode(
        trajectory, achieved_phase="EXPLOITATION", target_phase="PRIVILEGE_ESCALATION"
    )
    ok = True
    ok &= isinstance(relabeled, list)
    print(f"  relabel_episode returned {len(relabeled)} transitions")
    count = her.process_episode(
        trajectory, target_phase="PRIVILEGE_ESCALATION", achieved_phase="EXPLOITATION"
    )
    ok &= isinstance(count, int)
    print(f"  process_episode returned {count} synthetic transitions")
    return ok


# ---------------------------------------------------------------------------
# B2 — ProgressiveExpander
# API: should_expand(episode, explained_variance), get_target_dims(),
#      record_expansion()
# ---------------------------------------------------------------------------
def test_progressive_net():
    print("\n=== ProgressiveExpander ===")
    from core.algorithms.progressive_net import ProgressiveExpander
    exp = ProgressiveExpander()  # config=None -> defaults
    should = exp.should_expand(episode=100, explained_variance=0.8)
    ok = True
    ok &= isinstance(should, bool)
    print(f"  should_expand(ep=100, ev=0.8): {should}")
    dims = exp.get_target_dims()
    print(f"  get_target_dims: {dims}")
    return ok


# ---------------------------------------------------------------------------
# B8 — CoTCache
# API: put(phase, fp, reasoning_chain, ...), get(phase, fp),
#      compute_fingerprint(phase, state_dict), get_stats(), invalidate_phase()
# ---------------------------------------------------------------------------
def test_cot_cache():
    print("\n=== CoTCache ===")
    from core.llm.cot_cache import CoTCache
    cache = CoTCache()  # config=None -> defaults
    fp = CoTCache.compute_fingerprint("RECON", {"ports": [22, 80]})
    cache.put(
        "RECON", fp,
        reasoning_chain="nmap scan complete",
        command_suggestion="nmap -sV",
        confidence=0.9,
        model_used="local-llm",
        token_cost=50,
    )
    hit = cache.get("RECON", fp)
    ok = True
    ok &= (hit is not None)
    print(f"  Cache hit: {hit is not None}")
    miss = cache.get("RECON", "nonexistent_fp")
    ok &= (miss is None)
    print(f"  Cache miss: {miss is None}")
    stats = cache.get_stats()
    ok &= isinstance(stats, dict)
    print(f"  Stats: {stats}")
    return ok


# ---------------------------------------------------------------------------
# B9 — DAggerBuffer
# API: store(...), sample(batch_size), can_train(), get_stats(), decay_weights()
# ---------------------------------------------------------------------------
def test_dagger():
    print("\n=== DAggerBuffer ===")
    from core.training.dagger import DAggerBuffer, DAggerConfig
    cfg = DAggerConfig(min_samples_for_train=5)  # lower threshold for smoke test
    buf = DAggerBuffer(config=cfg)
    for i in range(10):
        state = [0.0] * 512
        state[i] = 1.0
        buf.store(
            state_hash=f"hash_{i}",
            state_vector=state,
            mentor_action_idx=i % 5,
            mentor_command=f"cmd_{i}",
            policy_action_idx=(i + 1) % 5,
            policy_command=f"cmd_{(i+1)%5}",
            mentor_confidence=0.9,
            phase="RECON",
            episode=1,
            step=i,
        )
    ok = True
    ok &= buf.can_train()
    print(f"  can_train: {buf.can_train()}")
    batch = buf.sample(batch_size=5)
    ok &= (len(batch) >= 1)
    print(f"  Sampled {len(batch)} DAgger transitions")
    stats = buf.get_stats()
    ok &= isinstance(stats, dict)
    print(f"  Stats: {stats}")
    return ok


# ---------------------------------------------------------------------------
# C4 — SelfPlayManager
# API: should_run_self_play(episode), compute_adversarial_rewards(...), get_stats()
# ---------------------------------------------------------------------------
def test_self_play():
    print("\n=== SelfPlayManager ===")
    from core.training.self_play import SelfPlayManager
    sp = SelfPlayManager()  # config=None -> defaults
    should = sp.should_run_self_play(episode=10)
    ok = True
    ok &= isinstance(should, bool)
    print(f"  should_run_self_play(ep=10): {should}")
    red_r, blue_r = sp.compute_adversarial_rewards(
        red_action="nmap -sV", blue_detected=False, red_success=True
    )
    ok &= isinstance(red_r, float) and isinstance(blue_r, float)
    print(f"  adversarial_rewards: red={red_r:.2f}, blue={blue_r:.2f}")
    stats = sp.get_stats()
    ok &= isinstance(stats, dict)
    print(f"  Stats: {stats}")
    return ok


# ---------------------------------------------------------------------------
# A3 — CommandPoolNarrower
# API: record_result(...), get_stats(), reset()
# ---------------------------------------------------------------------------
def test_pool_narrower():
    print("\n=== CommandPoolNarrower ===")
    from core.ops.pool_narrower import CommandPoolNarrower
    narrower = CommandPoolNarrower()  # config=None -> defaults
    narrower.record_result("nmap_tcp", success=True, reward=3.0, step=1)
    narrower.record_result("nmap_tcp", success=False, reward=0.0, step=2)
    stats = narrower.get_stats()
    ok = True
    ok &= isinstance(stats, dict)
    print(f"  Stats: {stats}")
    narrower.reset()
    print(f"  reset OK")
    return ok


# ---------------------------------------------------------------------------
# A4 — SSHSessionPool
# API: add_credentials(...), has_credentials(...), get_stats(), close_all()
# ---------------------------------------------------------------------------
def test_ssh_pool():
    print("\n=== SSHSessionPool ===")
    from core.execution.ssh_pool import SSHSessionPool
    pool = SSHSessionPool(keepalive_interval=30, connect_timeout=10, command_timeout=30)
    pool.add_credentials(username="root", password="toor", host="192.168.1.1")
    ok = True
    ok &= pool.has_credentials("192.168.1.1")
    print(f"  has_credentials: {pool.has_credentials('192.168.1.1')}")
    ok &= (pool.active_sessions() == 0)
    print(f"  active_sessions: {pool.active_sessions()}")
    stats = pool.get_stats()
    ok &= isinstance(stats, dict)
    print(f"  Stats: {stats}")
    pool.close_all()
    print(f"  close_all OK")
    return ok


# ---------------------------------------------------------------------------
# C5 — CTFModeTracker
# API: scan_output(...), get_stats(), reset()
# ---------------------------------------------------------------------------
def test_ctf_mode():
    print("\n=== CTFModeTracker ===")
    from core.execution.ctf_mode import CTFModeTracker
    ctf = CTFModeTracker()  # config=None -> defaults
    flags = ctf.scan_output(
        "Found flag: HTB{test_flag_12345}",
        command="cat flag.txt",
        agent="RedAgent",
    )
    ok = True
    ok &= isinstance(flags, list)
    print(f"  scan_output found {len(flags)} flags")
    stats = ctf.get_stats()
    ok &= isinstance(stats, dict)
    print(f"  Stats: {stats}")
    ctf.reset()
    print(f"  reset OK")
    return ok


# ---------------------------------------------------------------------------
# A5 — PhaseWeights
# API: get_weights(profile) -> PhaseWeights dataclass
# ---------------------------------------------------------------------------
def test_phase_weights():
    print("\n=== PhaseWeights ===")
    from core.config.phase_weights import get_weights, PhaseWeights
    w = get_weights()  # default profile
    ok = True
    ok &= isinstance(w, PhaseWeights)
    print(f"  PhaseWeights: {w}")
    return ok


# ---------------------------------------------------------------------------
# Warning-as-error import sweep
# ---------------------------------------------------------------------------
def test_warning_import():
    print("\n=== Warning-as-error import ===")
    try:
        import core.algorithms.hindsight_replay
        import core.algorithms.nstep_returns
        import core.algorithms.episodic_memory
        import core.algorithms.contrastive_state
        import core.algorithms.progressive_net
        import core.models.transformer_encoder
        import core.execution.ctf_mode
        import core.config.phase_weights
        import core.llm.cot_cache
        import core.training.dagger
        import core.training.self_play
        import core.llm.reflective_meta_learner
        import core.ops.pool_narrower
        import core.execution.ssh_pool
        print("  OK   All imports clean (no DeprecationWarning/RuntimeWarning)")
        return True
    except Warning as w:
        print(f"  FAIL Warning raised during import: {w}")
        return False


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------
def main():
    results = {}
    tests = [
        ("PPOAgent", test_ppo_agent),
        ("EpisodicMemory", test_episodic_memory),
        ("TransformerEncoder", test_transformer_encoder),
        ("ReflectiveMetaLearner", test_reflective_meta_learner),
        ("ContrastiveLoss", test_contrastive_state),
        ("NStepConfig", test_nstep_returns),
        ("HindsightReplay", test_her),
        ("ProgressiveExpander", test_progressive_net),
        ("CoTCache", test_cot_cache),
        ("DAggerBuffer", test_dagger),
        ("SelfPlayManager", test_self_play),
        ("PoolNarrower", test_pool_narrower),
        ("SSHPool", test_ssh_pool),
        ("CTFMode", test_ctf_mode),
        ("PhaseWeights", test_phase_weights),
        ("WarningImport", test_warning_import),
    ]

    all_ok = True
    for name, fn in tests:
        try:
            ok = fn()
            results[name] = "PASS" if ok else "FAIL"
            if not ok:
                all_ok = False
        except Exception as e:
            results[name] = f"ERROR: {e}"
            all_ok = False

    print("\n" + "=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)
    for name, status in results.items():
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} {name}: {status}")

    overall = "ALL PASS" if all_ok else "SOME FAILURES"
    print(f"\nOverall: {overall}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
