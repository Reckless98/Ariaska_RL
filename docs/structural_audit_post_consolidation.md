# Ariaska_RL — Post-Consolidation Structural Audit

**Date:** 2026-02-23
**Auditor:** Automated structural analysis
**Scope:** Full runtime orchestration autopsy of current codebase state
**Method:** Code tracing with exact file:line citations. No speculation.

---

## SECTION 1 — DECISION AUTHORITY

### Runtime Call Chain

```
SmartOrchestrator.run_training()           [smart_orchestrator.py:8171]
  └─ run_episode()                         [smart_orchestrator.py:8341 → L1985]
       └─ for step in range(max_steps):    [smart_orchestrator.py:1985]
            └─ _run_step()                 [smart_orchestrator.py:2015 → L4278]
                 └─ for agent in order:    [smart_orchestrator.py:4278]
                      └─ coach.decide()    [smart_orchestrator.py:4485]
                           └─ SmartCoach.decide()  [smart_coach.py:2601]
                                └─ return result   [smart_coach.py:4498]
```

### Where Candidates Are Generated

The `decide()` method (~1,900 lines) uses a **14-level priority cascade** at [smart_coach.py:3401-3746](core/training/smart_coach.py#L3401-L3746):

| Priority | Source | Lines |
|----------|--------|-------|
| 1 | Skill Library | L3401 |
| 2 | Web followup | L3402-3407 |
| 3 | Playbook | L3408 |
| 4 | MicroChain | L3409 |
| 5 | PhaseGuided | L3410 |
| 6 | Codex Meta | L3411 |
| 7 | CognitionNode | L3422 |
| 8 | Hypothesis Engine | L3424 |
| 9 | ActionArbitrator (P15) | L3466-3504 |
| 10 | Mentor-first coin flip | L3580-3596 |
| 11 | PPO-first | L3701-3714 |
| 12 | Mentor fallback | L3738-3742 |
| 13 | Registry fallback | L3744-3746 |
| 14 | Null-safety registry | L3754 |

Then **5 post-cascade mutators** can rewrite the result:

| Mutator | Lines | Effect |
|---------|-------|--------|
| Evidence Gate (enforce) | L3764-3809 | Replaces with registry fallback |
| TacticalCortex quality gate | L3826-3915 | Replaces command + source |
| Phase-stuck escalation (R52) | L3926-3990 | Overwrites with privesc command |
| Anti-repeat guard | L4053-4116 | Replaces with alternative |
| Smart reasoning check | L4188-4246 | Backward phase correction |

### Where Is `DecisionCore.arbitrate()` Called?

**`DecisionCore.arbitrate()` is NEVER called.** Zero invocations anywhere in `smart_coach.py`.

The `_decision_core` instance is used exclusively for **passive hit-counting** at [smart_coach.py:4428-4445](core/training/smart_coach.py#L4428-L4445):
```python
if self._decision_core is not None:
    # ... manually increment _hit_count[source] and _total_decisions
```

The fully-implemented weighted scoring method `arbitrate()` at [decision_core.py:221](core/decision/decision_core.py#L221) — with maturity-scaled source weights, hard constraint filtering, and win-rate factors — is **dead code in production**.

### Bypass Table

| Bypass | Condition | File:Symbol:Lines | What It Bypasses | Detection |
|--------|-----------|-------------------|------------------|-----------|
| CLOSEOUT early return | `current_phase == CLOSEOUT` | [smart_coach.py:decide():2656](core/training/smart_coach.py#L2656) | Entire pipeline: neuromod, MicroChain, PPO, mentor, anti-repeat, evidence gate, TacticalCortex, DecisionCore tracking | Monitor `source == "closeout"` but DecisionCore never counts it |
| Reflex override | `_reflex_override.triggered` | [smart_coach.py:decide():2887](core/training/smart_coach.py#L2887) | Entire pipeline same as above | `source == "reflex"` but DecisionCore never counts it |
| Stuck-escape (pre-decide) | `is_repeat_stuck` | [smart_orchestrator.py:_run_step():4400-4442](core/orchestration/smart_orchestrator.py#L4400) | `coach.decide()` never called | Not tracked anywhere |
| Followup queue (pre-decide) | `followup_queue` has entries | [smart_orchestrator.py:_run_step():4445-4472](core/orchestration/smart_orchestrator.py#L4445) | `coach.decide()` never called | Not tracked anywhere |

### Verdict

**Is DecisionCore the SINGLE action authority?**

**No.** `DecisionCore` is a **passive telemetry overlay** on top of the existing 14-level priority cascade + 5 post-cascade mutators. It records what won after the cascade decides. It never scores, never selects, never vetoes. The `arbitrate()` method is dead code. Four bypass paths completely evade even the hit-counting.

**Actual authority:** Hardcoded priority cascade with `if/elif/elif/else` waterfall, modified by post-cascade mutators.

---

## SECTION 2 — ENTROPY SOVEREIGNTY

### All Production Entropy Writers

**`entropy_coef` writes:**

| # | File:Line | Code | Guarded by `_entropy_locked`? | Runs During Training? | Can Override GMC? |
|----|-----------|------|-------------------------------|----------------------|-------------------|
| W1 | [ppo_agent.py:785](core/algorithms/ppo_agent.py#L785) | `self.entropy_coef = config.entropy_coef` | No (constructor) | Init only | N/A |
| W2 | [ppo_agent.py:1593](core/algorithms/ppo_agent.py#L1593) | `self.entropy_coef = base_entropy * multiplier` | **YES** (L1580) | Every `update()` | No — guarded |
| W3 | [ppo_agent.py:1605](core/algorithms/ppo_agent.py#L1605) | `self.entropy_coef *= stability_factor` | **YES** (same block) | Every `update()` | No — guarded |
| W4 | [ppo_agent.py:1608](core/algorithms/ppo_agent.py#L1608) | `self.entropy_coef *= min(1.2, ...)` | **YES** (same block) | Every `update()` | No — guarded |
| W5 | [ppo_agent.py:1623](core/algorithms/ppo_agent.py#L1623) | `self.entropy_coef = max(...)` (rebound) | **YES** (L1618) | Every `update()` | No — guarded |
| W6 | [ppo_agent.py:1781](core/algorithms/ppo_agent.py#L1781) | `self.entropy_coef = ckpt.get(...)` | **NO** | Checkpoint load | **VIOLATION** |
| W7 | [ppo_agent.py:1882](core/algorithms/ppo_agent.py#L1882) | `self.entropy_coef = max(...)` (neuromod) | **YES** (L1879) | Per-step | No — guarded |
| W8 | [schedule_coupler.py:230](core/training/schedule_coupler.py#L230) | `config.entropy_coef = state.entropy_coef` | **NO** | Episode start | **VIOLATION** (overwritten by GMC) |
| W9 | [maturity_controller.py:261-262](core/decision/maturity_controller.py#L261) | GMC writes both config + live | N/A (IS authority) | Episode start | N/A |

**`_entropy_adaptive_multiplier` writes:**

| # | File:Line | Guarded? | Can Override? |
|----|-----------|----------|---------------|
| M1 | [ppo_agent.py:1099](core/algorithms/ppo_agent.py#L1099) | **YES** (L1093) | No — guarded |
| M2 | [ppo_agent.py:1107](core/algorithms/ppo_agent.py#L1107) | **YES** (L1093) | No — guarded |
| M3 | [ppo_agent.py:1792](core/algorithms/ppo_agent.py#L1792) | **NO** | **VIOLATION** (checkpoint) |
| M4 | [smart_coach.py:3215](core/training/smart_coach.py#L3215) | Legacy fallback (`_maturity_controller is None`) | Partial — only when GMC absent |
| M5 | [smart_coach.py:3219](core/training/smart_coach.py#L3219) | Same legacy fallback | Partial — only when GMC absent |
| M6 | [maturity_controller.py:264](core/decision/maturity_controller.py#L264) | N/A (IS authority) | N/A |

### Runtime Entropy Timeline Per Episode

```
EPISODE START
  T0: ScheduleCoupler.update()              [SO:1575]
       └─ _apply_ppo(): config.entropy_coef ← interp    [SC:230] ⚠️ UNGUARDED
  T1: MaturityController.update(ep, metrics)             [SO:1602]
       └─ apply_to_ppo():                                [MC:249]
            ├─ config.entropy_coef ← state               [MC:261]
            ├─ entropy_coef ← state                      [MC:262] ← OVERWRITES T0
            ├─ _entropy_adaptive_multiplier ← 1.0        [MC:264]
            └─ _entropy_locked ← True                    [MC:266]

PER-STEP (×N):
  Neuromodulation → apply_neuromodulation()  [PPO:1882] BLOCKED ✓
  Legacy fallback → SmartCoach L3215         SKIPPED (GMC wired) ✓

EPISODE END:
  ppo.update() → cosine schedule             [PPO:1593] BLOCKED ✓
  ppo.update() → variance coupling           [PPO:1605] BLOCKED ✓
  ppo.update() → rebound                     [PPO:1623] BLOCKED ✓
  signal_episode_outcome() → multiplier      [PPO:1099] BLOCKED ✓
```

### Verdict

**Is GlobalMaturityController the ONLY entropy writer?**

**Conditionally yes at runtime.** When GMC is wired and `_entropy_locked = True`, all PPO internal entropy writes are blocked. Two real violations remain:

1. **[ppo_agent.py:1781+1792](core/algorithms/ppo_agent.py#L1781)** — `load_checkpoint()` restores `entropy_coef` and `_entropy_adaptive_multiplier` from disk **without** checking `_entropy_locked`. Window exists between checkpoint load and next `apply_to_ppo()`.
2. **[schedule_coupler.py:230](core/training/schedule_coupler.py#L230)** — Writes `config.entropy_coef` without lock check. Overwritten by GMC immediately after at T1, but architecturally unsound.

---

## SECTION 3 — REWARD PIPELINE

### Reward Flow Trace

```
CyberEnvironment.step(action) → info dict
  └─ SmartRewardCalculator.calculate_reward() → breakdown    [smart_coach.py:7931]
       └─ breakdown.total = base extrinsic reward

UnifiedRewardPipeline.compute(breakdown, rnd, scale)         [smart_coach.py:7973]
  └─ _unified_reward = UnifiedReward(total=...)
       └─ ⚠️ _unified_reward IS A DEAD LOCAL VARIABLE — NEVER READ
```

### Per-Learner Reward Table

| Learner | Store Line | Reward Scalar | Formula | Matches Unified? |
|---------|-----------|---------------|---------|------------------|
| **PPO** | [L7503](core/training/smart_coach.py#L7503) (batch at ep end) | `ppo_reward` | `breakdown.total` + mentor conformity (+5/floor 2) + macro alignment (±4/−1) + discovery reasoning (+1.5×n cap 8) + RND intrinsic + HTR relabeling | **NO** — most bonuses |
| **SAC** | [L8500](core/training/smart_coach.py#L8500) | `_sac_reward` | `breakdown.total` + RND intrinsic | **NO** — simpler |
| **DDQN** | [L8562](core/training/smart_coach.py#L8562) | `macro_reward` | `compute_macro_reward(breakdown.total)` — entirely reshaped with phase alignment, macro switching, discovery bonuses. **No RND.** | **NO** — completely reshaped |
| **CognitionNode** | [L8590](core/training/smart_coach.py#L8590) | `breakdown.total` | Raw extrinsic only | **NO** — raw only |
| **HybridMemory** | [L8624](core/training/smart_coach.py#L8624) | `breakdown.total` | Raw extrinsic only | **NO** — raw only |

### PPO Reward Construction (detailed)

Built incrementally at [smart_coach.py:8337-8450](core/training/smart_coach.py#L8337):

```python
ppo_reward = breakdown.total                           # base
  + 5.0     (if PPO agreed with mentor)                # L8347: conformity
  OR max(breakdown.total, 2.0)  (if mentor decision)   # L8358: DAgger-lite floor
  + 4.0     (if template ∈ active macro)               # L8372: macro alignment
  - 1.0     (if template ∉ active macro)               # L8378: macro misalignment
  + min(disc_count * 1.5, 8.0)  (if discoveries)       # L8392: reasoning bonus
  + _rnd_intrinsic                                     # L8431: RND
```

Then at episode end: R69 HTR relabeling retroactively adds bonuses at [L7470-7499](core/training/smart_coach.py#L7470).

### RND Intrinsic Injection Points

| Learner | Where | Present? |
|---------|-------|----------|
| PPO | [smart_coach.py:8431](core/training/smart_coach.py#L8431) | Yes |
| SAC | [smart_coach.py:8496](core/training/smart_coach.py#L8496) | Yes |
| DDQN | — | **No** |
| CognitionNode | — | **No** |
| HybridMemory | — | **No** |

### Verdict

**Is reward single-scalar unified?**

**No.** `UnifiedRewardPipeline.compute()` is called but its output is a dead local variable — never read by any learner. Each learner independently constructs its own reward scalar from `breakdown.total` with different per-learner bonuses applied upstream in `record_result()`. Five learners see five different reward values.

---

## SECTION 4 — SCHEDULE COUPLING

### MaturityController M Derivation

At [maturity_controller.py:170-174](core/decision/maturity_controller.py#L170):
```
M = 0.4 × success_rate + 0.3 × skill_coverage + 0.2 × disc_efficiency + 0.1 × (1 - stagnation_rate)
```

Clamped [0, 1].

### All Schedules Derived from M

| Schedule | Line | Method |
|----------|------|--------|
| entropy_coef | [MC:186-193](core/decision/maturity_controller.py#L186) | cosine anneal (0.08→0.003) |
| mentor_rate | [MC:195-197](core/decision/maturity_controller.py#L195) | cosine anneal (0.80→0.10) |
| bc_coef | [MC:200-202](core/decision/maturity_controller.py#L200) | cosine anneal (0.10→0.01) |
| kl_coef | [MC:205-207](core/decision/maturity_controller.py#L205) | cosine anneal (0.15→0.01) |
| prior_alpha | [MC:210-212](core/decision/maturity_controller.py#L210) | cosine anneal (0.50→0.02) |
| rnd_scale | [MC:215-217](core/decision/maturity_controller.py#L215) | cosine anneal (1.0→0.1) |
| sac_alpha_floor | [MC:220-222](core/decision/maturity_controller.py#L220) | cosine anneal (0.2→0.05) |
| reptile_active | [MC:225](core/decision/maturity_controller.py#L225) | boolean: M < 0.6 |
| eval_cadence | [MC:228-230](core/decision/maturity_controller.py#L228) | linear interp (5→20 ep) |

### Independent Annealing Still Active

| # | What | File:Line | Independent Of M? | Severity |
|----|------|-----------|-------------------|----------|
| 1 | `base_mentor_rate` linear decay by episode | [smart_coach.py:3526-3534](core/training/smart_coach.py#L3526) | **YES** — uses `self.current_episode * 0.002/0.005` | **HIGH** — actually controls mentor coin flip |
| 2 | `prior_alpha` + `kl_teacher_coef` internal cosine | [llm_policy_bridge.py:348-393](core/llm/llm_policy_bridge.py#L348) | **YES** — uses `self._step_count / self.total_anneal_steps` | **HIGH** — supersedes GMC's values |
| 3 | ScheduleCoupler parallel system | [schedule_coupler.py:115-141](core/training/schedule_coupler.py#L115) | **PARTIAL** — receives M but applies different formula | **MEDIUM** — overwritten by GMC |
| 4 | `bc_loss_coef` write in `apply_neuromodulation()` | [ppo_agent.py:1895](core/algorithms/ppo_agent.py#L1895) | **YES** — not gated by `_entropy_locked` | **MEDIUM** |
| 5 | PPO internal cosine entropy schedule | [ppo_agent.py:1580-1593](core/algorithms/ppo_agent.py#L1580) | **YES** — uses `total_steps/total_timesteps` | Guarded by `_entropy_locked` ✓ |

### Verdict: Are All Schedules Derived From Single M?

**No.** Two critical independent annealing systems remain:
1. SmartCoach `base_mentor_rate` at [L3526-3534](core/training/smart_coach.py#L3526) uses its own linear decay without consulting MaturityController. This is the **actual** mentor rate — `effective_mentor_rate` at [L3575](core/training/smart_coach.py#L3575) drives the coin flip at [L3593](core/training/smart_coach.py#L3593). GMC's `mentor_rate` schedule is computed but **never read** by the coin flip logic.
2. LLMPolicyBridge runs its own cosine anneal for `prior_alpha` and `kl_teacher_coef` independent of M.

---

## SECTION 5 — SAC STATUS

### Classification: **(C) Shadow-only**

SAC selects an action every step at [smart_coach.py:7301](core/training/smart_coach.py#L7301) via `sac_agent.select_action(state_tensor)`, gated by `FF_SAC_SHADOW` at [L2636-2639](core/training/smart_coach.py#L2636).

**What SAC contributes:**
- Populates `DecisionPacket.sac` fields at [L7331-7338](core/training/smart_coach.py#L7331) — action_idx, log_prob, q_value, confidence, alpha
- Stores off-policy transitions at [L8500](core/training/smart_coach.py#L8500)
- Runs `sac_agent.update()` every step at [L8509](core/training/smart_coach.py#L8509)

**What SAC does NOT do:**
- No `Advisory(source="sac")` is ever created
- Q-values never influence PPO logits, action selection, or arbitration
- SAC's selected action is never compared against the pipeline's choice for decision-making
- SAC never directly determines the final command

**Verdict:** SAC is a pure shadow learner. It observes all transitions, learns off-policy, and stores Q-value estimates on DecisionPacket for telemetry. It has **zero influence** on any decision or action selection path. Its advisory weight in `DEFAULT_SOURCE_WEIGHTS` (`sac=0.3` at [decision_core.py:173](core/decision/decision_core.py#L173)) is dead — `arbitrate()` is never called.

---

## SECTION 6 — REPTILE SANITY

### When `meta_step()` Is Called

[smart_orchestrator.py:3210](core/orchestration/smart_orchestrator.py#L3210):
```python
_rep_stats = self.reptile.meta_step(
    model=_network,            # PPO actor-critic network [SO:3193]
    inner_train_fn=_inner_train,
    global_step=self._reptile_global_step,
    maturity=_maturity_signal,
)
```

Called every episode boundary after warmup, gated by `self.reptile.should_run()` at [SO:3186](core/orchestration/smart_orchestrator.py#L3186) — which always returns True after warmup (no cooldown).

### What Weights It Mutates

1. **Snapshot:** `copy.deepcopy(model.state_dict())` at [reptile_meta.py:282](core/algorithms/reptile_meta.py#L282)
2. **Restore per scenario:** `model.load_state_dict(copy.deepcopy(snapshot))` at [L296](core/algorithms/reptile_meta.py#L296)
3. **Accumulate delta:** `(trained_weights - snapshot)` per scenario at [L305-310](core/algorithms/reptile_meta.py#L305)
4. **Final interpolation:** `weights = snapshot + ε × mean(deltas)` via `model.load_state_dict(final_state)` at [L316-321](core/algorithms/reptile_meta.py#L316)

### PPO Optimizer State Preservation

**NO.** `model.load_state_dict()` only affects model parameters. PPO's Adam optimizer (`self.optimizer`) retains stale momentum/variance buffers from pre-snapshot parameters. After Reptile interpolates new weights, Adam's state is mismatched. This is a known Reptile+Adam interaction issue.

### Double-Count Protection

**NONE.** Critical issues:

1. `should_run()` at [reptile_meta.py:250-254](core/algorithms/reptile_meta.py#L250) only checks `enabled` and `global_step >= warmup_steps` — always True after warmup.
2. No scenario dedup — `ScenarioSampler.sample()` at [L182-196](core/algorithms/reptile_meta.py#L182) randomly draws from pool; same scenario can repeat.
3. **The `inner_train_fn` ignores both `inner_steps` and `scenario_name`** — it simply calls `_ppo.update(last_value=0.0)` once on the **current buffer** at [SO:3198-3208](core/orchestration/smart_orchestrator.py#L3198). Every scenario iteration runs the same PPO update on the same data, making the meta-learning averaging meaningless.

### Verdict

Reptile is architecturally present but **functionally degenerate**: every scenario runs the same PPO update on the same buffer, the "meta-learning over scenario distributions" is effectively repeated identical PPO updates, and Adam state drift goes uncorrected.

---

## SECTION 7 — INTEGRITY ENFORCEMENT

### Where IntegrityCheck Runs

Import at [smart_orchestrator.py:8308](core/orchestration/smart_orchestrator.py#L8308), instantiation at [L8311](core/orchestration/smart_orchestrator.py#L8311), boot check at [L8312](core/orchestration/smart_orchestrator.py#L8312):
```python
_integrity = IntegrityCheck()
_boot_report = _integrity.check_boot(_first_coach, self)
```

### What It Asserts

**Boot-time `check_boot()`** ([integrity_check.py:90-200](core/decision/integrity_check.py#L90)):

| Check | Severity | What |
|-------|----------|------|
| `decision_core_wired` | error | `_decision_core is not None` |
| `maturity_controller_wired` | error | `_maturity_controller is not None` |
| `reward_pipeline_wired` | error | `_reward_pipeline is not None` |
| `harmony_metrics_wired` | error | `_harmony_metrics is not None` |
| `ppo_agent_exists` | error | `ppo_agent is not None` |
| `state_dim_512` | error | `config.state_dim == 512` |
| `orchestrator_maturity` | warning | `orchestrator._maturity_controller is not None` |

**Runtime `check_runtime()`** ([integrity_check.py:219-314](core/decision/integrity_check.py#L219)):

| Check | Severity | What |
|-------|----------|------|
| `single_entropy_writer` | error if >1 | `harmony_metrics.entropy_writer_count` |
| `ppo_has_updated` | warning | `ppo._update_count > 0` after warmup |
| `reward_pipeline_used` | error | `pipeline.step_count > 0` after warmup |
| `decision_core_used` | warning | `hit_distribution` non-empty after warmup |

### Critical Flaws

1. **Boot failure does NOT stop training.** At [SO:8313-8335](core/orchestration/smart_orchestrator.py#L8313), failures are **logged only** — no `RuntimeError`, no `sys.exit()`. The entire block is wrapped in `try/except Exception` at [L8307](core/orchestration/smart_orchestrator.py#L8307), which swallows all errors. Training proceeds regardless of integrity violations.

2. **`check_runtime()` is NEVER called in production.** Zero call sites outside tests. The runtime checks (entropy writer count, PPO liveness, pipeline usage, DecisionCore usage) are **dead code in production**.

3. **Entropy writer count is NOT checked at boot.** Only in `check_runtime()`, which is never called.

### Verdict

IntegrityCheck is **symbolic only** — it verifies attribute presence at boot (which always passes since `__init__` sets them) but never enforces consequences, and its runtime checks are never invoked.

---

## SECTION 8 — HARMONY METRICS

### Metrics Currently Logged

`log_to_tensorboard()` at [harmony_metrics.py:282-355](core/decision/harmony_metrics.py#L282) logs all 8 metrics:

| # | Metric | TensorBoard Key | Recording Call |
|----|--------|-----------------|----------------|
| 1 | Decision source entropy | `harmony/decision_source_entropy` | `record_decision()` at [L134](core/decision/harmony_metrics.py#L134) |
| 2 | Arbitration weights | `harmony/arb_weight/{src}` | `record_decision()` stores `_latest_weights` |
| 3 | Entropy writer count | `harmony/entropy_writer_count` | `register_entropy_writer()` at [L177](core/decision/harmony_metrics.py#L177) |
| 4 | Gradient norms | `harmony/grad_norm/{comp}` | `record_gradient_norms()` at [L193](core/decision/harmony_metrics.py#L193) |
| 5 | Macro switch rate | `harmony/macro_switch_rate` | `record_macro_step()` at [L203](core/decision/harmony_metrics.py#L203) |
| 6 | Mentor dependence | `harmony/mentor_dependence` | Derived from `_source_hits` |
| 7 | Reward variance | `harmony/reward_variance` | `record_reward()` at [L226](core/decision/harmony_metrics.py#L226) |
| 8 | KL drift | `harmony/kl_drift` | `record_kl_drift()` at [L241](core/decision/harmony_metrics.py#L241) |

### Critical Issue

**`log_to_tensorboard()` is NEVER called in production.** Zero call sites outside tests. The docstring example shows usage ([L97](core/decision/harmony_metrics.py#L97)), but no orchestrator or dashboard code invokes it. All 8 metrics are computed internally but never exported.

**Recording calls that ARE wired:**
- `record_decision()` — called at [smart_coach.py:4448](core/training/smart_coach.py#L4448) (only on normal path, not CLOSEOUT/reflex bypasses)
- `record_reward()` — called at [smart_coach.py:7987](core/training/smart_coach.py#L7987)
- `record_macro_step()` — called at [smart_coach.py:4456](core/training/smart_coach.py#L4456) (conditional on DDQN macro)
- `register_entropy_writer()` — called at [smart_coach.py:856](core/training/smart_coach.py#L856) (init)

**Recording calls NOT wired:**
- `record_gradient_norms()` — never called in production
- `record_kl_drift()` — never called in production
- `log_to_tensorboard()` — never called in production

---

## SECTION 9 — GENERALIZATION READINESS

### Will Behavior Degrade Gracefully on Unseen HTB?

**Assessment based on four structural factors:**

#### Shared Representation
- State encoder produces 512-dim vector at [state_encoder.py](core/models/state_encoder.py). Dims 0-90 encode phase, flags, ports, services, numerics, action history, LLM features, temporal features. Dims 91-511 are zero-padded.
- **Risk:** Port presence indicators (dims 27-46) and service type indicators (dims 47-58) are hardcoded to common services. Novel HTB services will fall into "unseen" space (zero-filled dims 91+), giving the encoder no discriminative signal for novel targets.
- **Prediction:** Moderate degradation. The encoder captures structural features (phase progress, detection risk, action history) that transfer reasonably, but service/port-specific knowledge won't transfer to novel targets.

#### Reward Shaping
- Five learners see five different reward scalars (see Section 3). PPO gets the most bonuses including mentor conformity (+5), macro alignment (±4), and discovery reasoning (+1.5×n).
- **Risk:** PPO's reward is heavily shaped toward confirming mentor decisions and aligning with DDQN macros. On unseen targets where mentor gives poor advice and macros misalign, these bonuses become noise or actively harmful shaping.
- **Prediction:** Reward shaping will cause PPO to over-rely on mentor agreement patterns that don't generalize. Expected oscillation between mentor-following and PPO's learned policy on unseen targets.

#### Meta-Learning Integration
- Reptile is functionally degenerate (see Section 6). `inner_train_fn` ignores scenario name, uses same buffer for every scenario.
- **Prediction:** No meta-learning benefit. Reptile currently functions as a weight-averaging perturbation, not scenario-adaptive generalization.

#### Eval Coverage
- `HeldOutEval` exists but `should_eval(episode)` in MaturityController gates it at [MC:228-230](core/decision/maturity_controller.py#L228). No evidence of held-out scenario evaluation actually running during training.
- **Prediction:** No OOD detection. System will degrade silently without metric visibility.

### Overall Prediction

**Behavior will oscillate, not collapse.** The system has enough structural scaffolding (phase ladder, evidence gate, anti-repeat, tactical cortex) to prevent total collapse. However:
- Novel services → weak state encoding → PPO uncertainty → high mentor reliance → high mentor dependence curve
- No meta-learning → no rapid adaptation → slow convergence on new targets
- Reward shaping bonuses → learned mentor-following bias → poor autonomous performance

Expected pattern: Early steps rely heavily on mentor/playbook (high mentor dependence), then slowly adapt via PPO's own exploration, with periodic phase-stuck escalation when the cascade fails to find working commands.

---

## FINAL VERDICT

### Architecture Classification

**Still layered cascade with cosmetic unification.**

The consolidation added a harmonization layer (`core/decision/`) with correct designs, but the new modules are **passive telemetry overlays** rather than actual authority points:

| Module | Intended Role | Actual Role |
|--------|--------------|-------------|
| `DecisionCore` | Single action authority | Passive hit counter; `arbitrate()` never called |
| `UnifiedRewardPipeline` | Single reward for all learners | Statistics tracker; output variable is dead |
| `GlobalMaturityController` | Single schedule authority | Entropy authority works (via `_entropy_locked`); mentor rate ignored by actual coin-flip |
| `HarmonyMetrics` | Observable 8-metric dashboard | Partially wired; 3 recording calls dead; TensorBoard export dead |
| `IntegrityCheck` | Boot-time enforcement | Symbolic-only; no hard stop; runtime checks dead |

The actual decision-making still flows through the 14-level priority cascade in `SmartCoach.decide()`, with 5 post-cascade mutators, 2 early-return bypasses, and 2 pre-decide bypasses in the orchestrator. The maturity controller correctly locks PPO entropy, but the mentor rate and LLMPolicyBridge annealing remain independent.

---

## STRUCTURAL RISK TABLE

| # | Risk | File:Symbol | Severity | Domain | P(Instability) | HTB Impact | Fix Complexity |
|----|------|------------|----------|--------|-----------------|------------|----------------|
| R1 | `DecisionCore.arbitrate()` never called — dead code | [decision_core.py:arbitrate():221](core/decision/decision_core.py#L221) | **Critical** | Decision | Low (passive) | High (no weighted scoring) | **High** — requires replacing cascade with arbitrate() calls |
| R2 | `UnifiedRewardPipeline.compute()` output dead — each learner gets different reward | [smart_coach.py:record_result():7973](core/training/smart_coach.py#L7973) | **Critical** | Reward | High (gradient conflict) | High (reward divergence) | **High** — requires refactoring all 5 learner reward paths |
| R3 | `base_mentor_rate` linear decay independent of M — actual mentor coin flip ignores GMC | [smart_coach.py:decide():3526-3534](core/training/smart_coach.py#L3526) | **High** | Scheduling | Medium | High (mentor over-reliance) | **Medium** — replace linear decay with `_maturity_controller.state.mentor_rate` |
| R4 | `LLMPolicyBridge` own cosine anneal ignores GMC's `prior_alpha` / `kl_coef` | [llm_policy_bridge.py:_compute_anneal_alpha():348](core/llm/llm_policy_bridge.py#L348) | **High** | Scheduling | Medium | Medium (KL/prior drift) | **Medium** — inject MaturityState into bridge |
| R5 | Reptile `inner_train_fn` ignores scenario — same buffer used for all scenarios | [smart_orchestrator.py:_inner_train():3198](core/orchestration/smart_orchestrator.py#L3198) | **High** | Meta | High (wasted cycles) | High (no meta-learning) | **High** — requires scenario-specific buffer collection |
| R6 | Reptile no Adam optimizer reset — stale momentum after `load_state_dict()` | [reptile_meta.py:meta_step():316](core/algorithms/reptile_meta.py#L316) | **High** | Meta | Medium (silent drift) | Medium (weight instability) | **Low** — snapshot/restore optimizer state_dict |
| R7 | 4 decision bypasses invisible to DecisionCore hit-counting | [smart_coach.py:2656,2887](core/training/smart_coach.py#L2656), [smart_orchestrator.py:4400,4445](core/orchestration/smart_orchestrator.py#L4400) | **Medium** | Decision | Low | Low (tracking gap) | **Low** — add hit-counting at bypass points |
| R8 | `IntegrityCheck` boot failure does not stop training | [smart_orchestrator.py:8313-8335](core/orchestration/smart_orchestrator.py#L8313) | **Medium** | Integrity | Low (rarely fails) | Medium (silent violation) | **Low** — add `raise RuntimeError` after error log |
| R9 | `check_runtime()` never called — entropy writer + liveness checks dead | [integrity_check.py:check_runtime():219](core/decision/integrity_check.py#L219) | **Medium** | Integrity | Low | Medium (no runtime guard) | **Low** — add periodic call in step loop |
| R10 | `HarmonyMetrics.log_to_tensorboard()` never called — metrics invisible | [harmony_metrics.py:282](core/decision/harmony_metrics.py#L282) | **Medium** | Metrics | Low | Medium (no observability) | **Low** — wire into episode-end or dashboard |
| R11 | `ScheduleCoupler` redundant with GMC — different formulas for same params | [schedule_coupler.py:115-141](core/training/schedule_coupler.py#L115) | **Medium** | Scheduling | Low (overwritten) | Low | **Medium** — remove or subordinate to GMC |
| R12 | PPO `load_checkpoint()` overwrites entropy without `_entropy_locked` check | [ppo_agent.py:1781,1792](core/algorithms/ppo_agent.py#L1781) | **Medium** | Entropy | Low (checkpoint rare) | Low | **Low** — add guard |
| R13 | `bc_loss_coef` write in `apply_neuromodulation()` not gated | [ppo_agent.py:1895](core/algorithms/ppo_agent.py#L1895) | **Low** | Scheduling | Low | Low | **Low** — add guard |
| R14 | PPO reward includes mentor conformity bonus (+5) — trains to follow mentor | [smart_coach.py:8347](core/training/smart_coach.py#L8347) | **Medium** | Reward | Medium (bias) | High (limits autonomy) | **Medium** — anneal with maturity |
| R15 | `record_gradient_norms()` and `record_kl_drift()` never called | [harmony_metrics.py:193,241](core/decision/harmony_metrics.py#L193) | **Low** | Metrics | None | Low | **Low** — wire into PPO update callback |

---

## TOP 7 HIGHEST-LEVERAGE CHANGES

Ranked by combined **impact × likelihood × (1/fix_cost)**:

### 1. Wire `UnifiedRewardPipeline.compute()` output into all learners
- **Risk:** R2 — Currently dead variable. Five learners see five different rewards.
- **Impact:** Eliminates gradient conflict between PPO/SAC/DDQN/CognitionNode. Foundational for multi-learner convergence.
- **Estimated effort:** High — must refactor 5 reward construction paths in `record_result()`.
- **Signal:** Immediate — PPO stops learning mentor-following bias, SAC/PPO/DDQN converge toward same value function.

### 2. Replace SmartCoach `base_mentor_rate` linear decay with GMC's `mentor_rate`
- **Risk:** R3 — Actual mentor coin flip at [L3593](core/training/smart_coach.py#L3593) uses independent linear decay, completely ignoring MaturityController's cosine-annealed `mentor_rate`.
- **Impact:** Mentor rate finally driven by maturity signal M. Single authority for the most expensive scheduling decision.
- **Estimated effort:** Medium — replace ~10 lines in `decide()` L3526-3575 with `self._maturity_controller.state.mentor_rate`.
- **Signal:** Immediate — mentor calls reduce as competence grows, budget savings compound.

### 3. Fix Reptile `inner_train_fn` to use scenario-specific data
- **Risk:** R5 — Reptile ignores scenario_name, uses same buffer for every inner iteration.
- **Impact:** Enables actual meta-learning over scenario distributions instead of repeated identical PPO updates.
- **Estimated effort:** High — requires per-scenario buffer collection or synthetic scenario generation.
- **Signal:** Deferred — requires curriculum/scenario infrastructure first.

### 4. Wire `DecisionCore.arbitrate()` into `decide()` hot path
- **Risk:** R1 — fully-implemented weighted scoring system is dead code.
- **Impact:** Replaces hardcoded priority cascade with maturity-sensitive source weighting. PPO influence grows with competence.
- **Estimated effort:** High — must collect Advisories from each source, call `arbitrate()`, replace cascade waterfall.
- **Signal:** Fundamental — transforms decision system from fixed priorities to adaptive multi-source scoring.

### 5. Make `IntegrityCheck` boot failure a hard stop
- **Risk:** R8 — Boot check only logs errors, never stops training.
- **Impact:** Prevents training with misconfigured consolidation modules (e.g., missing maturity controller, wrong state_dim).
- **Estimated effort:** Low — 2-line change: add `raise RuntimeError(msg)` after error count check.
- **Signal:** Safety — prevents silent training-with-violation.

### 6. Wire `HarmonyMetrics.log_to_tensorboard()` into episode-end
- **Risk:** R10 — All 8 harmony metrics computed but never exported.
- **Impact:** Provides observability into decision distribution, entropy authority, reward variance, mentor dependence.
- **Estimated effort:** Low — single call in `SmartOrchestrator._end_episode()` or dashboard update.
- **Signal:** Diagnostic — enables data-driven decisions about further consolidation.

### 7. Add Reptile Adam optimizer state snapshot/restore
- **Risk:** R6 — After `model.load_state_dict()`, Adam's momentum buffers are stale.
- **Impact:** Prevents Reptile from introducing silent weight drift through optimizer state mismatch.
- **Estimated effort:** Low — snapshot `optimizer.state_dict()` before meta_step, restore after.
- **Signal:** Stability — removes one source of unexplained training variance.
