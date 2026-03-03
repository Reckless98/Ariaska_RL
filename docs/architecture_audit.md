# Ariaska_RL — Full Structural Orchestration Audit

**Date:** 2025-07-14
**Author:** System Audit (automated)
**Scope:** Runtime execution paths, training signal flows, scheduling coherence, generalization capacity, structural harmony/instability, redesign proposals, missing metrics.
**Methodology:** Static analysis of all source files cited below. Every claim cites `file:symbol` or `file:L<line>`. "NOT FOUND" means the search returned zero results.

---

## Section A — Runtime Execution Path

### A.1 Entry Point

`run_training()` ([smart_orchestrator.py](../core/orchestration/smart_orchestrator.py#L8109)) forces **single continuous episode** (Phase 24 design). There is no multi-episode loop — training is one open-ended engagement against a live target. It calls `run_episode()` which iterates `_run_step()` for `max_steps`.

### A.2 Per-Step Call Graph

`_run_step()` ([smart_orchestrator.py](../core/orchestration/smart_orchestrator.py#L4165)) executes the following linear sequence per agent (agents iterated in phase-optimized activation order):

```
_run_step(step, agent_list)
│
├─ 1. Build SmartStepContext (L4180-4230)
│     state, phase, discoveries, command_history, etc.
│
├─ 2. RND intrinsic computation — BEFORE decide()  (L4290-4318)
│     rnd_curiosity.compute_intrinsic_reward(state_tensor)
│     Stashes _rnd_state_tensor for post-step predictor update
│
├─ 3. Build DecisionPacket (L4330)
│     DecisionPacket.from_step_context(step_ctx, rnd_intrinsic, coherence, macro_conf)
│
├─ 4. Stuck-escape check (L4340-4402)
│     If repeat-stuck → coach._force_novel_action() with 4 threshold levels
│     BYPASSES entire decide() pipeline
│
├─ 5. FollowupQueue consumption (L4404-4430)
│     Priority-sorted queued commands from previous steps
│     BYPASSES entire decide() pipeline
│
├─ 6. coach.decide(step_ctx, proposed_action, confidence, decision_packet)  (L4440)
│     ░░░ THE MAIN DECISION PIPELINE (see A.3) ░░░
│
├─ 7. Cross-agent command dedup (L4453-4470)
│     Prevents Orion+Red firing same nmap/gobuster in same step
│
├─ 8. _execute_env_step(red_action, blue_action)  (L4485)
│     CyberEnvironment.step() or live execution
│
├─ 9. Post-step: RND predictor update (L2458)
│     rnd_curiosity.update(_rnd_state_tensor) — trains predictor to match target
│
├─ 10. Post-step: coach.record_result()  (L7824)
│      Reward calculation, PPO/SAC/DDQN transition storage + updates
│
└─ 11. Episode boundary: Reptile meta_step (L3148) + HeldOutEval (L3167)
       ScheduleCoupler.update() in _start_episode (L1561)
```

### A.3 The Decision Cascade — `SmartCoach.decide()`

`decide()` ([smart_coach.py](../core/training/smart_coach.py#L2574)) is a **~1,100-line method** implementing a **12+-level priority cascade**. This is the core decision authority for every agent, every step.

**Cascade order (first non-None wins):**

| Priority | Source | Location | Bypass PPO? | Notes |
|----------|--------|----------|-------------|-------|
| 0 | CLOSEOUT hard gate | L2636 | YES | Terminal — forces closeout commands |
| 1 | Phase ladder gate | L2647 | YES | Blocks invalid phase transitions |
| 2 | Neuromodulator compute | L2660 | no (modifies PPO) | Modulates entropy_coef, lr, bc_weight |
| 3 | Aggression controller | L2773 | no (modifies PPO) | Adjusts exploration bounds |
| 4 | **Reflex policy** | L2812 | **YES — early RETURN** | Can return immediately, bypassing everything |
| 5 | Working memory | L2878 | no (context only) | Updates context for downstream |
| 6 | **Skill library** | L2910 | **YES** | Phase-matched skill cards from prior runs |
| 7 | **Playbook curriculum** | L2913 | **YES** | Anneals 60%→10%; when active, PPO not called |
| 8 | DDQN macro-intent | L3112 | no (soft filter) | Constrains `filtered_commands` but doesn't pick |
| 9 | **MicroChain** (3-stage LLM) | L3190 | **YES** | nano→mini→nano scoring; if score>threshold, wins |
| 10 | **PhaseGuidedLLM** | L3238 | **YES** | Structured JSON guidance; if confidence>0.45, wins |
| 11 | **Codex meta-layer** | L3280 | **YES** | Full GPT-5.2-codex call; if confident, wins |
| 12 | **CognitionNode** | L3293 | **YES** | Multi-brain fusion; confidence≥0.5 gate; if high, wins |
| 13 | **Hypothesis-driven** | L3398 | **YES** | Phase 14 hypothesis engine; if match, wins |
| 14 | **ActionArbitrator** | L3404 | **YES** | Collects PPO+registry+skill candidates; arbitrates |
| 15 | **Mentor vs PPO coin flip** | L3535 | **50/50** | `random() < effective_mentor_rate` decides who leads |
| 16 | **Registry fallback** | L3044 | YES (PPO absent) | Precondition-matched CommandRegistry (294 templates) |

**Critical observation:** PPO is priority 15 in a 16-level cascade. Every source from #4 through #14 can and does bypass PPO completely. When PPO does get called (level 15), it competes with the mentor via a random coin flip whose bias decays per-agent:

- **RedAgent:** 80%→45% mentor rate over 175 episodes ([smart_coach.py](../core/training/smart_coach.py#L3497))
- **OrionAgent:** 75%→40% ([smart_coach.py](../core/training/smart_coach.py#L3501))
- **Others:** 50%→15% over ~70 episodes ([smart_coach.py](../core/training/smart_coach.py#L3505))

### A.4 SAC's Role: Pure Shadow

`_sac_shadow_select()` ([smart_coach.py](../core/training/smart_coach.py#L7197)) runs every step. The docstring states:

> "SAC selects an action in parallel with PPO but does NOT override the final decision."

SAC populates the DecisionPacket with its Q-values and action, but **no code path in `decide()` ever reads `decision_packet.sac.*` to influence the chosen command.** SAC is a passive observer that trains from other agents' transitions.

### A.5 DDQN Macro's Role: Soft Influence

DDQN macro-intent selection at [smart_coach.py](../core/training/smart_coach.py#L3112) selects a high-level strategy (e.g., `SCAN`, `EXPLOIT`, `PRIVESC`). This constrains `filtered_commands` to phase-appropriate templates. However, DDQN does not pick the final command — it only narrows the candidate pool. If no candidates match, the constraint is relaxed.

---

## Section B — Training Signal Flow

### B.1 Five Independent Learners

The system has **5 independently parameterized neural networks** that all train from the same step outcomes but see **different reward signals** and have **no shared representations**:

| Learner | Networks | Reward Signal | Update Frequency | Called From |
|---------|----------|---------------|------------------|-------------|
| **PPO** | PPOActorCritic (shared backbone + actor/critic heads) | `breakdown.total` (normalized) | End of episode | `end_episode_ppo()` [smart_coach.py](../core/training/smart_coach.py#L7282) |
| **SAC** | DiscreteActor + TwinCritic + TargetCritic | `breakdown.total + RND intrinsic` (NOT normalized) | Every step | [smart_coach.py](../core/training/smart_coach.py#L8395) and [cognition_node.py](../core/algorithms/cognition_node.py#L548) |
| **DDQN** | online_net + target_net (MLP) | `compute_macro_reward()` — completely reshaped | Every step | [smart_coach.py](../core/training/smart_coach.py#L8457) |
| **RND** | target_net (frozen) + predictor_net | MSE(predictor, target) | Every step (post-step) | [smart_orchestrator.py](../core/orchestration/smart_orchestrator.py#L2458) |
| **CognitionNode** | ConfidenceGate (16→32→weights+confidence) | Cross-entropy on brain weights | Every step (observe) | [cognition_node.py](../core/algorithms/cognition_node.py#L494) |

### B.2 PPO's 9-Component Loss

`PPOAgent.update()` ([ppo_agent.py](../core/algorithms/ppo_agent.py#L1108)) computes:

1. **Policy loss** — clipped surrogate (clip ε=0.2, adaptive 0.15-0.25)
2. **Value loss** — Huber, clipped, ×0.5
3. **Entropy bonus** — ×`-entropy_coef` (cosine schedule)
4. **Phase prediction** — auxiliary head, ×0.1 (gated by `FF_PHASE_PRED_AUX`)
5. **BC loss** — from BCBuffer, ×0.1 (gated by `FF_BC_LOSS`)
6. **KL teacher distillation** (gated)
7. **Ranking margin loss** (gated)
8. **Value regularization** (gated)
9. **Contrastive state loss** (gated)

Components 5-9 are **gated by feature flags** — their activation depends on the feature flag profile (CLOUD/DETERMINISTIC/OFFLINE). In DETERMINISTIC mode (tests), all LLM-dependent components are OFF, so the loss degenerates to components 1-4.

### B.3 Reward Divergence Across Learners

**This is a critical structural issue.** Each learner optimizes a different objective:

**PPO** sees `breakdown.total` after normalization (mean-subtraction, std-division) in `store_transition()` ([ppo_agent.py](../core/algorithms/ppo_agent.py#L1046)).

**SAC** sees `breakdown.total + RND intrinsic bonus` at [smart_coach.py](../core/training/smart_coach.py#L8388):
```python
_sac_reward = breakdown.total
if _dp is not None and _dp.rnd.valid:
    _sac_reward += _dp.rnd.intrinsic_reward
```
This is NOT normalized. The magnitude mismatch between normalized PPO rewards and raw SAC rewards means their value estimates are in different scales.

**DDQN** sees `compute_macro_reward()` ([ddqn_macro.py](../core/algorithms/ddqn_macro.py#L828)) which reshapes `step_reward` with:
- +2.0 / -3.0 for phase alignment/misalignment
- -3.5 / +1.5 for macro switch penalty/hold bonus
- +5.0×phase_distance / -10.0 for phase regression
- +1.0×num_discoveries

This is a **completely different reward function** from what PPO and SAC see.

**Consequence:** Three learners optimizing three different objectives cannot converge to a coherent policy. The CognitionNode, which trains from `breakdown.total` ([cognition_node.py](../core/algorithms/cognition_node.py#L494)), adds a 4th reward perspective.

### B.4 Reptile: Direct PPO Weight Manipulation

`ReptileMeta.meta_step()` ([reptile_meta.py](../core/algorithms/reptile_meta.py#L253)) **directly modifies PPO's network weights** via state_dict interpolation:

```
snapshot = deepcopy(model.state_dict())
→ K=5 inner PPO updates per scenario (3 scenarios from 18 profiles)
→ Reptile interpolation: θ ← θ + ε×(θ' - θ)
→ load_state_dict(modified_weights)
```

outer_lr anneals 0.1→0.01 over 5000 steps. **ON by default** (`FF_REPTILE_META=True`, Phase 50).

When active, Reptile mutates PPO weights at episode boundaries. The Phase 50 inner loop uses **per-scenario buffer isolation**: each scenario configures the environment with scenario-specific ports/services/phase, collects fresh rollout data, runs `ppo.update()` on that isolated buffer, then restores the original buffer. Optimizer state (Adam momentum/variance) and EMA weights are snapshot/restored around the meta-step to prevent state corruption.

### B.5 Distillation Pipeline

```
Mentor call → MentorTrace → TeacherTrace → BCSample → BCBuffer (cap: 2000)
                                                           │
                                                    PPO BC Loss (component #5)
```

TeacherTrace records teacher-student divergence in `decide()` at [smart_coach.py](../core/training/smart_coach.py#L3629-3660). The BCBuffer feeds PPO's BC loss component, creating a supervised signal from LLM mentor decisions. **This couples PPO's policy to LLM quality** — if the mentor gives bad advice, PPO's BC loss actively pushes it toward bad actions.

### B.6 Signal Flow Summary Diagram

```
                  ┌─────────────────────────────────────────────────────┐
                  │              SmartOrchestrator._run_step()          │
                  │                                                     │
                  │  ┌──── RND computes intrinsic ──────┐              │
                  │  │                                    │              │
                  │  │  ┌── SmartCoach.decide() ──────────┤              │
                  │  │  │  SAC shadow (passive)           │              │
                  │  │  │  DDQN macro (soft filter)       │              │
                  │  │  │  12-level cascade (PPO at #15)  │              │
                  │  │  └─────────┬───────────────────────┘              │
                  │  │            │ chosen command                       │
                  │  │            ▼                                      │
                  │  │  CyberEnvironment.step()                         │
                  │  │            │                                      │
                  │  │            ▼                                      │
                  │  │  SmartCoach.record_result()                       │
                  │  │  ├─ RewardCalculator → breakdown                 │
                  │  │  ├─ PPO: store_transition(normalize(total))      │
                  │  │  ├─ SAC: store_transition(total+rnd) → update()  │
                  │  │  ├─ DDQN: store_transition(macro_reward) → update()│
                  │  │  └─ CognitionNode: observe(total)                │
                  │  │                                                   │
                  │  └── RND: update(state) ─── predictor training      │
                  │                                                     │
                  │  Episode end:                                        │
                  │  ├─ PPO: update(last_value) — 9-component loss      │
                  │  ├─ CognitionNode: end_episode() → SAC.update()     │
                  │  ├─ Reptile: meta_step() → mutate PPO weights       │
                  │  ├─ HeldOutEval: eval(policy) (if scheduled)        │
                  │  └─ ScheduleCoupler: update(maturity)               │
                  └─────────────────────────────────────────────────────┘
```

---

## Section C — Schedule & Maturity Coupling

### C.1 ScheduleCoupler Design

`ScheduleCoupler` ([schedule_coupler.py](../core/training/schedule_coupler.py)) coordinates 5 annealing targets from a single maturity signal via cosine interpolation:

1. Mentor call rate (via MentorPolicy)
2. LLM prior alpha (via LLMPolicyBridge)
3. PPO entropy_coef
4. BC coefficient
5. Teacher KL coefficient

**coupling_strength = 0.8** — schedules are 80% coupled, 20% independent.

Updated per episode in `_start_episode()` ([smart_orchestrator.py](../core/orchestration/smart_orchestrator.py#L1561)).

### C.2 The Three Entropy Writers — CONFLICT

PPO's `entropy_coef` is mutated by **three independent writers** with no serialization or arbitration:

| Writer | Location | Mechanism | When |
|--------|----------|-----------|------|
| **PPO internal** | [ppo_agent.py](../core/algorithms/ppo_agent.py#L1574) | Cosine schedule: `config.entropy_coef` → `config.entropy_coef_min` × `_entropy_adaptive_multiplier` | Every `update()` |
| **SmartCoach** | [smart_coach.py](../core/training/smart_coach.py#L3177-3183) | Directly mutates `ppo._entropy_adaptive_multiplier` (0.5-2.0 range) based on stagnation/progress | During `decide()`, based on stagnation detection |
| **ScheduleCoupler** | [schedule_coupler.py](../core/training/schedule_coupler.py#L229-230) | Sets `config.entropy_coef` (the BASE value) but only DECREASES it | Episode start |

**Conflict scenario:**
1. ScheduleCoupler reduces `config.entropy_coef` from 0.08 to 0.03 (maturity increasing)
2. SmartCoach detects stagnation → sets `_entropy_adaptive_multiplier = 2.0`
3. PPO internal cosine uses the reduced base (0.03) × boosted multiplier (2.0) = 0.06

The result is incoherent: three systems with different information are all fighting over the same parameter. Worst case: ScheduleCoupler keeps ratcheting the base DOWN (it never increases), while SmartCoach keeps spiking the multiplier UP, causing oscillating entropy that prevents stable convergence.

### C.3 Mentor Rate: Independent of Coupler

The mentor call rate in `decide()` ([smart_coach.py](../core/training/smart_coach.py#L3485-3535)) computes its own `effective_mentor_rate` using per-agent base rates, PPO confidence boost, dynamic floor/ceiling, and AutonomyScheduler suppression. **This is entirely independent of `ScheduleCoupler._apply_mentor()`.**

The ScheduleCoupler does call `mentor_policy.set_maturity()` ([schedule_coupler.py](../core/training/schedule_coupler.py#L174)), but `decide()` computes mentor rate from scratch using episode number, agent role, and PPO entropy — never reading the MentorPolicy's maturity-adjusted rate.

**Dead pathway:** `ScheduleCoupler`'s mentor rate coordination is effectively dead code for the `decide()` path. It only affects code paths that call `mentor_policy.should_call()`directly — which `decide()` does NOT do. The `decide()` method computes its own rate.

### C.4 Maturity Signal Composition

Maturity is computed in `_start_episode()` from:
```
maturity = 0.4×success_rate + 0.3×skill_coverage + 0.2×discovery_efficiency + 0.1×(1-stagnation_rate)
```

**Problem:** `success_rate` is the only component that reflects actual RL performance. `skill_coverage` counts skill library entries (accumulated monotonically), `discovery_efficiency` reflects the reward calculator's assessment, and `stagnation_rate` is a simple counter. The maturity signal is therefore **70% driven by non-RL metrics**, making it a poor proxy for actual policy quality.

---

## Section D — Generalization Analysis

### D.1 HeldOutEval: Insufficient

`EVAL_SCENARIOS` ([heldout_eval.py](../core/evaluation/heldout_eval.py#L39)) contains **only 3 scenarios**:

| Scenario | Target | OS | Max Steps | Difficulty |
|----------|--------|----|-----------|------------|
| eval_web_basic | 192.168.1.100 | Linux (implicit) | 40 | 0.3 |
| eval_ssh_pivot | 192.168.1.100 | Linux (implicit) | 50 | 0.5 |
| eval_full_chain | 192.168.1.100 | Linux (implicit) | 60 | 0.7 |

**Missing entirely:**
- Windows/Active Directory targets
- CTF-style flag hunting
- API-only / cloud targets
- Multi-host lateral movement
- Different IP ranges
- Different service profiles (all 3 use same IP → same service fingerprint)

**Contrast with Reptile:** `reptile_meta.py` defines 18 scenario profiles (different OS families, service configurations), and is **ON by default** (`FF_REPTILE_META=True`, Phase 50). The Phase 50 inner loop configures the environment per-scenario with different ports, services, and phases, collects scenario-specific rollout data in isolated buffers, and runs `ppo.update()` on each. This provides genuine domain randomization at the meta-learning level.

### D.2 Single-Target Training

The entire system trains against ONE target IP in ONE continuous episode. There is no multi-target curriculum, no domain randomization, and no environment variation. The 294 CommandTemplates ([command_registry.py](../core/commands/command_registry.py)) are all substituted with the same `{target}` at runtime.

**Prediction:** The trained policy would collapse on any target with different service configurations, OS, or network topology. The PPO policy has learned state→action mappings that are specific to the training target's response patterns.

### D.3 Reward Overfitting

The reward calculator ([reward_calculator.py](../core/llm/reward_calculator.py)) has **15+ additive reward components** including MS2-specific bonuses. The PPO policy will overfit to the reward function's preferred pathways rather than learning general penetration testing strategies.

---

## Section E — Harmony & Instability Map

### E.1 Top 10 Points of Harmony

| # | Harmony Point | Evidence |
|---|---------------|----------|
| H1 | **DecisionPacket as data bus** | [smart_orchestrator.py](../core/orchestration/smart_orchestrator.py#L4330): Clean dataclass carries RND, SAC, coherence, macro-confidence through entire pipeline. Good separation of data flow from control flow. |
| H2 | **RND pre-compute / post-update split** | Intrinsic computed BEFORE decide (L4290), predictor updated AFTER step (L2458). No circular dependency — clean temporal separation. |
| H3 | **SmartStepContext immutability** | Step context built once (L4180) and passed read-only through the cascade. No mutation during decision. |
| H4 | **Source attribution tracking** | `SmartDecisionResult.source` field traces every decision to its origin (playbook/ppo/mentor/micro_chain/etc.). Enables accurate win-rate tracking. |
| H5 | **Cross-agent dedup** | L4453 prevents duplicate commands from different agents in the same step. Simple and effective. |
| H6 | **Feature flag gating pattern** | All advanced features (#5-9 in PPO loss, MicroChain, PhaseGuided, etc.) are gated by flags with clean OFF paths. System remains functional with everything disabled. |
| H7 | **Anti-repeat guard** | Exact + prefix repeat blocking with per-role replacement pool prevents infinite loops. Multiple threshold levels for increasingly aggressive diversity. |
| H8 | **Reward breakdown structure** | `RewardBreakdown` carries itemized components (base, discovery, phase, efficiency, etc.) enabling per-component analysis. |
| H9 | **SAC shadow isolation** | SAC never influences decisions — cleanest interface possible. It learns from observations without polluting the policy. (Whether this is useful is a separate question.) |
| H10 | **Phase ladder enforcement** | `FF_STRICT_PHASE_LADDER` prevents backwards phase transitions. CyberEnvironment enforces forward-only progression. |

### E.2 Top 10 Points of Instability

| # | Instability Point | Risk Level | Evidence |
|---|-------------------|------------|----------|
| **I1** | **Three entropy writers with no coordinator** | **CRITICAL** | [ppo_agent.py](../core/algorithms/ppo_agent.py#L1574) (cosine schedule), [smart_coach.py](../core/training/smart_coach.py#L3177) (multiplier mutation), [schedule_coupler.py](../core/training/schedule_coupler.py#L229) (base reduction). Three independent systems fighting over one parameter. Can cause oscillating exploration that prevents convergence. |
| **I2** | **PPO buried at cascade level 15** | **CRITICAL** | [smart_coach.py](../core/training/smart_coach.py#L3535): PPO only drives when ALL 14 higher-priority sources fail AND the coin flip falls PPO's way. With mentor rates starting at 0.80 for RedAgent, PPO drives <20% of early decisions. PPO trains on ALL steps but drives <20% — the supervised signals (BC loss, teacher KL) actively pull PPO toward mentor behavior, but PPO never gets to verify its own learning through real execution. |
| **I3** | **Reward signal divergence across learners** | **HIGH** | PPO: normalized total. SAC: raw total + RND. DDQN: macro-shaped with ±3.5 switch penalties. Three learners optimizing three objectives while supposedly coordinating. See Section B.3. |
| **I4** | **`decide()` is 1,100 lines of cascading conditionals** | **HIGH** | L2574-3700: Any single bug in the cascade can silently redirect all decisions. No formal state machine or transition matrix. Untestable in isolation because each level depends on the failure of all prior levels. |
| **I5** | **SAC trains but never drives** | **MODERATE** | SAC updates every step ([smart_coach.py](../core/training/smart_coach.py#L8395)) consuming GPU cycles and memory, but its Q-values are never used for action selection. This is resource waste. The only benefit is populating the DecisionPacket for logging. |
| **I6** | **Reptile double-counts PPO buffer** | **MODERATE** | [smart_orchestrator.py](../core/orchestration/smart_orchestrator.py#L3140): Reptile's `_inner_train` calls `_ppo.update()` on the current buffer. Minutes later, `end_episode_ppo()` calls `ppo.update()` again on the same buffer. Buffer may be cleared by first update, causing second update to be a no-op, or both may consume same data. |
| **I7** | **ScheduleCoupler mentor path is dead code** | **LOW** | ScheduleCoupler writes mentor_policy maturity ([schedule_coupler.py](../core/training/schedule_coupler.py#L174)), but `decide()` computes its own mentor rate from scratch (L3485). The coupled mentor rate is never read in the decision path. |
| **I8** | **109 feature flags = 2^109 configurations** | **HIGH** | [feature_flags.py](../core/feature_flags.py): 109 boolean flags with 3 profiles (CLOUD/DETERMINISTIC/OFFLINE). Only 3 configurations are tested. Interactions between flags are untested. Any non-trivial interaction could produce silent behavioral changes. |
| **I9** | **PPO confidence-based mentor decay creates feedback loop** | **MODERATE** | [smart_coach.py](../core/training/smart_coach.py#L3509): PPO entropy → confidence boost → lower mentor rate → PPO drives more → PPO entropy changes → different confidence boost. Positive feedback loop: if PPO happens to have low entropy (even from degenerate collapse), mentor rate plummets and PPO drives exclusively — but with a collapsed policy. |
| **I10** | **CognitionNode gate is trainable but rarely drives** | **LOW** | [cognition_node.py](../core/algorithms/cognition_node.py#L870): ConfidenceGate trains from reward outcomes every step, but CognitionNode is cascade level 12 — it only drives when 11 higher-priority sources fail AND confidence ≥ 0.5. The gate trains on reward signals for decisions it rarely makes, creating a distribution mismatch between training and evaluation data. |

---

## Section F — Redesign Proposals

### F.1 Proposal: Single Decision Authority

**Problem:** 12-level priority cascade with no formal decision theory.

**Redesign:** Replace the cascade with a **3-tier architecture**:

```
Tier 1: OVERRIDES (non-negotiable)
  - Phase ladder gate
  - Closeout hardcode
  - Anti-repeat guard

Tier 2: POLICY (single authority chooses)
  - ActionArbitrator receives candidates from: PPO, Registry, Skill
  - Arbitrator scores and selects — ONE winner
  - No coin flip, no probabilistic fallthrough

Tier 3: ADVISORY (informs Tier 2 but never drives)
  - MicroChain → adds score feature to Arbitrator input
  - PhaseGuidedLLM → adds phase confidence to Arbitrator input
  - Mentor → provides TeacherTrace for distillation only
  - SAC → Q-values as auxiliary signal
```

**Impact:** PPO learns from ALL its own decisions (on-policy integrity). Mentor influence flows through distillation (BC loss), not through action hijacking. CognitionNode becomes a candidate provider for the Arbitrator, not a cascade level.

### F.2 Proposal: Unified Reward Signal

**Problem:** PPO/SAC/DDQN see different rewards.

**Redesign:**
```python
base_reward = reward_calculator.calculate(state, action, result)
ppo_reward  = normalize(base_reward)
sac_reward  = normalize(base_reward)  # Same normalization!
ddqn_reward = macro_shaping(base_reward)  # Keep shaping but from same base
rnd_bonus   = rnd.compute_intrinsic(state)
# Add RND to ALL learners uniformly
for learner in [ppo, sac]:
    learner.store_transition(reward=learner_reward + rnd_bonus * rnd_coef)
```

### F.3 Proposal: Delete SAC Shadow Mode

SAC trains every step but never influences decisions. Two options:
1. **Delete it** — save GPU/memory
2. **Activate it** — let SAC be a candidate in the Arbitrator with Q-value-based scoring

Current shadow mode is the worst of both worlds: cost without benefit.

### F.4 Proposal: Single Entropy Authority

**Problem:** Three entropy writers fighting.

**Redesign:** Only `ScheduleCoupler` controls `config.entropy_coef`. PPO's internal cosine schedule reads FROM the coupler's state rather than computing independently. SmartCoach's stagnation detection sends a SIGNAL to the coupler ("exploration_boost_requested") rather than directly mutating PPO internals.

```python
# In ScheduleCoupler
def request_exploration_boost(self, reason: str, magnitude: float):
    self._boost_requests.append((reason, magnitude))

def _compute_entropy(self, maturity, boosts):
    base = cosine_schedule(maturity)
    boost = max(b.magnitude for b in boosts) if boosts else 0
    return base * (1 + boost)
```

### F.5 Proposal: Expand HeldOutEval

Add at minimum:
- 3 different target IPs with distinct service profiles
- 1 Windows/AD scenario
- 1 CTF flag-hunting scenario
- 1 multi-host pivot scenario
- Variable max_steps (30/60/100)

This requires simulated environment profiles — not real targets, but parameterized `CyberEnvironment` configurations.

---

## Section G — Missing Metrics & Observability Gaps

### G.1 Metrics That Should Exist But Don't

| Metric | Why Missing Is Dangerous |
|--------|--------------------------|
| **PPO KL divergence from prior checkpoint** | No way to detect policy collapse or catastrophic forgetting |
| **Entropy collapse detector** | entropy_coef < 0.001 with no stagnation → degenerate policy |
| **Reward variance per source** | Cannot tell if PPO-driven steps get higher rewards than mentor-driven |
| **Per-cascade-level hit rate** | Don't know what % of decisions come from each cascade level |
| **SAC Q-value drift** | SAC trains every step — is its Q improving or diverging? No tracking |
| **DDQN macro switch rate** | No metric for how often DDQN changes macro-intent |
| **CognitionNode confidence calibration** | Gate confidence vs actual outcomes — is it calibrated? |
| **Mentor agreement rate** | When PPO→mentor→PPO both choose, how often do they agree? |
| **Curiosity decay curve** | RND novelty should decay as state space is explored — is it? |
| **BC loss / policy loss ratio** | If BC loss dominates policy loss, PPO is just imitating mentor |
| **Gradient norm per loss component** | Which of 9 loss components dominates gradient? |

### G.2 Observability Architecture

The LiveDashboard ([live_dashboard.py](../core/observability/live_dashboard.py)) shows operational metrics but lacks:
- Training signal decomposition (which loss components are active)
- Cross-learner agreement (PPO vs SAC vs DDQN agreement on best action)
- Cascade hit distribution (what % of decisions from each level)
- Entropy trajectory with writer attribution

---

## Final Verdict

### Blunt Assessment

**The system works despite itself, not because of deliberate architectural coherence.**

The 12-level cascade in `SmartCoach.decide()` is not a decision architecture — it's **archaeological stratification**. Each phase of development added a new decision source on top of the existing ones, without removing or consolidating previous layers. The result is a system where:

1. **PPO is nominally the "primary RL algorithm" but drives <20% of early decisions** (RedAgent: 80% mentor rate × additional bypass from 14 cascade levels). PPO trains on outcomes from decisions it didn't make, creating an off-policy training regime for what's supposed to be an on-policy algorithm.

2. **SAC exists but is completely inert** — it trains from other agents' transitions but never influences action selection. It's a warm body consuming resources.

3. **Three systems fight over PPO's entropy coefficient** with no arbitration protocol. This is the single most likely cause of training instability.

4. **The reward calculator produces a unified breakdown, then each learner transforms it differently**, destroying the unified training signal that's the entire point of having a shared reward function.

5. **109 feature flags create a combinatorial configuration space** of ~2^109, of which exactly 3 configurations are tested. The system under test is not the system in production.

### Top 5 Highest-Leverage Changes

| Rank | Change | Effort | Impact |
|------|--------|--------|--------|
| **1** | **Unify entropy control** — single ScheduleCoupler writer, remove SmartCoach and PPO-internal entropy mutations | Low (50 LOC) | Eliminates oscillating exploration, enables stable convergence |
| **2** | **Collapse cascade to 3 tiers** (Overrides → Arbitrator → Advisory) — every non-override source becomes a candidate, not a cascade level | High (500+ LOC) | Restores PPO's on-policy integrity; eliminates 11 bypass pathways |
| **3** | **Normalize reward uniformly** — all learners see same normalized reward + same RND bonus | Low (20 LOC) | Eliminates reward signal divergence; learners can meaningfully compare |
| **4** | **Delete or activate SAC** — shadow mode is pure waste | Low (remove ~200 LOC or add 30 LOC) | Eliminates wasted computation OR adds a useful Q-value signal to decisions |
| **5** | **Add per-cascade-level hit rate + reward-per-source metrics** | Medium (100 LOC) | Finally answers "what's actually driving decisions and how well?" Without this, optimization is blind. |

---

*End of audit. All claims cite file paths and line numbers verified against the codebase as of 2025-07-14.*
