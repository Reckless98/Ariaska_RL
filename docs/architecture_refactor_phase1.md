# Ariaska_RL — Refactor Phase 1: RL Core Integrity & Optional Mentorship

> Historical document. Current implementation contracts, recovery measurements and limitations are recorded in [recovery_validation.md](recovery_validation.md); older counts and architecture claims are not current verification.

**Date:** 2026-10-17
**Branch:** `claude/dreamy-ptolemy-tzek6r`
**Scope:** general-purpose reinforcement-learning core, learning-signal integrity,
optional/interchangeable LLM mentorship, offline regression tests, documentation.
**Out of scope (untouched):** domain-specific decision logic, command-template rendering,
target-specific playbooks, execution paths. One change that touched template rendering
was reverted before delivery (see §8).

---

## 1. Project identity (authoritative)

* Ariaska has **five custom agents** — Scout, Red, Blue, Shadow, Orion — implemented as
  software components with their own roles. Their intelligence is the learned RL stack:
  a per-coach **PPO Actor-Critic** (`core/algorithms/ppo_agent.py`), the per-agent
  `SmartCoach` decision pipeline with `DecisionCore` weighted arbitration, registry /
  playbook / skill-library candidates, and auxiliary learners (SAC shadow, DDQN macro,
  RND curiosity, CognitionNode).
* An LLM is only ever a **mentor** (Mentor2Apprentice): optional, selectively consulted,
  budget-controlled, interchangeable. Local Qwen fine-tunes are a *local substitute for a
  cloud mentor*; they are not the agents.
* **Mentor-free operation is a first-class mode.** Owner-reported baseline (not reproduced
  in CI): the PPO policies completed the two Metasploitable lab targets without mentor
  assistance. All changes below keep action selection on the mentor-free path unchanged.

## 2. Method

1. Read the live implementation (SmartCoach, SmartOrchestrator, PPO, learners, mentor and
   gateway modules, checkpoints, tests) and re-verified the two prior audits
   (`docs/architecture_audit.md`, `docs/structural_audit_post_consolidation.md`).
   Several prior findings are now **fixed** in code: `DecisionCore.arbitrate()` is called
   (`smart_coach.py::_p1_arbitrate_decision`), `UnifiedRewardPipeline.total` is the base
   scalar for every learner, `IntegrityCheck.check_runtime()` is invoked.
2. Ran 13 read-only subsystem readers; 10 completed before the session was switched to
   delivery mode. Findings below are split into **verified by hand** (cited and acted on)
   and **reader-reported** (recorded for follow-up, not independently verified).
3. Implemented the smallest behaviour-preserving fixes with regression tests; no PPO loss,
   reward constants, checkpoint formats or learning schedules were changed.

## 3. Findings verified by hand

| ID | Finding | Evidence | Status |
|----|---------|----------|--------|
| GW-01 | `GPTManager.is_offline()` returned True whenever no *local* server was detected, so a configured cloud provider could never serve as mentor; `is_local_only()` was hard-coded `True`, permanently selecting the CPU fast paths and the 1-call/5-steps mentor rate limit for every backend. | `core/gpt_manager.py` (pre-patch `is_offline`, `is_local_only`) | **Fixed** (§5.1) |
| GW-02 | Cloud detection knew only OpenAI-compatible endpoints; adding a provider meant editing the gateway. The strict-mode (`require_llm`) check ran before any provider was probed and could never pass. | `core/gpt_manager.py::_detect_cloud_provider`, `__init__` | **Fixed** (§5.1) |
| LS-01 | `_p1_arbitrate_decision` samples PPO (advisory "ppo") and then calls `_decide_from_registry`, whose PPO-first phase samples PPO **again** and overwrites `_ppo_pending`. When the first PPO advisory wins, the executed command and the stored `(state, action, log_prob, value)` tuple disagree — an on-policy integrity bug. | `smart_coach.py` arbitration step 2 vs. `_decide_from_registry` PPO phase | **Fixed** (§5.2, bookkeeping only) |
| LS-04 | `_ppo_pending["teacher_action"]` was PPO's **own** sampled action, so the ranking/KL "teacher" losses never saw the mentor's action. | `smart_coach.py::_ppo_select_command` | **Fixed** when a mentor template maps to an action; mentor-free behaviour unchanged (§5.2) |
| LS-05 / LS-17 | TeacherTrace was created and stored in `BCBuffer` at decision time, before the outcome was known; post-hoc quality fields were never written; rejected mentor answers left no artifact. | `smart_coach.py` arbitration step 10, `_decide_with_mentor` fallbacks | **Fixed** (§5.3) |
| LS-07 | SkillCard auto-promotion and `LLMPolicyBridge.record_step_outcome` lived inside `if self._ppo_pending is not None`, but `FF_PPO_REWARD_ATTRIBUTION_FIX` (default ON) clears `_ppo_pending` whenever a non-PPO source wins — so mentor-won steps never reached them. | `smart_coach.py::record_result` | **Fixed** (§5.4) |
| LS-09 | `end_episode_ppo` returned early on an empty trajectory, skipping the `finally` block that resets DDQN/CognitionNode state, signals memory/bus episode end and runs consolidation. | `smart_coach.py::end_episode_ppo` | **Fixed** (§5.5) |
| LS-10 | The postmortem transcript was built after `_ppo_trajectory.clear()`, so it was always empty. | same | **Fixed** (§5.5) |
| F3 | Results produced by `_decide_from_registry` and the Blue/Orion registry helpers had `source == "unknown"` inside `decide()`; attribution, win-rate feedback and the override-penalty table keyed on `"registry"` never matched. | `smart_coach.py` registry constructors | **Fixed** (label only, §5.6) |
| M-01 | `SmartCoach._init_smart_mentor` only constructed a `SmartMentor` when `GPTManager` exposed `async_client`/`client`; the current gateway exposes neither, so in production no `SmartMentor` existed and the mentor → TeacherTrace → BCBuffer → ledger chain was inert for every backend. | `smart_coach.py::_init_smart_mentor`, `core/gpt_manager.py` | **Fixed in R1.1** (`core/llm/gateway_chat_client.py`) |
| AL-01 | `record_result` reads `self._maturity_controller._state` (attribute is `state`), so the maturity-scaled RND factor always falls back to 1.0. | `smart_coach.py::record_result` | **Documented, not fixed** |

## 4. Reader-reported findings (not independently verified)

Recorded verbatim by subsystem for follow-up; treat as hypotheses with file anchors.

* **Decision authority:** four in-`decide()` early returns and four orchestrator pre-decide
  paths bypass arbitration; post-arbitration mutators overwrite the winner; advisory names
  diverge from `SmartDecisionResult.source` for several producers; producers mutate shared
  "used command" state before arbitration; `web_followup` advisory slot is dead.
* **Learning signals:** CognitionNode.observe writes a second PPO/SAC transition per step;
  three immediate PPO penalty stores bypass the trajectory and GAE ordering; the DAgger
  buffer is write-only with a degenerate dedup key; SAC shadow stores its own un-executed
  action; HybridMemory stores `next_state == state`; PPO importance ratios are computed
  without the sampling-time mask/bias/prior (PPO-core change — owner sign-off required).
* **Orchestrator lifecycle:** single continuous episode makes per-episode authorities fire
  at most once per process; CLI `--steps` is capped by `dynamic_max_steps`; no checkpoint
  autosave during the continuous run; MetaRLController is an unguarded entropy/RND writer.
* **PPO core / persistence:** `RolloutBuffer` pads teacher distributions with a hard-coded
  action dimension (latent crash); `PPOConfig` (with a live `BCBuffer`) is pickled into
  checkpoints; scheduler restore pins the learning rate; several checkpoint envelopes with
  no single authority; SAC/RND/GMC state never persisted.
* **Gateway:** `gpt_request` ignores `LLMResponse.error`; cache key ignores provider/model;
  response-cache and budget authorities are split; dashboard is blind to backend and cost
  (partially addressed by `get_stats()["mentor_backend"]` and cost tracking in §5.1).
* **Agents:** agent `act()/learn()` are not on the production path (SmartCoach decides);
  one agent hook makes a hidden per-step LLM call used only for telemetry; agent prompts
  bind identity to a model family.
* **Tests:** no direct tests for `decide()` stage ordering; real `GPTManager()` in tests
  performs hardware/network probes; orchestrator tests write into repo data directories.

## 5. Changes implemented

### 5.1 Mentor backend seam (`none | local | cloud | hybrid`)

* `core/llm/mentor_backend.py` (new) — pure resolver `resolve_mentor_backend(requested,
  local_available, cloud_available)`; `ARIASKA_MENTOR_BACKEND` (`auto` default).
  `auto` is behaviour-preserving: local present → `local` (even when cloud is also present,
  per the Phase 58 decision), nothing → `none`, and the previously impossible case cloud
  only → `cloud`. `hybrid` / `cloud`-with-local must be requested explicitly.
* `core/llm/cloud_provider_registry.py` (new) — lazy provider factories; nothing is
  imported unless its credentials exist; `ARIASKA_CLOUD_PROVIDER` preference.
* `core/llm/anthropic_provider.py` (new) — optional `LLMProvider` over the Anthropic SDK
  (lazy import). System messages → `system`, JSON mode → instruction, no sampling
  parameters, `max_tokens` floor so adaptive thinking cannot starve the answer,
  `anthropic-workspace-id` header from `ANTHROPIC_WORKSPACE_ID`, server-side refusal
  fallbacks with automatic downgrade if rejected, refusals returned as *error* responses
  (never exceptions), auth errors disable the provider, rate limits cool down, per-call
  USD cost from a static price table. Default model `claude-opus-5-5`, effort `medium`.
* `core/gpt_manager.py` — backend-aware detection (skips probes for excluded backends),
  `is_configured/is_offline/is_local_only` derived from the resolved backend,
  `mentor_backend` property, `get_mentor_backend_state()`, `_install_cloud_provider()`,
  provider-declared outer timeout cap, cloud-only routing branch, real cost accounting
  (`cost_by_model`, episode/cumulative USD, `get_cost_summary()`), strict-mode check moved
  after detection, backend-aware startup log.
* `core/llm/model_router.py` — `cloud_system3` option (hybrid: System 3 → cloud).
* `core/testing/fake_gpt_manager.py` — `is_local_only()`, `mentor_backend`,
  `get_mentor_backend_state()` so the fake mirrors the real surface.

### 5.2 PPO pairing and teacher index

* `_restore_ppo_pairing()` — after arbitration, when the winner **is** the original PPO
  advisory, the tuple that produced it is restored into `_ppo_pending`. Action selection is
  unchanged; only the stored sample is corrected. Corrections are counted
  (`MentorLedger.ppo_pairing_restored`).
* When a mentor answer maps to a valid action index, `_ppo_pending["teacher_action"]`
  carries that index; otherwise legacy behaviour is kept.

### 5.3 Outcome-aware TeacherTrace + MentorLedger

* `core/reasoning/teacher_trace.py` — optional fields `mentor_won`, `actual_reward`,
  `produced_discovery`, `teacher_was_correct`, `finalized` and `finalize_outcome()`.
  Defaults keep old serialized traces loadable; `BCSample` is unchanged.
* `core/training/mentor_ledger.py` (new) — proposals, rejections by reason, accepted,
  won/lost, correct-when-won, reward statistics for mentor-won vs. policy steps per phase,
  `summary()` exposed through `SmartCoach.get_stats()["mentor_ledger"]`.
* `SmartCoach`: the TeacherTrace is held as `_pending_teacher_trace` and stored into
  `BCBuffer` in `record_result` **after** finalization. Default weight is unchanged
  (confidence × 1.0). `FF_MENTOR_OUTCOME_WEIGHTING` (default OFF) halves the weight of
  mentor-won demonstrations that produced neither reward nor discovery.

### 5.4 Hooks that must see every executed step

SkillCard auto-promotion and `LLMPolicyBridge.record_step_outcome` moved out of the
`_ppo_pending` guard; the bridge now receives the unified step scalar.

### 5.5 Episode finalization

`end_episode_ppo` always executes its `finally` side effects (empty-trajectory case
returns from inside the `try`); the postmortem transcript is built before the clear.

### 5.6 Attribution

Every result constructed in `_decide_from_registry` and the Blue/Orion registry helpers
declares `source="registry"` (label only; no command or selection change).

## 6. Behaviour-preservation notes

* Mentor-free path: identical action selection; identical PPO buffer contents except that
  the pairing fix stores the sample that actually produced the executed command.
* Existing local-only setups: `auto` still resolves to `local`; `is_local_only()` still
  `True`; fast paths and rate limit unchanged.
* Hybrid keeps the CPU fast paths by default (`ARIASKA_HYBRID_FAST_PATH=1`).
* No PPO loss, reward constant, schedule or checkpoint format changed.

## 7. Tests and results

New tests (all deterministic, offline, no network):

| File | Covers | Result |
|------|--------|--------|
| `tests/llm/test_mentor_backend.py` | resolver matrix, GPTManager wiring for none/local/cloud/hybrid, cost accounting, strict mode, FakeGPTManager contract | pass |
| `tests/llm/test_anthropic_provider.py` | detection without key/SDK, request translation, refusal/error/cooldown handling, workspace header, fallback downgrade, registry preference, gateway end-to-end with a fake client | pass |
| `tests/training/coach/test_learning_signal_contract.py` | TeacherTrace outcome fields, MentorLedger, registry source, PPO pairing helper and end-to-end, mentor intervention measured, rejection accounting, episode-end cleanup, transcript ordering | pass; 1 skip |

The remaining skip (`test_mentor_intervention_is_measured`) occurs when the coach's mentor
gate does not open under the deterministic test profile; the ledger/BCBuffer contract is
still exercised through `test_rejected_mentor_answers_are_counted` and the unit tests.

Baseline before any change (full suite, `ARIASKA_DRY_RUN=1`): 3,083 collected, 3,018
passed, 42 failed, 14 errors, 3 skipped — every failure/error traced to a missing
`huggingface_hub` module in the fresh environment; after installing it the 258 tests in the
affected files passed. Collection after this change set: 3,150 tests. Focused re-runs after
the changes are reported in the pull request description with exact numbers.

## 8. Known limitations and deferred items

1. **SmartMentor construction (M-01) — resolved in follow-up R1.1.** `_init_smart_mentor` now builds the
   mentor over `GPTManager.gpt_request` via `core/llm/gateway_chat_client.py` for resolved
   `local` / `cloud` / `hybrid` backends. Validated with offline tests (adapter mapping, async path,
   construction matrix, SmartMentor end-to-end with a stubbed gateway); not yet exercised in a live run.
2. **Live cloud mentor test not performed.** The provider is validated with mocked SDK
   responses only. A user-scoped Anthropic key requires `ANTHROPIC_WORKSPACE_ID`; use a
   workspace-scoped key or set that variable in a later authorized session.
3. **PhaseGuided advisory command rendering (F12)** — the advisory still carries a template
   name as its command. A fix was implemented, then **reverted** because it touches
   command-template rendering, which is outside this session's scope. Deferred.
4. Reader-reported findings in §4 are unverified and unaddressed.
5. `AL-01` (`_state` vs `state`) left as-is: fixing it changes the RND scale every learner
   sees and needs a measured run.

## 9. Compatibility

* Checkpoints: untouched (no PPO state_dict, config or format change).
* `TeacherTrace`: additive optional fields; `from_dict` ignores unknown keys.
* Environment contract: additive (`ARIASKA_MENTOR_BACKEND`, `ARIASKA_CLOUD_PROVIDER`,
  `ARIASKA_HYBRID_FAST_PATH`, `ANTHROPIC_*`, `ARIASKA_ANTHROPIC_*`).
* `GPTManager` public surface: additive; `is_offline()` now returns False for a cloud-only
  configuration (previously always True without a local server).
* `requirements.txt`: `anthropic` is listed as an optional, commented dependency.

## 10. Roadmap (prioritized)

1. ~~Build `SmartMentor` over `GPTManager.gpt_request`~~ (done, R1.1); add a coach-level
   mentor-free inertness test and a live-backend smoke run.
2. Honour `LLMResponse.error` in `gpt_request` (typed outcome instead of fallback strings).
3. Route the immediate PPO penalty stores through the trajectory; gate CognitionNode's
   duplicate PPO/SAC writes behind a flag (default off) with a measured A/B.
4. Fix `RolloutBuffer` teacher-distribution padding to the real action dimension.
5. Fix `AL-01` and decouple maturity inputs from the LLM budget component.
6. Canonical decision-source names; record bypasses in DecisionCore/HarmonyMetrics.
7. Checkpoint hardening: config sanitisation, architecture fingerprint, atomic saves.

## 11. Items flagged for separate review

* `source="registry"` labels added inside `_decide_from_registry` (metadata only).
* `tests/integration/conftest.py` minimal-coach fixture gained attributes required by the
  episode-end path.
