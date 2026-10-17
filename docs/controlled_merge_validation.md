# Controlled recovery merge validation — 17 October 2026

**Decision: PASS for a controlled CPU research merge, conditional on the final PR head's offline CI succeeding and unchanged stack heads.** The owner subsequently authorized testing, fixing and merging when appropriate. This supersedes the initial integration report's blocked decision: the CLI/tool-test coverage gaps now have safe execution paths and the complete suite has run on an independent GitHub-hosted CPU runner. This is not GPU, live-provider, production or general learning certification.

The final merge ledger and post-merge Actions run must be checked before treating delivery as complete. This document is committed before merging, so it does not claim that the PRs have already been merged. Do not run training from a partially merged stack.

## Confirmed repairs preserved

PRs #6–#13 remain the original stacked changes. Follow-up [PR #14](https://github.com/Reckless98/Ariaska_RL/pull/14) preserves and completes them. The [initial hardening report](integration_merge_readiness.md) gives source-pinned line references, before/after evidence, and dispositions for all 33 original review comments.

| Defect | Repair and evidence |
|---|---|
| Reptile overwrote part of a live PPO rollout | Swap a complete temporary RolloutBuffer; bind actual inner behavior contexts; restore outer buffer/environment in finally, and weights/optimizer after failure. Regressions run two real inner updates and then update the preserved outer rollout. |
| Generic successful `[SIM]` results became failure rewards | Carry authoritative success separately from display text; preserve ordinary string storage and copying. Actual legacy orchestration records the correct outcome. |
| Unrestricted SAC and unified pickle loading | Use weights-only loading with a fixed local data allowlist for unified snapshots, restricted legacy container adapters, and no unsafe fallback. Marker-only malicious-global fixtures are rejected without execution. |
| Discarded DDQN proposal received a later action's reward | Clear pending/active macro/Q/confidence state and restore the previous executed macro. All five roles subsequently record an executed deterministic action with no DDQN replay write for the discarded proposal. |
| Checkpoint and outcome reliability | Restore v2 DDQN independently, deep-copy rollback tensors, preserve CUDA RNG with topology validation, restore SAC settings, omit multi-role broadcast aliases, fsync directory replacement, restore retryable preparation state, and reject retries after partial learner commits. |
| Other review-confirmed defects | Correct partial gradient accumulation, refresh critic spectral buffers, isolate standalone SIL, aggregate episode history/flags/TTF, release failed provider-construction capacity, reserve evaluation output directories, and retain complete experiment provenance. |

Two additional PR #14 findings arrived during the final validation pass:

| Review | Classification and fix |
|---|---|
| [4304381114 / P1](https://github.com/Reckless98/Ariaska_RL/pull/14#discussion_r4304381114) | **Confirmed and fixed.** `requirements.txt` and README now require PyTorch >=2.10.0. Both `UnifiedCheckpoint.load` and `SACAgent.load` call `core/checkpoints/safe_loading.py` before deserialization. Unsupported versions or a missing restricted API raise actionable upgrade guidance instead of AttributeError or unsafe fallback. The higher floor also excludes the weights-only vulnerability fixed in 2.10, per the [upstream advisory](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p). The tested lock remains 2.14.1+cpu; a full matrix of every version >=2.10 was not run. |
| [4304381123 / P2](https://github.com/Reckless98/Ariaska_RL/pull/14#discussion_r4304381123) | **Confirmed and fixed.** `SmartOrchestrator.run_training` ranks phase names using `SmartCoach.PHASE_ORDER`, so EXFILTRATION outranks POST_EXPLOITATION regardless of episode order. Both returned metrics and dashboard summary are asserted. |

Eleven new review regressions failed against the pre-fix source. After repair, 68 focused checkpoint/native-integration checks passed, with one physical-CUDA skip. A further missing-API regression was added and is included in the final full suite. Existing supported checkpoint round trips, rollback and continuation behavior remain exercised. No policy, optimizer objective or reward redesign was made in this final compatibility/metrics pass.

## Safe CLI and test coverage

`core/testing/guarded_cli.py` admits only the fixed production entry point `core.training.ariaska_trainer` with a plain-string option allowlist, at most two episodes and three steps. It invokes the current interpreter with `-I`, a clean credential-free environment and isolated temporary working directory. The child installs `core.testing.offline_guard` before importing the trainer. General Popen, sockets, DNS and os.system remain denied in parent and child. The parent fails closed if the child guard bindings have changed. Invalid options/budgets are rejected before process creation.

This is a trusted-code test harness, not an operating-system sandbox for hostile Python. Its only admitted subprocess is that bounded Python interpreter. Application network/tool attempts remain denied. Tool-registry tests mock availability probes and assert classification, unknown-name rejection and caching. They neither launch nor validate external security tools.

The CLI checks now execute the real parser/startup/offline path rather than requiring a nonexistent root `.venv`. The two-episode CLI test verifies all five role step records. This trainer currently uses the legacy Orchestrator; that smoke test is **not** evidence of SmartCoach PPO learning. Separate native SmartOrchestrator integration and numerical tests provide that evidence. A mentor-free run may correctly contain no mentor calls; its test does not manufacture them.

## Test evidence and reproducibility

| Check | Actual result and source |
|---|---|
| Complete local suite before the last two review fixes | **3,306 passed, 6 skipped, zero failures**, five warnings, 65.44 seconds. |
| Independent hosted suite before the last two review fixes | **3,306 passed, 6 skipped, zero failures**, five warnings, 67.20 seconds. [Actions run 39067576174](https://github.com/Reckless98/Ariaska_RL/actions/runs/39067576174), job 117498320895, success. |
| Hosted tested commit and tree | GitHub test merge `ee4ea8a8e71fdd185b9ca3723080698cbcb755e6`, head `9d04524222d3e862031c88b420204e9cb35df4b1`, tree `efa1cc5ab475108b7b519096e357c180bf899470`. Local tree, cumulative merge-tree against master and downloaded CI artifact all matched. |
| Final complete local suite | **3,318 passed, 6 skipped, zero failures**, five warnings, 65.69 seconds; 12 denied network/tool attempts and eight admitted isolated Python CLI runs. JUnit and skip reasons are retained in the evidence summary. |
| Final PR-head CI | Required to succeed before merge. Inspect the Offline recovery check on the actual final head; the older run above does not cover the later compatibility/phase repairs. |
| Mentor-free separate interpreter | All five roles completed two PPO updates each; zero provider requests and zero forbidden LLM-library imports. Repeated locally and independently in hosted CI. |
| PPO numerical consistency | All five role output dimensions × three phase heads; masks/biases/priors and phase gates exercised. Unchanged-parameter ratios 0.9999997616–1.0000004768 and zero error against an independently conditioned categorical oracle. |
| Executed attribution / Reptile / resume | Full suite includes discarded/replaced proposals for all roles, two real Reptile inner PPO updates plus restored outer update/failure rollback, and deterministic multi-episode save/load/continued updates. Fresh final-code verification also restored all 15 seed/role snapshots from update 40 and continued to 41; its output matched the earlier evidence. All 15 numerical role/head cases were rerun separately with zero guard denials. |

Hosted artifacts retain JUnit XML, exact pip freeze, commit/tree and mentor-free JSON for 14 days. Artifact 12113727219 has SHA256 `6077bbcfe3ad861671011e91cbb4c282a43c1998344188366b570cb2e1b3f7dd`; its bytes were downloaded and verified, and its JUnit reports 3,312 tests, six skipped, zero errors/failures. Durable summaries and exact skip reasons are under [evidence/merge_2026-10-17](evidence/merge_2026-10-17).

The pinned `.github/workflows/offline-recovery.yml` installs `requirements-recovery.lock` on Ubuntu 24.04/Python 3.12, records source identity, runs the whole guarded suite and then the fresh mentor-free probe. Official actions are pinned to full SHAs. Source checkout excludes historical models/checkpoints/data/logs/runs, matching the audited isolated checkout; synthetic tests create their own state. Thus historical research binary compatibility is not silently asserted. An initial invalid workflow created no jobs; a superseded run was cancelled. Neither counts as passed validation.

Reproduce in an isolated source checkout with the pinned dependencies:

```bash
export ARIASKA_DRY_RUN=1 ARIASKA_LIVE_MODE=false ARIASKA_MENTOR_BACKEND=none
export ARIASKA_DATA_DIR="$(mktemp -d)"
PYTHONPATH=. python -m pytest -p core.testing.offline_guard tests --timeout=90 -q --junitxml=pytest.xml
python scripts/verify_mentor_free.py --output mentor-free.json
```

The six explicit skips are: unavailable physical CUDA, two key-dependent live-provider cases, absent optional Optuna, a no-generated-SSH-hypothesis conditional, and an old mentor-gate conditional. No failures were relabeled as skips. Provider mocks and mentor outcome regressions run separately; these do not validate live provider behavior. Physical-CUDA generator replay now has an actual optional hardware test, in addition to mocked contract tests, and is skipped honestly on CPU hosts.

## Learning assessment and retained limitations

Two separate three-seed experiments previously produced byte-identical training curves (600 rows), held-out results (7,680 rows) and metrics. Trained mean accuracy was **96.18% ±3.58 percentage points across seed means**, versus frozen 49.31%, random 50.52% and constant-registry 48.76%. All five role means improved; Shadow seed 11 remained 73.44%. The exact experiment code remains `878586cffa2354cec4c843ba344309e4ce786845`, with its original manifest. This final test/CI/runtime-minimum/summary pass is not relabeled as a fresh learning experiment. It does not change those policy/reward/rollout computations. Details and hashes remain in the integration report.

The evidence supports useful learning by each independent PPO on this declared synthetic two-choice task. It does not prove default-arbitration learning, general cybersecurity capability, reward-learning efficacy, DAgger consumption, Reptile efficacy, every advanced auxiliary, or bitwise mid-environment resume. Greedy held-out inference uses base policy/mask/head 0, not every training-time bias/prior.

Checkpoint capture/restore still requires quiescent training and RNG restoration is process-global. Non-cooperative optional provider SDK threads may exhaust the four bounded slots; capacity fails closed, and operators must restart an isolated provider worker if I/O never returns. Preparation failures are retryable, but partial learner commits require checkpoint recovery. The general optional dependency set and physical CUDA are not validated by the pinned CPU CI. These are explicit operating/research boundaries, not reasons to weaken the confirmed fixes.

The 33 original inline threads and two new threads are classified in the reports; older branches naturally do not contain later fixes. No reviewer approval has been fabricated or dismissed. Sourcery exhausted its review budget, which is not an approval. Codex review status must be checked again on the final head. No confirmed unresolved P1 or learning-integrity defect is accepted as merge-ready merely because unit tests pass.

## Preserved stack and merge procedure

The unchanged original heads are:

| PR | Branch | Head SHA |
|---|---|---|
| [#6](https://github.com/Reckless98/Ariaska_RL/pull/6) | recovery/runtime-gateway | 9b6b116f55c174368e08fe425721600626908e6b |
| [#7](https://github.com/Reckless98/Ariaska_RL/pull/7) | recovery/ppo-contracts | a01c40acfa9479ed3623f5efedfc3cdec7ddb43d |
| [#8](https://github.com/Reckless98/Ariaska_RL/pull/8) | recovery/executed-transitions | c2caadcb8d9dbc9f3a4fc7a1eeef0db7da160b6e |
| [#9](https://github.com/Reckless98/Ariaska_RL/pull/9) | recovery/role-checkpoints | 4b94303275e6873555a8247a8351761ee269f682 |
| [#10](https://github.com/Reckless98/Ariaska_RL/pull/10) | recovery/offline-lifecycle | 86a55c68e8cbd2656caae1c85c775415394fb69b |
| [#11](https://github.com/Reckless98/Ariaska_RL/pull/11) | recovery/reliability-evidence | 688a0efd07825fd3e0f32aba52eb7f27c7b98106 |
| [#12](https://github.com/Reckless98/Ariaska_RL/pull/12) | recovery/proposal-lifecycle | 74ba71e025d6d4d3955fc15ce6ed7fdcc3edd59b |
| [#13](https://github.com/Reckless98/Ariaska_RL/pull/13) | recovery/research-validation | 1835612ea2687a1868c440090544ee7a2b6b3a1c |
| [#14](https://github.com/Reckless98/Ariaska_RL/pull/14) | recovery/integration-hardening | Latest final head shown on PR; includes all original heads above. |

Use merge commits in order **6 → 7 → 8 → 9 → 10 → 11 → 12 → 13 → 14**. After each parent merge, retarget the next PR to master, verify its expected head, base SHA and diff, then merge. Preserve branches and ancestry; no squash, destructive force-push or re-creation is needed. Stop if a head drifts, a confirmed defect appears or GitHub rejects a required gate. Verify final master tree equals the tested cumulative candidate, and require the master push CI to pass. Repository rules must not be weakened to complete delivery.

The resulting project is suitable for further **bounded CPU offline research** with these limitations. It remains experimental and is not production-ready.
