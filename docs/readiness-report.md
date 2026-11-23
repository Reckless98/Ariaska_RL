# Open-source readiness report

**Review date: 23 November 2026. Outcome: documentation and review readiness improved;
open-source relicensing, release and application eligibility remain unresolved.**

Scope: `Reckless98/Ariaska_RL`, baseline
`d2f72feeaf33b24e4f5d504aae9d5f8215cf98c2`, default branch `master`.
Work is isolated on `research/oss-readiness-2026-11-23`. The existing developer
checkout contains unrelated edits and was not reset, stashed or overwritten.
The task authorizes repository audit, offline review and a draft PR; it does not
authorize live-target operations, release publication or a default-branch merge.

## Verified GitHub baseline

| Field | Before this PR |
| --- | --- |
| Visibility | Public |
| Stars / forks | 1 / 0 |
| Pull requests | 14 total, 12 merged; all closed |
| Tags / GitHub releases | 0 / 0 |
| Reachable commits | 853 |
| Last push | 17 October 2026, 21:02:48 UTC |
| Default branch | `master` |
| License API classification | Other / `NOASSERTION` |
| Latest baseline offline CI | Success, [run 39069273198](https://github.com/Reckless98/Ariaska_RL/actions/runs/39069273198), source `d2f72fee` |
| Private vulnerability reporting | Disabled at audit time; SECURITY.md supplies the maintainer's existing public email |

Sources: GitHub repository, pulls, releases, tags and Actions APIs, plus local Git
history after fetching `origin/master`. These are a dated pre-change snapshot.
New commits and this PR will change activity counts and timestamps. Downloads,
external user counts and academic citations were not verified.

## Implemented changes

- A concise research README with architecture, source links, bounded evidence and
  explicit limitations. The complete prior README remains in an annotated history
  archive; older contribution notes point to the current guide.
- Root contribution, conduct, security, maintainer, authorship and citation files;
  issue and PR templates; a research roadmap and unreleased changelog.
- A [licensing decision](licensing-review.md), [asset inventory](third-party-inventory.md)
  and [dependency license metadata inventory](dependency-license-inventory.json).
  The current LICENSE and runtime dependency lock remain unchanged.
- A [reproducibility guide](reproducibility.md), machine-readable verification
  evidence, and documentation/evidence validation in CI. The existing full CPU
  job remains for runtime changes; this review's documentation route explicitly
  skips it. Workflow action revisions remain pinned and update checks are proposed
  through Dependabot without automatic merge.
- An unpublished [research-preview proposal](research-preview.md), proposed version
  `v0.1.0-rc.1`, and [English application drafts](openai-application.md), each capped
  at 500 characters and transparent about the license and limited adoption.

No application runtime, algorithm, reward, checkpoint or target-facing behavior
was changed. No models or data were deleted, no Git history was rewritten, no
license was replaced, and no release or application was submitted.

## Actual validation and its limits

The [verification record](evidence/readiness_2026-11-23/verification.json) preserves
counts, versions, source identifiers and advisory IDs. The fresh Python 3.12.12
CPU environment matched all 67 entries in `requirements-recovery.lock`; dependency
consistency checking passed.

| Check performed in this task | Result | Scope |
| --- | --- | --- |
| Isolated mathematical and protective tests | 47 passed, 0 failed, 0 skipped, 4.383 seconds in JUnit | N-step returns, PPO numerical contracts and rejection-only guarded-CLI tests; one expected denial, zero launched CLI children |
| Historical evidence recomputation | Earlier CSV aggregate 68.2161%; later saved-seed aggregate 96.1784% | Earlier study: 7,680 raw rows. Later study: metrics/seed summaries and hashes, not raw CSV reconstruction |
| Workflow syntax and pinned actions | actionlint v1.7.7 passed; three action tags matched their pinned SHAs | Static workflow validation, not an execution of the full job |
| CI routing fixtures | 19 cases passed | Trusted-base routing, forged head scripts, bootstrap integrity, runtime changes and invalid-scope failures |
| Negative evidence fixtures | 19 cases rejected as expected | Empty/duplicate baselines and seeds, malformed records, changed identities and lock metadata |
| Dependency advisory lookup | Exit 1: 17 advisory entries across 4 pinned packages | 66 packages audited against PyPI; exact `torch==2.14.1+cpu` unresolved by that source |
| Redacted source hygiene heuristics | 642 files checked; 3 example-like matches, no confirmed credential | 2,104 paths excluded/absent; not a history, binary, corpus or comprehensive privacy scan |

The 47-test run is not the full repository suite. The assisting model declined
running the full suite and mentor-free production-orchestrator probe because they
include offensive workflow integration. This was a model refusal, not a failing
test, missing credential, user denial or CI result. Those operations were not
retried through another model, tool or CI route. New training, live providers,
real targets, physical CUDA and stored checkpoint loading were not performed.

The successful October full-suite evidence remains historical. Its controlled
local report states 3,318 passed and six explicit skips; the post-merge hosted run
was separately rechecked for success. Documentation-only CI must not be described
as a fresh full-suite pass. See [reproducibility](reproducibility.md) for exact commands.

## Remaining blockers

1. **Relicensing:** contributor and third-party provenance is not established for
   the full tree. Walkthrough-derived corpora, embedded knowledge, generated data,
   weights and executable artifacts need exact rights decisions. Earlier GPL/AGPL
   license texts do not clear later additions.
2. **Release hygiene:** 350 checkpoints account for approximately 4.41 billion
   logical bytes. Sparse checkout and ignore rules do not remove them or other
   unresolved assets from GitHub source archives. A reviewed whole-tree candidate
   and disposition plan are needed.
3. **Dependencies:** the audit reports entries for certifi, filelock,
   opentelemetry-api and setuptools. Suggested fixed versions require compatibility
   verification before changing the recovery environment. The CPU PyTorch build
   and 32 optional declared packages lack complete advisory/rights coverage here.
   The pinned environment is a reproduction baseline, not a security recommendation.
4. **Research evidence:** the later study's raw CSVs are not tracked despite being
   referenced by historical reports. Recover and verify them or run a new declared
   experiment with its own provenance. Generalization, default arbitration,
   auxiliary-learning efficacy, GPU and live-provider behavior remain unverified.
5. **Adoption and program fit:** there is little measured community adoption, no
   published release and no current open-source license. The program's invitation
   to explain ecosystem importance does not guarantee acceptance or excuse inaccurate
   claims.

## Before and after

| Area | Baseline | Review proposal |
| --- | --- | --- |
| Landing page | Long phase/reference document with a misleading “Private” badge | Short research entry point; correct source-available status; history preserved |
| Governance | Historical nested contribution notes; no complete root set | Clear current contribution/disclosure/maintainer policies and citation metadata |
| Licensing | Broad custom grant with undocumented third-party provenance | Explicit blocker register and dependency/asset inventories; no premature relicensing |
| Evidence | Strong but mixed historical reports | Clear separation of historical full-suite/study evidence from current limited checks; missing raw-data caveat |
| CI | Guarded full CPU recovery job | Preserved full job plus scoped metadata/evidence validation and update configuration |
| Release/application | No published version; unsupported eligibility assumptions possible | Concrete unpublished preview gates and honest, length-checked application drafts |

## Next decisions

Review this PR's documentation and CI changes. Resolve contribution rights and the
asset manifest with the relevant rightsholders before authorizing a license change.
Address the dated dependency findings with compatible updates and appropriate
regression verification. Recover the missing study artifacts or publish a separately
identified reproduction. Only then review the exact candidate for tagging/release
and refresh the application evidence. This report grants no merge or publication approval.
