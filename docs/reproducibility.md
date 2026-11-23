# Reproducing the research checks

The November 23 review verified 47 CPU math and execution-guard tests and recomputed archived statistics. The latest complete suite count in the repository comes from October. The [dated verification record](evidence/readiness_2026-11-23/verification.json) includes test counts, timings, package identity and dependency advisories.

## Check the published results

From this branch's repository root, run the following with Python 3.12 or later. Both commands use the standard library, read local files and leave application code unloaded.

```bash
python3 scripts/check_repository_metadata.py --evidence-only
python3 scripts/check_repository_metadata.py
```

The first command checks two different historical experiments. The second also checks the public research documents, simple Markdown file links, the four application response lengths, and the dated verification and dependency inventory records. It reconstructs the historical lock digest from all 67 recorded distribution names and versions. Link checks cover file paths in the current public documents. Remote URLs, Markdown anchors and the archived README are outside their scope.

| October 17 experiment | Input available here | Recomputed trained accuracy |
| --- | --- | ---: |
| Initial recovery | 7,680 held-out CSV rows, covering four baselines × three seeds × 128 episodes × five roles | 68.2161% |
| Integration at `878586c` | Three saved seed means per baseline and a SHA-256 manifest for the metrics file | 96.1784% |

Both experiments used the same two-choice synthetic task, with seeds 11, 22 and 33. Training used 40 episodes per seed. Evaluation used eight decisions per episode and role. The later frozen, random and constant baselines averaged 49.3099%, 50.5208% and 48.7630% respectively. Its trained sample standard deviation was 3.58 percentage points **across three seed means**. Roles shared cue streams. These figures describe that task under its reported arbitration and evaluation settings. [Initial raw observations](evidence/recovery_2026-10-17/heldout_results.csv) · [Integration metrics and settings](evidence/integration_2026-10-17/metrics.json)

The October integration manifest records two runs with matching hashes for their curves, held-out CSVs and metrics. This checkout retains the later metrics and the CSV hashes. Its committed CSV rows are from the earlier study. The November check verified the later metrics hash and recomputed its mean and sample SD from saved seed means. The earlier aggregate was recomputed from the individual CSV rows. Both output tables were identical on Python 3.12 and 3.14. [October repeatability manifest](evidence/integration_2026-10-17/repeatability.json)

## Repeat the focused CPU tests

Use Linux x86-64 with CPython 3.12, a fresh virtual environment, and the sparse source checkout described in the [README](../README.md). These three test modules use generated fixtures. They need no model weights, checkpoint files, provider accounts, project data or additional services.

**The historical lock has known advisories.** On November 23, pip-audit reported 17 advisories in four packages. The scan covered 66 of 67 entries. PyPI could not match the exact PyTorch CPU build. The affected pins and available fixes are in the [audit results](evidence/readiness_2026-11-23/verification.json). The pinned environment below reproduces the historical software versions for this isolated test selection. The standard-library checks above require none of those packages.

Dependency setup needs access to public package registries. Install the CPU wheel from the PyTorch index first, then sync the rest from PyPI. This sequence was verified in a fresh environment with uv 0.9.18. The lock specifies versions and omits wheel hashes.

```bash
uv --no-config venv --python 3.12 .venv-recovery
uv --no-config pip install --python .venv-recovery/bin/python --only-binary=:all: --no-deps \
  --default-index https://download.pytorch.org/whl/cpu 'torch==2.14.1+cpu'
uv --no-config pip sync --python .venv-recovery/bin/python --only-binary=:all: \
  --default-index https://pypi.org/simple requirements-recovery.lock
uv --no-config pip check --python .venv-recovery/bin/python
```

Run from the repository root. The test process starts with an empty inherited environment. The explicit pytest plugin blocks covered network and process calls. This Python guard serves these trusted fixtures and provides no operating-system isolation for untrusted code.

```bash
out=$(mktemp -d)
env -i PATH=/usr/bin:/bin LANG=C.UTF-8 PYTHONPATH=. \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  ARIASKA_DRY_RUN=1 ARIASKA_LIVE_MODE=false ARIASKA_MENTOR_BACKEND=none \
  ARIASKA_DATA_DIR="$out/data" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-recovery/bin/python -m pytest -p pytest_timeout -p core.testing.offline_guard -p no:cacheprovider \
  tests/algorithms/test_nstep_returns.py tests/recovery/test_ppo_contracts.py \
  tests/recovery/test_guarded_cli.py --timeout=90 -q -ra --junitxml="$out/pytest.xml"
```

The review ran on Ubuntu 26.04.1 x86-64, CPython 3.12.12, PyTorch 2.14.1+cpu, NumPy 2.5.4 and pytest 9.1.1. All 67 installed distribution names and versions matched the recovery lock. The selection passed 13 n-step return tests, 24 PPO likelihood/rollout tests and 10 process-guard tests, with zero failures, errors and skips. JUnit measured 4.383 seconds in the first run. A deliberate invalid subprocess request triggered one guard denial. No trainer child launched. [Machine-readable results](evidence/readiness_2026-11-23/verification.json)

## Historical full-suite and mentor evidence

The October local merge validation recorded 3,318 passes, six skips and five warnings in 65.69 seconds. The skip reasons cover one unavailable physical CUDA device, two live-provider tests requiring a key, missing optional Optuna, a conditional hypothesis fixture and a conditional mentor gate. A separate historical mentor-free check completed two synthetic episodes, with two updates per role, zero provider requests and no blocked SDK imports loaded. The November review left both broader checks unexecuted. It also excluded live targets, paid model calls, physical CUDA, checkpoint loading and retraining. [October local suite and skips](evidence/merge_2026-10-17/validation_summary.json) · [October mentor-free result](evidence/merge_2026-10-17/mentor_free.json)

GitHub recorded a successful post-merge run on source commit `d2f72feeaf33b24e4f5d504aae9d5f8215cf98c2` on October 17. The historical local count above refers to its separately identified local tree. [Post-merge Actions run](https://github.com/Reckless98/Ariaska_RL/actions/runs/39069273198)

## Continuous integration

The metadata job checks docs and archived aggregates without importing the application. Pull requests limited to recognized documentation and governance paths show the full CPU job as skipped. For an ordinary PR, the scope step reads the selector from the GitHub-reported base commit and runs it with isolated Python before executing the PR’s metadata checker. The initial change requires an exact baseline, a pinned selector digest, the reviewed 29 paths and the retained CPU job. Later workflow, checker, selector, dependency and runtime changes request the full CPU job. Pushes to `master` also request it. An unavailable or ambiguous scope fails the metadata job. The CPU job starts only after successful metadata validation and an explicit full-job decision. The ordinary full job records its tested commit/tree, installed packages, JUnit XML and mentor result with 14-day artifact retention. Official actions use commit SHA pins. Dependabot checks those action references monthly. [Workflow](../.github/workflows/offline-recovery.yml) · [Dependency update configuration](../.github/dependabot.yml)

GitHub executes the workflow definition proposed by the PR. Maintainers must review changes to that definition. Protection against a contributor replacing the workflow itself requires a trusted check or repository rule administered outside that PR.

The November dependency audit used a separate environment for pip-audit 2.9.0. The command was `pip-audit -r requirements-recovery.lock --no-deps --disable-pip --vulnerability-service pypi --timeout 10 --progress-spinner off --format json`. It exited 1 for known advisories in certifi, filelock, opentelemetry-api and setuptools. The result depends on the advisory database at query time. Compatibility of newer package versions and the advisory status of the exact PyTorch CPU build remain open.
