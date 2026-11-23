# Third-party rights and artifact inventory

This inventory covers baseline `d2f72feeaf33b24e4f5d504aae9d5f8215cf98c2` on
23 November 2026. **Unknown means unresolved, not permitted.** The root license is
unchanged; no entry below is cleared for relicensing merely by appearing here.

## Inputs and generated material

| Material | Observed evidence | Status and next action |
| --- | --- | --- |
| `data/htb_walkthroughs/`, `data/htb_extractions/` | Four PDFs, twelve Markdown files including a README, and twelve extracted JSON files; commit `18307001` added writeups alongside knowledge/training changes | Unresolved third-party content and derivative rights; trace origins, dates, terms and permissions |
| Embedded and generated knowledge | `scripts/extract_htb_walkthroughs.py` documents extraction; `scripts/gpu_pipeline/generate_htb_data.py` accepts third-party writeup directories | Review embedded Python/playbook content and downstream datasets; excluding a directory alone is insufficient |
| `data/distill_prep/`, `data/finetune/`, `data/unified/`, `data/knowledge_sources/` | 811 paths in distillation preparation, training/registry exports and other corpus material | Per-record source and rights mapping needed, including generated-output provider terms and privacy review |
| `models/` and root `Modelfile*` | 350 `.pt` files, two temporary checkpoint-like files, progress labels and a placeholder; seven model definitions reference local GGUF paths | No checkpoint loading performed; establish exact base models, inputs, conversions, hashes and redistribution terms |
| Root executable files | Two executable artifacts first appear at `da347084` | Exclude from a future candidate until origin and redistribution are resolved; no execution or deletion performed |
| `results/` and other run state | 835 result paths, plus target-named reports and state backups | Review sensitive identifiers and source permissions before redistribution; preserve reviewed aggregates and historical provenance |

The Qwen3-4B model card identifies Apache-2.0 for that upstream model. It does not
establish the origin or rights of Ariaska's individual weights or modified model
names. [Qwen3-4B card](https://huggingface.co/Qwen/Qwen3-4B) ·
[Apache-2.0 obligations](https://apache.org/licenses/LICENSE-2.0.html)

Hack The Box's current restrictions are a concrete clearance issue for covered
inputs and derivatives, subject to any documented agreement and the terms applicable
at the time. See the [license review](licensing-review.md) and
[official acceptable use policy](https://www.hackthebox.com/hubfs/Legal/AUP.pdf?hsLang=en).

## Libraries and distributions

The tested recovery environment, full optional runtime and dashboard dependencies
are separate sets. See [requirements-recovery.lock](../requirements-recovery.lock),
[requirements.txt](../requirements.txt) and
[requirements-streamlit.txt](../requirements-streamlit.txt). The
[dependency metadata inventory](dependency-license-inventory.json) records the
installed pinned recovery distributions and unresolved declared optional packages.
Metadata identifies review leads; it is not a compatibility verdict or a complete SBOM.

Specific review points include `python-nmap>=0.7.1` in the optional stack (the 0.7.1
package identifies GPLv3 licensing), multi-license notices in PyTorch distributions,
and externally referenced container images with floating tags. Referencing a package
or image is distinct from distributing its contents. A redistributed environment
must carry applicable notices and source offers/availability where required.
[python-nmap 0.7.1](https://pypi.org/project/python-nmap/0.7.1/) ·
[PyTorch license](https://github.com/pytorch/pytorch/blob/main/LICENSE)

A prospective AGPL-covered combination must be checked against exact upstream
terms. Do not relabel dependencies, model weights or documentation as project-owned.
There was no committed third-party notice bundle or SBOM in the baseline.

## Repository hygiene

The Git tree contains **2,746 tracked paths and 4,486,933,769 logical bytes**.
The 350 `.pt` checkpoints account for **4,408,404,997 bytes**. These are sums by
tracked path, before compression or deduplication, not a download-size measurement.
They explain why the offline workflow uses a sparse checkout.

The modernization preserves these tracked artifacts. Ignore rules protect against
new accidental additions; they do not remove existing files or cure rights issues.
The existing Dockerfile copies the whole build context, and the baseline has no
`.dockerignore`. Do not interpret an ordinary image build as a cleared release.

A redacted heuristic scan covered 642 currently checked-out, baseline-tracked
source/document/configuration files; 2,104 paths were absent or excluded. It checked
four high-confidence token/key shapes and found three example-like literal matches:
a private-key marker fragment in `core/knowledge/knowledge_packs.py`, and a
placeholder key envelope plus a standard documentation access-key example in
`core/orchestration/simulated_output.py`. No values are reproduced in the evidence.
These were not confirmed credentials. Real environment/credential files, corpora,
models, generated output and Git history were not scanned. This limited check is
not a clean bill for secrets or personal data across the repository.

## Cleared-release manifest to prepare

For each selected file or asset, record: path, hash/revision, original source,
rightsholder, license/contract reference, permitted uses, required notices, derivative
inputs, privacy disposition, reviewer and decision date. Keep confidential permission
records private; publish only the reviewed summary and needed notices.

Use a forward, reviewable cleanup or a separately scoped cleared distribution once
authorized. Inspect the complete tag because GitHub automatically exposes its source
archives. Preserve original research evidence through an appropriate retention process;
this PR does not delete data, rewrite history or publish new assets.
[GitHub release archive behavior](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
