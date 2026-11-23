# Research preview proposal

**Proposed version: `v0.1.0-rc.1`. Status: not tagged, not released, blocked on rights clearance and maintainer review.**

This would be the first versioned research preview, not a stability commitment.
The `0.1.0` core version reflects an experimental interface; `rc.1` marks a candidate
that must satisfy the gates below. No existing versioned GitHub releases or tags
are assumed from phase numbers, model filenames or the internal PPO “v3.0” label.

## Proposed release notes

Ariaska_RL explores independent role policies, optional model mentorship and
executed-outcome attribution. The preview would expose a documented research
baseline with source-pinned evidence, a concise architecture guide, contribution
and disclosure policies, and reproducibility instructions.

The October 2026 study recorded 96.18% trained mean accuracy on a two-choice
synthetic task, against a 49.31% frozen baseline, with three fixed seeds. These are
historical results, with role variance and tightly limited inference conditions.
They do not demonstrate live cybersecurity performance, production readiness or
broad task generalization. Link the [original evidence](integration_merge_readiness.md)
and identify any independently rerun protocol separately in final release notes.

The license for a released candidate cannot be announced until the
[rights review](licensing-review.md) is resolved. Do not copy an AGPL badge onto this
proposal or describe it as an already available open-source release.

## Required gates

1. **Rights and provenance:** approve an exact source-and-asset manifest, resolve
   contribution ownership, identify third-party terms, and remove or replace
   unresolved material in a reviewed forward change. Models, extracted writeups,
   embedded corpora, executable payloads and raw target reports need explicit
   disposition. Preserve research history through a rights-appropriate process;
   this proposal does not authorize deletion or history rewriting.
2. **Whole-tree review:** inspect every file in the proposed tag, not only uploaded
   attachments. GitHub automatically offers source archives of the tagged tree.
   Sparse checkout and `.gitignore` do not exclude already tracked material from
   those archives. A “source-only” label cannot fix a contaminated tag.
3. **License decision:** after clearance, the maintainer reviews commercial-use
   permission, reciprocal distribution obligations and network-source obligations.
   If AGPL-3.0-only is approved, use the canonical unmodified text, preserve copyright
   and separate upstream notices, and align current metadata. This PR does not do so.
4. **Validation:** run the appropriate offline checks on the exact candidate with
   recorded source identity, resolved dependencies and all skips. Obtain an authorized
   independent full-suite result where this modernization could not rerun it.
   Record known advisories and exceptions; do not treat a successful install as a
   vulnerability audit. Review generated outputs before publication.
5. **Reproducibility:** include an environment record, documented commands, seed/task
   settings, machine-readable results and hashes. Avoid embedding local paths,
   credentials or raw target data. Report GPU/provider checks only if actually run.
6. **Review and publication:** obtain explicit maintainer approval of the final
   diff, tag, license and release notes. Only then create the pre-release. Do not
   enable automatic publishing or silently merge the readiness PR.

## Candidate contents

Use an allowlist of cleared source, independently authored synthetic fixtures,
necessary configuration, licenses/notices and reviewed aggregate evidence. Treat
full-container builds separately: the existing Dockerfile copies the build context,
so building from the present mixed tree is not a cleared distribution route.

Do not attach model weights, third-party corpora, runtime databases, session logs or
executables until their permissions and disclosure risks have been reviewed. Removing
files from a future tag does not remove earlier Git history or cure past redistribution.
No package publishing or external asset upload is part of this proposal.
