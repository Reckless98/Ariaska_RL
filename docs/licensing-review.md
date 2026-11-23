# License migration decision

**23 November 2026 — repository-wide AGPL-3.0-only migration is blocked.**
The reviewed baseline is `d2f72feeaf33b24e4f5d504aae9d5f8215cf98c2`.
The root [LICENSE](../LICENSE) remains byte-for-byte unchanged. No AGPL source
headers, package declarations or open-source badges are applied by this proposal.

The current Source Available Non-Commercial License names Filip Volf and covers
code, documentation, configuration, trained models and the knowledge corpus. Its
commercial-use and redistribution restrictions conflict with sections 1 and 6 of
the [Open Source Definition](https://opensource.org/osd). Public source access does
not make the current revision open source.

## What the history establishes

| Date | Commit | Root license |
| --- | --- | --- |
| 25 April 2025 | `439d3ab2` | GNU GPL version 3 text added |
| 20 July 2025 | `74d8318a`, `0e341450` | License removed, then the same GPL text restored |
| 7 February 2026 | `91ef8f3c` | GNU Affero GPL version 3 text substituted |
| 16 February 2026 | `9d2c1e65` | Present non-commercial text substituted |

The earlier text is evidence of historical licensing, not proof that later
additions can be relicensed. Earlier grants must be assessed for the particular
copies and rights involved; this PR does not attempt to revoke them. See the
[GNU license FAQ](https://www.gnu.org/licenses/gpl-faq.html) and
[AGPL text](https://www.gnu.org/licenses/agpl-3.0.html).

The current license's section 5 asserts a copyright assignment on contribution.
No separate assignment records or per-contribution evidence were identified in
the reviewed tree. That clause alone does not settle third-party rights or the
status of earlier contributions. Git author labels and repository administration
rights likewise do not prove copyright ownership. The [authorship inventory](../AUTHORS.md)
keeps those distinctions explicit; see also the
[FSF contributor FAQ](https://www.fsf.org/licensing/contributor-faq).

## Blocking questions and required evidence

| Blocker | Evidence needed | Decision owner |
| --- | --- | --- |
| Contributor rights | Reconcile author aliases and automated-account operators; document ownership, applicable grants or written consent for affected contributions | Maintainer and relevant rightsholders |
| Walkthroughs and embedded knowledge | Exact origins, applicable terms and permissions; trace extracted JSON, embedded Python content, generated datasets and derivative weights | Maintainer and source rightsholders |
| Trained models | Base-model revision/license, conversion record, training inputs and permissions, artifact hashes and redistribution terms | Maintainer and model/data rightsholders |
| Bundled binaries and run data | Origin, license, authorization to distribute and privacy/disclosure review | Maintainer |
| Dependencies and distributions | Exact versions/digests, bundled notices and corresponding-source obligations where applicable; assess the actual combined distribution | Maintainer, with specialist review where necessary |

The [third-party inventory](third-party-inventory.md) identifies concrete paths and
limits. Hack The Box's current acceptable use policy restricts AI training,
evaluation, dataset compilation and derivative synthetic training data using its
content unless expressly authorized by a service agreement. No such agreement was
available for this audit. Determine which terms applied to historical activity
separately; do not infer that a current policy proves every past act was prohibited.
Removing a PDF, its attribution or a dataset label does not clear its derivatives.
[Hack The Box acceptable use policy, section 6.2](https://www.hackthebox.com/hubfs/Legal/AUP.pdf?hsLang=en)

## A prospective AGPL migration

If the selected distribution is cleared and the maintainer approves, AGPL-3.0-only
would permit commercial use under its terms while preserving copyright ownership.
It imposes reciprocal source and notice obligations on covered distributions; its
network-interaction provision applies to modified versions as specified in section
13. It is not a non-commercial license, an automatic assignment of copyright or a
guarantee that every separate dependency can be relabeled AGPL.
[Canonical GNU AGPLv3](https://www.gnu.org/licenses/agpl-3.0.html)

A future migration should:

1. Approve the exact scope of cleared code and assets, with separate notices for
   third-party material. Resolve conflicting terms or substitute independently
   authored material before choosing the release tree.
2. Record the required rightsholder approvals and preserve Filip Volf's existing
   copyright notice and other applicable notices.
3. Use the canonical unmodified [AGPLv3 text](https://www.gnu.org/licenses/agpl-3.0.txt),
   and explicitly choose `AGPL-3.0-only` in applicable metadata and source notices.
   Do not silently choose `-or-later` or add a non-commercial restriction.
4. Update current documentation, citation metadata and contribution terms together.
   Keep historical snapshots clearly historical; do not rewrite Git history.
5. Review the complete proposed tag and source archives, plus any packages, images
   and release attachments. Obtain maintainer review before merging or publishing.

No project package manifest was found in the baseline (`pyproject.toml`, `setup.py`
and `setup.cfg` are absent), so no fictitious package-license field was added.
A candidate canonical license file is not installed while compatibility remains
unresolved. The [release proposal](research-preview.md) records the outstanding gates.

## Scope of this assessment

This is a repository evidence and compatibility assessment, not a legal opinion
or a file-by-file originality determination. It used license history, commit labels,
all tracked-path metadata, declarations and selected provenance code. Private
contracts, excluded corpus contents and model files were not reviewed. The audit
can identify blockers; it cannot infer missing permission.
