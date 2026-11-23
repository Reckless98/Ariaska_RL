# Security policy

Ariaska_RL is experimental research software with tool-execution and model-provider
boundaries. It is not recommended for production security operations. The current
[license](LICENSE) and applicable law remain separate from this reporting policy.

## Supported code

| Version or branch | Security maintenance |
| --- | --- |
| Latest `master` | Best-effort triage and fixes by the maintainer |
| Older commits, checkpoints and untagged experiments | No backport commitment |
| Proposed research preview | Not released; no supported release series yet |

There were no GitHub releases when this policy was prepared on 23 November 2026.
A passing offline test suite is not a security certification.

## Report a vulnerability privately

Email **[filipwolf@gmail.com](mailto:filipwolf@gmail.com)**, the public contact on
[Filip Volf's GitHub profile](https://github.com/Reckless98), with subject
`Ariaska_RL security report`. GitHub private vulnerability reporting was disabled
when checked on 23 November 2026, so do not rely on an unavailable reporting button.

Include the affected commit, impact, relevant component and a minimal inert or
synthetic reproduction. Explain the permissions and assumptions required. Do not
send credentials, working payloads, private target data or large archives in the
initial message. If sensitive supporting material is necessary, agree on a transfer
method first. Do not open a public issue with exploit details before coordination.

The maintainer will triage reports on a best-effort basis, discuss remediation and
coordinate disclosure with the reporter. No bounty, response deadline or legal
safe harbor is offered by this policy. Reporters should not infer permission to
test someone else's infrastructure.

## Authorized research and trust boundaries

Routine repository validation uses synthetic observations, fake providers and
inert execution outputs. The Python offline guard denies selected network and
process operations in trusted test code; it is not a containment boundary for
hostile Python. Use operating-system isolation for untrusted code and artifacts.

Do not run live scans, exploitation, credential checks or target-facing commands
without explicit authorization from the relevant system owner and a written scope.
Documentation, sample IP addresses and authorization flags do not grant permission.
Production provider calls, cloud costs and real target operations are outside the
default offline test procedure.

Treat model output, retrieved documents, tool output, checkpoints and user-supplied
paths as untrusted. Restricted checkpoint loading reduces certain deserialization
risks but does not establish provenance or make unknown model files safe. Keep
credentials out of Git, issues, CI logs and research evidence.

## Dependency and release review

Dependency updates require compatible licenses and offline regression evidence.
Advisory scanning, when available, is a dated signal rather than proof that all
vulnerabilities are absent. Follow the [release gates](docs/research-preview.md)
and [third-party rights review](docs/licensing-review.md) before redistributing
models, corpora or other research artifacts.
