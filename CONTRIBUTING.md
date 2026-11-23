# Contributing to Ariaska_RL

Start with a small, reproducible research or maintenance improvement. This is an
experimental project maintained by [Filip Volf](https://github.com/Reckless98).
Please read the [code of conduct](CODE_OF_CONDUCT.md) and
[security reporting policy](SECURITY.md) before opening an issue.

## License and contribution rights

The current terms are the [Source Available Non-Commercial License](LICENSE),
not AGPL. In particular, section 5 contains a copyright-assignment provision.
Read it before submitting code; ask the maintainer to resolve any uncertainty
before contributing. This guide does not replace, broaden or validate that clause.
The [licensing review](docs/licensing-review.md) explains why migration remains pending.

Submit only work you are entitled to contribute. Identify copied or adapted code,
its exact upstream revision, copyright holders and license. Preserve notices.
Disclose any third-party material or generated asset that needs separate review.
A sign-off, an automated author name or public availability alone does not prove
ownership. Do not upload datasets, weights, scraped corpora or training traces
without documented redistribution and use rights.

## A useful pull request

1. Check existing issues and [the roadmap](ROADMAP.md). Discuss a proposed change
   to scientific objectives, architecture or licensing before implementing it.
2. Create a branch from current `master`. Keep unrelated formatting and generated
   output out of the change.
3. Explain the concrete problem and the intended behavior. For research, state
   the hypothesis, baseline, fixed budget and evaluation protocol before comparing
   results. Report failures and uncertainty as well as positive outcomes.
4. Use the [reproducibility guide](docs/reproducibility.md). Record the command,
   source revision, environment, actual result and every skip. Do not label
   historical evidence or an unexecuted check as a new pass.
5. Add a focused regression for a behavioral correction where appropriate.
   Documentation changes need metadata/link checks, not application execution.
6. Open a PR using the template. The maintainer reviews scope, evidence, rights
   and safety before deciding whether to merge. There is no response-time promise.

## Engineering boundaries

Preserve the independent policy identities and executed-action ownership contracts.
Route model access through `GPTManager`. Keep learning and reward changes narrow
and support them with numerical evidence. Do not remove guards or weaken assertions
to obtain a green check.

Routine verification uses fake providers, synthetic observations and inert outputs.
It must not launch security tools, call paid providers or touch live targets. A
repository example or a target address is not authorization. Any separate live
research needs explicit target-owner permission and a defined scope.

Never submit credentials, real `.env` files, session captures, identifiable target
data, executable payloads or bulk checkpoints in a PR. Keep logs sanitized and
minimal. Preserve existing research records; discuss artifact migration before
moving or deleting them. Use [private disclosure](SECURITY.md) for vulnerabilities.

## Good first research contributions

- Recompute published tables from the retained CSVs and check provenance links.
- Clarify uncertainty, obsolete commands and undocumented dependency assumptions.
- Propose a new synthetic task or ablation with a fixed protocol and explicit limits.
- Improve protective tests and trust-boundary documentation without expanding
  offensive execution.

The older [engineering notes](core/documentation/CONTRIBUTING.md) are historical;
this root guide is the current contribution policy.
