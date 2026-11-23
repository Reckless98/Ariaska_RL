# OpenAI Codex for Open Source application draft

**Prepared 23 November 2026 for maintainer review. Not submitted.**
The current repository is source-available non-commercial. This is a material
eligibility weakness; do not describe the AGPL migration or proposed release as
completed. The strongest honest case is sustained maintenance and a reproducible,
limited research contribution, with early community adoption disclosed.

## Official program and fit

The [official program page](https://developers.openai.com/community/codex-for-oss)
describes six months of ChatGPT Pro with Codex, conditional Codex Security access
and potential API credits for core open-source maintenance. Benefits are subject
to selection and verification; the [program terms](https://learn.chatgpt.com/docs/codex-for-oss-terms)
do not guarantee selection, duration or all benefits for every applicant.

The [application form](https://openai.com/form/codex-for-oss/) asks for a public
GitHub profile/repository and a primary/core maintainer role. It considers meaningful
usage, ecosystem relevance and active maintenance, without publishing a numeric
star threshold. A project outside the usual criteria may explain its importance;
that is not an exemption from accurate disclosure or proof of eligibility.

| Signal | Verified baseline before this PR | Interpretation |
| --- | --- | --- |
| Maintainer | Repository owned by @Reckless98; account has admin/maintain/push rights; LICENSE names Filip Volf | Strong evidence of maintainer role, not sole ownership of every third-party asset |
| Maintenance | 853 reachable commits; 14 PRs, 12 merged; last push 17 October 2026 | Real maintenance history, not 853 independent contributors or a current usage metric |
| Adoption | 1 star, 0 forks | Early adoption; downloads, external users and academic citations are not verified |
| Releases | No tags or GitHub releases | Proposed preview is not released |
| Validation | Successful historical offline CI and published synthetic evidence | Bounded research value; no claim of live capability |
| Licensing | Non-commercial restrictions; rights review unresolved | Do not claim current open-source eligibility |

The counts are a dated snapshot from the GitHub repository, pulls, releases, tags
and Actions APIs. This PR will change maintenance counts and timestamps; refresh
before submitting. See the [readiness report](readiness-report.md).

## Form choices

**Role: Primary maintainer.** The current role question is a choice, not a prose field.
Optional explanation if requested: “I created Ariaska_RL and maintain its architecture,
code review, offline validation, documentation and release decisions.”

Select Codex Security and API credits only if those benefits are wanted. Enter account
email and organization ID directly in the form; they are intentionally absent from
these repository drafts. Do not submit confidential permission documents or private
research data with the application.

## Draft responses

The four response texts below are each below 500 characters, matching the displayed
limits for the main prose fields. The Security response is prepared for the
conditional field shown when that benefit is selected; its current dynamic limit
was not independently visible in the static page, so the same 500-character cap is
used. [Machine-readable source](application-responses.json).

### Why does this repository qualify?

I created and maintain Ariaska_RL to study independent RL policies, optional LLM mentorship and agent reliability. It has 853 baseline commits, 12 merged PRs, public synthetic evaluation data and successful historical offline CI. Adoption is early: 1 star and 0 forks. Its potential contribution is reproducible evidence about learning and safety boundaries. The current license is source-available; open-source relicensing is pending third-party rights clearance.

Character count: **464 / 500**.

### Why does your project need Codex Security?

Ariaska_RL combines model-provider calls, tool-execution boundaries, checkpoint loading and generated artifacts. I would use Codex Security to review this repository for unsafe deserialization, credential exposure, dependency risks and permission failures, then add protective regression tests. The scope is maintaining the project and improving its safety, not testing external systems or expanding offensive capabilities.

Character count: **423 / 500**.

### How will you use API credits for your project?

I would use API credits for Ariaska_RL maintenance: reviewing scoped PRs, explaining test failures, improving documentation, checking release evidence and developing protective regressions. Any optional model evaluation would use cleared synthetic inputs, fixed budgets and a mentor-free baseline. I would publish reproducible methods and aggregate results where permissions allow, with no credentials or private target data.

Character count: **425 / 500**.

### Anything else we should know?

I am the creator and primary maintainer. The October 2026 recovery work made policy ownership, executed-outcome attribution and checkpoint behavior auditable. This readiness proposal adds governance and a rights review while preserving negative results and limitations. There is no published release, verified download count or broad adoption claim. A research preview and AGPL migration remain subject to rights clearance and my review.

Character count: **437 / 500**.

## Before submission

Resolve or explicitly disclose the licensing blocker. Refresh public metrics and
link the final reviewed changes and any actual release. Only describe a new test
run when its source identity and results are available. Do not claim novelty,
external adoption, download counts, citations or production readiness without
evidence. Submit the application yourself after reviewing its terms and wording.
