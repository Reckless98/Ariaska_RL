"""Select the CPU job from git objects in a tested pull-request merge."""

import argparse
import os
from pathlib import Path
import subprocess


BOOTSTRAP_BASE = "d2f72feeaf33b24e4f5d504aae9d5f8215cf98c2"
METADATA = {
    ".gitignore", "CITATION.cff", "README.md", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md",
    "SECURITY.md", "CHANGELOG.md", "ROADMAP.md", "AUTHORS.md", "MAINTAINERS.md",
    ".github/dependabot.yml", ".github/pull_request_template.md", ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/research_question.yml", ".github/ISSUE_TEMPLATE/config.yml",
    "core/documentation/CONTRIBUTING.md", "docs/application-responses.json",
    "docs/dependency-license-inventory.json", "docs/evidence/readiness_2026-11-23/verification.json",
}
# During initial adoption there is no base selector. The workflow authenticates
# this file by SHA-256. Only this exact change set is eligible for bootstrap.
BOOTSTRAP_PATHS = METADATA | {
    ".github/workflows/offline-recovery.yml", "scripts/select_ci_scope.py", "scripts/check_repository_metadata.py",
    "docs/history/architecture-and-phases.md", "docs/research-preview.md", "docs/readiness-report.md",
    "docs/reproducibility.md", "docs/openai-application.md", "docs/licensing-review.md", "docs/third-party-inventory.md",
}


def requires_full_job(repo, base):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo)

    parents = git("rev-list", "--parents", "-n", "1", "HEAD").decode("ascii").split()
    if len(parents) != 3 or parents[1] != base:
        raise ValueError("Cannot establish the tested merge and its expected base parent")
    raw = git("diff", "--no-ext-diff", "--no-renames", "--name-only", "-z", base, "HEAD")
    if not raw or not raw.endswith(b"\0"):
        raise ValueError("Changed-path list is empty or incomplete")
    paths = {os.fsdecode(p) for p in raw[:-1].split(b"\0")}
    if base == BOOTSTRAP_BASE:
        if paths != BOOTSTRAP_PATHS:
            raise ValueError("Changes differ from the reviewed metadata bootstrap")
        workflow = ".github/workflows/offline-recovery.yml"
        before = git("show", f"{base}:{workflow}")
        after = git("show", f"HEAD:{workflow}")
        header, cpu = before.split(b"jobs:\n  cpu-offline:\n")
        gate = (b"  cpu-offline:\n    needs: metadata\n"
                b"    if: ${{ success() && needs.metadata.result == 'success' && needs.metadata.outputs.run_full == 'true' }}\n")
        if not after.startswith(header + b"jobs:\n  metadata:\n") or not after.endswith(
            gate + cpu.replace(b"git rev-parse HEAD^{tree}", b"git rev-parse 'HEAD^{tree}'")
        ):
            raise ValueError("Bootstrap changed existing workflow settings or CPU job")
        return False
    return any(p not in METADATA and not (p.startswith("docs/") and p.endswith(".md")) for p in paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("base")
    args = parser.parse_args()
    try:
        print("true" if requires_full_job(args.repo, args.base) else "false")
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Validation scope unavailable: {error}\n")
