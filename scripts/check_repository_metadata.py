"""Validate public research metadata and recompute the archived accuracy summaries."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from math import isclose
import os
from pathlib import Path
import re
from statistics import mean, stdev
import subprocess
import sys
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
ROOT_DOCS = {
    "README.md", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "SECURITY.md",
    "CHANGELOG.md", "ROADMAP.md", "AUTHORS.md", "MAINTAINERS.md",
}
RESEARCH_DOCS = {
    "docs/research-preview.md", "docs/reproducibility.md", "docs/readiness-report.md",
    "docs/licensing-review.md", "docs/third-party-inventory.md", "docs/openai-application.md",
}

def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def check_documents():
    tracked = {os.fsdecode(path) for path in git("ls-files", "-z").split(b"\0") if path}
    for filename in sorted(ROOT_DOCS | RESEARCH_DOCS):
        doc = ROOT / filename
        for link in re.findall(r"\]\(([^\s()]+)\)", doc.read_text()):
            url = urlsplit(link)
            if url.scheme or url.netloc or not url.path:
                continue
            target = (doc.parent / unquote(url.path)).resolve()
            relative = target.relative_to(ROOT).as_posix()
            if not target.exists() and relative not in tracked and not any(p.startswith(relative + "/") for p in tracked):
                raise ValueError(f"{filename}: unresolved local link {link!r}")
    for filename in ("LICENSE", "CITATION.cff", "docs/history/architecture-and-phases.md"):
        if not (ROOT / filename).is_file():
            raise ValueError(f"Missing {filename}")
    draft = json.loads((ROOT / "docs/application-responses.json").read_text())
    if not isinstance(draft.get("role"), str) or not draft["role"].strip():
        raise ValueError("The application draft needs a role")
    responses = draft["responses"]
    expected = {"why_repository_qualifies", "why_codex_security", "how_api_credits_will_be_used", "additional_information"}
    if not isinstance(responses, dict) or set(responses) != expected:
        raise ValueError("Expected four named application responses")
    for key, answer in responses.items():
        if not isinstance(answer, str) or not answer.strip() or len(answer) > 500:
            raise ValueError(f"Invalid application response {key!r}")


def check_readiness_records():
    docs = ROOT / "docs"
    record = json.loads((docs / "evidence/readiness_2026-11-23/verification.json").read_text())
    inventory = json.loads((docs / "dependency-license-inventory.json").read_text())
    if record["date"] != "2026-11-23" or inventory["review_date"] != record["date"]:
        raise ValueError("Readiness record dates disagree")
    if record["runtime_source_base"] != "d2f72feeaf33b24e4f5d504aae9d5f8215cf98c2" or (
        inventory["assessed_repository_commit"] != record["runtime_source_base"]
    ):
        raise ValueError("Readiness record source commits disagree")
    packages = inventory["recovery_packages"]
    names = {re.sub(r"[-_.]+", "-", p["name"]).lower() for p in packages}
    environment = record["environment"]
    if len(names) != 67 or len(packages) != 67 or (
        environment["exact_distribution_count"] != len(names)
        or inventory["summary"]["installed_recovery_packages"] != len(names)
        or inventory["python_version"] != environment["python"].removeprefix("CPython ")
    ):
        raise ValueError("Recovery distributions or interpreter identity disagree")
    # Reconstruct the historical lock from its names and versions. Later runtime
    # dependency changes should not force edits to a dated research record.
    lock = "".join(f"{p['name']}=={p['version']}\n" for p in sorted(packages, key=lambda p: p["name"].lower()))
    digest = hashlib.sha256(lock.encode()).hexdigest()
    locked = inventory["requirement_lists"]["requirements-recovery.lock"]
    if digest != "6ed737d1d3d06ff2c892bfa529a0995a9b2f558823db11159655c301c1bcfc3d" or (
        locked["sha256"] != digest or environment["requirements_lock_sha256"] != digest
        or locked["requirements"] != 67
    ):
        raise ValueError("Package versions do not reproduce the historical recovery lock")
    tests = record["current_focused_pytest"]
    if tests["passed"] != sum(tests["passed_by_module"].values()) or tests["passed"] != 47 or (
        (tests["failed"], tests["errors"], tests["skipped"]) != (0, 0, 0) or tests["junit_seconds"] <= 0
    ):
        raise ValueError("Focused validation counts disagree")
    audit = record["dependency_advisory_scan"]
    if audit["audited"] + len(audit["unresolved"]) != len(names) or (
        sum(len(row["advisories"]) for row in audit["affected"]) != audit["advisory_count"]
    ):
        raise ValueError("Dependency audit coverage or advisory count disagrees")


BASELINES = {"random", "registry", "frozen", "trained"}
SEEDS = {11, 22, 33}
ROLES = {"RedAgent", "BlueAgent", "ScoutAgent", "ShadowAgent", "OrionAgent"}


def check_ids(rows, key, expected):
    if len(rows) != len(expected) or {row[key] for row in rows} != expected:
        raise ValueError(f"Missing, duplicate or unexpected {key} in historical summary")


def check_historical_evidence():
    evidence = ROOT / "docs/evidence"
    recovery = evidence / "recovery_2026-10-17"
    integration = evidence / "integration_2026-10-17"
    groups = defaultdict(dict)
    with (recovery / "heldout_results.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            group = (row["baseline"], int(row["seed"]))
            observation = (int(row["episode"]), row["role"])
            accuracy = float(row["accuracy"])
            if (group[0] not in BASELINES or group[1] not in SEEDS or observation[0] not in range(128)
                    or observation[1] not in ROLES or observation in groups[group] or not 0 <= accuracy <= 1):
                raise ValueError("Unexpected or duplicate recovery CSV observation")
            groups[group][observation] = accuracy
    if len(groups) != 12 or any(len(rows) != 640 for rows in groups.values()):
        raise ValueError("Expected 128 episodes and five roles per baseline and seed")
    manifest = json.loads((integration / "repeatability.json").read_text())
    metrics = (integration / "metrics.json").read_bytes()
    if hashlib.sha256(metrics).hexdigest() != manifest["metrics.json"]["sha256"]:
        raise ValueError("Integration metrics hash differs from its published manifest")
    for label, raw in (("Recovery", (recovery / "metrics.json").read_bytes()), ("Integration", metrics)):
        summary = json.loads(raw)["summary"]
        check_ids(summary, "baseline", BASELINES)
        for item in summary:
            baseline = item["baseline"]
            check_ids(item["seeds"], "seed", SEEDS)
            accuracies = []
            for seed in item["seeds"]:
                accuracy = seed["accuracy"]
                if label == "Recovery":
                    actual = mean(groups[baseline, seed["seed"]].values())
                    if not isclose(actual, accuracy, abs_tol=1e-12):
                        raise ValueError("Recovery CSV and recorded per-seed accuracy disagree")
                    accuracy = actual
                accuracies.append(accuracy)
            if not isclose(mean(accuracies), item["mean_accuracy"], abs_tol=1e-12) or not isclose(
                stdev(accuracies), item["sd_accuracy"], abs_tol=1e-12
            ):
                raise ValueError(f"{label} seed means and summary statistics disagree")
            note = " (from archived seed means)" if label == "Integration" else ""
            print(f"{label} {baseline}: {mean(accuracies):.4%}{note}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-only", action="store_true", help="check historical aggregates only")
    args = parser.parse_args()
    try:
        if not args.evidence_only:
            check_documents()
            check_readiness_records()
        check_historical_evidence()
    except (OSError, KeyError, TypeError, ValueError) as error:
        sys.exit(str(error))
