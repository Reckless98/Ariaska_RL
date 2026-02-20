#!/usr/bin/env python3
"""Validate distillation prep artifacts against schema.

Checks every JSONL line for:
  - Schema conformance (required fields, valid enums)
  - No NaN/Inf floats
  - Command families in registry
  - Reward ranges sane
  - Phases in enum set
  - Discovery types valid

Returns non-zero exit code on any failure.

Usage:
    python -m scripts.distill_prep.validate_artifacts \\
        --traces-dir data/distill_prep/synthetic_traces \\
        --trajectories-dir data/distill_prep/teacher_trajectories
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger("ariaska.distill_prep.validate_artifacts")


def validate_file(filepath: Path) -> Tuple[int, int, List[str]]:
    """Validate a single JSONL file.

    Returns (total_lines, error_count, error_messages).
    """
    from scripts.distill_prep.trace_schema import validate_jsonl_line

    total = 0
    errors: List[str] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            line_errors = validate_jsonl_line(line)
            for err in line_errors:
                errors.append(f"{filepath.name}:{line_num}: {err}")
    return total, len(errors), errors


def validate_directory(dirpath: Path, label: str = "") -> Tuple[int, int, int, List[str]]:
    """Validate all JSONL files in a directory.

    Returns (files_checked, total_lines, error_count, error_messages).
    """
    if not dirpath.exists():
        return 0, 0, 0, [f"Directory does not exist: {dirpath}"]

    files = sorted(dirpath.glob("*.jsonl"))
    if not files:
        return 0, 0, 0, [f"No JSONL files found in {dirpath}"]

    files_checked = 0
    total_lines = 0
    total_errors = 0
    all_errors: List[str] = []

    for fp in files:
        lines, errs, msgs = validate_file(fp)
        files_checked += 1
        total_lines += lines
        total_errors += errs
        all_errors.extend(msgs)

    return files_checked, total_lines, total_errors, all_errors


def validate_all(
    traces_dir: str = "data/distill_prep/synthetic_traces",
    trajectories_dir: str = "data/distill_prep/teacher_trajectories",
    max_errors_shown: int = 50,
) -> bool:
    """Validate all distill prep artifacts.

    Returns True if all valid, False otherwise.
    """
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        has_rich = True
    except ImportError:
        console = None  # type: ignore[assignment]
        has_rich = False

    all_ok = True
    results: List[Tuple[str, int, int, int, List[str]]] = []

    # Validate synthetic traces
    fc, tl, te, errs = validate_directory(Path(traces_dir), "Synthetic Traces")
    results.append(("Synthetic Traces", fc, tl, te, errs))
    if te > 0:
        all_ok = False

    # Validate teacher trajectories
    fc, tl, te, errs = validate_directory(
        Path(trajectories_dir), "Teacher Trajectories"
    )
    results.append(("Teacher Trajectories", fc, tl, te, errs))
    if te > 0:
        all_ok = False

    # Report
    if has_rich and console is not None:
        table = Table(title="Artifact Validation Results")
        table.add_column("Category", style="cyan")
        table.add_column("Files", justify="right")
        table.add_column("Lines", justify="right")
        table.add_column("Errors", justify="right")
        table.add_column("Status")

        for label, fc, tl, te, _ in results:
            status = "[bold green]PASS[/bold green]" if te == 0 else "[bold red]FAIL[/bold red]"
            table.add_row(label, str(fc), str(tl), str(te), status)

        console.print(table)

        # Show errors
        for label, _fc, _tl, _te, errs in results:
            if errs:
                console.print(f"\n[bold red]{label} errors:[/bold red]")
                for err in errs[:max_errors_shown]:
                    console.print(f"  {err}")
                if len(errs) > max_errors_shown:
                    console.print(
                        f"  ... and {len(errs) - max_errors_shown} more"
                    )
    else:
        for label, fc, tl, te, errs in results:
            status = "PASS" if te == 0 else "FAIL"
            logger.info(
                "%s: files=%d lines=%d errors=%d status=%s",
                label, fc, tl, te, status,
            )
            for err in errs[:max_errors_shown]:
                logger.error("  %s", err)

    return all_ok


def main() -> int:
    """CLI entry point. Returns 0 on success, 1 on failure."""
    parser = argparse.ArgumentParser(description="Validate distill prep artifacts")
    parser.add_argument(
        "--traces-dir",
        type=str,
        default="data/distill_prep/synthetic_traces",
    )
    parser.add_argument(
        "--trajectories-dir",
        type=str,
        default="data/distill_prep/teacher_trajectories",
    )
    parser.add_argument(
        "--max-errors", type=int, default=50, help="Max errors to display"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

    ok = validate_all(args.traces_dir, args.trajectories_dir, args.max_errors)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
