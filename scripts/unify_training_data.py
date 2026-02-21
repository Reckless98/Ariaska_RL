"""Convert ALL Ariaska training data into one canonical unified format.

Supports 5 source formats:
1. Knowledge Corpus v2  (data/knowledge_candidates_v2/*.jsonl)
2. Teacher Trajectories (data/distill_prep/teacher_trajectories/*.jsonl)
3. GPU Training Traces  (traces/h200_distill/*.jsonl)
4. Learned Commands     (data/learned_commands.json)
5. Scenario Profiles    (data/distill_prep/scenarios/*.json)

Output goes to ``data/unified/`` in sorted JSONL files.

Usage::

    python -m scripts.unify_training_data convert-all
    python -m scripts.unify_training_data convert-knowledge
    python -m scripts.unify_training_data convert-traces
    python -m scripts.unify_training_data convert-commands
    python -m scripts.unify_training_data convert-trajectories
    python -m scripts.unify_training_data validate
    python -m scripts.unify_training_data stats
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from rich.console import Console
from rich.table import Table

from scripts.unified_data_schema import (
    DIFFICULTY_ORDER,
    PHASE_ORDER,
    UNIFIED_DATA_DIR,
    UNIFIED_SCHEMA_VERSION,
    VALID_KINDS,
    VALID_SOURCES,
    UnifiedRecord,
    gpu_trace_sort_key,
    knowledge_sort_key,
    learned_command_sort_key,
    load_manifest,
    trajectory_sort_key,
    validate_record,
    wrap_record,
    write_manifest,
)

logger = logging.getLogger("ariaska.unify_data")
console = Console()

# ═══════════════════════════════════════════════════════════════
# Source paths
# ═══════════════════════════════════════════════════════════════

KNOWLEDGE_DIR = Path("data/knowledge_candidates_v2")
TRAJECTORIES_DIR = Path("data/distill_prep/teacher_trajectories")
TRACES_DIR = Path("traces/h200_distill")
LEARNED_COMMANDS_PATH = Path("data/learned_commands.json")
SCENARIOS_DIR = Path("data/distill_prep/scenarios")
SYNTHETIC_TRACES_DIR = Path("data/distill_prep/synthetic_traces")

# Output sub-dirs under UNIFIED_DATA_DIR
KNOW_OUT = UNIFIED_DATA_DIR / "knowledge" / "by_phase"
TRAJ_OUT = UNIFIED_DATA_DIR / "trajectories" / "by_difficulty"
GPU_OUT = UNIFIED_DATA_DIR / "gpu_runs"
CMD_OUT = UNIFIED_DATA_DIR / "learned_commands"
SCEN_OUT = UNIFIED_DATA_DIR / "scenarios"
IDX_OUT = UNIFIED_DATA_DIR / "indices"


# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════


def _ensure_dirs() -> None:
    """Create all output directories."""
    for d in [KNOW_OUT, TRAJ_OUT, GPU_OUT, CMD_OUT, SCEN_OUT, IDX_OUT]:
        d.mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> int:
    """Write records as compact JSONL. Returns number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, default=str, separators=(",", ":")) + "\n")
    return len(records)


def _read_jsonl_safe(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file, skipping malformed lines."""
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.debug("Skipped malformed line %d in %s", lineno, path)
    return records


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ═══════════════════════════════════════════════════════════════
# 1. Knowledge Corpus v2 → unified (107K entries, 18 JSONL files)
# ═══════════════════════════════════════════════════════════════


# Map knowledge killchain_step values to canonical phase names
_KC_PHASE_MAP: Dict[str, str] = {
    "recon": "RECON",
    "reconnaissance": "RECON",
    "enumeration": "ENUMERATION",
    "enum": "ENUMERATION",
    "foothold": "EXPLOITATION",
    "exploitation": "EXPLOITATION",
    "exploit": "EXPLOITATION",
    "privesc": "PRIVILEGE_ESCALATION",
    "privilege_escalation": "PRIVILEGE_ESCALATION",
    "privilege-escalation": "PRIVILEGE_ESCALATION",
    "lateral": "LATERAL_MOVEMENT",
    "lateral_movement": "LATERAL_MOVEMENT",
    "lateral-movement": "LATERAL_MOVEMENT",
    "persist": "POST_EXPLOITATION",
    "post_exploitation": "POST_EXPLOITATION",
    "post-exploitation": "POST_EXPLOITATION",
    "exfil": "EXFILTRATION",
    "exfiltration": "EXFILTRATION",
    "closeout": "CLOSEOUT",
}


def _normalize_phase(raw: str) -> str:
    """Normalize a killchain_step or phase string to canonical form."""
    return _KC_PHASE_MAP.get(raw.lower().strip(), raw.upper().strip())


def convert_knowledge() -> Dict[str, int]:
    """Convert knowledge corpus v2 → unified/knowledge/by_phase/*.jsonl."""
    console.print("[cyan]Converting knowledge corpus...[/cyan]")
    _ensure_dirs()

    # Collect all records
    all_records: List[Dict[str, Any]] = []
    if not KNOWLEDGE_DIR.exists():
        console.print("[yellow]  No knowledge dir found, skipping[/yellow]")
        return {}

    for jsonl_file in sorted(KNOWLEDGE_DIR.glob("*.jsonl")):
        entries = _read_jsonl_safe(jsonl_file)
        for entry in entries:
            # Determine phase from taxonomy.killchain_step
            taxonomy = entry.get("taxonomy", {})
            ks = taxonomy.get("killchain_step", "")
            phase = _normalize_phase(ks) if ks else "RECON"

            rec = wrap_record(
                kind="knowledge",
                source="knowledge_v2",
                data=entry,
                phase=phase,
                timestamp=entry.get("source", {}).get("ingested_at", _iso_now()),
            )
            all_records.append(rec.to_dict())

    # Sort by killchain phase then evidence_coverage desc
    all_records.sort(key=knowledge_sort_key)

    # Partition by phase
    phase_buckets: Dict[str, List[Dict[str, Any]]] = {p: [] for p in PHASE_ORDER}
    phase_buckets["OTHER"] = []

    for rec in all_records:
        phase = rec.get("phase", "RECON")
        bucket = phase_buckets.get(phase, phase_buckets["OTHER"])
        bucket.append(rec)

    counts: Dict[str, int] = {}
    for phase, recs in phase_buckets.items():
        if not recs:
            continue
        fname = phase.lower() + ".jsonl"
        n = _write_jsonl(KNOW_OUT / fname, recs)
        counts[phase] = n
        console.print(f"  [green]✓[/green] {fname}: {n:,} records")

    total = sum(counts.values())
    console.print(f"  [bold]Total knowledge: {total:,} records[/bold]")
    return counts


# ═══════════════════════════════════════════════════════════════
# 2. Teacher Trajectories → unified (300 JSONL files)
# ═══════════════════════════════════════════════════════════════


def convert_trajectories() -> Dict[str, int]:
    """Convert teacher trajectories → unified/trajectories/by_difficulty/*.jsonl."""
    console.print("[cyan]Converting teacher trajectories...[/cyan]")
    _ensure_dirs()

    all_records: List[Dict[str, Any]] = []
    if not TRAJECTORIES_DIR.exists():
        console.print("[yellow]  No trajectories dir found, skipping[/yellow]")
        return {}

    for jsonl_file in sorted(TRAJECTORIES_DIR.glob("*.jsonl")):
        entries = _read_jsonl_safe(jsonl_file)
        for entry in entries:
            # Each teacher trajectory entry has kind: episode_start|step|episode_end
            kind_raw = entry.get("kind", "teacher_step")
            # Map to unified kinds
            kind_map = {
                "episode_start": "trajectory_start",
                "step": "teacher_step",
                "episode_end": "trajectory_end",
            }
            kind = kind_map.get(kind_raw, "teacher_step")

            rec = wrap_record(
                kind=kind,
                source="teacher_gen",
                data=entry,
                run_id=entry.get("trajectory_id", entry.get("episode_id", "")),
                phase=entry.get("phase", entry.get("highest_phase", "")),
                timestamp=entry.get("timestamp", _iso_now()),
            )
            all_records.append(rec.to_dict())

    # Sort by difficulty then reward desc
    all_records.sort(key=trajectory_sort_key)

    # Partition by difficulty
    diff_buckets: Dict[str, List[Dict[str, Any]]] = {d: [] for d in DIFFICULTY_ORDER}
    diff_buckets["other"] = []

    for rec in all_records:
        diff = rec.get("data", {}).get("difficulty", "medium")
        bucket = diff_buckets.get(diff.lower(), diff_buckets["other"])
        bucket.append(rec)

    counts: Dict[str, int] = {}
    for diff, recs in diff_buckets.items():
        if not recs:
            continue
        fname = diff + ".jsonl"
        n = _write_jsonl(TRAJ_OUT / fname, recs)
        counts[diff] = n
        console.print(f"  [green]✓[/green] {fname}: {n:,} records")

    total = sum(counts.values())
    console.print(f"  [bold]Total trajectories: {total:,} records[/bold]")
    return counts


# ═══════════════════════════════════════════════════════════════
# 3. GPU Training Traces → unified (one file per run_id)
# ═══════════════════════════════════════════════════════════════


def _extract_run_id_from_trace(path: Path) -> str:
    """Extract run_id from trace filename: h200_distill_{run_id}.jsonl"""
    name = path.stem  # e.g., h200_distill_20260221T183438Z
    parts = name.split("_", 2)
    if len(parts) >= 3:
        return parts[2]
    return name


def convert_gpu_traces() -> Dict[str, int]:
    """Convert GPU training traces → unified/gpu_runs/{run_id}.jsonl."""
    console.print("[cyan]Converting GPU training traces...[/cyan]")
    _ensure_dirs()

    counts: Dict[str, int] = {}
    if not TRACES_DIR.exists():
        console.print("[yellow]  No traces dir found, skipping[/yellow]")
        return counts

    for trace_file in sorted(TRACES_DIR.glob("*.jsonl")):
        run_id = _extract_run_id_from_trace(trace_file)
        entries = _read_jsonl_safe(trace_file)

        records: List[Dict[str, Any]] = []
        for entry in entries:
            # Check if already in unified format
            if entry.get("schema_version"):
                records.append(entry)
                continue

            # Wrap raw trace record
            kind = entry.get("kind", "step")
            if kind not in VALID_KINDS:
                kind = "step"

            rec = wrap_record(
                kind=kind,
                source="gpu_distill",
                data=entry,
                run_id=run_id,
                phase=entry.get("phase", entry.get("max_phase", "")),
                timestamp=(
                    datetime.fromtimestamp(entry["timestamp"], tz=timezone.utc).isoformat()
                    if isinstance(entry.get("timestamp"), (int, float))
                    else str(entry.get("timestamp", _iso_now()))
                ),
            )
            records.append(rec.to_dict())

        # Sort chronologically
        records.sort(key=gpu_trace_sort_key)

        out_path = GPU_OUT / f"{run_id}.jsonl"
        n = _write_jsonl(out_path, records)
        counts[run_id] = n
        console.print(f"  [green]✓[/green] {run_id}: {n:,} records")

    # Create 'latest' symlink
    if counts:
        latest_run = max(counts.keys())
        latest_link = GPU_OUT / "latest.jsonl"
        latest_target = f"{latest_run}.jsonl"
        latest_link.unlink(missing_ok=True)
        try:
            latest_link.symlink_to(latest_target)
        except OSError:
            pass  # Windows doesn't support symlinks without admin

    total = sum(counts.values())
    console.print(f"  [bold]Total GPU traces: {total:,} records across {len(counts)} runs[/bold]")
    return counts


def convert_single_gpu_run(trace_path: Path, run_id: str = "") -> Optional[Path]:
    """Convert a single GPU trace file to unified format.

    Called from ``h200_run_distill_3h.py`` after training completes.
    """
    if not trace_path.exists():
        return None

    _ensure_dirs()
    if not run_id:
        run_id = _extract_run_id_from_trace(trace_path)

    entries = _read_jsonl_safe(trace_path)
    records: List[Dict[str, Any]] = []

    for entry in entries:
        if entry.get("schema_version"):
            records.append(entry)
            continue

        kind = entry.get("kind", "step")
        if kind not in VALID_KINDS:
            kind = "step"

        rec = wrap_record(
            kind=kind,
            source="gpu_distill",
            data=entry,
            run_id=run_id,
            phase=entry.get("phase", ""),
            timestamp=(
                datetime.fromtimestamp(entry["timestamp"], tz=timezone.utc).isoformat()
                if isinstance(entry.get("timestamp"), (int, float))
                else str(entry.get("timestamp", _iso_now()))
            ),
        )
        records.append(rec.to_dict())

    records.sort(key=gpu_trace_sort_key)
    out_path = GPU_OUT / f"{run_id}.jsonl"
    _write_jsonl(out_path, records)

    # Update latest symlink
    latest_link = GPU_OUT / "latest.jsonl"
    latest_link.unlink(missing_ok=True)
    try:
        latest_link.symlink_to(f"{run_id}.jsonl")
    except OSError:
        pass

    logger.info("Converted GPU run %s: %d records -> %s", run_id, len(records), out_path)
    return out_path


# ═══════════════════════════════════════════════════════════════
# 4. Learned Commands → unified
# ═══════════════════════════════════════════════════════════════


def convert_learned_commands() -> Dict[str, int]:
    """Convert learned_commands.json → unified/learned_commands/all.jsonl."""
    console.print("[cyan]Converting learned commands...[/cyan]")
    _ensure_dirs()

    if not LEARNED_COMMANDS_PATH.exists():
        console.print("[yellow]  No learned_commands.json found, skipping[/yellow]")
        return {}

    with open(LEARNED_COMMANDS_PATH) as f:
        data = json.load(f)

    commands = data.get("commands", {})
    records: List[Dict[str, Any]] = []

    for key, cmd in commands.items():
        phase = cmd.get("phase", "")
        rec = wrap_record(
            kind="learned_command",
            source="learned_runtime",
            data=cmd,
            phase=_normalize_phase(phase) if phase else "",
            timestamp=cmd.get("last_used", cmd.get("discovered_at", _iso_now())),
        )
        records.append(rec.to_dict())

    # Sort by phase order, then avg_reward desc, then success_count desc
    records.sort(key=learned_command_sort_key)

    out_path = CMD_OUT / "all.jsonl"
    n = _write_jsonl(out_path, records)
    console.print(f"  [green]✓[/green] all.jsonl: {n:,} commands")
    return {"learned_commands": n}


# ═══════════════════════════════════════════════════════════════
# 5. Scenario Profiles → unified
# ═══════════════════════════════════════════════════════════════


def convert_scenarios() -> Dict[str, int]:
    """Convert scenario profiles → unified/scenarios/all.jsonl."""
    console.print("[cyan]Converting scenario profiles...[/cyan]")
    _ensure_dirs()

    if not SCENARIOS_DIR.exists():
        console.print("[yellow]  No scenarios dir found, skipping[/yellow]")
        return {}

    records: List[Dict[str, Any]] = []
    for scenario_file in sorted(SCENARIOS_DIR.glob("*.json")):
        try:
            with open(scenario_file) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logger.warning("Skipped malformed scenario: %s", scenario_file)
            continue

        rec = wrap_record(
            kind="scenario",
            source="scenario_gen",
            data=data,
            run_id=data.get("scenario_id", scenario_file.stem),
            timestamp=_iso_now(),
        )
        records.append(rec.to_dict())

    # Sort by difficulty
    _diff_rank = {"easy": 0, "medium": 1, "hard": 2}
    records.sort(key=lambda r: _diff_rank.get(
        r.get("data", {}).get("difficulty", "medium").lower(), 1
    ))

    out_path = SCEN_OUT / "all.jsonl"
    n = _write_jsonl(out_path, records)
    console.print(f"  [green]✓[/green] all.jsonl: {n:,} scenarios")
    return {"scenarios": n}


# ═══════════════════════════════════════════════════════════════
# Build indices + manifest
# ═══════════════════════════════════════════════════════════════


def _build_indices(
    record_counts: Dict[str, int],
    source_counts: Dict[str, int],
    phase_counts: Dict[str, int],
) -> None:
    """Build index files under data/unified/indices/."""
    _ensure_dirs()

    # stats.json
    stats = {
        "schema_version": UNIFIED_SCHEMA_VERSION,
        "generated_at": _iso_now(),
        "record_counts": record_counts,
        "source_counts": source_counts,
        "phase_counts": phase_counts,
    }
    with open(IDX_OUT / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    # by_phase.json — count records per phase
    with open(IDX_OUT / "by_phase.json", "w") as f:
        json.dump(phase_counts, f, indent=2)

    logger.info("Indices written to %s", IDX_OUT)


# ═══════════════════════════════════════════════════════════════
# Aggregate commands
# ═══════════════════════════════════════════════════════════════


def convert_all() -> None:
    """Convert ALL 5 source formats into unified data."""
    console.print("[bold cyan]═══ Converting ALL training data to unified format ═══[/bold cyan]")
    t0 = time.monotonic()

    record_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}
    phase_counts: Dict[str, int] = {}

    # 1. Knowledge
    kc = convert_knowledge()
    for phase, count in kc.items():
        record_counts["knowledge"] = record_counts.get("knowledge", 0) + count
        source_counts["knowledge_v2"] = source_counts.get("knowledge_v2", 0) + count
        phase_counts[phase] = phase_counts.get(phase, 0) + count

    # 2. Trajectories
    tc = convert_trajectories()
    for diff, count in tc.items():
        record_counts["teacher_step"] = record_counts.get("teacher_step", 0) + count
        source_counts["teacher_gen"] = source_counts.get("teacher_gen", 0) + count

    # 3. GPU traces
    gc = convert_gpu_traces()
    for run_id, count in gc.items():
        record_counts["step"] = record_counts.get("step", 0) + count
        source_counts["gpu_distill"] = source_counts.get("gpu_distill", 0) + count

    # 4. Learned commands
    lc = convert_learned_commands()
    for key, count in lc.items():
        record_counts["learned_command"] = record_counts.get("learned_command", 0) + count
        source_counts["learned_runtime"] = source_counts.get("learned_runtime", 0) + count

    # 5. Scenarios
    sc = convert_scenarios()
    for key, count in sc.items():
        record_counts["scenario"] = record_counts.get("scenario", 0) + count
        source_counts["scenario_gen"] = source_counts.get("scenario_gen", 0) + count

    # Build indices
    _build_indices(record_counts, source_counts, phase_counts)

    # Write manifest
    manifest = write_manifest(
        UNIFIED_DATA_DIR,
        record_counts=record_counts,
        source_counts=source_counts,
        phase_counts=phase_counts,
    )

    elapsed = time.monotonic() - t0
    console.print(
        f"\n[bold green]✓ All conversions complete in {elapsed:.1f}s[/bold green]"
        f" — {manifest.total_records:,} total records"
    )


# ═══════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════


def validate() -> bool:
    """Validate every JSONL line under data/unified/ against schema."""
    console.print("[cyan]Validating unified data...[/cyan]")

    if not UNIFIED_DATA_DIR.exists():
        console.print("[red]data/unified/ does not exist — run convert-all first[/red]")
        return False

    total = 0
    errors_total = 0
    files_checked = 0

    for jsonl_file in sorted(UNIFIED_DATA_DIR.rglob("*.jsonl")):
        records = _read_jsonl_safe(jsonl_file)
        file_errors = 0

        for i, rec in enumerate(records):
            total += 1
            ok, errs = validate_record(rec)
            if not ok:
                file_errors += len(errs)
                if file_errors <= 3:  # Show first 3 errors per file
                    rel = jsonl_file.relative_to(UNIFIED_DATA_DIR)
                    for e in errs:
                        console.print(f"  [red]✗[/red] {rel}:{i+1}: {e}")

        errors_total += file_errors
        files_checked += 1

        status = "[green]✓[/green]" if file_errors == 0 else f"[red]✗ {file_errors} errors[/red]"
        rel = jsonl_file.relative_to(UNIFIED_DATA_DIR)
        console.print(f"  {status} {rel} ({len(records):,} records)")

    console.print(
        f"\n[bold]Validation: {files_checked} files, {total:,} records, "
        f"{errors_total} errors[/bold]"
    )

    if errors_total == 0:
        console.print("[bold green]✓ All records valid[/bold green]")
    else:
        console.print(f"[bold red]✗ {errors_total} validation errors found[/bold red]")

    return errors_total == 0


# ═══════════════════════════════════════════════════════════════
# Stats
# ═══════════════════════════════════════════════════════════════


def stats() -> None:
    """Print summary statistics for the unified data directory."""
    console.print("[cyan]Unified data statistics[/cyan]")

    manifest = load_manifest(UNIFIED_DATA_DIR)
    if manifest is None:
        console.print("[yellow]No manifest.json found — run convert-all first[/yellow]")
        return

    # Overview table
    t = Table(title="Unified Data Overview")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", style="bold")
    t.add_row("Schema Version", manifest.schema_version)
    t.add_row("Total Records", f"{manifest.total_records:,}")
    t.add_row("Created", manifest.created_at[:19])
    t.add_row("Updated", manifest.updated_at[:19])
    t.add_row("Files", str(len(manifest.file_checksums)))
    console.print(t)

    # By kind
    if manifest.record_counts:
        t2 = Table(title="Records by Kind")
        t2.add_column("Kind", style="cyan")
        t2.add_column("Count", justify="right")
        for kind, count in sorted(manifest.record_counts.items(), key=lambda x: -x[1]):
            t2.add_row(kind, f"{count:,}")
        console.print(t2)

    # By source
    if manifest.source_counts:
        t3 = Table(title="Records by Source")
        t3.add_column("Source", style="cyan")
        t3.add_column("Count", justify="right")
        for src, count in sorted(manifest.source_counts.items(), key=lambda x: -x[1]):
            t3.add_row(src, f"{count:,}")
        console.print(t3)

    # By phase
    if manifest.phase_counts:
        t4 = Table(title="Knowledge by Phase")
        t4.add_column("Phase", style="cyan")
        t4.add_column("Count", justify="right")
        for phase in PHASE_ORDER:
            count = manifest.phase_counts.get(phase, 0)
            if count:
                t4.add_row(phase, f"{count:,}")
        other = sum(
            c for p, c in manifest.phase_counts.items()
            if p not in PHASE_ORDER
        )
        if other:
            t4.add_row("OTHER", f"{other:,}")
        console.print(t4)

    # File sizes
    if manifest.file_line_counts:
        t5 = Table(title="Files")
        t5.add_column("File", style="dim")
        t5.add_column("Lines", justify="right")
        for fname, lc in sorted(manifest.file_line_counts.items()):
            t5.add_row(fname, f"{lc:,}")
        console.print(t5)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="unify_training_data",
        description="Convert all Ariaska training data to unified format",
    )
    parser.add_argument(
        "command",
        choices=[
            "convert-all",
            "convert-knowledge",
            "convert-traces",
            "convert-commands",
            "convert-trajectories",
            "convert-scenarios",
            "validate",
            "stats",
        ],
        help="Subcommand to run",
    )
    args = parser.parse_args(argv)

    if args.command == "convert-all":
        convert_all()
    elif args.command == "convert-knowledge":
        convert_knowledge()
    elif args.command == "convert-traces":
        convert_gpu_traces()
    elif args.command == "convert-commands":
        convert_learned_commands()
    elif args.command == "convert-trajectories":
        convert_trajectories()
    elif args.command == "convert-scenarios":
        convert_scenarios()
    elif args.command == "validate":
        ok = validate()
        return 0 if ok else 1
    elif args.command == "stats":
        stats()
    return 0


if __name__ == "__main__":
    sys.exit(main())
