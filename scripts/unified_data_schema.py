"""Unified data schema for ALL Ariaska training data.

Provides a common envelope wrapping every data format produced or consumed
by the training pipeline: knowledge corpus, teacher trajectories, GPU traces,
learned commands, and scenario profiles.

Usage::

    from scripts.unified_data_schema import (
        UnifiedRecord, UnifiedManifest, wrap_record, validate_record,
        write_manifest, load_manifest,
    )

    rec = wrap_record(
        kind="step", source="gpu_distill",
        data={"command": "nmap ...", "reward": 5.0},
        run_id="20260221T183438Z", phase="RECON",
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

logger = logging.getLogger("ariaska.unified_schema")

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

UNIFIED_SCHEMA_VERSION = "3.0.0"

UNIFIED_DATA_DIR = Path("data/unified")

VALID_KINDS: FrozenSet[str] = frozenset({
    # GPU training trace records
    "step",
    "episode_start",
    "episode_end",
    # Teacher trajectory records
    "teacher_step",
    "trajectory_start",
    "trajectory_end",
    # Corpus / intelligence
    "knowledge",
    "learned_command",
    "scenario",
})

VALID_SOURCES: FrozenSet[str] = frozenset({
    "gpu_distill",
    "teacher_gen",
    "live_train",
    "knowledge_v2",
    "learned_runtime",
    "scenario_gen",
    "synthetic_trace",
})

# Canonical kill-chain phase order (shared with trace_schema.py)
PHASE_ORDER: List[str] = [
    "RECON",
    "ENUMERATION",
    "EXPLOITATION",
    "PRIVILEGE_ESCALATION",
    "LATERAL_MOVEMENT",
    "POST_EXPLOITATION",
    "EXFILTRATION",
    "CLOSEOUT",
]

_PHASE_RANK: Dict[str, int] = {p: i for i, p in enumerate(PHASE_ORDER)}

DIFFICULTY_ORDER: List[str] = ["easy", "medium", "hard"]
_DIFF_RANK: Dict[str, int] = {d: i for i, d in enumerate(DIFFICULTY_ORDER)}


# ═══════════════════════════════════════════════════════════════
# Core dataclasses
# ═══════════════════════════════════════════════════════════════


@dataclass
class UnifiedRecord:
    """Common envelope for ALL training data.

    Every JSONL line in the unified data directory is one of these.
    The ``kind`` field determines what payload lives in ``data``.
    """

    schema_version: str = UNIFIED_SCHEMA_VERSION
    kind: str = ""
    source: str = ""
    timestamp: str = ""
    run_id: str = ""
    phase: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return asdict(self)

    def to_json(self) -> str:
        """Compact JSON line (no trailing newline)."""
        return json.dumps(self.to_dict(), default=str, separators=(",", ":"))


@dataclass
class UnifiedManifest:
    """Global index of the unified data directory."""

    schema_version: str = UNIFIED_SCHEMA_VERSION
    created_at: str = ""
    updated_at: str = ""
    total_records: int = 0
    record_counts: Dict[str, int] = field(default_factory=dict)
    source_counts: Dict[str, int] = field(default_factory=dict)
    phase_counts: Dict[str, int] = field(default_factory=dict)
    file_checksums: Dict[str, str] = field(default_factory=dict)
    file_line_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════


def wrap_record(
    kind: str,
    source: str,
    data: Dict[str, Any],
    run_id: str = "",
    phase: str = "",
    timestamp: Optional[str] = None,
) -> UnifiedRecord:
    """Create a ``UnifiedRecord`` with defaults filled in."""
    if timestamp is None:
        timestamp = datetime.now(tz=timezone.utc).isoformat()
    return UnifiedRecord(
        schema_version=UNIFIED_SCHEMA_VERSION,
        kind=kind,
        source=source,
        timestamp=timestamp,
        run_id=run_id,
        phase=phase,
        data=data,
    )


def validate_record(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a raw dict against the ``UnifiedRecord`` schema.

    Returns:
        (is_valid, list_of_error_strings)
    """
    errors: List[str] = []

    sv = record.get("schema_version", "")
    if not sv:
        errors.append("missing schema_version")

    kind = record.get("kind", "")
    if kind not in VALID_KINDS:
        errors.append(f"invalid kind={kind!r}, expected one of {sorted(VALID_KINDS)}")

    source = record.get("source", "")
    if source not in VALID_SOURCES:
        errors.append(f"invalid source={source!r}, expected one of {sorted(VALID_SOURCES)}")

    if "data" not in record or not isinstance(record.get("data"), dict):
        errors.append("missing or non-dict 'data' field")

    ts = record.get("timestamp", "")
    if not ts:
        errors.append("missing timestamp")

    return (len(errors) == 0, errors)


# ═══════════════════════════════════════════════════════════════
# Sort key helpers
# ═══════════════════════════════════════════════════════════════


def knowledge_sort_key(record: Dict[str, Any]) -> Tuple[int, float]:
    """Sort knowledge by killchain_step order, then evidence_coverage descending."""
    data = record.get("data", {})
    # KnowledgeCandidate has taxonomy.killchain_step and governance.quality.evidence_coverage
    taxonomy = data.get("taxonomy", {})
    ks = taxonomy.get("killchain_step", "")
    phase_rank = _PHASE_RANK.get(ks.upper(), 99)

    governance = data.get("governance", {})
    quality = governance.get("quality", {})
    evidence = quality.get("evidence_coverage", 0.0)
    # Negate for descending sort
    return (phase_rank, -float(evidence))


def trajectory_sort_key(record: Dict[str, Any]) -> Tuple[int, float]:
    """Sort trajectories by difficulty, then total_reward descending."""
    data = record.get("data", {})
    diff = data.get("difficulty", "medium")
    diff_rank = _DIFF_RANK.get(diff.lower(), 1)
    reward = data.get("total_reward", 0.0)
    return (diff_rank, -float(reward))


def learned_command_sort_key(record: Dict[str, Any]) -> Tuple[int, float, float]:
    """Sort learned commands by phase order, then avg_reward desc, then success_count desc."""
    data = record.get("data", {})
    phase = data.get("phase", "")
    phase_rank = _PHASE_RANK.get(phase.upper(), 99)
    avg_reward = data.get("avg_reward", 0.0)
    success_count = data.get("success_count", 0)
    return (phase_rank, -float(avg_reward), -float(success_count))


def gpu_trace_sort_key(record: Dict[str, Any]) -> str:
    """Sort GPU traces chronologically by timestamp."""
    return record.get("timestamp", "")


# ═══════════════════════════════════════════════════════════════
# Manifest I/O
# ═══════════════════════════════════════════════════════════════


def _file_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_lines(path: Path) -> int:
    """Count lines in a file efficiently."""
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


def write_manifest(
    data_dir: Path,
    record_counts: Optional[Dict[str, int]] = None,
    source_counts: Optional[Dict[str, int]] = None,
    phase_counts: Optional[Dict[str, int]] = None,
) -> UnifiedManifest:
    """Build and write ``manifest.json`` for the unified data directory.

    Scans all ``.jsonl`` files under *data_dir* to compute checksums and
    line counts.  Caller should supply aggregated counts; if omitted they
    are initialised empty.
    """
    now = datetime.now(tz=timezone.utc).isoformat()
    manifest_path = data_dir / "manifest.json"

    # Scan for existing manifest to preserve created_at
    created_at = now
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                old = json.load(f)
            created_at = old.get("created_at", now)
        except Exception:
            pass

    checksums: Dict[str, str] = {}
    line_counts: Dict[str, int] = {}
    total = 0

    for jsonl in sorted(data_dir.rglob("*.jsonl")):
        rel = str(jsonl.relative_to(data_dir))
        checksums[rel] = _file_sha256(jsonl)
        lc = _count_lines(jsonl)
        line_counts[rel] = lc
        total += lc

    manifest = UnifiedManifest(
        schema_version=UNIFIED_SCHEMA_VERSION,
        created_at=created_at,
        updated_at=now,
        total_records=total,
        record_counts=record_counts or {},
        source_counts=source_counts or {},
        phase_counts=phase_counts or {},
        file_checksums=checksums,
        file_line_counts=line_counts,
    )

    data_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2, default=str)
    logger.info("Manifest written: %s (%d total records)", manifest_path, total)
    return manifest


def load_manifest(data_dir: Path) -> Optional[UnifiedManifest]:
    """Load ``manifest.json`` from *data_dir*, or return ``None``."""
    path = data_dir / "manifest.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            d = json.load(f)
        return UnifiedManifest(**{
            k: v for k, v in d.items()
            if k in UnifiedManifest.__dataclass_fields__
        })
    except Exception as exc:
        logger.warning("Failed to load manifest %s: %s", path, exc)
        return None
