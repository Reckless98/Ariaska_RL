"""Tests for scripts.unified_data_schema — unified envelope + helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")

from scripts.unified_data_schema import (
    DIFFICULTY_ORDER,
    PHASE_ORDER,
    UNIFIED_SCHEMA_VERSION,
    VALID_KINDS,
    VALID_SOURCES,
    UnifiedManifest,
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


# ── UnifiedRecord ──────────────────────────────────────────────


class TestUnifiedRecord:
    def test_create_default(self) -> None:
        r = UnifiedRecord()
        assert r.schema_version == UNIFIED_SCHEMA_VERSION
        assert r.kind == ""
        assert r.data == {}

    def test_to_dict_roundtrip(self) -> None:
        r = UnifiedRecord(kind="step", source="gpu_distill", data={"x": 1})
        d = r.to_dict()
        assert isinstance(d, dict)
        assert d["kind"] == "step"
        assert d["data"] == {"x": 1}

    def test_to_json_compact(self) -> None:
        r = UnifiedRecord(kind="step", source="gpu_distill", data={"x": 1})
        j = r.to_json()
        assert "\n" not in j
        parsed = json.loads(j)
        assert parsed["kind"] == "step"


# ── wrap_record ────────────────────────────────────────────────


class TestWrapRecord:
    def test_basic_wrap(self) -> None:
        r = wrap_record(kind="step", source="gpu_distill", data={"cmd": "nmap"})
        assert r.kind == "step"
        assert r.source == "gpu_distill"
        assert r.schema_version == UNIFIED_SCHEMA_VERSION
        assert r.timestamp  # auto-generated

    def test_optional_fields(self) -> None:
        r = wrap_record(
            kind="knowledge",
            source="knowledge_v2",
            data={"id": "abc"},
            run_id="run_1",
            phase="RECON",
            timestamp="2025-01-01T00:00:00Z",
        )
        assert r.run_id == "run_1"
        assert r.phase == "RECON"
        assert r.timestamp == "2025-01-01T00:00:00Z"


# ── validate_record ───────────────────────────────────────────


class TestValidateRecord:
    def test_valid_record(self) -> None:
        rec = wrap_record(kind="step", source="gpu_distill", data={"x": 1}).to_dict()
        ok, errs = validate_record(rec)
        assert ok
        assert errs == []

    def test_missing_schema_version(self) -> None:
        rec = {"kind": "step", "source": "gpu_distill", "data": {}, "timestamp": "now"}
        ok, errs = validate_record(rec)
        assert not ok
        assert any("schema_version" in e for e in errs)

    def test_invalid_kind(self) -> None:
        rec = {
            "schema_version": "3.0.0",
            "kind": "bogus",
            "source": "gpu_distill",
            "data": {},
            "timestamp": "now",
        }
        ok, errs = validate_record(rec)
        assert not ok
        assert any("kind" in e for e in errs)

    def test_invalid_source(self) -> None:
        rec = {
            "schema_version": "3.0.0",
            "kind": "step",
            "source": "unknown_source",
            "data": {},
            "timestamp": "now",
        }
        ok, errs = validate_record(rec)
        assert not ok
        assert any("source" in e for e in errs)

    def test_missing_data(self) -> None:
        rec = {
            "schema_version": "3.0.0",
            "kind": "step",
            "source": "gpu_distill",
            "timestamp": "now",
        }
        ok, errs = validate_record(rec)
        assert not ok
        assert any("data" in e for e in errs)

    def test_missing_timestamp(self) -> None:
        rec = {
            "schema_version": "3.0.0",
            "kind": "step",
            "source": "gpu_distill",
            "data": {},
        }
        ok, errs = validate_record(rec)
        assert not ok
        assert any("timestamp" in e for e in errs)

    def test_all_valid_kinds(self) -> None:
        for kind in VALID_KINDS:
            rec = {
                "schema_version": "3.0.0",
                "kind": kind,
                "source": "gpu_distill",
                "data": {},
                "timestamp": "now",
            }
            ok, _ = validate_record(rec)
            assert ok, f"kind={kind} should be valid"

    def test_all_valid_sources(self) -> None:
        for source in VALID_SOURCES:
            rec = {
                "schema_version": "3.0.0",
                "kind": "step",
                "source": source,
                "data": {},
                "timestamp": "now",
            }
            ok, _ = validate_record(rec)
            assert ok, f"source={source} should be valid"


# ── Sort keys ──────────────────────────────────────────────────


class TestSortKeys:
    def test_knowledge_sort_by_phase(self) -> None:
        rec_recon = {"data": {"taxonomy": {"killchain_step": "RECON"}}}
        rec_exploit = {"data": {"taxonomy": {"killchain_step": "EXPLOITATION"}}}
        assert knowledge_sort_key(rec_recon) < knowledge_sort_key(rec_exploit)

    def test_knowledge_sort_by_evidence(self) -> None:
        """Higher evidence_coverage should come first (negative for desc)."""
        rec_high = {
            "data": {
                "taxonomy": {"killchain_step": "RECON"},
                "governance": {"quality": {"evidence_coverage": 0.9}},
            }
        }
        rec_low = {
            "data": {
                "taxonomy": {"killchain_step": "RECON"},
                "governance": {"quality": {"evidence_coverage": 0.3}},
            }
        }
        assert knowledge_sort_key(rec_high) < knowledge_sort_key(rec_low)

    def test_trajectory_sort_by_difficulty(self) -> None:
        rec_easy = {"data": {"difficulty": "easy", "total_reward": 10}}
        rec_hard = {"data": {"difficulty": "hard", "total_reward": 10}}
        assert trajectory_sort_key(rec_easy) < trajectory_sort_key(rec_hard)

    def test_trajectory_sort_by_reward(self) -> None:
        """Higher reward should come first within same difficulty."""
        rec_high = {"data": {"difficulty": "medium", "total_reward": 50}}
        rec_low = {"data": {"difficulty": "medium", "total_reward": 10}}
        assert trajectory_sort_key(rec_high) < trajectory_sort_key(rec_low)

    def test_learned_command_sort(self) -> None:
        rec_recon = {"data": {"phase": "RECON", "avg_reward": 5.0, "success_count": 10}}
        rec_privesc = {"data": {"phase": "PRIVILEGE_ESCALATION", "avg_reward": 5.0, "success_count": 10}}
        assert learned_command_sort_key(rec_recon) < learned_command_sort_key(rec_privesc)

    def test_gpu_trace_sort_chronological(self) -> None:
        rec_early = {"timestamp": "2025-01-01T00:00:00Z"}
        rec_late = {"timestamp": "2025-12-31T00:00:00Z"}
        assert gpu_trace_sort_key(rec_early) < gpu_trace_sort_key(rec_late)


# ── Manifest I/O ──────────────────────────────────────────────


class TestManifest:
    def test_write_and_load(self, tmp_path: Path) -> None:
        # Create a dummy JSONL file
        data_dir = tmp_path / "unified"
        data_dir.mkdir()
        jsonl = data_dir / "test.jsonl"
        rec = wrap_record(kind="step", source="gpu_distill", data={"x": 1})
        jsonl.write_text(rec.to_json() + "\n")

        # Write manifest
        m = write_manifest(
            data_dir,
            record_counts={"step": 1},
            source_counts={"gpu_distill": 1},
        )
        assert m.total_records == 1
        assert m.schema_version == UNIFIED_SCHEMA_VERSION
        assert "test.jsonl" in m.file_checksums
        assert m.file_line_counts["test.jsonl"] == 1

        # Load manifest
        loaded = load_manifest(data_dir)
        assert loaded is not None
        assert loaded.total_records == 1
        assert loaded.record_counts == {"step": 1}

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        assert load_manifest(tmp_path) is None

    def test_preserves_created_at(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "unified"
        data_dir.mkdir()
        jsonl = data_dir / "test.jsonl"
        jsonl.write_text('{"schema_version":"3.0.0","kind":"step","source":"gpu_distill","data":{},"timestamp":"x"}\n')

        m1 = write_manifest(data_dir)
        created = m1.created_at

        # Write again — created_at should be preserved
        m2 = write_manifest(data_dir)
        assert m2.created_at == created


# ── Constants ──────────────────────────────────────────────────


class TestConstants:
    def test_phase_order_length(self) -> None:
        assert len(PHASE_ORDER) == 8

    def test_difficulty_order(self) -> None:
        assert DIFFICULTY_ORDER == ["easy", "medium", "hard"]

    def test_valid_kinds_count(self) -> None:
        assert len(VALID_KINDS) == 9

    def test_valid_sources_count(self) -> None:
        assert len(VALID_SOURCES) == 7

    def test_schema_version(self) -> None:
        assert UNIFIED_SCHEMA_VERSION == "3.0.0"
