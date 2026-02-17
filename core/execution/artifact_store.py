#!/usr/bin/env python3
"""
core/execution/artifact_store.py — Binary-safe artifact capture pipeline.

When live tools produce binary blobs (PCAP, downloaded files, archives),
this module captures them byte-for-byte with SHA-256 digests and metadata.

Used by SandboxedExecutor and LiveCommandExecutor when `capture_binary=True`
is set, or when output contains known binary signatures.

Architecture:
    LiveCommandExecutor.execute(cmd, capture_binary=True)
        → ArtifactStore.store(raw_bytes, metadata)
            → writes to artifacts/<episode>/<step>/<sha256[:12]>.<ext>

Author: Filip Volf / Ariaska System
Phase: HTB Capability Upgrade — T0.1
"""

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariaska.artifact_store")

# ─── Binary file signatures ─────────────────────────────────────────────
# Used for auto-detection when `capture_binary` is not explicitly set.
BINARY_SIGNATURES: Dict[bytes, str] = {
    b"\xd4\xc3\xb2\xa1": "pcap",       # PCAP (little-endian)
    b"\xa1\xb2\xc3\xd4": "pcap",       # PCAP (big-endian)
    b"\x0a\x0d\x0d\x0a": "pcapng",     # PCAPNG
    b"\x7fELF": "elf",                   # ELF binary
    b"PK\x03\x04": "zip",               # ZIP archive
    b"\x1f\x8b": "gz",                   # Gzip
    b"BZ": "bz2",                        # Bzip2
    b"\xfd7zXZ": "xz",                  # XZ
    b"%PDF": "pdf",                      # PDF
    b"\x89PNG": "png",                   # PNG image
}


@dataclass
class StoredArtifact:
    """Metadata for a stored binary artifact."""
    sha256: str
    path: str
    size_bytes: int
    file_type: str           # Extension/type hint: "pcap", "elf", "zip", etc.
    source_command: str = ""
    agent_name: str = ""
    episode_id: int = 0
    step_idx: int = 0
    target_ip: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for logging/JSON."""
        return {
            "sha256": self.sha256,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "file_type": self.file_type,
            "source_command": self.source_command[:120],
            "agent_name": self.agent_name,
            "episode_id": self.episode_id,
            "step_idx": self.step_idx,
            "target_ip": self.target_ip,
            "timestamp": self.timestamp,
        }


def detect_binary_type(data: bytes) -> Optional[str]:
    """Detect file type from magic bytes.

    Args:
        data: First 8+ bytes of the file.

    Returns:
        File type string (e.g., 'pcap', 'elf') or None if unknown.
    """
    if len(data) < 2:
        return None
    for sig, file_type in BINARY_SIGNATURES.items():
        if data[:len(sig)] == sig:
            return file_type
    return None


def is_likely_binary(data: bytes, threshold: float = 0.30) -> bool:
    """Heuristic check: is this data likely binary (not text)?

    Args:
        data: Raw bytes to check.
        threshold: Fraction of non-printable bytes to trigger binary classification.

    Returns:
        True if data appears to be binary content.
    """
    if not data:
        return False
    # Check magic bytes first
    if detect_binary_type(data) is not None:
        return True
    # Heuristic: count non-printable bytes in first 512 bytes
    sample = data[:512]
    non_printable = sum(
        1 for b in sample
        if b < 0x20 and b not in (0x09, 0x0A, 0x0D)  # tab, newline, carriage return
    )
    return (non_printable / max(len(sample), 1)) > threshold


class ArtifactStore:
    """
    Binary-safe artifact storage.

    Captures bytes from command output, writes them to disk with SHA-256
    naming, and maintains an in-memory manifest for the current episode.

    Usage:
        store = ArtifactStore(base_dir="artifacts")
        artifact = store.store(
            data=raw_pcap_bytes,
            file_type="pcap",
            command="curl -o /tmp/cap.pcap http://target/data/0",
            agent_name="ScoutAgent",
            episode_id=42,
            step_idx=7,
        )
        # artifact.path → "artifacts/ep042/s007/a1b2c3d4e5f6.pcap"
    """

    def __init__(self, base_dir: str = "artifacts/captures"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._manifest: List[StoredArtifact] = []
        self._hashes_seen: set = set()  # Dedup by SHA-256

    def store(
        self,
        data: bytes,
        file_type: str = "",
        command: str = "",
        agent_name: str = "",
        episode_id: int = 0,
        step_idx: int = 0,
        target_ip: str = "",
    ) -> Optional[StoredArtifact]:
        """
        Store a binary artifact to disk.

        Args:
            data: Raw bytes to store.
            file_type: File extension hint (auto-detected if empty).
            command: The command that produced this data.
            agent_name: Agent that executed the command.
            episode_id: Current episode number.
            step_idx: Current step number.
            target_ip: Target IP address.

        Returns:
            StoredArtifact with path and metadata, or None if duplicate/empty.
        """
        if not data:
            return None

        # Compute SHA-256
        sha256 = hashlib.sha256(data).hexdigest()

        # Dedup: skip if we already have this exact content
        if sha256 in self._hashes_seen:
            logger.debug(f"[ARTIFACT] Duplicate skipped: {sha256[:12]}")
            return None
        self._hashes_seen.add(sha256)

        # Auto-detect file type if not specified
        if not file_type:
            file_type = detect_binary_type(data) or "bin"

        # Build output path: artifacts/captures/ep042/s007/a1b2c3d4e5f6.pcap
        ep_dir = self.base_dir / f"ep{episode_id:03d}" / f"s{step_idx:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{sha256[:12]}.{file_type}"
        filepath = ep_dir / filename

        # Write bytes
        try:
            filepath.write_bytes(data)
        except Exception as e:
            logger.error(f"[ARTIFACT] Failed to write {filepath}: {e}")
            return None

        artifact = StoredArtifact(
            sha256=sha256,
            path=str(filepath),
            size_bytes=len(data),
            file_type=file_type,
            source_command=command,
            agent_name=agent_name,
            episode_id=episode_id,
            step_idx=step_idx,
            target_ip=target_ip,
        )
        self._manifest.append(artifact)

        logger.info(
            f"[ARTIFACT] Stored {file_type} artifact: {filepath} "
            f"({len(data)} bytes, sha256={sha256[:12]})"
        )
        return artifact

    def get_manifest(self) -> List[StoredArtifact]:
        """Return all artifacts stored this session."""
        return list(self._manifest)

    def get_by_type(self, file_type: str) -> List[StoredArtifact]:
        """Return all artifacts of a given type (e.g., 'pcap')."""
        return [a for a in self._manifest if a.file_type == file_type]

    def get_latest(self, file_type: str = "") -> Optional[StoredArtifact]:
        """Return the most recent artifact, optionally filtered by type."""
        candidates = self._manifest if not file_type else self.get_by_type(file_type)
        return candidates[-1] if candidates else None

    def reset_episode(self) -> None:
        """Clear manifest for new episode (files remain on disk)."""
        self._manifest.clear()
        self._hashes_seen.clear()
        logger.debug("[ARTIFACT] Episode manifest reset")

    def total_bytes(self) -> int:
        """Total bytes stored across all artifacts."""
        return sum(a.size_bytes for a in self._manifest)
