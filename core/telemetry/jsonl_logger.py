#!/usr/bin/env python3
"""
core/telemetry/jsonl_logger.py — Phase 9.7: Buffered JSONL event writer

Writes StepEvent and EpisodeEvent records to a JSONL file with:
- Line-buffered writes (flush every N events or on episode end)
- Uses orjson if available (5-10x faster), falls back to json
- Thread-safe
- Configurable output path
- Rotation support (new file per run)

Controlled by FF_JSONL_TELEMETRY feature flag.
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("ariaska.telemetry")

# Try orjson for performance
try:
    import orjson
    def _dumps(obj: Any) -> str:
        return orjson.dumps(obj).decode("utf-8")
except ImportError:
    def _dumps(obj: Any) -> str:
        return json.dumps(obj, separators=(",", ":"), default=str)


class JSONLLogger:
    """Buffered JSONL event logger for training telemetry.

    Usage:
        logger = JSONLLogger(run_id="run_20260215", output_dir="logs/telemetry")
        logger.log_step(step_event)       # buffers
        logger.log_episode(episode_event) # flushes
        logger.close()                    # final flush
    """

    def __init__(
        self,
        run_id: str = "",
        output_dir: str = "logs/telemetry",
        buffer_size: int = 50,
        enabled: bool = True,
    ):
        self.run_id = run_id or f"run_{int(time.time())}"
        self.output_dir = Path(output_dir)
        self.buffer_size = buffer_size
        self.enabled = enabled
        self._lock = threading.Lock()
        self._buffer: List[str] = []
        self._file = None
        self._path: Optional[Path] = None
        self._total_events = 0
        self._total_bytes = 0

        if self.enabled:
            self._open()

    def _open(self):
        """Open the JSONL output file."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._path = self.output_dir / f"{self.run_id}_telemetry.jsonl"
            self._file = open(self._path, "a", encoding="utf-8")
            logger.info(f"[TELEMETRY] JSONL logger opened: {self._path}")
        except Exception as e:
            logger.warning(f"[TELEMETRY] Failed to open JSONL file: {e}")
            self.enabled = False

    def log_step(self, event: Any) -> None:
        """Log a step event (buffered).

        Args:
            event: StepEvent or dict-like with .to_dict() method
        """
        if not self.enabled:
            return
        try:
            d = event.to_dict() if hasattr(event, "to_dict") else event
            line = _dumps(d)
            with self._lock:
                self._buffer.append(line)
                if len(self._buffer) >= self.buffer_size:
                    self._flush_unsafe()
        except Exception as e:
            logger.debug(f"[TELEMETRY] Failed to log step: {e}")

    def log_episode(self, event: Any) -> None:
        """Log an episode summary event (flushes immediately).

        Args:
            event: EpisodeEvent or dict-like with .to_dict() method
        """
        if not self.enabled:
            return
        try:
            d = event.to_dict() if hasattr(event, "to_dict") else event
            line = _dumps(d)
            with self._lock:
                self._buffer.append(line)
                self._flush_unsafe()
        except Exception as e:
            logger.debug(f"[TELEMETRY] Failed to log episode: {e}")

    def log_raw(self, record: Dict[str, Any]) -> None:
        """Log a raw dict as JSONL (for transparent mode events)."""
        if not self.enabled:
            return
        try:
            line = _dumps(record)
            with self._lock:
                self._buffer.append(line)
                if len(self._buffer) >= self.buffer_size:
                    self._flush_unsafe()
        except Exception as e:
            logger.debug(f"[TELEMETRY] Failed to log raw: {e}")

    def flush(self) -> None:
        """Explicitly flush buffer to disk."""
        with self._lock:
            self._flush_unsafe()

    def _flush_unsafe(self) -> None:
        """Flush without acquiring lock (caller must hold lock)."""
        if not self._buffer or self._file is None:
            return
        try:
            data = "\n".join(self._buffer) + "\n"
            self._file.write(data)
            self._file.flush()
            self._total_events += len(self._buffer)
            self._total_bytes += len(data.encode("utf-8"))
            self._buffer.clear()
        except Exception as e:
            logger.debug(f"[TELEMETRY] Flush failed: {e}")

    def close(self) -> None:
        """Flush and close the output file."""
        with self._lock:
            self._flush_unsafe()
            if self._file is not None:
                try:
                    self._file.close()
                except Exception:
                    pass
                self._file = None
        logger.info(
            f"[TELEMETRY] Closed. {self._total_events} events, "
            f"{self._total_bytes:,} bytes → {self._path}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Return logger stats for metrics."""
        return {
            "total_events": self._total_events + len(self._buffer),
            "total_bytes": self._total_bytes,
            "buffer_pending": len(self._buffer),
            "path": str(self._path) if self._path else None,
            "enabled": self.enabled,
        }

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
