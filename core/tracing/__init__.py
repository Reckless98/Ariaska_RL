"""
ARIASKA Tracing Module

Provides structured JSONL logging for training runs.
Phase 8: Added DecisionLogger for multi-brain telemetry.
"""

from core.tracing.episode_trace import (
    StepTrace,
    EpisodeTrace,
    RunTrace,
    TraceWriter,
    TraceReader,
    TraceEventType,
    create_trace_writer,
    load_trace,
    validate_step_trace,
    validate_event_id_format,
    parse_event_id,
    validate_evidence_refs,
    STEP_TRACE_SCHEMA,
)
from core.tracing.jsonl_logger import (
    DecisionLogEntry,
    DecisionLogger,
    create_decision_logger,
)

__all__ = [
    "StepTrace",
    "EpisodeTrace", 
    "RunTrace",
    "TraceWriter",
    "TraceReader",
    "TraceEventType",
    "create_trace_writer",
    "load_trace",
    "validate_step_trace",
    "validate_event_id_format",
    "parse_event_id",
    "validate_evidence_refs",
    "STEP_TRACE_SCHEMA",
    # Phase 8: Multi-brain decision logger
    "DecisionLogEntry",
    "DecisionLogger",
    "create_decision_logger",
]
