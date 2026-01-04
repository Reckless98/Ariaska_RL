#!/usr/bin/env python3
"""
core/tracing/episode_trace.py — ARIASKA Episode Tracing System v1.0
Structured JSONL logging for training runs with deterministic reproducibility.

Features:
- EpisodeTrace: Per-episode structured logging
- RunTrace: Aggregates all episodes in a training run
- TraceReader: Load and normalize traces for analysis
- Schema validation for all trace events
"""

import os
import json
import time
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger("ariaska.tracing")


class TraceEventType(str, Enum):
    """Types of trace events."""
    STEP = "step"
    EPISODE_START = "episode_start"
    EPISODE_END = "episode_end"
    RUN_START = "run_start"
    RUN_END = "run_end"
    MENTOR_CALL = "mentor_call"
    PHASE_TRANSITION = "phase_transition"
    ERROR = "error"
    CHECKPOINT = "checkpoint"


@dataclass
class StepTrace:
    """
    Trace of a single training step.
    
    Captures all relevant information for reproducibility and analysis.
    
    Canonical fields:
    - event_id: Deterministic ID format: "{episode_id}:{step:04d}:{agent}"
    - agent: Agent identifier (not agent_id)
    - chosen_action: Final action taken (not action_final)
    """
    episode_id: str
    step: int
    agent: str  # Canonical name (replaces agent_id)
    phase: str
    
    # Action information
    proposed_action: str  # Agent's neural network proposal
    chosen_action: str    # Final action taken (may differ if mentor overrides)
    
    # Mentor/GPT information
    mentor_call: bool = False
    model_used: Optional[str] = None  # "gpt-5-mini", "gpt-5-nano", None if no mentor
    
    # Outcome
    reward: Optional[float] = None
    done: bool = False
    
    # Optional fields (excluded from canonical trace if None)
    action_params: Optional[Dict[str, Any]] = None
    command: Optional[str] = None  # Actual command if applicable
    observation_summary: str = ""
    output_summary: str = ""
    mentor_response: Optional[str] = None
    confidence: float = 0.5  # Agent's confidence in its proposal
    q_values: Optional[List[float]] = None
    epsilon: float = 1.0
    error_flag: bool = False
    error_message: Optional[str] = None
    
    # Token tracking (NEW: for observability)
    tokens_used_step: int = 0           # Tokens used in this step
    tokens_used_episode: int = 0        # Cumulative tokens in episode
    
    # Reward breakdown (NEW: for observability)
    reward_breakdown: Optional[Dict[str, float]] = None  # Detailed reward components
    
    # Timestamp stored separately (not part of deterministic trace)
    _timestamp: Optional[float] = field(default=None, repr=False)
    
    @property
    def event_id(self) -> str:
        """Deterministic event ID format: {episode_id}:{step:04d}:{agent}"""
        return f"{self.episode_id}:{self.step:04d}:{self.agent}"
    
    @property
    def timestamp(self) -> float:
        """Get timestamp (or current time if not set)."""
        return self._timestamp if self._timestamp is not None else time.time()
    
    @timestamp.setter
    def timestamp(self, value: float):
        self._timestamp = value
    
    def to_dict(self, include_timestamp: bool = True) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "event_id": self.event_id,
            "episode_id": self.episode_id,
            "step": self.step,
            "agent": self.agent,
            "phase": self.phase,
            "proposed_action": self.proposed_action,
            "chosen_action": self.chosen_action,
            "mentor_call": self.mentor_call,
            "model_used": self.model_used,
            "reward": self.reward,
            "done": self.done,
        }
        
        # Add optional fields if present
        if self.action_params:
            result["action_params"] = self.action_params
        if self.command:
            result["command"] = self.command
        if self.observation_summary:
            result["observation_summary"] = self.observation_summary
        if self.output_summary:
            result["output_summary"] = self.output_summary
        if self.mentor_response:
            result["mentor_response"] = self.mentor_response
        if self.confidence != 0.5:
            result["confidence"] = self.confidence
        if self.q_values:
            result["q_values"] = self.q_values
        if self.epsilon != 1.0:
            result["epsilon"] = self.epsilon
        if self.error_flag:
            result["error_flag"] = self.error_flag
            result["error_message"] = self.error_message
        
        # Token tracking (NEW: for observability)
        if self.tokens_used_step > 0:
            result["tokens_used_step"] = self.tokens_used_step
        if self.tokens_used_episode > 0:
            result["tokens_used_episode"] = self.tokens_used_episode
        
        # Reward breakdown (NEW: for observability)
        if self.reward_breakdown:
            result["reward_breakdown"] = self.reward_breakdown
        
        # Timestamp is optional (for determinism)
        if include_timestamp and self._timestamp is not None:
            result["timestamp"] = self._timestamp
        
        return result
    
    def to_json(self, include_timestamp: bool = True) -> str:
        """Convert to JSON string (for JSONL)."""
        return json.dumps(self.to_dict(include_timestamp=include_timestamp))
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepTrace":
        """Create StepTrace from dictionary."""
        # Handle field name migrations
        agent = data.get("agent") or data.get("agent_id", "unknown")
        chosen_action = data.get("chosen_action") or data.get("action_final", "")
        proposed_action = data.get("proposed_action") or data.get("action_proposed", "")
        model_used = data.get("model_used") or data.get("mentor_model")
        
        step = cls(
            episode_id=data.get("episode_id", ""),
            step=data.get("step", 0),
            agent=agent,
            phase=data.get("phase", "unknown"),
            proposed_action=proposed_action,
            chosen_action=chosen_action,
            mentor_call=data.get("mentor_call", False),
            model_used=model_used,
            reward=data.get("reward"),
            done=data.get("done", False),
            action_params=data.get("action_params"),
            command=data.get("command"),
            observation_summary=data.get("observation_summary", ""),
            output_summary=data.get("output_summary", ""),
            mentor_response=data.get("mentor_response"),
            confidence=data.get("confidence", 0.5),
            q_values=data.get("q_values"),
            epsilon=data.get("epsilon", 1.0),
            error_flag=data.get("error_flag", False),
            error_message=data.get("error_message"),
        )
        step._timestamp = data.get("timestamp")
        return step


@dataclass
class EpisodeTrace:
    """
    Trace of a complete episode.
    
    Contains all steps and episode-level metrics.
    """
    episode_id: str
    run_id: str
    episode_number: int
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    # Episode metrics
    total_reward: float = 0.0
    total_steps: int = 0
    success: bool = False
    final_phase: str = "unknown"
    
    # Mentor usage
    mentor_calls: int = 0
    mentor_models_used: Dict[str, int] = field(default_factory=dict)
    
    # Learning metrics
    avg_confidence: float = 0.5
    confidence_distribution: List[float] = field(default_factory=list)
    
    # Steps (not serialized to JSONL - kept separate)
    steps: List[StepTrace] = field(default_factory=list)
    
    # Event ID index for validation
    event_ids: set = field(default_factory=set)
    
    def add_step(self, step: StepTrace):
        """Add a step trace to this episode."""
        self.steps.append(step)
        self.event_ids.add(step.event_id)
        self.total_steps += 1
        self.total_reward += step.reward if step.reward else 0.0
        self.confidence_distribution.append(step.confidence)
        
        if step.mentor_call:
            self.mentor_calls += 1
            model = step.model_used or "unknown"
            self.mentor_models_used[model] = self.mentor_models_used.get(model, 0) + 1
    
    def finalize(self, success: bool = False, final_phase: str = "unknown"):
        """Finalize the episode trace."""
        self.end_time = time.time()
        self.success = success
        self.final_phase = final_phase
        if self.confidence_distribution:
            self.avg_confidence = sum(self.confidence_distribution) / len(self.confidence_distribution)
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary (excludes individual steps)."""
        return {
            "episode_id": self.episode_id,
            "run_id": self.run_id,
            "episode_number": self.episode_number,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_reward": self.total_reward,
            "total_steps": self.total_steps,
            "success": self.success,
            "final_phase": self.final_phase,
            "mentor_calls": self.mentor_calls,
            "mentor_models_used": self.mentor_models_used,
            "avg_confidence": self.avg_confidence,
            "duration": (self.end_time - self.start_time) if self.end_time else 0
        }


@dataclass 
class RunTrace:
    """
    Trace of a complete training run.
    
    Aggregates all episodes and provides run-level metrics.
    """
    run_id: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None
    
    # Episodes
    total_episodes: int = 0
    successful_episodes: int = 0
    episodes: List[EpisodeTrace] = field(default_factory=list)
    
    # Aggregated metrics
    total_reward: float = 0.0
    total_steps: int = 0
    total_mentor_calls: int = 0
    mentor_models_used: Dict[str, int] = field(default_factory=dict)
    
    # Learning progress
    reward_history: List[float] = field(default_factory=list)
    mentor_call_rate: List[float] = field(default_factory=list)  # Per episode
    avg_confidence_history: List[float] = field(default_factory=list)
    
    def add_episode(self, episode: EpisodeTrace):
        """Add an episode trace to this run."""
        self.episodes.append(episode)
        self.total_episodes += 1
        self.total_reward += episode.total_reward
        self.total_steps += episode.total_steps
        self.total_mentor_calls += episode.mentor_calls
        
        if episode.success:
            self.successful_episodes += 1
        
        # Update model usage
        for model, count in episode.mentor_models_used.items():
            self.mentor_models_used[model] = self.mentor_models_used.get(model, 0) + count
        
        # Track history
        self.reward_history.append(episode.total_reward)
        rate = episode.mentor_calls / max(episode.total_steps, 1)
        self.mentor_call_rate.append(rate)
        self.avg_confidence_history.append(episode.avg_confidence)
    
    def get_all_event_ids(self) -> set:
        """Get all event IDs from all episodes in this run."""
        all_ids = set()
        for episode in self.episodes:
            all_ids.update(episode.event_ids)
        return all_ids
    
    def finalize(self):
        """Finalize the run trace."""
        self.end_time = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "config": self.config,
            "seed": self.seed,
            "total_episodes": self.total_episodes,
            "successful_episodes": self.successful_episodes,
            "total_reward": self.total_reward,
            "total_steps": self.total_steps,
            "total_mentor_calls": self.total_mentor_calls,
            "mentor_models_used": self.mentor_models_used,
            "success_rate": self.successful_episodes / max(self.total_episodes, 1),
            "avg_reward_per_episode": self.total_reward / max(self.total_episodes, 1),
            "avg_mentor_calls_per_episode": self.total_mentor_calls / max(self.total_episodes, 1),
            "duration": (self.end_time - self.start_time) if self.end_time else 0,
            "reward_history": self.reward_history,
            "mentor_call_rate": self.mentor_call_rate,
            "avg_confidence_history": self.avg_confidence_history
        }


class TraceWriter:
    """
    Writes trace events to JSONL files.
    
    Features:
    - Per-run trace files
    - Step-by-step JSONL logging
    - Atomic writes for crash safety
    """
    
    def __init__(self, output_dir: str = "traces", run_id: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.run_id = run_id or self._generate_run_id()
        self.run_dir = self.output_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.steps_file = self.run_dir / "steps.jsonl"
        self.episodes_file = self.run_dir / "episodes.jsonl"
        self.run_file = self.run_dir / "run.json"
        
        # Current state
        self.current_run: Optional[RunTrace] = None
        self.current_episode: Optional[EpisodeTrace] = None
        
        logger.info(f"TraceWriter initialized: {self.run_dir}")
    
    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
        return f"run_{timestamp}_{random_suffix}"
    
    def start_run(self, config: Dict[str, Any] = None, seed: Optional[int] = None):
        """Start a new training run."""
        self.current_run = RunTrace(
            run_id=self.run_id,
            config=config or {},
            seed=seed
        )
        
        # Write run start event
        self._write_event({
            "type": TraceEventType.RUN_START.value,
            "run_id": self.run_id,
            "timestamp": time.time(),
            "config": config,
            "seed": seed
        })
        
        logger.info(f"Run started: {self.run_id}")
    
    def start_episode(self, episode_number: int) -> str:
        """Start a new episode."""
        episode_id = f"{self.run_id}_ep{episode_number:04d}"
        
        self.current_episode = EpisodeTrace(
            episode_id=episode_id,
            run_id=self.run_id,
            episode_number=episode_number
        )
        
        # Write episode start event
        self._write_event({
            "type": TraceEventType.EPISODE_START.value,
            "episode_id": episode_id,
            "episode_number": episode_number,
            "timestamp": time.time()
        })
        
        return episode_id
    
    def log_step(self, step: StepTrace):
        """Log a training step."""
        if self.current_episode:
            self.current_episode.add_step(step)
        
        # Write step to JSONL
        with open(self.steps_file, "a") as f:
            f.write(step.to_json() + "\n")
    
    def log_step_dict(self, **kwargs):
        """Log a step from keyword arguments (convenience method)."""
        step = StepTrace(**kwargs)
        self.log_step(step)
        return step
    
    def end_episode(self, success: bool = False, final_phase: str = "unknown") -> EpisodeTrace:
        """End the current episode."""
        if not self.current_episode:
            raise ValueError("No episode in progress")
        
        self.current_episode.finalize(success=success, final_phase=final_phase)
        
        # Add to run
        if self.current_run:
            self.current_run.add_episode(self.current_episode)
        
        # Write episode summary
        with open(self.episodes_file, "a") as f:
            f.write(json.dumps(self.current_episode.to_summary_dict()) + "\n")
        
        episode = self.current_episode
        self.current_episode = None
        
        logger.debug(f"Episode ended: {episode.episode_id}, reward={episode.total_reward:.2f}")
        return episode
    
    def end_run(self) -> RunTrace:
        """End the current run and save final summary."""
        if not self.current_run:
            raise ValueError("No run in progress")
        
        self.current_run.finalize()
        
        # Write run summary
        with open(self.run_file, "w") as f:
            json.dump(self.current_run.to_dict(), f, indent=2)
        
        # Write run end event
        self._write_event({
            "type": TraceEventType.RUN_END.value,
            "run_id": self.run_id,
            "timestamp": time.time(),
            "summary": self.current_run.to_dict()
        })
        
        run = self.current_run
        logger.info(f"Run ended: {run.run_id}, episodes={run.total_episodes}, reward={run.total_reward:.2f}")
        return run
    
    def _write_event(self, event: Dict[str, Any]):
        """Write a generic event to the events log."""
        events_file = self.run_dir / "events.jsonl"
        with open(events_file, "a") as f:
            f.write(json.dumps(event) + "\n")


class TraceReader:
    """
    Reads and normalizes trace files for analysis.
    
    Features:
    - Load steps, episodes, or full runs
    - Normalize events for analysis
    - Filter and query capabilities
    """
    
    def __init__(self, trace_dir: str):
        self.trace_dir = Path(trace_dir)
        if not self.trace_dir.exists():
            raise FileNotFoundError(f"Trace directory not found: {trace_dir}")
    
    def load_run(self) -> Dict[str, Any]:
        """Load the run summary."""
        run_file = self.trace_dir / "run.json"
        if not run_file.exists():
            raise FileNotFoundError(f"Run file not found: {run_file}")
        
        with open(run_file, "r") as f:
            return json.load(f)
    
    def load_episodes(self) -> List[Dict[str, Any]]:
        """Load all episode summaries."""
        episodes_file = self.trace_dir / "episodes.jsonl"
        if not episodes_file.exists():
            return []
        
        episodes = []
        with open(episodes_file, "r") as f:
            for line in f:
                if line.strip():
                    episodes.append(json.loads(line))
        return episodes
    
    def load_steps(self, episode_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load step traces, optionally filtered by episode."""
        steps_file = self.trace_dir / "steps.jsonl"
        if not steps_file.exists():
            return []
        
        steps = []
        with open(steps_file, "r") as f:
            for line in f:
                if line.strip():
                    step = json.loads(line)
                    if episode_id is None or step.get("episode_id") == episode_id:
                        steps.append(step)
        return steps
    
    def get_all_event_ids(self) -> set:
        """Get all event IDs from the trace."""
        steps = self.load_steps()
        return {step.get("event_id") for step in steps if step.get("event_id")}
    
    def get_mentor_call_stats(self) -> Dict[str, Any]:
        """Get statistics about mentor calls."""
        episodes = self.load_episodes()
        
        total_mentor_calls = sum(ep.get("mentor_calls", 0) for ep in episodes)
        total_steps = sum(ep.get("total_steps", 0) for ep in episodes)
        
        # Mentor calls by model
        model_usage = {}
        for ep in episodes:
            for model, count in ep.get("mentor_models_used", {}).items():
                model_usage[model] = model_usage.get(model, 0) + count
        
        return {
            "total_mentor_calls": total_mentor_calls,
            "total_steps": total_steps,
            "mentor_call_rate": total_mentor_calls / max(total_steps, 1),
            "model_usage": model_usage
        }
    
    def get_learning_curve(self) -> Dict[str, List[float]]:
        """Get learning curve data (reward over episodes)."""
        run = self.load_run()
        return {
            "reward_history": run.get("reward_history", []),
            "mentor_call_rate": run.get("mentor_call_rate", []),
            "avg_confidence_history": run.get("avg_confidence_history", [])
        }


# Convenience functions
def create_trace_writer(output_dir: str = "traces", run_id: Optional[str] = None) -> TraceWriter:
    """Create a new trace writer."""
    return TraceWriter(output_dir=output_dir, run_id=run_id)


def load_trace(trace_dir: str) -> TraceReader:
    """Load traces from a directory."""
    return TraceReader(trace_dir)


# Schema validation
STEP_TRACE_SCHEMA = {
    "required": ["event_id", "episode_id", "step", "agent", "phase", "chosen_action", "mentor_call", "done"],
    "optional": ["proposed_action", "model_used", "reward", "action_params", "command", 
                 "observation_summary", "output_summary", "mentor_response", "confidence", 
                 "q_values", "epsilon", "error_flag", "error_message", "timestamp"]
}


def validate_step_trace(data: Dict[str, Any]) -> bool:
    """Validate a step trace against schema."""
    for field in STEP_TRACE_SCHEMA["required"]:
        if field not in data:
            logger.warning(f"Missing required field in step trace: {field}")
            return False
    return True


def validate_event_id_format(event_id: str) -> bool:
    """
    Validate event_id format: {episode_id}:{step:04d}:{agent}
    Returns True if valid.
    """
    import re
    pattern = r'^.+:\d{4}:.+$'
    return bool(re.match(pattern, event_id))


def parse_event_id(event_id: str) -> Optional[Dict[str, Any]]:
    """
    Parse an event_id into components.
    Returns dict with episode_id, step, agent or None if invalid.
    """
    parts = event_id.rsplit(':', 2)
    if len(parts) != 3:
        return None
    try:
        return {
            "episode_id": parts[0],
            "step": int(parts[1]),
            "agent": parts[2]
        }
    except ValueError:
        return None


def validate_evidence_refs(evidence_refs: List[str], valid_event_ids: set) -> List[str]:
    """
    Validate that evidence_refs exist in the trace.
    Returns list of invalid refs (empty if all valid).
    """
    invalid = []
    for ref in evidence_refs:
        if ref not in valid_event_ids:
            invalid.append(ref)
    return invalid


if __name__ == "__main__":
    # Test the tracing system
    from rich.console import Console
    console = Console()
    
    console.print("[bold cyan]Testing EpisodeTrace System[/bold cyan]")
    
    # Create writer
    writer = create_trace_writer(output_dir="test_traces")
    
    # Start run
    writer.start_run(config={"episodes": 3, "max_steps": 10}, seed=42)
    
    # Simulate episodes
    for ep in range(3):
        episode_id = writer.start_episode(ep)
        
        for step in range(5):
            writer.log_step(StepTrace(
                episode_id=episode_id,
                step=step,
                agent="RedAgent",
                phase="recon",
                proposed_action="nmap -sV 10.10.10.10",
                chosen_action="nmap -sV 10.10.10.10",
                reward=10.0 if step == 4 else 1.0,
                mentor_call=step == 0,
                model_used="gpt-5-mini" if step == 0 else None,
                confidence=0.5 + (step * 0.1)
            ))
        
        writer.end_episode(success=True, final_phase="exploit")
    
    # End run
    run = writer.end_run()
    
    console.print(f"[green]✓ Run completed: {run.run_id}[/green]")
    console.print(f"  Episodes: {run.total_episodes}")
    console.print(f"  Total Reward: {run.total_reward:.2f}")
    console.print(f"  Mentor Calls: {run.total_mentor_calls}")
    
    # Test reader
    reader = load_trace(writer.run_dir)
    stats = reader.get_mentor_call_stats()
    console.print(f"\n[cyan]Mentor Stats:[/cyan] {stats}")
    
    console.print("\n[bold green]✓ EpisodeTrace system test passed![/bold green]")
