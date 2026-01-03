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
    """
    episode_id: str
    step: int
    timestamp: float
    agent_id: str
    phase: str
    
    # Action information
    action_proposed: str  # Agent's neural network proposal
    action_final: str     # Final action taken (may differ if mentor overrides)
    action_params: Optional[Dict[str, Any]] = None
    command: Optional[str] = None  # Actual command if applicable
    
    # Observation and outcome
    observation_summary: str = ""
    output_summary: str = ""
    reward: float = 0.0
    done: bool = False
    
    # Mentor/GPT information
    mentor_call: bool = False
    mentor_model: Optional[str] = None  # "gpt-5-mini", "gpt-5-nano", etc.
    mentor_response: Optional[str] = None
    
    # Learning metrics
    confidence: float = 0.5  # Agent's confidence in its proposal
    q_values: Optional[List[float]] = None
    epsilon: float = 1.0
    
    # Error tracking
    error_flag: bool = False
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def to_json(self) -> str:
        """Convert to JSON string (for JSONL)."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepTrace":
        """Create StepTrace from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


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
    
    def add_step(self, step: StepTrace):
        """Add a step trace to this episode."""
        self.steps.append(step)
        self.total_steps += 1
        self.total_reward += step.reward
        self.confidence_distribution.append(step.confidence)
        
        if step.mentor_call:
            self.mentor_calls += 1
            model = step.mentor_model or "unknown"
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
    "required": ["episode_id", "step", "timestamp", "agent_id", "phase", "action_proposed", "action_final"],
    "optional": ["action_params", "command", "observation_summary", "output_summary", "reward", 
                 "done", "mentor_call", "mentor_model", "mentor_response", "confidence", 
                 "q_values", "epsilon", "error_flag", "error_message"]
}


def validate_step_trace(data: Dict[str, Any]) -> bool:
    """Validate a step trace against schema."""
    for field in STEP_TRACE_SCHEMA["required"]:
        if field not in data:
            logger.warning(f"Missing required field in step trace: {field}")
            return False
    return True


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
                timestamp=time.time(),
                agent_id="RedAgent",
                phase="recon",
                action_proposed="nmap -sV 10.10.10.10",
                action_final="nmap -sV 10.10.10.10",
                reward=10.0 if step == 4 else 1.0,
                mentor_call=step == 0,
                mentor_model="gpt-5-mini" if step == 0 else None,
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
