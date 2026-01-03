#!/usr/bin/env python3
"""
core/observability/live_dashboard.py — ARIASKA Live Training Dashboard v2.0

Rich-powered live dashboard showing step-by-step agent actions, environment state,
and training trends with multi-panel layout.

Panels:
A) Global: run_id, episode progress, total steps, avg reward trend, mentor rate, token usage
B) Agents table: agent, phase, last_action, last_reward, episode_reward, confidence, mentor_call, tokens
C) Environment snapshot: target, discovered ports/services, phase, key flags
D) Event feed: last 10 events (mentor failures, token warnings, repeated-action warnings)
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import deque

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.style import Style
from rich import box

console = Console()


@dataclass
class RewardBreakdown:
    """Detailed reward breakdown for explainability."""
    base: float = 0.0
    novelty_bonus: float = 0.0
    redundancy_penalty: float = 0.0
    phase_bonus: float = 0.0
    step_cost: float = 0.0
    total: float = 0.0
    reason: str = ""
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RewardBreakdown":
        if not d:
            return cls()
        return cls(
            base=d.get("base", 0.0),
            novelty_bonus=d.get("novelty_bonus", 0.0),
            redundancy_penalty=d.get("redundancy_penalty", 0.0),
            phase_bonus=d.get("phase_bonus", 0.0),
            step_cost=d.get("step_cost", 0.0),
            total=d.get("total", d.get("base", 0.0)),
            reason=d.get("reason", ""),
        )


@dataclass
class EventRecord:
    """Event for the event feed."""
    timestamp: float
    event_type: str  # "mentor_fail", "budget_warn", "repeat_warn", "phase_change", "stuck"
    message: str
    agent: Optional[str] = None


@dataclass
class DashboardConfig:
    """Configuration for live dashboard."""
    
    # Display settings
    enabled: bool = True
    mode: str = "live"  # "off", "summary", "live"
    watch_rate: float = 1.0  # Print every N steps (1.0 = every step, 0.5 = every 2 steps)
    
    # Trend window
    trend_window: int = 10  # Steps to average for trends
    
    # Table settings
    max_action_width: int = 35
    show_guidance: bool = False  # Show mentor guidance column
    show_reward_breakdown: bool = True
    max_event_feed: int = 10
    
    # Colors
    mentor_color: str = "yellow"
    changed_color: str = "cyan"
    kept_color: str = "green"
    warning_color: str = "red"


@dataclass
class StepRecord:
    """Record of a single step for dashboard."""
    step: int
    phase: str
    agent: str
    proposed: str
    chosen: str
    mentor: bool
    mentor_success: bool
    model: Optional[str]
    confidence: float
    reward: float
    reward_breakdown: Optional[RewardBreakdown] = None
    tokens_used: int = 0
    done: bool = False
    delta: str = "kept"


class LiveDashboard:
    """
    Live training dashboard using Rich tables.
    
    Panels:
    A) Global: run_id, episode progress, total steps, avg reward trend, mentor rate, token usage
    B) Agents table: agent, phase, last_action, last_reward, episode_reward, confidence, mentor_call, tokens
    C) Environment snapshot: target, discovered ports/services, phase, key flags
    D) Event feed: last 10 events (mentor failures, token warnings, repeated-action warnings)
    """
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
        
        # Step tracking
        self.steps: List[StepRecord] = []
        self.step_counter = 0
        self.last_print_step = -1
        
        # Trend tracking (sliding window)
        self.reward_history: deque = deque(maxlen=self.config.trend_window)
        self.confidence_history: deque = deque(maxlen=self.config.trend_window)
        self.mentor_history: deque = deque(maxlen=self.config.trend_window)
        
        # Per-agent tracking
        self.agent_stats: Dict[str, Dict[str, Any]] = {}
        
        # Episode tracking
        self.current_episode = 0
        self.episode_rewards: List[float] = []
        self.skill_library_size = 0
        
        # Run info
        self.run_id: Optional[str] = None
        self.total_episodes: int = 0
        self.tokens_total: int = 0
        self.tokens_by_agent: Dict[str, int] = {}
        
        # Environment snapshot
        self.env_snapshot: Dict[str, Any] = {}
        
        # Event feed
        self.events: deque = deque(maxlen=self.config.max_event_feed)
        
        # Action history for repeat detection
        self.action_history: Dict[str, deque] = {}  # agent -> last N actions
    
    def set_run_info(self, run_id: str, total_episodes: int):
        """Set run metadata for display."""
        self.run_id = run_id
        self.total_episodes = total_episodes
    
    def should_print(self, step: int) -> bool:
        """Check if dashboard should print at this step."""
        if not self.config.enabled or self.config.mode == "off":
            return False
        
        if self.config.mode == "summary":
            return False  # Only print summaries
        
        if self.config.watch_rate >= 1.0:
            return True
        
        # Throttle: print every N steps
        interval = int(1.0 / self.config.watch_rate)
        return step % interval == 0
    
    def add_event(self, event_type: str, message: str, agent: Optional[str] = None):
        """Add event to the feed."""
        self.events.append(EventRecord(
            timestamp=time.time(),
            event_type=event_type,
            message=message,
            agent=agent,
        ))
    
    def update_env_snapshot(self, env_state: Dict[str, Any]):
        """Update environment snapshot for display."""
        self.env_snapshot = {
            "target": env_state.get("target_ip", "10.10.10.10"),
            "phase": env_state.get("phase", "recon"),
            "ports": env_state.get("discovered_ports", []),
            "services": env_state.get("discovered_services", {}),
            "root": env_state.get("root_achieved", False),
            "creds_found": len(env_state.get("credentials", [])) > 0,
            "vulns_found": len(env_state.get("discovered_vulns", [])) > 0,
            "detection_risk": env_state.get("detection_risk", 0.0),
        }
    
    def record_step(
        self,
        step: int,
        phase: str,
        agent_results: List[Dict[str, Any]],
        global_reward: float,
        done: bool,
        reward_breakdown: Optional[Dict[str, Any]] = None,
    ):
        """Record step data from orchestrator."""
        rb = RewardBreakdown.from_dict(reward_breakdown) if reward_breakdown else None
        
        for result in agent_results:
            agent_name = result.get("agent", result.get("agent_name", "?"))
            chosen = result.get("chosen_action", "?")
            proposed = result.get("proposed_action", "?")
            tokens = result.get("tokens_used", 0)
            mentor_success = result.get("mentor_success", result.get("mentor_call", False))
            
            record = StepRecord(
                step=step,
                phase=phase,
                agent=agent_name,
                proposed=self._truncate(proposed),
                chosen=self._truncate(chosen),
                mentor=result.get("mentor_call", False),
                mentor_success=mentor_success,
                model=result.get("model_used"),
                confidence=result.get("confidence", 0.5),
                reward=global_reward,
                reward_breakdown=rb,
                tokens_used=tokens,
                done=done,
                delta=result.get("mentor_delta", "kept"),
            )
            self.steps.append(record)
            
            # Update per-agent stats
            if agent_name not in self.agent_stats:
                self.agent_stats[agent_name] = {
                    "episode_reward": 0.0,
                    "episode_mentor_calls": 0,
                    "episode_tokens": 0,
                    "last_action": "",
                    "last_reward": 0.0,
                    "confidence": 0.5,
                    "phase": phase,
                }
            self.agent_stats[agent_name]["episode_reward"] += global_reward
            self.agent_stats[agent_name]["last_action"] = chosen
            self.agent_stats[agent_name]["last_reward"] = global_reward
            self.agent_stats[agent_name]["episode_tokens"] += tokens
            self.agent_stats[agent_name]["confidence"] = result.get("confidence", 0.5)
            self.agent_stats[agent_name]["phase"] = phase
            if result.get("mentor_call"):
                self.agent_stats[agent_name]["episode_mentor_calls"] += 1
            
            # Track tokens
            self.tokens_total += tokens
            self.tokens_by_agent[agent_name] = self.tokens_by_agent.get(agent_name, 0) + tokens
            
            # Detect repeated actions
            if agent_name not in self.action_history:
                self.action_history[agent_name] = deque(maxlen=5)
            self.action_history[agent_name].append(chosen)
            
            # Check for repeats (stuck detection signal)
            if len(self.action_history[agent_name]) >= 3:
                recent = list(self.action_history[agent_name])[-3:]
                if len(set(recent)) == 1:
                    self.add_event("repeat_warn", f"Repeated action x3: {chosen[:30]}", agent_name)
            
            # Check mentor failures
            if result.get("mentor_call") and not mentor_success:
                error = result.get("error", "unknown")
                self.add_event("mentor_fail", f"Mentor call failed: {error[:40]}", agent_name)
        
        # Update trends
        self.reward_history.append(global_reward)
        avg_conf = sum(r.get("confidence", 0.5) for r in agent_results) / max(len(agent_results), 1)
        self.confidence_history.append(avg_conf)
        mentor_rate = sum(1 for r in agent_results if r.get("mentor_call")) / max(len(agent_results), 1)
        self.mentor_history.append(mentor_rate)
        
        self.step_counter = step
    
    def print_step_table(self, step: int):
        """Print the step table to console."""
        if not self.should_print(step):
            return
        
        # Get records for this step
        step_records = [r for r in self.steps if r.step == step]
        if not step_records:
            return
        
        # Build table with episode progress
        title = f"Step {step}"
        if self.run_id:
            title = f"Step {step} | Ep {self.current_episode}/{self.total_episodes}"
        
        table = Table(
            title=title,
            show_header=True,
            header_style="bold blue",
            border_style="dim",
            box=box.ROUNDED,
        )
        
        table.add_column("Agent", style="bold", width=12)
        table.add_column("Phase", width=8)
        table.add_column("Action", width=self.config.max_action_width)
        table.add_column("Mentor", width=8, justify="center")
        table.add_column("Conf", width=6, justify="right")
        table.add_column("Reward", width=10, justify="right")
        table.add_column("Tokens", width=6, justify="right")
        
        for r in step_records:
            # Mentor indicator with success status
            if r.mentor:
                if r.mentor_success:
                    mentor_text = Text("✓", style=self.config.mentor_color)
                else:
                    mentor_text = Text("✗", style=self.config.warning_color)
            else:
                mentor_text = Text("", style="dim")
            
            # Action styling based on delta
            if r.delta == "changed":
                action_style = self.config.changed_color
            else:
                action_style = self.config.kept_color if not r.mentor else ""
            
            # Reward with styling for negative
            reward_text = f"{r.reward:+.2f}"
            if r.reward < 0:
                reward_text = Text(f"{r.reward:+.2f}", style="red")
            
            table.add_row(
                r.agent,
                r.phase,
                Text(r.chosen, style=action_style),
                mentor_text,
                f"{r.confidence:.2f}",
                reward_text if isinstance(reward_text, Text) else str(reward_text),
                str(r.tokens_used) if r.tokens_used else "",
            )
        
        console.print(table)
        
        # Add reward explanation if available
        if step_records and step_records[0].reward_breakdown:
            rb = step_records[0].reward_breakdown
            if rb.reason:
                console.print(f"  [dim]↳ {rb.reason}[/dim]")
        
        self.last_print_step = step
        
        # Print recent events if any new ones
        self._print_recent_events()
    
    def _print_recent_events(self):
        """Print recent events (warnings, failures)."""
        if not self.events:
            return
        
        recent = list(self.events)[-3:]
        for event in recent:
            style = "dim"
            icon = "ℹ"
            if event.event_type == "mentor_fail":
                style = "red"
                icon = "⚠"
            elif event.event_type == "repeat_warn":
                style = "yellow"
                icon = "🔁"
            elif event.event_type == "budget_warn":
                style = "yellow"
                icon = "💰"
            elif event.event_type == "stuck":
                style = "red"
                icon = "🔒"
            elif event.event_type == "phase_change":
                style = "green"
                icon = "📍"
            
            agent_prefix = f"[{event.agent}] " if event.agent else ""
            console.print(f"  [{style}]{icon} {agent_prefix}{event.message}[/{style}]")
    
    def print_episode_summary(
        self,
        episode: int,
        total_reward: float,
        total_steps: int,
        mentor_calls: int,
    ):
        """Print episode summary and trends."""
        self.current_episode = episode
        self.episode_rewards.append(total_reward)
        
        # Calculate trends
        reward_trend = self._calculate_trend(list(self.reward_history))
        conf_trend = self._calculate_trend(list(self.confidence_history))
        mentor_trend = self._calculate_trend(list(self.mentor_history))
        
        # Build trends table
        table = Table(
            title=f"Episode {episode} Summary",
            show_header=True,
            header_style="bold green",
            border_style="dim",
            box=box.ROUNDED,
        )
        
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_column("Trend", justify="center")
        
        # Reward
        avg_reward = sum(self.reward_history) / max(len(self.reward_history), 1)
        table.add_row(
            "Avg Reward",
            f"{avg_reward:.2f}",
            self._trend_arrow(reward_trend),
        )
        
        # Confidence
        avg_conf = sum(self.confidence_history) / max(len(self.confidence_history), 1)
        table.add_row(
            "Avg Confidence",
            f"{avg_conf:.2f}",
            self._trend_arrow(conf_trend),
        )
        
        # Mentor rate
        avg_mentor = sum(self.mentor_history) / max(len(self.mentor_history), 1)
        table.add_row(
            "Mentor Rate",
            f"{avg_mentor:.1%}",
            self._trend_arrow(mentor_trend, invert=True),  # Lower is better
        )
        
        # Totals
        table.add_row("Episode Reward", f"{total_reward:.2f}", "")
        table.add_row("Episode Steps", str(total_steps), "")
        table.add_row("Mentor Calls", str(mentor_calls), "")
        table.add_row("Tokens Used", str(self.tokens_total), "")
        table.add_row("Skill Library", str(self.skill_library_size), "")
        
        console.print(table)
        
        # Reset per-episode stats
        self._reset_episode_stats()
    
    def _reset_episode_stats(self):
        """Reset per-episode agent stats."""
        for agent in self.agent_stats:
            self.agent_stats[agent] = {
                "episode_reward": 0.0,
                "episode_mentor_calls": 0,
                "episode_tokens": 0,
                "last_action": "",
                "last_reward": 0.0,
                "confidence": 0.5,
                "phase": "",
            }
        self.events.clear()
        self.action_history.clear()
    
    def print_run_summary(
        self,
        run_id: str,
        total_episodes: int,
        total_time: float,
        final_metrics: Dict[str, Any],
    ):
        """Print final run summary."""
        console.rule("[bold green]Training Complete")
        
        table = Table(
            title=f"Run: {run_id}",
            show_header=True,
            header_style="bold cyan",
            box=box.ROUNDED,
        )
        
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        
        table.add_row("Total Episodes", str(total_episodes))
        table.add_row("Total Time", f"{total_time:.1f}s")
        table.add_row("Avg Reward", f"{final_metrics.get('avg_reward_recent', 0):.2f}")
        table.add_row("Reward Trend", final_metrics.get('reward_trend', 'stable'))
        table.add_row("Avg Confidence", f"{final_metrics.get('avg_confidence_recent', 0):.2f}")
        table.add_row("Mentor Rate", f"{final_metrics.get('avg_mentor_rate_recent', 0):.1%}")
        table.add_row("Total Tokens", str(self.tokens_total))
        table.add_row("Skill Library", str(final_metrics.get('skill_library_size', 0)))
        
        console.print(table)
        
        # Per-agent token usage
        if self.tokens_by_agent:
            token_table = Table(title="Token Usage by Agent", box=box.SIMPLE)
            token_table.add_column("Agent", style="bold")
            token_table.add_column("Tokens", justify="right")
            for agent, tokens in sorted(self.tokens_by_agent.items()):
                token_table.add_row(agent, str(tokens))
            console.print(token_table)
        
        # Print next commands
        self._print_next_commands(run_id)
    
    def _print_next_commands(self, run_id: str):
        """Print helpful next commands."""
        console.print("\n[bold cyan]📋 Next Commands:[/bold cyan]")
        console.print(f"  View traces:     [dim]cat traces/{run_id}/steps.jsonl | head[/dim]")
        console.print(f"  View mentor log: [dim]cat traces/{run_id}/mentor.jsonl | head[/dim]")
        console.print("  Re-run offline:  [dim]./scripts/aria offline[/dim]")
        console.print("  Run tests:       [dim]./scripts/aria test[/dim]")
        console.print("  Watch last run:  [dim]./scripts/aria last[/dim]")
        console.print()
    
    def _truncate(self, s: str, max_len: Optional[int] = None) -> str:
        """Truncate string for display."""
        max_len = max_len or self.config.max_action_width
        if len(s) > max_len:
            return s[:max_len-3] + "..."
        return s
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values."""
        if len(values) < 3:
            return "stable"
        
        # Compare first half to second half
        mid = len(values) // 2
        first_half = sum(values[:mid]) / max(mid, 1)
        second_half = sum(values[mid:]) / max(len(values) - mid, 1)
        
        diff = second_half - first_half
        threshold = 0.05 * max(abs(first_half), 0.1)
        
        if diff > threshold:
            return "increasing"
        elif diff < -threshold:
            return "decreasing"
        return "stable"
    
    def _trend_arrow(self, trend: str, invert: bool = False) -> Text:
        """Get arrow for trend direction."""
        if trend == "increasing":
            if invert:
                return Text("↑", style="yellow")
            return Text("↑", style="green")
        elif trend == "decreasing":
            if invert:
                return Text("↓", style="green")
            return Text("↓", style="red")
        return Text("→", style="dim")
    
    def reset_episode(self):
        """Reset for new episode."""
        self.steps.clear()
        self.step_counter = 0
        self.last_print_step = -1
        self._reset_episode_stats()
    
    def set_skill_library_size(self, size: int):
        """Update skill library size for display."""
        self.skill_library_size = size
