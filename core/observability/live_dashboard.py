#!/usr/bin/env python3
"""
core/observability/live_dashboard.py — ARIASKA Live Training Dashboard v3.0

Phase 6.5: Unified Rich terminal UI showing EVERYTHING:
  • Per-agent commands with full output, reasoning, discoveries
  • ASCII sparkline reward trends across episodes
  • Phase progression timeline
  • Strategic agent activation display (who acts and why)
  • Discovery board with real-time updates
  • Improvement tracking with block-char bar charts

Author: Filip Volf — Phase 6.5
"""

import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from collections import deque

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

# Force Rich terminal rendering — ensures full Rich UI even if accidentally piped
console = Console(force_terminal=True, width=140)

# ─── Sparkline characters ────────────────────────────────────────────────────
SPARK_CHARS = "▁▂▃▄▅▆▇█"
PHASE_ICONS = {
    "RECON": "🔍",
    "ENUMERATION": "📋",
    "EXPLOITATION": "💥",
    "PRIVILEGE_ESCALATION": "👑",
    "LATERAL_MOVEMENT": "🔀",
    "POST_EXPLOITATION": "🏴",
    "EXFILTRATION": "📤",
    "CLOSEOUT": "🧹",
}
AGENT_ICONS = {
    "ScoutAgent": ("🔍", "Recon", "bold cyan"),
    "RedAgent": ("⚔️", "Attack", "bold red"),
    "BlueAgent": ("🛡️", "Defense", "bold blue"),
    "OrionAgent": ("🎯", "Strategy", "bold yellow"),
    "ShadowAgent": ("👤", "Stealth", "bold magenta"),
}
SOURCE_STYLES = {
    "ppo": ("🤖", "green"),
    "mentor": ("📡", "yellow"),
    "playbook": ("📖", "cyan"),
    "registry": ("📦", "blue"),
    "anti_repeat": ("🔁", "red"),
    "anti_repe": ("🔁", "red"),
    "skill": ("🧠", "magenta"),
    "forced_novel": ("🆕", "bright_yellow"),
    "forced_no": ("🆕", "bright_yellow"),
    "closeout_gate": ("🧹", "green"),
    "closeout_": ("🧹", "green"),
    "closeout": ("🧹", "green"),
    "difficulty_gate": ("🛡️", "blue"),
    "difficul": ("🛡️", "blue"),
    "fallback": ("⚡", "dim"),
    "unknown": ("❓", "dim"),
}


def sparkline(values: List[float], width: int = 20) -> str:
    """Generate an ASCII sparkline from a list of values."""
    if not values:
        return ""
    vals = values[-width:]
    mn, mx = min(vals), max(vals)
    rng = mx - mn if mx != mn else 1.0
    chars = []
    for v in vals:
        idx = int((v - mn) / rng * (len(SPARK_CHARS) - 1))
        idx = max(0, min(idx, len(SPARK_CHARS) - 1))
        chars.append(SPARK_CHARS[idx])
    return "".join(chars)


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
    event_type: str
    message: str
    agent: Optional[str] = None


@dataclass
class AgentStepInfo:
    """Everything about one agent's action in a step — used by print_step."""
    agent_name: str
    command: str
    command_output: str
    mentor_reasoning: str
    source: str
    confidence: float
    reward: float
    mentor_call: bool
    tokens_used: int
    discoveries: Dict[str, List[str]] = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class DashboardConfig:
    """Configuration for live dashboard."""
    enabled: bool = True
    mode: str = "live"  # "off", "summary", "live"
    watch_rate: float = 1.0
    trend_window: int = 20
    max_action_width: int = 80
    max_output_lines: int = 3
    max_output_width: int = 80
    show_guidance: bool = True
    show_reward_breakdown: bool = True
    show_discoveries: bool = True
    show_output: bool = True
    max_event_feed: int = 10
    mentor_color: str = "yellow"
    changed_color: str = "cyan"
    kept_color: str = "green"
    warning_color: str = "red"


class LiveDashboard:
    """
    ARIASKA Live Training Dashboard v3.0 — Phase 6.5

    Unified Rich terminal UI displaying EVERYTHING in one clean, readable flow.

    Per step:
      ┌─ STEP HEADER ── step / episode / mode / phase / reward ────────┐
      │  AGENT TABLE: agent, source, command, reward, confidence       │
      │  DETAIL PANEL: full output, reasoning, discoveries per agent   │
      │  SKIPPED: agents not active + reason                           │
      │  DISCOVERY BOARD: ports, services, credentials, shells         │
      │  REWARD BREAKDOWN: base, novelty, penalty, phase bonus         │
      └────────────────────────────────────────────────────────────────┘

    Per episode:
      • Summary table with trends
      • Phase timeline with icons
      • ASCII bar chart of reward trajectory
      • Discovery summary
      • Per-agent snapshot
    """

    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()

        # Step tracking
        self.step_counter = 0
        self.last_print_step = -1

        # Trend tracking (sliding window)
        self.step_rewards: deque = deque(maxlen=100)
        self.confidence_history: deque = deque(maxlen=self.config.trend_window)
        self.mentor_history: deque = deque(maxlen=self.config.trend_window)
        self.reward_history: deque = deque(maxlen=self.config.trend_window)

        # Per-agent tracking
        self.agent_stats: Dict[str, Dict[str, Any]] = {}

        # Episode tracking
        self.current_episode = 0
        self.episode_rewards: List[float] = []
        self.episode_phases: List[str] = []
        self.skill_library_size = 0

        # Per-episode command tracking
        self.episode_commands: Dict[str, List[str]] = {}
        self.episode_unique_commands: Set[str] = set()
        self.episode_repeat_count: int = 0
        self.episode_discoveries: Dict[str, Set[str]] = {}

        # Run info
        self.run_id: Optional[str] = None
        self.total_episodes: int = 0
        self.tokens_total: int = 0
        self.tokens_by_agent: Dict[str, int] = {}

        # Environment snapshot
        self.env_snapshot: Dict[str, Any] = {}

        # Event feed
        self.events: deque = deque(maxlen=self.config.max_event_feed)

        # Phase timeline for this episode
        self.phase_timeline: List[tuple] = []

        # Action history for repeat detection
        self.action_history: Dict[str, deque] = {}

        # Per-episode mentor tracking (Phase 6.9.3)
        self.episode_mentor_calls_total: int = 0
        self.episode_active_steps: int = 0

        # Phase 10.0: KG flow + LLM usage tracking
        self.kg_queries_total: int = 0
        self.kg_hits_total: int = 0
        self.kg_queries_episode: int = 0
        self.kg_hits_episode: int = 0
        self.llm_calls_episode: int = 0
        self.llm_calls_total: int = 0
        self.llm_tokens_episode: int = 0
        self.llm_tokens_total: int = 0
        self.llm_model_usage: Dict[str, int] = {}     # model → call count
        self.parser_stage_counts: Dict[str, int] = {}  # stage → hit count
        self.cloud_role_calls: Dict[str, int] = {}     # role → call count
        self.venice_calls_episode: int = 0
        self.venice_calls_total: int = 0
        self.runtime_profile: str = "UNKNOWN"

    # ─── Run metadata ────────────────────────────────────────────────────────

    def set_run_info(self, run_id: str, total_episodes: int):
        self.run_id = run_id
        self.total_episodes = total_episodes

    def should_print(self, step: int) -> bool:
        if not self.config.enabled or self.config.mode == "off":
            return False
        if self.config.mode == "summary":
            return False
        if self.config.watch_rate >= 1.0:
            return True
        interval = int(1.0 / self.config.watch_rate)
        return step % interval == 0

    def add_event(self, event_type: str, message: str, agent: Optional[str] = None):
        self.events.append(EventRecord(
            timestamp=time.time(), event_type=event_type,
            message=message, agent=agent,
        ))

    # ─── Phase 10.0: KG + LLM usage updates ─────────────────────────────

    def update_kg_stats(self, queries: int = 0, hits: int = 0):
        """Update knowledge graph query stats."""
        self.kg_queries_episode += queries
        self.kg_queries_total += queries
        self.kg_hits_episode += hits
        self.kg_hits_total += hits

    def update_llm_stats(
        self,
        calls: int = 0,
        tokens: int = 0,
        model: str = "",
        role: str = "",
        venice_calls: int = 0,
    ):
        """Update LLM usage stats."""
        self.llm_calls_episode += calls
        self.llm_calls_total += calls
        self.llm_tokens_episode += tokens
        self.llm_tokens_total += tokens
        if model:
            self.llm_model_usage[model] = self.llm_model_usage.get(model, 0) + calls
        if role:
            self.cloud_role_calls[role] = self.cloud_role_calls.get(role, 0) + calls
        self.venice_calls_episode += venice_calls
        self.venice_calls_total += venice_calls

    def update_parser_stats(self, stage: str):
        """Record which parser stage produced a discovery."""
        self.parser_stage_counts[stage] = self.parser_stage_counts.get(stage, 0) + 1

    def set_runtime_profile(self, profile: str):
        """Set the detected runtime profile (CLOUD/OFFLINE/DETERMINISTIC)."""
        self.runtime_profile = profile

    def update_env_snapshot(self, env_state: Dict[str, Any]):
        self.env_snapshot = {
            "target": env_state.get("target_ip", "10.10.10.10"),
            "phase": env_state.get("phase", "recon"),
            "ports": env_state.get("discovered_ports", []),
            "services": env_state.get("discovered_services", {}),
            "root": env_state.get("root_achieved", False),
            "creds_found": len(env_state.get("credentials", [])) > 0,
            "detection_risk": env_state.get("detection_risk", 0.0),
        }

    # =========================================================================
    # UNIFIED STEP DISPLAY — The one and only step printer (Phase 6.5)
    # =========================================================================
    def print_step(
        self,
        step: int,
        phase: str,
        mode_tag: str,
        agent_infos: List[AgentStepInfo],
        skipped_agents: Dict[str, str],
        global_reward: float,
        done: bool,
        reward_breakdown: Optional[Dict[str, Any]] = None,
        discovery_board: Optional[Dict[str, Any]] = None,
    ):
        """
        Print unified step display with Rich agent table. Phase 6.9.3.

        Layout:
          ┄┄┄ Step 3 │ Ep 1/5 │ LIVE │ 🧹 CLOSEOUT │ R:+87.0 │ 📡 2 │ ▁█▁
          ┃ Agent     ┃ Source     ┃ Command                  ┃ Output              ┃ Reward ┃ Discoveries    ┃
          ┃ 👤Shadow  ┃ 🧹closeout ┃ sed -i "/attacker/d"... ┃ Connected to 172... ┃ +87.0  ┃ service:ssh    ┃
          ┃ 💤Scout   ┃ skip       ┃ inactive in CLOSEOUT     ┃                     ┃        ┃                ┃
          🔓 7 ports │ ⚙️ 7 svcs │ 💀 4 shells │ 💰 base:+12.0 +nov:+7.5
        """
        if not self.should_print(step):
            return

        # Update episode step counter
        self.episode_active_steps += 1

        phase_upper = phase.upper()
        phase_icon = PHASE_ICONS.get(phase_upper, "❓")
        reward_color = "green" if global_reward > 0 else "red" if global_reward < 0 else "dim"
        ep_str = (
            f"Ep {self.current_episode}/{self.total_episodes}"
            if self.total_episodes else f"Ep {self.current_episode}"
        )
        step_spark = sparkline(list(self.step_rewards), width=12)

        # Mentor stats
        mentor_str = f"📡 {self.episode_mentor_calls_total}" if self.episode_mentor_calls_total else ""
        done_tag = " [bold green]✅ DONE[/bold green]" if done else ""

        # ── STEP HEADER ──────────────────────────────────────────────
        console.print(
            f"\n[dim]{'┄' * 3}[/dim] "
            f"[bold cyan]Step {step + 1:2d}[/bold cyan] │ "
            f"[cyan]{ep_str}[/cyan] │ "
            f"[bold yellow]{mode_tag}[/bold yellow] │ "
            f"{phase_icon} [bold]{phase_upper}[/bold] │ "
            f"[bold {reward_color}]R:{global_reward:+.1f}[/bold {reward_color}] │ "
            f"[dim]{mentor_str}[/dim] │ "
            f"[dim]{step_spark}[/dim]{done_tag}"
        )

        # ── AGENT TABLE ──────────────────────────────────────────────
        table = Table(
            box=box.SIMPLE_HEAVY, show_header=True,
            header_style="bold", padding=(0, 1), expand=True,
        )
        table.add_column("Agent", style="bold", width=12)
        table.add_column("Source", width=12)
        table.add_column("Command", width=34, no_wrap=True)
        table.add_column("Output", width=30, no_wrap=True)
        table.add_column("Reward", width=8, justify="right")
        table.add_column("Conf", width=5, justify="center")
        table.add_column("Discoveries", width=18)

        # Active agents
        active = [a for a in agent_infos if not a.skipped]
        for a in active:
            icon, _role, style = AGENT_ICONS.get(
                a.agent_name, ("🤖", "Agent", "dim")
            )
            # Better source key matching — try full name, truncated, and prefix
            src_raw = a.source or "unknown"
            src_key = src_raw[:8]
            src_icon, src_style = SOURCE_STYLES.get(
                src_raw, SOURCE_STYLES.get(
                    src_key, SOURCE_STYLES.get(
                        src_raw.split("_")[0],
                        ("❓", "dim"),
                    )
                )
            )

            agent_label = f"{icon} {a.agent_name.replace('Agent', '')}"
            source_label = f"{src_icon}{src_raw[:9]}"
            if a.mentor_call:
                source_label += " 📡"

            cmd = (a.command or "(none)")[:32]

            # Output: flatten to single line, truncate
            out = ""
            if a.command_output:
                out = a.command_output.strip().replace("\n", " │ ")[:28]

            # Reward display
            if a.reward > 0:
                r_str = f"[green]{a.reward:+.1f}[/green]"
            elif a.reward < 0:
                r_str = f"[red]{a.reward:+.1f}[/red]"
            else:
                r_str = "[dim]-[/dim]"

            # Confidence display
            conf = a.confidence
            if conf >= 0.7:
                conf_str = f"[green]{conf:.0%}[/green]"
            elif conf >= 0.4:
                conf_str = f"[yellow]{conf:.0%}[/yellow]"
            else:
                conf_str = f"[red]{conf:.0%}[/red]"

            # Discoveries
            disc = ""
            if a.discoveries:
                parts = []
                for dtype, items in a.discoveries.items():
                    if items and isinstance(items, (list, set, tuple)):
                        parts.append(
                            f"{dtype}:{','.join(str(i) for i in list(items)[:3])}"
                        )
                    elif items and isinstance(items, str):
                        parts.append(f"{dtype}:{items}")
                disc = "; ".join(parts)[:18] if parts else ""

            table.add_row(
                f"[{style}]{agent_label}[/{style}]",
                f"[{src_style}]{source_label}[/{src_style}]",
                f"[white]{cmd}[/white]",
                f"[dim]{out}[/dim]",
                r_str,
                conf_str,
                f"[green]{disc}[/green]" if disc else "[dim]-[/dim]",
            )

        # Skipped agents
        for name, reason in skipped_agents.items():
            icon = AGENT_ICONS.get(name, ("💤", "", "dim"))[0]
            table.add_row(
                f"[dim]{icon} {name.replace('Agent', '')}[/dim]",
                "[dim]💤 skip[/dim]",
                f"[dim]{reason}[/dim]",
                "", "", "", "",
            )

        console.print(table)

        # ── FOOTER: Discovery board + Reward breakdown ───────────────
        footer = []
        if discovery_board and self.config.show_discoveries:
            db = discovery_board
            for key, icon_s in [("ports", "🔓"), ("services", "⚙️"),
                                ("credentials", "🔑"), ("shells", "💀")]:
                items = db.get(key, set())
                if items:
                    footer.append(f"{icon_s} {len(items)} {key[:5]}")

        if reward_breakdown and self.config.show_reward_breakdown:
            rb = reward_breakdown
            rp = []
            if rb.get("base", 0):
                rp.append(f"base:{rb['base']:+.1f}")
            if rb.get("novelty_bonus", 0):
                rp.append(f"[green]+nov:{rb['novelty_bonus']:+.1f}[/green]")
            if rb.get("redundancy_penalty", 0):
                rp.append(f"[red]rep:{rb['redundancy_penalty']:+.1f}[/red]")
            if rb.get("phase_bonus", 0):
                rp.append(f"[cyan]phase:{rb['phase_bonus']:+.1f}[/cyan]")
            if rp:
                footer.append(f"💰 {' '.join(rp)}")

        if footer:
            console.print(f"  [dim]{' │ '.join(footer)}[/dim]")

        # ── EVENTS (last 3 seconds, max 2) ───────────────────────────
        now = time.time()
        recent = [e for e in self.events if now - e.timestamp < 3]
        if recent:
            self._print_events(recent[-2:])

        self.last_print_step = step

    # ─── Event printer ───────────────────────────────────────────────────────

    def _print_events(self, events: List[EventRecord]):
        EV = {
            "stuck": ("🔴", "bold red"), "stuck_abort": ("🔴", "bold red"),
            "mentor_fail": ("❌", "red"), "parse_fail": ("❌", "red"),
            "repeat_warn": ("🔁", "yellow"), "budget_warn": ("⚠️", "yellow"),
            "phase_change": ("🔵", "bold blue"), "new_discovery": ("🟢", "bold green"),
            "mentor_call": ("📡", "cyan"), "forced_novel": ("🆕", "bright_yellow"),
            "watchdog": ("🐕", "red"), "neg_streak": ("🟡", "yellow"),
        }
        for event in events:
            icon, style = EV.get(event.event_type, ("ℹ️", "dim"))
            ap = f"[{event.agent}] " if event.agent else ""
            console.print(f"  [{style}]{icon} {ap}{event.message}[/{style}]")

    # =========================================================================
    # AGENT SNAPSHOT TABLE
    # =========================================================================
    def print_agent_snapshot(self):
        if not self.agent_stats:
            return
        table = Table(
            title="[bold]Agent Performance[/bold]",
            show_header=True, header_style="bold magenta",
            border_style="dim", box=box.ROUNDED, padding=(0, 1),
        )
        table.add_column("Agent", style="bold", width=14)
        table.add_column("Phase", width=14)
        table.add_column("Cmds", width=5, justify="right")
        table.add_column("Uniq", width=5, justify="right")
        table.add_column("Ep Reward", width=10, justify="right")
        table.add_column("Conf", width=6, justify="center")
        table.add_column("Mentor", width=6, justify="center")
        table.add_column("Top Source", width=12)

        for agent_name, stats in sorted(self.agent_stats.items()):
            icon = AGENT_ICONS.get(agent_name, ("🤖", "", ""))[0]
            ep_r = stats.get("episode_reward", 0.0)
            ep_text = Text(
                f"{ep_r:+.1f}",
                style="green" if ep_r > 0 else "red" if ep_r < 0 else "dim",
            )
            mc = stats.get("episode_mentor_calls", 0)
            total_cmds = stats.get("total_commands", 0)
            unique_cmds = len(stats.get("unique_commands", set()))
            conf = stats.get("confidence", 0.5)
            conf_color = "green" if conf >= 0.7 else "yellow" if conf >= 0.4 else "red"
            sources = stats.get("decision_sources", {})
            top_src = max(sources, key=sources.get) if sources else "-"
            src_icon = SOURCE_STYLES.get(top_src[:8], SOURCE_STYLES.get(top_src.split("_")[0] if top_src != "-" else "unknown", ("", "")))[0]
            phase_name = str(stats.get("phase", "?"))
            phase_icon = PHASE_ICONS.get(phase_name.upper(), "")
            table.add_row(
                f"{icon} {agent_name.replace('Agent', '')}",
                f"{phase_icon} {phase_name[:12]}",
                str(total_cmds), str(unique_cmds), ep_text,
                f"[{conf_color}]{conf:.0%}[/{conf_color}]",
                f"📡{mc}" if mc > 0 else "[dim]-[/dim]",
                f"{src_icon}{top_src[:10]}",
            )
        console.print(table)

    # =========================================================================
    # EPISODE SUMMARY with sparklines and phase timeline
    # =========================================================================
    def print_episode_summary(
        self,
        episode: int,
        total_reward: float,
        total_steps: int,
        mentor_calls: int,
        highest_phase: str = "",
        ppo_metrics: Optional[Dict[str, float]] = None,
    ):
        self.current_episode = episode
        self.episode_rewards.append(total_reward)
        if highest_phase:
            self.episode_phases.append(highest_phase)

        console.print()
        console.rule(f"[bold green]Episode {episode} Complete", style="green")

        # ── METRICS TABLE with deltas ────────────────────────────────
        table = Table(
            title=f"[bold]Episode {episode} Summary[/bold]",
            show_header=True, header_style="bold green",
            border_style="green", box=box.ROUNDED,
        )
        table.add_column("Metric", style="bold cyan", width=22)
        table.add_column("Value", justify="right", width=14)
        table.add_column("Δ / Trend", width=28)

        avg_step = (
            sum(list(self.step_rewards)[-10:]) /
            max(len(list(self.step_rewards)[-10:]), 1)
        )

        # Episode reward with delta from previous
        prev_reward = self.episode_rewards[-2] if len(self.episode_rewards) >= 2 else None
        delta_str = ""
        if prev_reward is not None:
            delta = total_reward - prev_reward
            delta_pct = (delta / max(abs(prev_reward), 1)) * 100
            if delta > 0:
                delta_str = f"[green]▲ +{delta:.0f} ({delta_pct:+.0f}%)[/green]"
            elif delta < 0:
                delta_str = f"[red]▼ {delta:.0f} ({delta_pct:+.0f}%)[/red]"
            else:
                delta_str = "[dim]→ same[/dim]"
        trend_spark = self._trend_display(self.episode_rewards)

        table.add_row("Episode Reward", f"{total_reward:+.1f}",
                       f"{delta_str}  {trend_spark}")
        table.add_row("Avg Step Reward", f"{avg_step:+.2f}",
                       sparkline(list(self.step_rewards)))
        table.add_row("Steps", str(total_steps), "")

        phase_icon = PHASE_ICONS.get(highest_phase, "")
        table.add_row("Highest Phase", f"{phase_icon} {highest_phase}", "")

        # Mentor rate with trend
        if self.episode_active_steps > 0:
            mentor_rate = mentor_calls / self.episode_active_steps
        else:
            mentor_rate = 0.0
        self.mentor_history.append(mentor_rate)
        avg_mentor = (
            sum(self.mentor_history) / max(len(self.mentor_history), 1)
        )
        mentor_delta = ""
        if len(self.mentor_history) >= 2:
            prev_mr = self.mentor_history[-2]
            if mentor_rate > prev_mr:
                mentor_delta = f"[yellow]▲ {(mentor_rate - prev_mr):.0%}[/yellow]"
            elif mentor_rate < prev_mr:
                mentor_delta = f"[dim]▼ {(mentor_rate - prev_mr):.0%}[/dim]"
        table.add_row("Mentor Calls", str(mentor_calls),
                       f"rate: {mentor_rate:.0%}  avg: {avg_mentor:.0%}  {mentor_delta}")

        total_disc = sum(len(v) for v in self.episode_discoveries.values())
        table.add_row("Discoveries", str(total_disc), "")

        # Command diversity with quality signal
        unique_count = len(self.episode_unique_commands)
        diversity = unique_count / max(total_steps, 1)
        diversity_color = "green" if diversity > 0.6 else "yellow" if diversity > 0.3 else "red"
        table.add_row("Unique Cmds", str(unique_count),
                       f"[{diversity_color}]diversity: {diversity:.0%}[/{diversity_color}]  repeats: {self.episode_repeat_count}")
        table.add_row("Tokens", str(self.tokens_total), "")

        # Decision source breakdown for this episode
        all_sources: Dict[str, int] = {}
        for stats in self.agent_stats.values():
            for src, cnt in stats.get("decision_sources", {}).items():
                all_sources[src] = all_sources.get(src, 0) + cnt
        total_decisions = sum(all_sources.values()) or 1
        src_parts = []
        for src, cnt in sorted(all_sources.items(), key=lambda x: -x[1])[:4]:
            pct = cnt / total_decisions * 100
            src_icon = SOURCE_STYLES.get(src[:8], ("", "dim"))[0]
            src_parts.append(f"{src_icon}{src}:{pct:.0f}%")
        if src_parts:
            table.add_row("Decision Mix", " ".join(src_parts), "")

        if ppo_metrics:
            if ppo_metrics.get("updates"):
                table.add_row("PPO Updates", str(ppo_metrics["updates"]), "")
            if ppo_metrics.get("policy_loss"):
                table.add_row("PPO π Loss", f"{ppo_metrics['policy_loss']:.4f}", "")
            if ppo_metrics.get("value_loss"):
                table.add_row("PPO V Loss", f"{ppo_metrics['value_loss']:.4f}", "")
            if ppo_metrics.get("entropy"):
                table.add_row("PPO Entropy", f"{ppo_metrics['entropy']:.4f}", "")

        # Phase 10.0: KG + LLM usage rows
        if self.kg_queries_episode > 0:
            kg_hit_pct = (self.kg_hits_episode / max(self.kg_queries_episode, 1)) * 100
            table.add_row(
                "📚 KG Queries",
                f"{self.kg_queries_episode}",
                f"hits: {self.kg_hits_episode} ({kg_hit_pct:.0f}%)"
            )
        if self.llm_calls_episode > 0:
            models_str = ", ".join(
                f"{m}:{c}" for m, c in sorted(self.llm_model_usage.items(), key=lambda x: -x[1])[:3]
            )
            table.add_row(
                "🤖 LLM Calls",
                f"{self.llm_calls_episode}",
                f"tokens: {self.llm_tokens_episode:,}  [{models_str}]"
            )
        if self.venice_calls_episode > 0:
            table.add_row("🌊 Venice", f"{self.venice_calls_episode}", "")
        if self.cloud_role_calls:
            roles_str = ", ".join(
                f"{r}:{c}" for r, c in sorted(self.cloud_role_calls.items(), key=lambda x: -x[1])
            )
            table.add_row("☁️ Cloud Roles", roles_str, f"profile: {self.runtime_profile}")
        if self.parser_stage_counts:
            parser_str = " ".join(
                f"{s}:{c}" for s, c in sorted(self.parser_stage_counts.items(), key=lambda x: -x[1])
            )
            table.add_row("🔍 Parser", parser_str, "")

        console.print(table)

        # ── PHASE TIMELINE ───────────────────────────────────────────
        if self.phase_timeline:
            parts = []
            for _step_num, pname in self.phase_timeline:
                ic = PHASE_ICONS.get(pname, "")
                parts.append(f"[bold]{ic} {pname}[/bold] (s{_step_num})")
            console.print(Panel(
                " → ".join(parts),
                title="[bold]Phase Timeline[/bold]",
                border_style="blue", padding=(0, 1),
            ))

        # ── KILL CHAIN PROGRESS BAR (Phase 6.7) ─────────────────────
        phases_reached = {pname for _, pname in self.phase_timeline}
        self.print_kill_chain_bar(highest_phase, phases_reached)
        self.print_cost_ticker(self.tokens_total)

        # ── REWARD CHART ─────────────────────────────────────────────
        if len(self.episode_rewards) >= 2:
            console.print(Panel(
                self._ascii_reward_chart(),
                title="[bold]Reward Trend[/bold]",
                border_style="yellow", padding=(0, 1),
            ))

        # ── DISCOVERY SUMMARY ────────────────────────────────────────
        if self.episode_discoveries:
            parts = []
            for dtype, items in sorted(self.episode_discoveries.items()):
                istr = ", ".join(str(i)[:20] for i in sorted(items)[:8])
                parts.append(f"[bold]{dtype}[/bold]: {istr}")
            console.print(Panel(
                "\n".join(parts),
                title="[bold green]Discoveries This Episode[/bold green]",
                border_style="green", padding=(0, 1),
            ))

        # Per-agent snapshot
        self.print_agent_snapshot()
        console.rule(style="dim green")

        # Reset
        self._reset_episode_stats()

    def _ascii_reward_chart(self) -> str:
        rewards = self.episode_rewards
        if not rewards:
            return "No data yet"
        width = min(len(rewards), 40)
        vals = rewards[-width:]
        mn, mx = min(min(vals), 0), max(max(vals), 1)
        rng = mx - mn if mx != mn else 1.0
        avg = sum(vals) / len(vals)

        trend = (
            "📈" if len(vals) >= 2 and vals[-1] > vals[-2]
            else "📉" if len(vals) >= 2 and vals[-1] < vals[-2]
            else "→"
        )
        lines = [
            f"  {trend} Eps {max(1, len(rewards)-width+1)}-{len(rewards)}"
            f"  avg:{avg:+.1f}  last:{vals[-1]:+.1f}  best:{max(vals):+.1f}"
        ]
        for i, v in enumerate(vals):
            ep_num = len(rewards) - len(vals) + i + 1
            bar_len = max(int((v - mn) / rng * 30), 1)
            color = "green" if v > avg else "yellow" if v > 0 else "red"
            ph = (
                self.episode_phases[len(rewards) - len(vals) + i]
                if (len(rewards) - len(vals) + i) < len(self.episode_phases)
                else "?"
            )
            lines.append(
                f"  Ep{ep_num:3d} [{color}]{'█' * bar_len}[/{color}]"
                f" {v:+7.1f}  {PHASE_ICONS.get(ph, '')} {ph[:6]}"
            )
        lines.append(f"\n  Sparkline: {sparkline(list(self.episode_rewards), 40)}")
        return "\n".join(lines)

    def _trend_display(self, values: List[float]) -> str:
        if len(values) < 2:
            return "[dim]→ (need data)[/dim]"
        spark = sparkline(values, width=12)
        last, prev = values[-1], values[-2]
        if last > prev * 1.05:
            return f"[green]↑ improving[/green] {spark}"
        elif last < prev * 0.95:
            return f"[red]↓ declining[/red] {spark}"
        return f"[yellow]→ stable[/yellow] {spark}"

    # =========================================================================
    # RUN SUMMARY
    # =========================================================================
    def print_run_summary(self, run_id: str, total_episodes: int,
                          total_time: float, final_metrics: Dict[str, Any]):
        console.print()
        self.print_ariaska_banner()
        console.rule("[bold green]🏁 Training Complete", style="green")
        table = Table(title=f"[bold]Run: {run_id}[/bold]",
                      show_header=True, header_style="bold cyan", box=box.ROUNDED)
        table.add_column("Metric", style="bold", width=20)
        table.add_column("Value", justify="right", width=15)
        table.add_row("Total Episodes", str(total_episodes))
        table.add_row("Time", f"{total_time:.1f}s ({total_time/max(total_episodes,1):.1f}s/ep)")
        table.add_row("Avg Reward", f"{final_metrics.get('avg_reward_recent', 0):+.2f}")
        table.add_row("Best Reward", f"{max(self.episode_rewards) if self.episode_rewards else 0:+.1f}")
        table.add_row("Tokens", str(self.tokens_total))
        console.print(table)

        if self.episode_rewards:
            console.print(Panel(self._ascii_reward_chart(),
                                title="[bold]Final Reward Trend[/bold]",
                                border_style="yellow"))
        if self.tokens_by_agent:
            tt = Table(title="[bold]Tokens by Agent[/bold]", box=box.SIMPLE)
            tt.add_column("Agent", style="bold")
            tt.add_column("Tokens", justify="right")
            for ag, tok in sorted(self.tokens_by_agent.items()):
                tt.add_row(ag, str(tok))
            console.print(tt)
        self.print_cost_ticker(self.tokens_total)
        console.rule(style="green")

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================
    def _reset_episode_stats(self):
        for agent in self.agent_stats:
            self.agent_stats[agent] = {
                "episode_reward": 0.0, "episode_mentor_calls": 0,
                "episode_tokens": 0, "last_action": "", "last_reward": 0.0,
                "confidence": 0.5, "phase": "", "total_commands": 0,
                "unique_commands": set(), "decision_sources": {},
            }
        self.events.clear()
        self.episode_commands.clear()
        self.episode_unique_commands.clear()
        self.episode_repeat_count = 0
        self.episode_discoveries.clear()
        self.phase_timeline.clear()
        self.episode_mentor_calls_total = 0
        self.episode_active_steps = 0

    def reset_episode(self):
        self.step_counter = 0
        self.last_print_step = -1
        self.step_rewards.clear()
        self.confidence_history.clear()
        self.mentor_history.clear()
        self.reward_history.clear()
        self._reset_episode_stats()
        self.episode_mentor_calls_total = 0
        self.episode_active_steps = 0

    def set_skill_library_size(self, size: int):
        self.skill_library_size = size

    # =========================================================================
    # INVESTOR-READY DISPLAY (Phase 6.7)
    # =========================================================================

    def print_ariaska_banner(self):
        """Print the ARIASKA ASCII logo banner — investor-demo mode."""
        logo = r"""
[bold red]     █████╗ ██████╗ ██╗ █████╗ ███████╗██╗  ██╗ █████╗ [/bold red]
[bold red]    ██╔══██╗██╔══██╗██║██╔══██╗██╔════╝██║ ██╔╝██╔══██╗[/bold red]
[bold red]    ███████║██████╔╝██║███████║███████╗█████╔╝ ███████║[/bold red]
[bold red]    ██╔══██║██╔══██╗██║██╔══██║╚════██║██╔═██╗ ██╔══██║[/bold red]
[bold red]    ██║  ██║██║  ██║██║██║  ██║███████║██║  ██╗██║  ██║[/bold red]
[bold red]    ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝[/bold red]
[bold cyan]    ⚡ Autonomous Multi-Agent Reinforcement Learning for Cybersecurity ⚡[/bold cyan]
[dim]    5 Agents  •  PPO + GPT Hybrid  •  8-Phase Kill Chain  •  Live Pentesting[/dim]
"""
        console.print(Panel(
            logo,
            border_style="red",
            padding=(0, 2),
        ))

    def print_kill_chain_bar(self, current_phase: str, phases_reached: Optional[set] = None):
        """Print a colored kill-chain progress bar showing 8 phase segments.

        Each phase that's been reached fills in with its icon and color;
        unreached phases show as dim outlines.
        """
        phases_reached = phases_reached or set()
        PHASE_ORDER = [
            ("RECON", "🔍", "cyan"),
            ("ENUMERATION", "📋", "blue"),
            ("EXPLOITATION", "💥", "yellow"),
            ("PRIVILEGE_ESCALATION", "👑", "bright_yellow"),
            ("LATERAL_MOVEMENT", "🔀", "magenta"),
            ("POST_EXPLOITATION", "🏴", "red"),
            ("EXFILTRATION", "📤", "bright_red"),
            ("CLOSEOUT", "🧹", "green"),
        ]
        current_upper = current_phase.upper()
        parts = []
        for phase_name, icon, color in PHASE_ORDER:
            reached = phase_name in phases_reached or phase_name == current_upper
            if phase_name == current_upper:
                parts.append(f"[bold {color} on dark_blue]▓{icon}{phase_name[:5]}▓[/bold {color} on dark_blue]")
            elif reached:
                parts.append(f"[{color}]█{icon}{phase_name[:5]}█[/{color}]")
            else:
                parts.append(f"[dim]░░{phase_name[:5]}░░[/dim]")

        bar = "→".join(parts)
        console.print(f"  [bold]Kill Chain:[/bold] {bar}")

    def print_cost_ticker(self, tokens_used: int, estimated_cost_per_1k: float = 0.01):
        """Print a running cost estimate based on token usage.

        Args:
            tokens_used: Total tokens consumed this run.
            estimated_cost_per_1k: Cost per 1K tokens (default $0.01 for GPT-5-mini).
        """
        cost = (tokens_used / 1000.0) * estimated_cost_per_1k
        console.print(
            f"  [dim]💰 Tokens: {tokens_used:,}  |  Est. cost: ${cost:.4f}  |  "
            f"Rate: ${estimated_cost_per_1k:.3f}/1K tok[/dim]"
        )

    # =========================================================================
    # LEGACY COMPATIBILITY — record_step + print_step_table stubs
    # =========================================================================
    def record_step(self, step, phase, agent_results, global_reward, done,
                    reward_breakdown=None, agent_discoveries=None,
                    skipped_agents=None):
        """Legacy record — just track stats (print_step handles display)."""
        agent_discoveries = agent_discoveries or {}
        skipped_agents = skipped_agents or {}

        for result in agent_results:
            agent_name = result.get("agent", result.get("agent_name", "?"))
            chosen = result.get("chosen_action", "?")
            tokens = result.get("tokens_used", 0)

            if agent_name not in self.agent_stats:
                self.agent_stats[agent_name] = {
                    "episode_reward": 0.0, "episode_mentor_calls": 0,
                    "episode_tokens": 0, "last_action": "", "last_reward": 0.0,
                    "confidence": 0.5, "phase": phase, "total_commands": 0,
                    "unique_commands": set(), "decision_sources": {},
                }
            s = self.agent_stats[agent_name]
            s["episode_reward"] += global_reward
            s["last_action"] = chosen
            s["last_reward"] = global_reward
            s["episode_tokens"] += tokens
            s["confidence"] = result.get("confidence", 0.5)
            s["phase"] = phase
            s["total_commands"] = s.get("total_commands", 0) + 1
            s.setdefault("unique_commands", set()).add(chosen)

            source = result.get("source", "unknown")
            s.setdefault("decision_sources", {})
            s["decision_sources"][source] = s["decision_sources"].get(source, 0) + 1
            if result.get("mentor_call"):
                s["episode_mentor_calls"] += 1
                self.episode_mentor_calls_total += 1

            self.tokens_total += tokens
            self.tokens_by_agent[agent_name] = self.tokens_by_agent.get(agent_name, 0) + tokens

            if agent_name not in self.episode_commands:
                self.episode_commands[agent_name] = []
            if chosen in self.episode_commands[agent_name]:
                self.episode_repeat_count += 1
            self.episode_commands[agent_name].append(chosen)
            self.episode_unique_commands.add(chosen)

        for aname, disc_dict in agent_discoveries.items():
            for disc_type, items in disc_dict.items():
                if disc_type not in self.episode_discoveries:
                    self.episode_discoveries[disc_type] = set()
                for item in items:
                    self.episode_discoveries[disc_type].add(str(item))

        self.step_rewards.append(global_reward)
        if not self.phase_timeline or self.phase_timeline[-1][1] != phase.upper():
            self.phase_timeline.append((step, phase.upper()))
        self.step_counter = step

    def print_step_table(self, step: int):
        """Legacy no-op — print_step is the unified printer now."""
        pass
