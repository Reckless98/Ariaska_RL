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
    "difficulty_gate": ("🛡️", "blue"),
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
    max_output_lines: int = 6
    max_output_width: int = 90
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
        Print ONE comprehensive step display.  This is the ONLY step printer.
        """
        if not self.should_print(step):
            return

        phase_upper = phase.upper()
        phase_icon = PHASE_ICONS.get(phase_upper, "❓")
        reward_style = "bold green" if global_reward > 0 else "bold red" if global_reward < 0 else "dim"
        ep_str = (
            f"Ep {self.current_episode}/{self.total_episodes}"
            if self.total_episodes else f"Ep {self.current_episode}"
        )

        # Step reward sparkline (last 15 steps)
        step_spark = sparkline(list(self.step_rewards), width=15)

        # ── HEADER ───────────────────────────────────────────────────
        hdr_parts = [
            f"[bold cyan]Step {step + 1:2d}[/bold cyan]",
            f"[bold cyan]{ep_str}[/bold cyan]",
            f"[bold yellow]{mode_tag}[/bold yellow]",
            f"{phase_icon} [bold]{phase_upper}[/bold]",
            f"[{reward_style}]R:{global_reward:+.1f}[/{reward_style}]",
        ]
        if done:
            hdr_parts.append("[bold magenta]✅ DONE[/bold magenta]")
        hdr_parts.append(f"[dim]{step_spark}[/dim]")
        header = " │ ".join(hdr_parts)

        console.print(f"\n[cyan]┌{'─' * 95}┐[/cyan]")
        console.print(f"│ {header}")
        console.print(f"[dim cyan]├{'─' * 95}┤[/dim cyan]")

        # ── COMPACT AGENT TABLE ──────────────────────────────────────
        active_agents = [a for a in agent_infos if not a.skipped]

        if active_agents:
            table = Table(
                show_header=True,
                header_style="bold white on dark_blue",
                border_style="dim",
                box=box.SIMPLE_HEAVY,
                padding=(0, 1),
                expand=True,
            )
            table.add_column("Agent", style="bold", width=14, no_wrap=True)
            table.add_column("Src", width=10, no_wrap=True)
            table.add_column("Command", width=50, overflow="fold")
            table.add_column("R", width=7, justify="right")
            table.add_column("Conf", width=5, justify="right")

            for a in active_agents:
                icon, _role, _style = AGENT_ICONS.get(
                    a.agent_name, ("🤖", "Agent", "dim")
                )
                src_key = a.source[:8] if a.source else "unknown"
                src_icon, src_style = SOURCE_STYLES.get(
                    src_key, SOURCE_STYLES.get(a.source.split("_")[0] if a.source else "unknown", ("❓", "dim"))
                )

                r_text = Text(
                    f"{a.reward:+.1f}",
                    style="green" if a.reward > 0 else "red" if a.reward < 0 else "dim",
                )
                c_text = Text(
                    f"{a.confidence:.0%}",
                    style="green" if a.confidence > 0.7 else "red" if a.confidence < 0.3 else "yellow",
                )
                cmd_disp = (a.command or "(none)")[:70]

                table.add_row(
                    f"{icon} {a.agent_name.replace('Agent', '')}",
                    Text(f"{src_icon} {a.source[:8]}", style=src_style),
                    Text(cmd_disp, style="white"),
                    r_text,
                    c_text,
                )
            console.print(table)

        # ── DETAILED PER-AGENT OUTPUT ────────────────────────────────
        for a in active_agents:
            icon, _role, style = AGENT_ICONS.get(
                a.agent_name, ("🤖", "Agent", "dim")
            )
            src_key = a.source[:8] if a.source else "unknown"
            src_icon, src_style = SOURCE_STYLES.get(
                src_key, SOURCE_STYLES.get(a.source.split("_")[0] if a.source else "unknown", ("❓", "dim"))
            )
            mentor_tag = " [yellow]📡 MENTOR[/yellow]" if a.mentor_call else ""

            console.print(
                f"  [{style}]{icon} {a.agent_name}[/{style}]"
                f" [{src_style}][{src_icon} {a.source}][/{src_style}]"
                f"{mentor_tag}"
                f"  [dim]tok:{a.tokens_used}[/dim]"
            )

            # Full command
            if a.command:
                console.print(f"    [bold white]CMD:[/bold white] [cyan]{a.command}[/cyan]")

            # Command output (multi-line)
            if a.command_output and self.config.show_output:
                lines = a.command_output.strip().split("\n")
                max_l = self.config.max_output_lines
                for line in lines[:max_l]:
                    console.print(f"    [dim]OUT: {line[:self.config.max_output_width]}[/dim]")
                if len(lines) > max_l:
                    console.print(f"    [dim]    ... (+{len(lines) - max_l} more lines)[/dim]")

            # Reasoning
            if a.mentor_reasoning:
                console.print(f"    [yellow]WHY: {a.mentor_reasoning[:150]}[/yellow]")

            # Discoveries
            if a.discoveries:
                disc_parts = []
                for dtype, items in a.discoveries.items():
                    if items and isinstance(items, (list, set, tuple)):
                        disc_parts.append(f"{dtype}: {', '.join(str(i) for i in list(items)[:5])}")
                    elif items and isinstance(items, str):
                        disc_parts.append(f"{dtype}: {items}")
                if disc_parts:
                    console.print(f"    [green bold]FOUND: {' │ '.join(disc_parts)}[/green bold]")

        # ── SKIPPED AGENTS ───────────────────────────────────────────
        if skipped_agents:
            parts = []
            for aname, reason in skipped_agents.items():
                ic = AGENT_ICONS.get(aname, ("🤖", "", ""))[0]
                parts.append(f"{ic} {aname.replace('Agent', '')}: {reason}")
            console.print(f"  [dim]💤 Skipped: {' │ '.join(parts)}[/dim]")

        # ── DISCOVERY BOARD (compact) ────────────────────────────────
        if discovery_board and self.config.show_discoveries:
            db = discovery_board
            parts = []
            ports = db.get("ports", set())
            if ports:
                pl = sorted(ports if isinstance(ports, (set, list)) else [])[:10]
                parts.append(f"🔓 Ports({len(ports)}): {','.join(str(p) for p in pl)}")
            svcs = db.get("services", set())
            if svcs:
                parts.append(
                    f"⚙️  Svcs({len(svcs)}): {','.join(str(s)[:15] for s in list(svcs)[:5])}"
                )
            creds = db.get("credentials", set())
            if creds:
                parts.append(f"🔑 Creds: {len(creds)}")
            shells = db.get("shells", set())
            if shells:
                parts.append(f"💀 Shells: {len(shells)}")
            flags = db.get("flags_set", set())
            if flags:
                parts.append(f"🚩 Flags: {','.join(str(f)[:18] for f in list(flags)[:4])}")
            if parts:
                console.print(f"  [bold]{' │ '.join(parts)}[/bold]")

        # ── REWARD BREAKDOWN ─────────────────────────────────────────
        if reward_breakdown and self.config.show_reward_breakdown:
            rb = reward_breakdown
            rp = []
            if rb.get("base", 0):
                rp.append(f"base:{rb['base']:+.1f}")
            if rb.get("novelty_bonus", 0):
                rp.append(f"[green]novelty:{rb['novelty_bonus']:+.1f}[/green]")
            if rb.get("redundancy_penalty", 0):
                rp.append(f"[red]repeat:{rb['redundancy_penalty']:+.1f}[/red]")
            if rb.get("phase_bonus", 0):
                rp.append(f"[cyan]phase:{rb['phase_bonus']:+.1f}[/cyan]")
            reason = rb.get("reason", "")
            if rp or reason:
                detail = " ".join(rp)
                if reason:
                    detail += f"  [dim]({reason[:80]})[/dim]"
                console.print(f"  [dim]💰 {detail}[/dim]")

        # ── EVENTS (last 3 seconds) ─────────────────────────────────
        now = time.time()
        recent = [e for e in self.events if now - e.timestamp < 5]
        if recent:
            self._print_events(recent[-3:])

        console.print(f"[dim cyan]└{'─' * 95}┘[/dim cyan]")
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
            title="[bold]Agent Snapshot[/bold]",
            show_header=True, header_style="bold magenta",
            border_style="dim", box=box.SIMPLE, padding=(0, 1),
        )
        table.add_column("Agent", style="bold", width=14)
        table.add_column("Phase", width=12)
        table.add_column("Cmds", width=5, justify="right")
        table.add_column("Uniq", width=5, justify="right")
        table.add_column("Ep R", width=9, justify="right")
        table.add_column("Mentor", width=6, justify="center")
        table.add_column("Tok", width=7, justify="right")
        table.add_column("Top Src", width=10)

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
            sources = stats.get("decision_sources", {})
            top_src = max(sources, key=sources.get) if sources else "-"
            table.add_row(
                f"{icon} {agent_name.replace('Agent', '')}",
                str(stats.get("phase", "?"))[:12],
                str(total_cmds), str(unique_cmds), ep_text,
                f"📡{mc}" if mc > 0 else "-",
                str(stats.get("episode_tokens", 0)),
                top_src[:10],
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

        # ── METRICS TABLE ────────────────────────────────────────────
        table = Table(
            title=f"[bold]Episode {episode} Summary[/bold]",
            show_header=True, header_style="bold green",
            border_style="green", box=box.ROUNDED,
        )
        table.add_column("Metric", style="bold cyan", width=22)
        table.add_column("Value", justify="right", width=14)
        table.add_column("Trend", width=24)

        avg_step = (
            sum(list(self.step_rewards)[-10:]) /
            max(len(list(self.step_rewards)[-10:]), 1)
        )
        table.add_row("Episode Reward", f"{total_reward:+.1f}",
                       self._trend_display(self.episode_rewards))
        table.add_row("Avg Step Reward", f"{avg_step:+.2f}",
                       sparkline(list(self.step_rewards)))
        table.add_row("Steps", str(total_steps), "")

        phase_icon = PHASE_ICONS.get(highest_phase, "")
        table.add_row("Highest Phase", f"{phase_icon} {highest_phase}", "")

        avg_mentor = (
            sum(self.mentor_history) / max(len(self.mentor_history), 1)
        )
        table.add_row("Mentor Calls", str(mentor_calls), f"rate: {avg_mentor:.0%}")

        total_disc = sum(len(v) for v in self.episode_discoveries.values())
        table.add_row("Discoveries", str(total_disc), "")
        table.add_row("Unique Cmds", str(len(self.episode_unique_commands)),
                       f"repeats: {self.episode_repeat_count}")
        table.add_row("Tokens", str(self.tokens_total), "")

        if ppo_metrics:
            if ppo_metrics.get("updates"):
                table.add_row("PPO Updates", str(ppo_metrics["updates"]), "")
            if ppo_metrics.get("policy_loss"):
                table.add_row("PPO π Loss", f"{ppo_metrics['policy_loss']:.4f}", "")
            if ppo_metrics.get("value_loss"):
                table.add_row("PPO V Loss", f"{ppo_metrics['value_loss']:.4f}", "")
            if ppo_metrics.get("entropy"):
                table.add_row("PPO Entropy", f"{ppo_metrics['entropy']:.4f}", "")

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

    def reset_episode(self):
        self.step_counter = 0
        self.last_print_step = -1
        self.step_rewards.clear()
        self.confidence_history.clear()
        self.mentor_history.clear()
        self.reward_history.clear()
        self._reset_episode_stats()

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
