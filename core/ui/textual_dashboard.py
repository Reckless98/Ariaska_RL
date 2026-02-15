#!/usr/bin/env python3
"""
core/ui/textual_dashboard.py — ARIASKA Textual TUI Dashboard v2.0 (Phase 6.3)

Persistent 6-pane terminal dashboard for overnight training runs.
Subscribes to EventBus and renders live updates using Textual.

Layout:
  ┌──────────────────────────────────────────────────────────────────┐
  │  HEADER: Ep · Step · Phase · Reward · Mentor% · Cost · Target   │
  ├──────────────────────┬───────────────────────┬───────────────────┤
  │  AGENT TABLE         │ DISCOVERY BOARD        │ STATUS PANEL      │
  │  DataTable: 5 rows   │ Ports, Creds, Vulns,  │ Watchdog 🟢/🟡/🔴│
  │  (agent, source,     │ Shells, Services      │ PPO loss, entropy │
  │   reward, discs)     │                        │ Token budget      │
  ├──────────────────────┴───────────────────────┴───────────────────┤
  │  PER-AGENT LOGS (tabbed) + REASONING TRACE                      │
  ├──────────────────────────────────────────────────────────────────┤
  │  GLOBAL EVENT LOG                                                │
  └──────────────────────────────────────────────────────────────────┘

Requires: textual>=0.47.0
"""

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("ariaska.textual_dashboard")

try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical
    from textual.widgets import (
        Header,
        Footer,
        Static,
        DataTable,
        RichLog,
        TabbedContent,
        TabPane,
    )
    from textual.reactive import reactive

    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False
    logger.debug("textual not installed — TextualDashboard disabled")


# Guard all class definitions behind availability check
if TEXTUAL_AVAILABLE:

    class StatusBar(Static):
        """Top status bar showing key metrics including cost."""

        episode_num: reactive[int] = reactive(0)
        step_num: reactive[int] = reactive(0)
        phase: reactive[str] = reactive("RECON")
        reward: reactive[float] = reactive(0.0)
        mentor_pct: reactive[float] = reactive(0.0)
        cost_usd: reactive[float] = reactive(0.0)
        token_budget_pct: reactive[float] = reactive(100.0)
        target: reactive[str] = reactive("")
        mode: reactive[str] = reactive("sim")

        def render(self) -> str:
            mode_icon = "🔴 LIVE" if self.mode == "live" else "🔵 SIM"
            budget_color = "green" if self.token_budget_pct > 50 else ("yellow" if self.token_budget_pct > 20 else "red")
            return (
                f" {mode_icon}  │  "
                f"Ep: [bold cyan]{self.episode_num}[/]  │  "
                f"Step: [bold]{self.step_num}[/]  │  "
                f"Phase: [bold yellow]{self.phase}[/]  │  "
                f"Reward: [bold green]{self.reward:+.1f}[/]  │  "
                f"Mentor: [bold magenta]{self.mentor_pct:.1f}%[/]  │  "
                f"Cost: [bold]${self.cost_usd:.4f}[/]  │  "
                f"Budget: [bold {budget_color}]{self.token_budget_pct:.0f}%[/]  │  "
                f"Target: {self.target}"
            )

    class DiscoveryBoard(Static):
        """Panel showing cross-episode discovery state."""

        ports: reactive[str] = reactive("—")
        services: reactive[str] = reactive("—")
        credentials: reactive[str] = reactive("—")
        vulns: reactive[str] = reactive("—")
        shells: reactive[str] = reactive("—")

        def render(self) -> str:
            return (
                f"[bold underline]Discovery Board[/]\n"
                f"  [cyan]Ports:[/]   {self.ports}\n"
                f"  [green]Svc:[/]     {self.services}\n"
                f"  [yellow]Creds:[/]  {self.credentials}\n"
                f"  [red]Vulns:[/]   {self.vulns}\n"
                f"  [magenta]Shells:[/] {self.shells}"
            )

    class StatusPanel(Static):
        """Panel showing watchdog, PPO, and system status."""

        watchdog_status: reactive[str] = reactive("🟢 OK")
        ppo_loss: reactive[str] = reactive("—")
        ppo_entropy: reactive[str] = reactive("—")
        ppo_value_loss: reactive[str] = reactive("—")
        decision_source: reactive[str] = reactive("—")

        def render(self) -> str:
            return (
                f"[bold underline]System Status[/]\n"
                f"  Watchdog: {self.watchdog_status}\n"
                f"  [cyan]PPO π-loss:[/] {self.ppo_loss}\n"
                f"  [cyan]PPO V-loss:[/] {self.ppo_value_loss}\n"
                f"  [cyan]Entropy:[/]    {self.ppo_entropy}\n"
                f"  [dim]Last src:[/]   {self.decision_source}"
            )

    class AriaskaDashboard(App):
        """
        Textual TUI application for Ariaska training monitoring.
        Phase 6.3: Extended with discovery board, cost tracking,
        watchdog status, PPO metrics, and reasoning trace.

        Usage:
            dashboard = AriaskaDashboard()
            event_bus.subscribe(dashboard.on_event)
            dashboard.run()  # Blocks — run in thread or async
        """

        CSS = """
        StatusBar {
            dock: top;
            height: 1;
            background: $surface;
            color: $text;
            padding: 0 1;
        }

        #main-layout {
            height: 1fr;
        }

        #left-pane {
            width: 35%;
            min-width: 35;
        }

        #mid-pane {
            width: 30%;
            min-width: 28;
        }

        #right-pane {
            width: 35%;
            min-width: 30;
        }

        #agent-table {
            height: 1fr;
        }

        DiscoveryBoard {
            height: auto;
            padding: 1;
            border: solid $accent;
        }

        StatusPanel {
            height: auto;
            padding: 1;
            border: solid $accent;
        }

        #reasoning-log {
            height: 8;
            border-top: dashed $accent;
        }

        #global-log {
            dock: bottom;
            height: 10;
            border-top: solid $accent;
        }

        DataTable {
            height: 1fr;
        }

        RichLog {
            height: 1fr;
        }

        TabbedContent {
            height: 1fr;
        }
        """

        BINDINGS = [
            ("q", "quit", "Quit"),
            ("d", "toggle_dark", "Dark Mode"),
        ]

        TITLE = "ARIASKA RL — Training Dashboard"
        SUB_TITLE = "Phase 6.3"

        AGENTS = ["ScoutAgent", "RedAgent", "BlueAgent", "OrionAgent", "ShadowAgent"]

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._event_count = 0

        def compose(self) -> ComposeResult:
            yield Header()
            yield StatusBar(id="status-bar")

            with Horizontal(id="main-layout"):
                with Vertical(id="left-pane"):
                    yield DataTable(id="agent-table")

                with Vertical(id="mid-pane"):
                    yield DiscoveryBoard(id="discovery-board")
                    yield StatusPanel(id="status-panel")
                    yield RichLog(
                        id="reasoning-log",
                        highlight=True,
                        markup=True,
                        max_lines=100,
                    )

                with Vertical(id="right-pane"):
                    with TabbedContent():
                        for agent in self.AGENTS:
                            with TabPane(agent, id=f"tab-{agent}"):
                                yield RichLog(
                                    id=f"log-{agent}",
                                    highlight=True,
                                    markup=True,
                                    max_lines=500,
                                )

            yield RichLog(id="global-log", highlight=True, markup=True, max_lines=300)
            yield Footer()

        def on_mount(self) -> None:
            """Initialize table columns."""
            table = self.query_one("#agent-table", DataTable)
            table.add_columns(
                "Agent", "Source", "Phase", "Reward", "Disc", "ms",
            )
            # Add initial rows
            for agent in self.AGENTS:
                table.add_row(agent, "—", "—", "—", "—", "—", key=agent)

            # Write welcome messages
            glog = self.query_one("#global-log", RichLog)
            glog.write("[bold green]✓ ARIASKA Dashboard v2.0 (Phase 6.3) started[/]")

            rlog = self.query_one("#reasoning-log", RichLog)
            rlog.write("[dim]Reasoning traces will appear here...[/]")

        # -- EventBus callback --

        def on_event(self, event: Any) -> None:
            """
            EventBus subscriber callback.

            Dispatches to appropriate handler based on event kind.
            Thread-safe: Textual handles cross-thread message posting.
            """
            self._event_count += 1
            kind = getattr(event, "kind", None)

            if kind is None:
                return

            kind_str = kind.value if hasattr(kind, "value") else str(kind)

            if kind_str == "step":
                self.call_from_thread(self._handle_step_event, event)
            elif kind_str in ("phase_transition", "mentor_call", "checkpoint",
                              "warning", "error", "target_health"):
                self.call_from_thread(self._handle_generic_event, event)
            elif kind_str == "episode_start":
                self.call_from_thread(self._handle_episode_start, event)
            elif kind_str == "episode_end":
                self.call_from_thread(self._handle_episode_end, event)

        # -- Event handlers --

        def _handle_step_event(self, event: Any) -> None:
            """Process a StepEvent — update table, discoveries, PPO, reasoning."""
            # Update status bar
            status = self.query_one("#status-bar", StatusBar)
            status.episode_num = getattr(event, "episode_num", 0)
            status.step_num = getattr(event, "step_num", 0)
            status.phase = getattr(event, "phase_after", "?")
            status.reward = getattr(event, "episode_reward_so_far", 0.0)
            status.target = getattr(event, "target_ip", "")
            status.mode = getattr(event, "mode", "sim")

            # Compute mentor %
            ep_steps = max(getattr(event, "episode_steps_so_far", 1), 1)
            ep_mentor = getattr(event, "episode_mentor_calls_so_far", 0)
            status.mentor_pct = (ep_mentor / ep_steps) * 100

            # Cost + budget
            data = getattr(event, "data", {}) or {}
            status.cost_usd = data.get("cumulative_cost_usd", status.cost_usd)
            budget_remaining = data.get("token_budget_remaining_pct", None)
            if budget_remaining is not None:
                status.token_budget_pct = budget_remaining

            # Discovery board update
            disc_data = data.get("discovery_board", None)
            if disc_data:
                try:
                    board = self.query_one("#discovery-board", DiscoveryBoard)
                    ports = disc_data.get("ports", set())
                    board.ports = ", ".join(sorted(str(p) for p in ports)[:10]) if ports else "—"
                    svcs = disc_data.get("services", set())
                    board.services = ", ".join(sorted(str(s) for s in svcs)[:8]) if svcs else "—"
                    creds = disc_data.get("credentials", set())
                    board.credentials = str(len(creds)) + " found" if creds else "—"
                    vulns = disc_data.get("vulns", set())
                    board.vulns = str(len(vulns)) + " known" if vulns else "—"
                    shells = disc_data.get("shells", set())
                    board.shells = ", ".join(sorted(str(s) for s in shells)[:5]) if shells else "—"
                except Exception:
                    pass

            # Status panel: watchdog + PPO
            try:
                panel = self.query_one("#status-panel", StatusPanel)
                wd = data.get("watchdog_status", None)
                if wd:
                    panel.watchdog_status = wd
                ppo = data.get("ppo_metrics", None)
                if ppo:
                    panel.ppo_loss = f"{ppo.get('policy_loss', 0):.4f}"
                    panel.ppo_value_loss = f"{ppo.get('value_loss', 0):.4f}"
                    panel.ppo_entropy = f"{ppo.get('entropy', 0):.4f}"
            except Exception:
                pass

            # Update agent table + per-agent logs
            table = self.query_one("#agent-table", DataTable)
            records = getattr(event, "agent_records", [])

            for rec in records:
                agent = getattr(rec, "agent_name", "?")
                source = getattr(rec, "decision_source", "?")
                phase = getattr(rec, "phase", "?")
                reward = getattr(rec, "reward", 0.0)
                discoveries = getattr(rec, "discoveries", [])
                exec_ms = getattr(rec, "exec_ms", 0.0)
                command = getattr(rec, "command", "?")
                stdout = getattr(rec, "stdout_snippet", "")
                mentor_call = getattr(rec, "mentor_call", False)
                mentor_tier = getattr(rec, "mentor_tier", None)
                reasoning = getattr(rec, "reasoning", "")

                # Update last decision source on status panel
                try:
                    panel = self.query_one("#status-panel", StatusPanel)
                    panel.decision_source = f"{agent[:5]}:{source}"
                except Exception:
                    pass

                # Update table row
                disc_str = ", ".join(discoveries[:3]) if discoveries else "—"
                try:
                    table.update_cell(agent, "Source", source[:8])
                    table.update_cell(agent, "Phase", phase[:6])
                    table.update_cell(agent, "Reward", f"{reward:+.1f}")
                    table.update_cell(agent, "Disc", disc_str[:15])
                    table.update_cell(agent, "ms", f"{exec_ms:.0f}")
                except Exception:
                    pass  # Row may not exist yet

                # Write to per-agent log
                try:
                    log = self.query_one(f"#log-{agent}", RichLog)
                    mentor_str = f" [magenta]🧠{mentor_tier}[/]" if mentor_call else ""
                    log.write(
                        f"[dim]S{getattr(event, 'step_num', 0):03d}[/] "
                        f"[cyan]{source:10s}[/]{mentor_str} "
                        f"[bold]{command[:60]}[/]"
                    )
                    if stdout:
                        log.write(f"  [dim]{stdout[:120]}[/]")
                    if discoveries:
                        log.write(f"  [green]→ {', '.join(discoveries)}[/]")
                except Exception:
                    pass

                # Reasoning trace log
                if reasoning:
                    try:
                        rlog = self.query_one("#reasoning-log", RichLog)
                        rlog.write(
                            f"[dim]S{getattr(event, 'step_num', 0):03d}[/] "
                            f"[bold]{agent[:5]}[/] {reasoning[:150]}"
                        )
                    except Exception:
                        pass

            # Phase transition → global log
            phase_before = getattr(event, "phase_before", "")
            phase_after = getattr(event, "phase_after", "")
            if phase_before and phase_after and phase_before != phase_after:
                glog = self.query_one("#global-log", RichLog)
                glog.write(
                    f"[bold yellow]⚡ PHASE: {phase_before} → {phase_after}[/] "
                    f"(Step {getattr(event, 'step_num', 0)})"
                )

        def _handle_generic_event(self, event: Any) -> None:
            """Process a GenericEvent — write to global log."""
            glog = self.query_one("#global-log", RichLog)
            kind_str = event.kind.value if hasattr(event.kind, "value") else str(event.kind)
            msg = getattr(event, "message", "")

            style_map = {
                "mentor_call": "[magenta]🧠 MENTOR[/]",
                "checkpoint": "[blue]💾 CHECKPOINT[/]",
                "warning": "[yellow]⚠️  WARNING[/]",
                "error": "[red]❌ ERROR[/]",
                "target_health": "[cyan]🏥 TARGET[/]",
                "phase_transition": "[yellow]⚡ PHASE[/]",
            }

            prefix = style_map.get(kind_str, f"[dim]{kind_str}[/]")
            glog.write(f"{prefix} {msg}")

            # Watchdog warnings → also update status panel
            if kind_str == "warning":
                data = getattr(event, "data", {}) or {}
                trigger = data.get("watchdog_trigger", None)
                if trigger:
                    try:
                        panel = self.query_one("#status-panel", StatusPanel)
                        panel.watchdog_status = f"🟡 {trigger}"
                    except Exception:
                        pass
            elif kind_str == "error":
                try:
                    panel = self.query_one("#status-panel", StatusPanel)
                    panel.watchdog_status = f"🔴 {msg[:30]}"
                except Exception:
                    pass

        def _handle_episode_start(self, event: Any) -> None:
            """Episode start → reset panels + global log."""
            glog = self.query_one("#global-log", RichLog)
            data = getattr(event, "data", {})
            ep = data.get("episode", getattr(event, "episode_num", "?"))
            glog.write(f"\n[bold green]{'='*50}[/]")
            glog.write(f"[bold green]📍 EPISODE {ep} STARTED[/]")

            # Reset watchdog status
            try:
                panel = self.query_one("#status-panel", StatusPanel)
                panel.watchdog_status = "🟢 OK"
            except Exception:
                pass

        def _handle_episode_end(self, event: Any) -> None:
            """Episode end → summary in global log with cost."""
            glog = self.query_one("#global-log", RichLog)
            data = getattr(event, "data", {})
            ep = data.get("episode", getattr(event, "episode_num", "?"))
            reward = data.get("total_reward", 0)
            phase = data.get("highest_phase", "?")
            mentor_calls = data.get("mentor_calls", 0)
            ep_cost = data.get("episode_cost_usd", 0.0)
            cum_cost = data.get("cumulative_cost_usd", 0.0)
            discoveries = data.get("total_discoveries", 0)
            glog.write(
                f"[bold cyan]📊 EPISODE {ep} DONE[/] — "
                f"Reward: {reward:+.1f}, Phase: {phase}, "
                f"Disc: {discoveries}, Mentor: {mentor_calls}, "
                f"Cost: ${ep_cost:.4f} (cum: ${cum_cost:.4f})"
            )
            glog.write(f"[bold green]{'='*50}[/]\n")


# ---------------------------------------------------------------------------
# Factory / wrapper for non-Textual environments
# ---------------------------------------------------------------------------

class NullDashboard:
    """No-op dashboard when Textual is unavailable."""

    def on_event(self, event: Any) -> None:
        pass

    def run(self) -> None:
        pass


def create_textual_dashboard() -> Any:
    """
    Factory: returns AriaskaDashboard if textual is available, else NullDashboard.
    """
    if TEXTUAL_AVAILABLE:
        return AriaskaDashboard()
    else:
        logger.warning("textual not installed — using NullDashboard")
        return NullDashboard()
