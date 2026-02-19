#!/usr/bin/env python3
"""
core/observability/live_dashboard.py — ARIASKA Live Training Dashboard v5.0

Phase 11.0: Full Visibility upgrade with teaching annotations:
  • Per-agent commands with full output, reasoning, discoveries
  • ASCII sparkline reward trends across episodes
  • Phase progression timeline with kill chain bar
  • Strategic agent activation display (who acts and why)
  • Discovery board with real-time updates
  • Improvement tracking with block-char bar charts
  ─── NEW in v4.0 ───
  • Training start banner with full configuration display
  • Unified algorithm panel (PPO + DDQN + CognitionNode + SIL + RND)
  • Decision pipeline visualization (4-stage: Playbook → PPO → Registry → GPT)
  • Per-coach PPO training metrics (policy loss, value loss, entropy sparklines)
  • DDQN macro-intent distribution chart
  • Discovery board heatmap panel
  • Agent coordination matrix display
  • Enhanced run summary with algorithm trend analysis
  ─── NEW in v5.0 (Phase 11.0) ───
  • Teaching points inline annotations
  • Budget pressure indicator with color-coded status
  • Parse explanation annotations per discovery
  • Phase ladder state display
  • Unified step trace integration

Author: Filip Volf — Phase 6.5 → Phase 11.0
"""

import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from collections import deque

from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich.layout import Layout
from rich import box

# Force Rich terminal rendering — ensures full Rich UI even if accidentally piped
console = Console(force_terminal=True)

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
    "dual_mentor": ("📡", "bright_yellow"),
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
    "codex_meta": ("🧬", "bright_magenta"),
    "codex_me": ("🧬", "bright_magenta"),
    "sil": ("💎", "bright_green"),
    "ddqn": ("🎲", "bright_cyan"),
    "cognition": ("🧠", "bright_white"),
    "gpt_codex": ("🧠", "bright_green"),
    "gpt_primary": ("🧠", "bright_green"),
    "arbitrator_ppo": ("🤖", "green"),
    "arbitrato": ("🤖", "green"),
    "regex_fallback": ("🔍", "dim"),
    "regex_p19": ("🔍", "dim"),
    "web_followup": ("🌐", "bright_cyan"),
    "web_follo": ("🌐", "bright_cyan"),
    "cred_reuse": ("🔑", "bright_yellow"),
    "cred_reus": ("🔑", "bright_yellow"),
    "followup_queue": ("📋", "bright_yellow"),
    "followup_": ("📋", "bright_yellow"),
    "fallback": ("⚡", "dim"),
    "unknown": ("❓", "dim"),
}

# Algorithm icons for the unified algorithm panel
ALGO_ICONS = {
    "PPO": ("🤖", "green", "Proximal Policy Optimization"),
    "DDQN": ("🎲", "cyan", "Double DQN Macro-Actions"),
    "SIL": ("💎", "bright_green", "Self-Imitation Learning (PPO integrated)"),
    "RND": ("🔮", "magenta", "Random Network Distillation"),
    "SAC": ("🌊", "blue", "Soft Actor-Critic"),
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
    # Phase 29: observability extensions
    evidence_gate_result: str = ""          # "", "pass", "log_reject", "enforce_reject"
    micro_chain_escalated: bool = False     # True if MicroChain escalated to codex
    self_debug_fix: str = ""                # corrected command from SelfDebugger (empty = no fix)
    retry_count: int = 0                    # number of transient-failure retries


@dataclass
class DashboardConfig:
    """Configuration for live dashboard."""
    enabled: bool = True
    mode: str = "live"  # "off", "summary", "live"
    watch_rate: float = 1.0
    trend_window: int = 20
    max_action_width: int = 0  # 0 = no limit
    max_output_lines: int = 0  # 0 = no limit
    max_output_width: int = 0  # 0 = no limit
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

        # Phase 10.2: Algorithm-level tracking for unified display
        self.ppo_loss_history: Dict[str, deque] = {}   # coach → deque of policy losses
        self.ppo_vloss_history: Dict[str, deque] = {}  # coach → deque of value losses
        self.ppo_entropy_history: Dict[str, deque] = {}  # coach → deque of entropies
        self.ddqn_history: deque = deque(maxlen=50)  # macro distribution per episode
        self.decision_source_history: List[Dict[str, int]] = []  # per-episode source counts
        self.discovery_board_snapshot: Dict[str, Any] = {}
        self.algo_active: Dict[str, bool] = {  # which algorithms are active
            "PPO": False, "DDQN": False,
            "SIL": False, "RND": False, "SAC": False,
        }
        self.training_config: Dict[str, Any] = {}  # stored at training start

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
    # INIT PROGRESS — Rich module loading display
    # =========================================================================
    def print_init_progress(self, modules: list):
        """
        Print a compact Rich table showing all initialized modules.

        Args:
            modules: List of (name, status, detail) tuples.
                     status: 'ok', 'warn', 'fail', 'skip'
        """
        STATUS_ICONS = {
            "ok": "[green]✅[/green]",
            "warn": "[yellow]⚠️[/yellow]",
            "fail": "[red]❌[/red]",
            "skip": "[dim]⏭️[/dim]",
        }

        table = Table(
            box=box.ROUNDED, show_header=True, header_style="bold cyan",
            padding=(0, 1), expand=False, title="Module Initialization",
            title_style="bold white",
        )
        table.add_column("", width=3, justify="center")
        table.add_column("Module", style="white", width=28)
        table.add_column("Detail", style="dim", max_width=55)

        ok_count = warn_count = fail_count = 0
        for name, status, detail in modules:
            icon = STATUS_ICONS.get(status, STATUS_ICONS["ok"])
            if status == "ok":
                ok_count += 1
            elif status == "warn":
                warn_count += 1
            elif status == "fail":
                fail_count += 1
            # Truncate detail to keep table compact
            detail_str = str(detail)[:48] if detail else ""
            table.add_row(icon, name, detail_str)

        summary_parts = [f"[green]{ok_count} loaded[/green]"]
        if warn_count:
            summary_parts.append(f"[yellow]{warn_count} warnings[/yellow]")
        if fail_count:
            summary_parts.append(f"[red]{fail_count} failed[/red]")

        panel = Panel(
            Group(table, Text.from_markup(f"\n  {' │ '.join(summary_parts)}", style="bold")),
            title="[bold bright_blue]⚡ Ariaska System Init[/bold bright_blue]",
            border_style="bright_blue",
            box=box.ROUNDED,
            padding=(0, 2),
        )
        console.print(panel)

    # =========================================================================
    # TRAINING START BANNER — Phase 10.2
    # =========================================================================
    def print_training_start(
        self,
        config: Dict[str, Any],
        agents: List[str],
        algorithms: Dict[str, bool],
        target: str = "",
    ):
        """Print a polished training start banner with full system overview.

        Shows all agents, active algorithms, configuration, and target info
        in a professional multi-panel layout.

        Args:
            config: Training configuration dict (episodes, steps, seed, env, etc.)
            agents: List of active agent names
            algorithms: Dict of algorithm name → active boolean
            target: Target IP/hostname
        """
        self.training_config = config
        self.algo_active.update(algorithms)

        # ── ARIASKA BANNER ───────────────────────────────────────────
        self.print_ariaska_banner()

        # ── CONFIG + AGENTS PANEL (side by side) ─────────────────────
        # Left: Training Configuration
        config_lines = []
        config_lines.append(f"[bold cyan]Target:[/bold cyan]      {target or config.get('target', '?')}")
        config_lines.append(f"[bold cyan]Mode:[/bold cyan]        [bold bright_green]CONTINUOUS LIVE[/bold bright_green]")
        config_lines.append(f"[bold cyan]Max Steps:[/bold cyan]   {config.get('steps_per_episode', 500)}")
        auto_close = config.get('auto_close', '')
        if auto_close:
            _ac_style = "bright_green" if "CTF" in auto_close else "yellow"
            config_lines.append(f"[bold cyan]Auto-Close:[/bold cyan]  [{_ac_style}]{auto_close}[/{_ac_style}]")
        if config.get('mentor_budget'):
            config_lines.append(f"[bold cyan]Mentor:[/bold cyan]      {config.get('mentor_budget')}% budget")
        # Phase 23: GPT model info
        if config.get('gpt_primary'):
            config_lines.append(f"[bold cyan]GPT Model:[/bold cyan]   [bright_green]{config['gpt_primary']}[/bright_green]")
        if config.get('gpt_nano') and config.get('gpt_nano') != config.get('gpt_primary'):
            config_lines.append(f"[bold cyan]GPT Nano:[/bold cyan]    [dim]{config['gpt_nano']}[/dim]")
        if config.get('gpt_postmortem') and config.get('gpt_postmortem') != config.get('gpt_primary'):
            config_lines.append(f"[bold cyan]GPT Post:[/bold cyan]    [dim]{config['gpt_postmortem']}[/dim]")
        if config.get('gpt_token_limit'):
            config_lines.append(f"[bold cyan]Token Lim:[/bold cyan]   {config['gpt_token_limit']:,}/episode")
        config_panel = Panel(
            "\n".join(config_lines),
            title="[bold bright_cyan]⚙️  Configuration[/bold bright_cyan]",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(1, 2),
        )

        # Right: Active Agents
        agent_lines = []
        for agent_name in agents:
            icon, role, style = AGENT_ICONS.get(agent_name, ("🤖", "Agent", "dim"))
            agent_lines.append(f"[{style}]{icon} {agent_name:<14}[/{style}] │ {role}")
        agent_panel = Panel(
            "\n".join(agent_lines) if agent_lines else "[dim]No agents[/dim]",
            title=f"[bold bright_yellow]🎯  Agents ({len(agents)})[/bold bright_yellow]",
            border_style="yellow",
            box=box.ROUNDED,
            padding=(1, 2),
        )

        console.print(Columns([config_panel, agent_panel], equal=True, expand=True))

        # ── ALGORITHM STACK ──────────────────────────────────────────
        algo_table = Table(
            title="[bold]🧬  Unified Algorithm Stack[/bold]",
            show_header=True, header_style="bold bright_white on dark_blue",
            border_style="bright_blue", box=box.ROUNDED,
            expand=True, padding=(0, 2),
        )
        algo_table.add_column("Algorithm", style="bold", width=20)
        algo_table.add_column("Status", width=10, justify="center")
        algo_table.add_column("Role", width=40)
        algo_table.add_column("Config", width=40)

        algo_configs = {
            "PPO": "clip=0.2, γ=0.99, λ=0.97, lr=3e-4→1e-5",
            "DDQN": "macro-actions, ε-greedy, target network",

            "SIL": "500-entry golden buffer, top-K replay",
            "RND": "intrinsic motivation, novelty bonus",
            "SAC": "entropy-regularized, dual Q-networks",
        }

        for algo_name, (icon, color, description) in ALGO_ICONS.items():
            active = algorithms.get(algo_name, False)
            status = f"[bold green]● ACTIVE[/bold green]" if active else "[dim]○ idle[/dim]"
            cfg = algo_configs.get(algo_name, "")
            algo_table.add_row(
                f"[{color}]{icon} {algo_name}[/{color}]",
                status,
                description,
                f"[dim]{cfg}[/dim]",
            )

        console.print(algo_table)

        # ── GPT MODEL ROUTING ────────────────────────────────────────
        gpt_lines = []
        _gpt_primary = config.get('gpt_primary', '?')
        _gpt_nano = config.get('gpt_nano', '?')
        _gpt_post = config.get('gpt_postmortem', '?')
        _tok_lim = config.get('gpt_token_limit', 0)
        gpt_lines.append(f"[bright_green]🧠 Primary:[/bright_green]    {_gpt_primary}")
        if _gpt_nano != _gpt_primary:
            gpt_lines.append(f"[dim]⚡ Nano:[/dim]       {_gpt_nano}")
        if _gpt_post != _gpt_primary:
            gpt_lines.append(f"[dim]📊 Postmortem:[/dim] {_gpt_post}")
        if _tok_lim:
            gpt_lines.append(f"[cyan]📏 Budget:[/cyan]     {_tok_lim:,} tokens/episode")
        gpt_lines.append(f"[cyan]💳 Tracking:[/cyan]   Per-call prompt/response visibility")
        if gpt_lines:
            console.print(Panel(
                "\n".join(gpt_lines),
                title="[bold]🤖  GPT Intelligence Layer[/bold]",
                border_style="bright_green",
                padding=(0, 2),
            ))

        # ── DECISION PIPELINE ────────────────────────────────────────
        pipeline = (
            "[bold cyan]📖 PLAYBOOK[/bold cyan] [dim](60→10%)[/dim] "
            "[bright_white]━━▶[/bright_white] "
            "[bold green]🤖 PPO[/bold green] [dim](RL)[/dim] "
            "[bright_white]━━▶[/bright_white] "
            "[bold blue]📦 REGISTRY[/bold blue] [dim](144+)[/dim] "
            "[bright_white]━━▶[/bright_white] "
            "[bold yellow]📡 MENTOR[/bold yellow] [dim](GPT)[/dim] "
            "[bright_white]━━▶[/bright_white] "
            "[bold red]🔁 ANTI-REPEAT[/bold red]\n"
            "\n"
            "[dim]  Intelligence Layers:  "
            "[bright_magenta]🧭 PhaseGuide[/bright_magenta] · "
            "[bright_cyan]⛓️ MicroChain[/bright_cyan] · "
            "[bright_red]🛡️ EvidenceGate[/bright_red] · "
            "[bright_yellow]🎯 TacticalCortex[/bright_yellow][/dim]"
        )
        console.print(Panel(
            pipeline,
            title="[bold]🔄  4-Stage Decision Pipeline[/bold]",
            border_style="bright_white",
            padding=(0, 2),
        ))

        # ── KILL CHAIN PHASES ────────────────────────────────────────
        phase_parts = []
        for phase_name, icon in PHASE_ICONS.items():
            phase_parts.append(f"[dim]{icon} {phase_name[:5]}[/dim]")
        chain = " [bright_white]━▶[/bright_white] ".join(phase_parts)
        console.print(Panel(
            f"  {chain}",
            title="[bold bright_white]🔗  Kill Chain Progression[/bold bright_white]",
            border_style="dim",
            box=box.ROUNDED,
            padding=(0, 2),
        ))
        console.print()
        console.rule("[bold bright_green]▶ Training Started[/bold bright_green]", style="bright_green")
        console.print()

    # =========================================================================
    # ALGORITHM METRICS PANELS — Phase 10.2
    # =========================================================================
    def _build_ppo_panel(
        self,
        ppo_metrics: Optional[Dict[str, Any]] = None,
        per_coach_ppo: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Panel:
        """Build a Rich panel showing PPO training metrics with sparklines.

        Args:
            ppo_metrics: Aggregate PPO metrics (updates, avg losses, entropy)
            per_coach_ppo: Per-coach PPO breakdown {coach_name: {policy_loss, value_loss, entropy}}

        Returns:
            Rich Panel with PPO visualization
        """
        lines = []

        if ppo_metrics:
            updates = ppo_metrics.get("updates", 0)
            pi_loss = ppo_metrics.get("avg_policy_loss", 0.0)
            v_loss = ppo_metrics.get("avg_value_loss", 0.0)
            entropy = ppo_metrics.get("avg_entropy", 0.0)

            lines.append(f"[bold]Updates:[/bold]      {updates}")
            pi_color = "green" if pi_loss < 0.1 else "yellow" if pi_loss < 0.5 else "red"
            lines.append(f"[bold]π Loss:[/bold]       [{pi_color}]{pi_loss:.6f}[/{pi_color}]")
            v_color = "green" if v_loss < 1.0 else "yellow" if v_loss < 5.0 else "red"
            lines.append(f"[bold]V Loss:[/bold]       [{v_color}]{v_loss:.6f}[/{v_color}]")
            ent_color = "green" if entropy > 0.5 else "yellow" if entropy > 0.1 else "red"
            lines.append(f"[bold]Entropy:[/bold]      [{ent_color}]{entropy:.6f}[/{ent_color}]")
        else:
            lines.append("[dim]No PPO updates this episode[/dim]")

        # Per-coach breakdown with sparkline trends
        if per_coach_ppo:
            lines.append("")
            lines.append("[bold underline]Per-Coach Breakdown:[/bold underline]")
            for coach_name, metrics in sorted(per_coach_ppo.items()):
                icon = AGENT_ICONS.get(coach_name, ("🤖", "", "dim"))[0]
                pi = metrics.get("policy_loss", 0.0)
                vl = metrics.get("value_loss", 0.0)
                ent = metrics.get("entropy", 0.0)

                # Track history for sparklines
                if coach_name not in self.ppo_loss_history:
                    self.ppo_loss_history[coach_name] = deque(maxlen=20)
                    self.ppo_vloss_history[coach_name] = deque(maxlen=20)
                    self.ppo_entropy_history[coach_name] = deque(maxlen=20)
                self.ppo_loss_history[coach_name].append(pi)
                self.ppo_vloss_history[coach_name].append(vl)
                self.ppo_entropy_history[coach_name].append(ent)

                pi_spark = sparkline(list(self.ppo_loss_history[coach_name]))
                short = coach_name.replace("Agent", "")
                lines.append(
                    f"  {icon} [bold]{short:<8}[/bold] "
                    f"π:{pi:.5f} V:{vl:.4f} H:{ent:.4f}  {pi_spark}"
                )

        return Panel(
            "\n".join(lines),
            title="[bold green]🤖  PPO Actor-Critic[/bold green]",
            border_style="green",
            box=box.ROUNDED,
            padding=(0, 2),
        )

    def _build_ddqn_panel(
        self,
        ddqn_metrics: Optional[Dict[str, Any]] = None,
    ) -> Panel:
        """Build a Rich panel showing DDQN macro-action metrics.

        Args:
            ddqn_metrics: DDQN stats (macros, switches, epsilon, distribution)

        Returns:
            Rich Panel with DDQN visualization
        """
        lines = []

        if ddqn_metrics and ddqn_metrics.get("macros", 0) > 0:
            macros = ddqn_metrics.get("macros", 0)
            switches = ddqn_metrics.get("switches", 0)
            epsilon = ddqn_metrics.get("epsilon", 0.0)
            dist = ddqn_metrics.get("distribution", {})

            eps_color = "green" if epsilon < 0.3 else "yellow" if epsilon < 0.7 else "red"
            lines.append(f"[bold]Macros:[/bold]     {macros}")
            lines.append(f"[bold]Switches:[/bold]   {switches}")
            lines.append(f"[bold]Epsilon:[/bold]    [{eps_color}]{epsilon:.3f}[/{eps_color}]")

            # Distribution bar chart
            if dist:
                lines.append("")
                lines.append("[bold underline]Macro Distribution:[/bold underline]")
                total = sum(dist.values()) or 1
                max_count = max(dist.values()) if dist else 1
                for macro_name, count in sorted(dist.items(), key=lambda x: -x[1]):
                    pct = count / total * 100
                    bar_len = max(int(count / max_count * 20), 1)
                    lines.append(
                        f"  {macro_name[:14]:<14} [cyan]{'█' * bar_len}[/cyan] {count} ({pct:.0f}%)"
                    )

            self.ddqn_history.append(dist)
        else:
            lines.append("[dim]No DDQN macro-actions this episode[/dim]")

        return Panel(
            "\n".join(lines),
            title="[bold cyan]🎲  DDQN Macro-Actions[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 2),
        )

    def _build_llm_bridge_panel(
        self,
        bridge_data: Dict[str, Any],
    ) -> Panel:
        """Build Phase 37 Level 5 GPT↔RL Integration panel.

        Shows real-time LLM influence metrics including:
        - Teacher anneal progress (% detachment from GPT)
        - Prior alpha (logit injection weight)
        - Auxiliary losses (KL teacher, ranking, value reg)
        - Maturity signal components

        Args:
            bridge_data: Snapshot from LLMPolicyBridge.get_influence_snapshot()

        Returns:
            Rich Panel with GPT↔RL integration visualization
        """
        lines = []

        # Anneal progress bar
        anneal_pct = bridge_data.get("teacher_anneal_pct", 1.0)
        alpha = bridge_data.get("prior_alpha", 0.5)
        maturity = bridge_data.get("maturity_signal", 0.0)
        enabled = bridge_data.get("enabled", True)

        status = "[bold green]ACTIVE[/bold green]" if enabled else "[bold red]DISABLED[/bold red]"
        lines.append(f"[bold]Status:[/bold]     {status}")

        # Anneal / Detachment bar
        detach_pct = 1.0 - anneal_pct
        detach_bar_len = 20
        filled = int(detach_pct * detach_bar_len)
        bar = "█" * filled + "░" * (detach_bar_len - filled)
        detach_color = "green" if detach_pct > 0.7 else "yellow" if detach_pct > 0.3 else "red"
        lines.append(
            f"[bold]Detach:[/bold]     [{detach_color}]{bar}[/{detach_color}] "
            f"{detach_pct*100:.0f}% autonomous"
        )
        lines.append(f"[bold]α Prior:[/bold]    {alpha:.4f}  [dim](logit injection weight)[/dim]")

        # Maturity bar
        mat_bar_len = 20
        mat_filled = int(maturity * mat_bar_len)
        mat_bar = "█" * mat_filled + "░" * (mat_bar_len - mat_filled)
        mat_color = "green" if maturity > 0.7 else "yellow" if maturity > 0.3 else "cyan"
        lines.append(
            f"[bold]Maturity:[/bold]   [{mat_color}]{mat_bar}[/{mat_color}] {maturity:.3f}"
        )

        # Component signals
        sr = bridge_data.get("success_rate", 0.0)
        rv = bridge_data.get("reward_velocity", 0.0)
        de = bridge_data.get("discovery_efficiency", 0.0)
        esr = bridge_data.get("exploit_success_rate", 0.0)
        lines.append(
            f"  [dim]SR={sr:.2f} RV={rv:.2f} DE={de:.2f} ESR={esr:.2f}[/dim]"
        )

        # Auxiliary losses
        bc = bridge_data.get("bc_loss", 0.0)
        kl = bridge_data.get("kl_teacher_loss", 0.0)
        rl = bridge_data.get("ranking_loss", 0.0)
        vr = bridge_data.get("value_reg_loss", 0.0)

        lines.append("")
        lines.append("[bold underline]Auxiliary Losses:[/bold underline]")
        kl_c = bridge_data.get("kl_teacher_coef", 0.15)
        rl_c = bridge_data.get("ranking_loss_coef", 0.05)
        vr_c = bridge_data.get("value_reg_coef", 0.10)
        lines.append(f"  KL Teacher:  {kl:.6f}  [dim](coef={kl_c:.3f})[/dim]")
        lines.append(f"  Ranking:     {rl:.6f}  [dim](coef={rl_c:.3f})[/dim]")
        lines.append(f"  Value Reg:   {vr:.6f}  [dim](coef={vr_c:.3f})[/dim]")
        lines.append(f"  BC Loss:     {bc:.6f}")

        steps = bridge_data.get("total_steps", 0)
        lines.append(f"\n[dim]Steps: {steps}[/dim]")

        return Panel(
            "\n".join(lines),
            title="[bold magenta]🧬  GPT↔RL Integration (Level 5)[/bold magenta]",
            border_style="magenta",
            box=box.ROUNDED,
            padding=(0, 2),
        )

    def _build_decision_pipeline_panel(
        self,
        source_counts: Optional[Dict[str, int]] = None,
    ) -> Panel:
        """Build a visualization of the 4-stage decision pipeline distribution.

        Args:
            source_counts: Decision source → count mapping

        Returns:
            Rich Panel with pipeline visualization
        """
        lines = []

        if source_counts:
            total = sum(source_counts.values()) or 1
            # Store for trend tracking
            self.decision_source_history.append(dict(source_counts))

            # Pipeline stages in order
            stage_order = ["playbook", "ppo", "registry", "mentor", "dual_mentor",
                           "anti_repeat", "codex_meta", "sil", "fallback"]
            max_count = max(source_counts.values()) if source_counts else 1

            for src in stage_order:
                count = source_counts.get(src, 0)
                if count == 0:
                    continue
                pct = count / total * 100
                bar_len = max(int(count / max_count * 25), 1)
                src_icon, src_style = SOURCE_STYLES.get(src, ("❓", "dim"))
                lines.append(
                    f"  {src_icon} [{src_style}]{src:<12}[/{src_style}] "
                    f"[{src_style}]{'█' * bar_len}[/{src_style}] {count:3d} ({pct:4.1f}%)"
                )

            # Other sources not in the standard list
            for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
                if src not in stage_order and count > 0:
                    pct = count / total * 100
                    bar_len = max(int(count / max_count * 25), 1)
                    lines.append(
                        f"  ❓ [dim]{src:<12}[/dim] [dim]{'█' * bar_len}[/dim] {count:3d} ({pct:4.1f}%)"
                    )

            # Trend comparison with previous episode
            if len(self.decision_source_history) >= 2:
                prev = self.decision_source_history[-2]
                lines.append("")
                shifts = []
                for src in stage_order:
                    curr = source_counts.get(src, 0)
                    prev_c = prev.get(src, 0)
                    if curr != prev_c and (curr > 0 or prev_c > 0):
                        delta = curr - prev_c
                        arrow = "↑" if delta > 0 else "↓"
                        color = "green" if (src == "ppo" and delta > 0) or (src == "anti_repeat" and delta < 0) else "yellow"
                        shifts.append(f"[{color}]{src}:{arrow}{abs(delta)}[/{color}]")
                if shifts:
                    lines.append(f"  [dim]Δ[/dim] {' '.join(shifts[:5])}")
        else:
            lines.append("[dim]No decision data[/dim]")

        return Panel(
            "\n".join(lines),
            title="[bold bright_white]🔄  Decision Pipeline[/bold bright_white]",
            border_style="bright_white",
            box=box.ROUNDED,
            padding=(0, 2),
        )

    def _build_discovery_board_panel(
        self,
        discovery_board: Optional[Dict[str, Any]] = None,
    ) -> Panel:
        """Build a visual discovery board showing all findings.

        Args:
            discovery_board: Current discovery board state

        Returns:
            Rich Panel with discovery heatmap
        """
        lines = []

        if discovery_board:
            # Snapshot for trending (skip internal keys and unhashable types)
            self.discovery_board_snapshot = {}
            for k, v in discovery_board.items():
                if k.startswith("_") or k == "phase":
                    continue
                if isinstance(v, (set, list)):
                    # Only convert to set if all items are hashable
                    try:
                        self.discovery_board_snapshot[k] = set(v)
                    except TypeError:
                        self.discovery_board_snapshot[k] = list(v)
                else:
                    self.discovery_board_snapshot[k] = v

            DISC_ICONS = {
                "ports": ("🔓", "cyan"),
                "services": ("⚙️", "blue"),
                "credentials": ("🔑", "yellow"),
                "vulns": ("💀", "red"),
                "shells": ("💥", "bright_red"),
                "users": ("👤", "magenta"),
                "web_paths": ("🌐", "bright_cyan"),
                "flags_set": ("🏁", "bright_green"),
            }

            for key, (icon, color) in DISC_ICONS.items():
                items = discovery_board.get(key, set())
                if isinstance(items, (set, list)):
                    count = len(items)
                    if count > 0:
                        # P34-EXT: Truncate large sets to top 20 + count
                        _max_display = 20
                        try:
                            item_strs = [str(i) for i in sorted(items) if str(i)]
                        except TypeError:
                            item_strs = [str(i) for i in items if str(i)]
                        if len(item_strs) > _max_display:
                            _remaining = len(item_strs) - _max_display
                            item_strs = item_strs[:_max_display]
                            item_strs.append(f"... +{_remaining} more")
                        lines.append(
                            f"  {icon} [{color}]{key:<12}[/{color}] "
                            f"[bold]{count:3d}[/bold]  "
                            f"[dim]{', '.join(item_strs)}[/dim]"
                        )
                    else:
                        lines.append(
                            f"  {icon} [dim]{key:<12} {'—':>3}[/dim]"
                        )

            # Current phase
            phase = discovery_board.get("phase", "RECON")
            phase_icon = PHASE_ICONS.get(str(phase).upper(), "❓")
            lines.append(f"\n  [bold]Phase:[/bold] {phase_icon} {phase}")
        else:
            lines.append("[dim]No discoveries yet[/dim]")

        return Panel(
            "\n".join(lines),
            title="[bold bright_green]🗺️  Discovery Board[/bold bright_green]",
            border_style="bright_green",
            box=box.ROUNDED,
            padding=(0, 2),
        )

    def _build_algo_activity_bar(
        self,
        source: str,
        ppo_conf: float = 0.0,
        ddqn_macro: str = "",
    ) -> str:
        """Build a compact algorithm activity indicator for per-step display.

        Args:
            source: Decision source string
            ppo_conf: PPO confidence/value estimate
            ddqn_macro: Current DDQN macro name

        Returns:
            Formatted string showing algorithm activity
        """
        parts = []
        src_icon, src_style = SOURCE_STYLES.get(source, SOURCE_STYLES.get(source[:8], ("❓", "dim")))

        if source == "ppo":
            parts.append(f"[green]🤖 PPO[/green]")
            if ppo_conf > 0:
                parts.append(f"[dim]v={ppo_conf:.2f}[/dim]")
        elif source == "playbook":
            parts.append(f"[cyan]📖 PLAY[/cyan]")
        elif source in ("mentor", "dual_mentor"):
            parts.append(f"[yellow]📡 GPT[/yellow]")
        elif source == "registry":
            parts.append(f"[blue]📦 REG[/blue]")
        elif source == "anti_repeat":
            parts.append(f"[red]🔁 A/R[/red]")
        elif source == "codex_meta":
            parts.append(f"[bright_magenta]🧬 CDX[/bright_magenta]")
        else:
            parts.append(f"[{src_style}]{src_icon}[/{src_style}]")

        if ddqn_macro:
            parts.append(f"[cyan]🎲{ddqn_macro[:6]}[/cyan]")

        return " ".join(parts)

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
        parser_stats: Optional[Dict[str, Any]] = None,
        reasoning_events: Optional[List[Dict[str, str]]] = None,
        # Phase 11.0: New parameters
        teaching_points: Optional[List[str]] = None,
        budget_snapshot: Optional[Dict[str, Any]] = None,
        parse_explanations: Optional[List[Dict[str, Any]]] = None,
        phase_state: Optional[Dict[str, Any]] = None,
        # Phase 23: GPT call visibility
        gpt_activity: Optional[Dict[str, Any]] = None,
        # P34-EXT: Learning metrics snapshot
        learning_snapshot: Optional[Dict[str, Any]] = None,
        # Phase 37: GPT↔RL bridge metrics
        llm_bridge_snapshot: Optional[Dict[str, Any]] = None,
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

        # Phase 23: GPT cost in header
        gpt_cost_str = ""
        if gpt_activity:
            _cum = gpt_activity.get("cumulative_cost_usd", 0.0)
            if _cum > 0:
                _cost_clr = "red" if _cum > 1.0 else "yellow" if _cum > 0.25 else "green"
                gpt_cost_str = f" │ [{_cost_clr}]💳${_cum:.3f}[/{_cost_clr}]"

        # ── STEP HEADER ──────────────────────────────────────────────
        # Phase 37: Clean gradient-style header with Rich Rule
        console.print()  # breathing room
        console.rule(
            f"[bold bright_white]▶ Step {step + 1:2d}[/bold bright_white]  │  "
            f"[bright_cyan]{ep_str}[/bright_cyan]  │  "
            f"[bold bright_yellow]{mode_tag}[/bold bright_yellow]  │  "
            f"{phase_icon} [bold bright_white]{phase_upper}[/bold bright_white]  │  "
            f"[bold {reward_color}]R:{global_reward:+.1f}[/bold {reward_color}]  │  "
            f"[dim]{mentor_str}[/dim]  [dim]{step_spark}[/dim]{gpt_cost_str}{done_tag}",
            style="bright_blue",
        )

        # ── AGENT TABLE ──────────────────────────────────────────────
        # Phase 36: ROUNDED box for beautiful demo appearance
        table = Table(
            box=box.ROUNDED, show_header=True,
            header_style="bold bright_white on dark_blue", padding=(0, 2),
            expand=True, show_lines=True,
            border_style="bright_blue",
            title="[bold bright_white]⚡ Agent Actions[/bold bright_white]",
            title_style="bold bright_blue",
        )
        table.add_column("Agent", style="bold", ratio=1, no_wrap=True)
        table.add_column("Source", ratio=1, no_wrap=True)
        table.add_column("Command", ratio=3, no_wrap=False, overflow="fold")
        table.add_column("Reward", ratio=1, justify="right", no_wrap=True)

        # Active agents — each gets a row in the summary table,
        # then a full verbose output+discovery panel below
        active = [a for a in agent_infos if not a.skipped]
        _verbose_panels = []  # Collect panels for after the table

        for a in active:
            icon, _role, style = AGENT_ICONS.get(
                a.agent_name, ("🤖", "Agent", "dim")
            )
            # Better source key matching
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
            source_label = f"{src_icon}{src_raw}"
            if a.mentor_call:
                source_label += " 📡"

            cmd = a.command or "(none)"

            # Reward display
            if a.reward > 0:
                r_str = f"[bold green]{a.reward:+.1f}[/bold green]"
            elif a.reward < 0:
                r_str = f"[bold red]{a.reward:+.1f}[/bold red]"
            else:
                r_str = "[dim]+0.0[/dim]"

            table.add_row(
                f"[{style}]{agent_label}[/{style}]",
                f"[{src_style}]{source_label}[/{src_style}]",
                f"[white]{cmd}[/white]",
                r_str,
            )

            # ── Build verbose output + discoveries panel per agent ──
            panel_parts = []

            # Full command output — multi-line, no truncation
            if a.command_output and a.command_output.strip():
                _out_lines = a.command_output.strip().split("\n")
                # Show ALL lines (investor demo = full verbosity)
                _out_text = "\n".join(_out_lines)
                panel_parts.append(f"[bold white]📋 Output:[/bold white]")
                panel_parts.append(f"[dim]{_out_text}[/dim]")
            else:
                panel_parts.append("[dim]📋 Output: (no output)[/dim]")

            # Mentor reasoning if present
            if a.mentor_reasoning:
                panel_parts.append(f"")
                panel_parts.append(f"[bright_yellow]💬 Mentor:[/bright_yellow] {a.mentor_reasoning}")

            # Discoveries — HIGHLIGHTED and verbose, ALL items shown
            if a.discoveries:
                panel_parts.append(f"")
                _disc_icon_map = {
                    "open_port": "🔓", "service": "⚙️", "credential": "🔑",
                    "version": "📌", "vulnerability": "💀", "shell": "💀",
                    "user": "👤", "web_path": "🌐", "hostname": "🏷️",
                    "os": "💻", "technology": "🔧", "flag": "🚩",
                }
                for dtype, items in a.discoveries.items():
                    _di = _disc_icon_map.get(dtype, "🟢")
                    if items and isinstance(items, (list, set, tuple)):
                        _item_list = list(items)
                        for _item in _item_list:
                            panel_parts.append(
                                f"  [bold bright_green]{_di} {dtype.upper()}:[/bold bright_green] "
                                f"[bold white on dark_green] {_item} [/bold white on dark_green]"
                            )
                    elif items and isinstance(items, str):
                        panel_parts.append(
                            f"  [bold bright_green]{_di} {dtype.upper()}:[/bold bright_green] "
                            f"[bold white on dark_green] {items} [/bold white on dark_green]"
                        )

            # Confidence — visual bar indicator
            _conf = a.confidence
            _conf_pct = int(_conf * 20)  # 0-20 blocks
            _conf_bar_full = "━" * _conf_pct
            _conf_bar_empty = "╌" * (20 - _conf_pct)
            if _conf >= 0.7:
                _conf_style = "bold green"
                _conf_bar_color = "green"
            elif _conf >= 0.4:
                _conf_style = "yellow"
                _conf_bar_color = "yellow"
            else:
                _conf_style = "red"
                _conf_bar_color = "red"
            panel_parts.append(f"")
            panel_parts.append(
                f"[dim]Confidence:[/dim] [{_conf_style}]{_conf:.0%}[/{_conf_style}] "
                f"[{_conf_bar_color}]{_conf_bar_full}[/{_conf_bar_color}]"
                f"[dim]{_conf_bar_empty}[/dim] │ "
                f"[dim]Tokens:[/dim] {a.tokens_used}"
            )

            # Evidence gate status (if available)
            _ev_gate = getattr(a, 'evidence_gate_result', None)
            if _ev_gate and _ev_gate != "pass":
                _gate_color = "bold red" if "enforce" in str(_ev_gate) else "yellow"
                panel_parts.append(
                    f"  [{_gate_color}]🛡️ Evidence Gate: {_ev_gate}[/{_gate_color}]"
                )

            _panel_border = "bright_green" if a.discoveries else ("cyan" if a.reward > 0 else "dim")
            _verbose_panels.append(Panel(
                "\n".join(panel_parts),
                title=f"[bold]{icon} {a.agent_name}[/bold]",
                border_style=_panel_border,
                box=box.ROUNDED,
                expand=True,
                padding=(0, 2),
            ))

        # Skipped agents
        for name, reason in skipped_agents.items():
            icon = AGENT_ICONS.get(name, ("💤", "", "dim"))[0]
            table.add_row(
                f"[dim]{icon} {name.replace('Agent', '')}[/dim]",
                "[dim]💤 skip[/dim]",
                f"[dim]{reason}[/dim]",
                "",
            )

        console.print(table)

        # ── Per-agent verbose panels (output + discoveries) ──────────
        for _vp in _verbose_panels:
            console.print(_vp)

        # ── DISCOVERY BOARD — Full highlighted panel ─────────────────
        if discovery_board and self.config.show_discoveries:
            db = discovery_board
            _db_parts = []
            _db_icon_map = {
                "ports": ("🔓", "bright_cyan", "PORTS"),
                "services": ("⚙️", "bright_green", "SERVICES"),
                "credentials": ("🔑", "bold bright_yellow", "CREDENTIALS"),
                "vulns": ("💀", "bold red", "VULNS"),
                "shells": ("💀", "bold bright_red", "SHELLS"),
                "users": ("👤", "bright_magenta", "USERS"),
                "web_paths": ("🌐", "bright_blue", "WEB PATHS"),
            }
            _total_discoveries = 0
            for key, (icon_s, style_s, label) in _db_icon_map.items():
                items = db.get(key, set())
                if items:
                    count = len(items) if isinstance(items, (set, list)) else 1
                    _total_discoveries += count
                    if isinstance(items, (set, list)):
                        _items_str = ", ".join(str(i) for i in sorted(items, key=str))
                    else:
                        _items_str = str(items)
                    _db_parts.append(
                        f"  [{style_s}]{icon_s} {label} ({count}):[/{style_s}] {_items_str}"
                    )
            if _db_parts:
                _db_title = (
                    f"[bold bright_green]🗺️  Discovery Board[/bold bright_green]"
                    f"[dim]  ─  {_total_discoveries} total findings[/dim]"
                )
                console.print(Panel(
                    "\n".join(_db_parts),
                    title=_db_title,
                    border_style="bright_green",
                    box=box.ROUNDED,
                    expand=True,
                    padding=(0, 2),
                ))

        # ══════════════════════════════════════════════════════════════
        # METRICS PANEL — Reward, Parser, Budget, GPT, Phase in one box
        # ══════════════════════════════════════════════════════════════
        _metrics_lines: List[str] = []

        # ── Reward breakdown row ──
        if reward_breakdown and self.config.show_reward_breakdown:
            rb = reward_breakdown
            rp = []
            if rb.get("base", 0):
                rp.append(f"base:[bold]{rb['base']:+.1f}[/bold]")
            if rb.get("novelty_bonus", 0):
                rp.append(f"[bold green]+novelty:{rb['novelty_bonus']:+.1f}[/bold green]")
            if rb.get("redundancy_penalty", 0):
                rp.append(f"[bold red]repeat:{rb['redundancy_penalty']:+.1f}[/bold red]")
            if rb.get("phase_bonus", 0):
                rp.append(f"[bold cyan]phase:{rb['phase_bonus']:+.1f}[/bold cyan]")
            if rp:
                _metrics_lines.append(f"💰 [bold]Reward[/bold]    {' │ '.join(rp)}")

        # ── Parser row ──
        if parser_stats:
            total = parser_stats.get("total_calls", 0)
            if total > 0:
                parts = [f"calls:{total}"]
                gpt_hits = parser_stats.get("stage3_hits", 0) + parser_stats.get("gpt_fallback_successes", 0)
                regex_hits = parser_stats.get("stage1_hits", 0)
                lessons = parser_stats.get("lessons_produced", 0)
                patterns = parser_stats.get("patterns_learned", 0)
                if gpt_hits:
                    parts.append(f"[bright_green]🧠gpt:{gpt_hits}[/bright_green]")
                if regex_hits:
                    parts.append(f"🔍regex:{regex_hits}")
                if lessons:
                    parts.append(f"[cyan]📚lessons:{lessons}[/cyan]")
                if patterns:
                    parts.append(f"[magenta]🧬patterns:{patterns}[/magenta]")
                empty = parser_stats.get("empty_outputs", 0)
                if empty:
                    parts.append(f"empty:{empty}")
                _metrics_lines.append(f"🔬 [bold]Parser[/bold]    {' │ '.join(parts)}")

        # ── Budget row ──
        if budget_snapshot:
            pressure = budget_snapshot.get("budget_pressure", 0)
            mentor_rem = budget_snapshot.get("mentor_budget_remaining", budget_snapshot.get("mentor_remaining", 0))
            mentor_tot = budget_snapshot.get("mentor_budget_total", budget_snapshot.get("mentor_total", 0))
            venice_rem = budget_snapshot.get("venice_budget_remaining", budget_snapshot.get("venice_remaining", 0))
            if pressure > 0.8:
                p_style = "bold red"
                p_icon = "🔴"
            elif pressure > 0.5:
                p_style = "yellow"
                p_icon = "🟡"
            else:
                p_style = "green"
                p_icon = "🟢"
            _metrics_lines.append(
                f"[{p_style}]{p_icon} Budget[/{p_style}]    "
                f"pressure:[bold]{pressure:.0%}[/bold] │ "
                f"mentor:{mentor_rem}/{mentor_tot} │ "
                f"venice:{venice_rem}"
            )

        # ── GPT cost row ──
        if gpt_activity:
            cum_cost = gpt_activity.get("cumulative_cost_usd", 0.0)
            ep_cost = gpt_activity.get("episode_cost_usd", 0.0)
            total_req = gpt_activity.get("total_requests", 0)
            total_tok = gpt_activity.get("total_tokens", 0)
            cache_hits = gpt_activity.get("cache_hits", 0)
            if cum_cost > 1.0:
                cost_style = "bold red"
            elif cum_cost > 0.25:
                cost_style = "yellow"
            else:
                cost_style = "green"
            _metrics_lines.append(
                f"💳 [bold]GPT[/bold]       "
                f"[{cost_style}]${cum_cost:.4f}[/{cost_style}] │ "
                f"ep:${ep_cost:.4f} │ "
                f"calls:{total_req} │ tokens:{total_tok:,} │ "
                f"cache:{cache_hits}"
            )

        # ── P36.1: Fast-Learn metrics row ──
        if learning_snapshot:
            _fl_line = learning_snapshot.get("fast_learn_line", "")
            if _fl_line:
                _metrics_lines.append(f"🧪 [bold]FastLearn[/bold] {_fl_line}")

        # ── Phase ladder row ──
        if phase_state:
            tp = phase_state.get("teaching_point", "")
            current = phase_state.get("current_phase", "?")
            if tp:
                if "SMART LADDER" in tp:
                    _metrics_lines.append(f"[bright_red]🪜 Phase[/bright_red]     [yellow]{tp}[/yellow]")
                elif "not yet ready" in tp:
                    _metrics_lines.append(f"[yellow]🪜 Phase {current}[/yellow]  {tp}")
                else:
                    _metrics_lines.append(f"[green]🪜 Phase {current}[/green]  {tp}")

        # ── Teaching points rows ──
        if teaching_points:
            for tp in teaching_points:
                _metrics_lines.append(f"[bright_yellow]📚 Teaching[/bright_yellow]  {tp}")

        # ── Render the unified metrics panel ──
        if _metrics_lines:
            console.print(Panel(
                "\n".join(_metrics_lines),
                title="[bold bright_white]📊 Step Metrics[/bold bright_white]",
                subtitle="[dim]phase · budget · cost · learning[/dim]",
                border_style="bright_blue",
                box=box.ROUNDED,
                expand=True,
                padding=(0, 2),
            ))

        # ── DECISION REASONING VISIBILITY (P36: ALWAYS VISIBLE) ──────
        if reasoning_events:
            # Separate mentor Q/A events from other events
            _mentor_events = [ev for ev in reasoning_events if ev.get("type") == "mentor_reason"]
            _other_events = [ev for ev in reasoning_events if ev.get("type") != "mentor_reason"]
            
            _reasoning_lines = []
            for ev in _other_events[:6]:
                ev_type = ev.get("type", "")
                agent = ev.get("agent", "")
                msg = ev.get("message", "")
                # P37: Skip empty/malformed reasoning entries
                if not msg and not ev_type:
                    continue
                if ev_type == "tc_block":
                    _reasoning_lines.append(f"[yellow]🛡️ [{agent}] TC Block:[/yellow] {msg}")
                elif ev_type == "phase_gate":
                    _reasoning_lines.append(f"[cyan]🚪 [{agent}] Phase Gate:[/cyan] {msg}")
                elif ev_type == "codex_meta":
                    _reasoning_lines.append(f"[bright_magenta]🧬 [{agent}] Codex:[/bright_magenta] {msg}")
                elif ev_type == "decision":
                    # P36: Structured reasoning — parse EVIDENCE|GOAL|WHY_THIS|STOP|CONF
                    _parts = {}
                    for _seg in msg.split(" | "):
                        if ":" in _seg:
                            _k, _v = _seg.split(":", 1)
                            _parts[_k.strip()] = _v.strip()
                    _ev_str = _parts.get("EVIDENCE", "?")
                    _goal_str = _parts.get("GOAL", "?")
                    _why_str = _parts.get("WHY_THIS", "?")
                    _stop_str = _parts.get("STOP", "")
                    _conf_str = _parts.get("CONF", "?")
                    _line = (
                        f"[bold cyan]🎯 [{agent}][/bold cyan] "
                        f"[green]EV:[/green]{_ev_str[:80]} "
                        f"[yellow]GOAL:[/yellow]{_goal_str[:50]} "
                        f"[bright_magenta]WHY:[/bright_magenta]{_why_str[:80]}"
                    )
                    if _stop_str:
                        _line += f" [dim]STOP:{_stop_str[:40]}[/dim]"
                    _line += f" [dim]CONF:{_conf_str}[/dim]"
                    _reasoning_lines.append(_line)
                elif ev_type == "phase_guided":
                    _reasoning_lines.append(
                        f"[bold bright_green]🧭 [{agent}] P34 Guide:[/bold bright_green] {msg}"
                    )
                else:
                    _reasoning_lines.append(f"[dim]📎 [{agent}] {ev_type}: {msg}[/dim]")
            
            if _reasoning_lines:
                console.print(Panel(
                    "\n".join(_reasoning_lines),
                    title="[bold bright_magenta]🧠 Decision Reasoning[/bold bright_magenta]",
                    border_style="bright_magenta",
                    box=box.ROUNDED,
                    expand=True,
                    padding=(0, 2),
                ))

            # Render LLM Q/A as Rich table (Phase 10.4)
            if _mentor_events:
                qa_table = Table(
                    box=box.ROUNDED,
                    show_header=True,
                    header_style="bold bright_yellow",
                    expand=True,
                    padding=(0, 2),
                    border_style="bright_yellow",
                )
                qa_table.add_column("Agent", style="bold cyan", min_width=12, no_wrap=True)
                qa_table.add_column("Q (Prompt)", style="white", no_wrap=False)
                qa_table.add_column("A (LLM Response)", style="bright_green", no_wrap=False)
                for ev in _mentor_events[:3]:
                    _agent = ev.get("agent", "")[:12]
                    _msg = ev.get("message", "")
                    # Parse Q=... → A=... format from mentor reasoning
                    if "→ A=" in _msg:
                        _q_part, _a_part = _msg.split("→ A=", 1)
                        _q = _q_part.replace("Q=", "").strip()
                        _a = _a_part.strip()
                    elif "Q=" in _msg:
                        _q = _msg.replace("Q=", "").strip()
                        _a = "-"
                    else:
                        _q = _msg
                        _a = "-"
                    qa_table.add_row(_agent, _q, _a)
                console.print(Panel(
                    qa_table,
                    title="[bold bright_yellow]💬 LLM Communication[/bold bright_yellow]",
                    border_style="bright_yellow",
                    box=box.ROUNDED,
                    expand=True,
                    padding=(0, 1),
                ))

        # ── GPT PER-STEP CALL DETAILS ────────────────────────────────
        if gpt_activity:
            step_calls = gpt_activity.get("step_calls", [])
            step_api = gpt_activity.get("step_api_calls", 0)
            step_cache = gpt_activity.get("step_cache_hits", 0)
            step_tok = gpt_activity.get("step_tokens", 0)
            step_cost = gpt_activity.get("step_cost_usd", 0.0)

            if step_calls:
                _gpt_lines = [
                    f"[dim]{step_api} API + {step_cache} cached │ "
                    f"tokens:{step_tok:,} │ ${step_cost:.5f}[/dim]"
                ]
                for call in step_calls[:4]:
                    _model = call.get("model", "?").replace("gpt-", "").replace("-codex", "c")
                    _agent = call.get("agent_id", "?")[:8]
                    _task = call.get("task_type", "?")[:8]
                    _tok = call.get("tokens", 0)
                    _in_tok = call.get("input_tokens", 0)
                    _out_tok = call.get("output_tokens", 0)
                    _cost = call.get("cost_usd", 0.0)
                    _lat = call.get("latency_ms", 0)
                    _cached = call.get("cache_hit", False)
                    _prompt = call.get("prompt_snippet", "")
                    _resp = call.get("response_snippet", "")

                    if _cached:
                        _gpt_lines.append(
                            f"[dim]📦 CACHE │ {_agent}/{_task} │ Q: {_prompt}[/dim]"
                        )
                        _gpt_lines.append(f"[dim]  └─ A: {_resp}[/dim]")
                    else:
                        tok_detail = f"{_in_tok}→{_out_tok}" if _in_tok or _out_tok else str(_tok)
                        _gpt_lines.append(
                            f"[bright_green]🧠 {_model}[/bright_green] "
                            f"[cyan]{_agent}/{_task}[/cyan] "
                            f"[dim]{tok_detail}tok ${_cost:.4f} {_lat}ms[/dim]"
                        )
                        _gpt_lines.append(f"[dim]  Q: {_prompt}[/dim]")
                        _gpt_lines.append(f"[bright_green]  A: {_resp}[/bright_green]")

                if len(step_calls) > 4:
                    _gpt_lines.append(f"[dim]  ... +{len(step_calls) - 4} more calls[/dim]")
                console.print(Panel(
                    "\n".join(_gpt_lines),
                    title="[bold bright_cyan]🔌 GPT Calls This Step[/bold bright_cyan]",
                    border_style="bright_cyan",
                    box=box.ROUNDED,
                    expand=True,
                    padding=(0, 2),
                ))

        # ── GPT INTERPRETATION REASONING ─────────────────────────────
        if parse_explanations:
            _pe_lines = []
            for pe in parse_explanations[:3]:
                stage = pe.get("stage", "?")
                dtype = pe.get("discovery_type", "?")
                dval = pe.get("discovery_value", "?")
                reason = pe.get("reasoning", "")
                chain = pe.get("interpretation_chain", "")
                if stage in ("gpt_codex", "gpt_fallback", "gpt"):
                    _pe_lines.append(
                        f"[bright_green]🧠 GPT:[/bright_green] "
                        f"[green]{dtype}={dval}[/green] {reason}"
                    )
                    if chain:
                        _pe_lines.append(f"  [dim]└─ {chain}[/dim]")
                else:
                    _pe_lines.append(
                        f"🔬 Parse ({stage}): "
                        f"[green]{dtype}={dval}[/green] {reason}"
                    )
            if _pe_lines:
                console.print(Panel(
                    "\n".join(_pe_lines),
                    title="[bold bright_cyan]🔬 Parser Interpretations[/bold bright_cyan]",
                    border_style="cyan",
                    box=box.ROUNDED,
                    expand=True,
                    padding=(0, 2),
                ))

        # ── P34-EXT: LEARNING METRICS PANEL ─────────────────────────
        if learning_snapshot:
            self._print_learning_panel(learning_snapshot, step)

        # ── EVENTS (last 3 seconds, max 2) ───────────────────────────
        now = time.time()
        recent = [e for e in self.events if now - e.timestamp < 3]
        if recent:
            self._print_events(recent[-2:])

        self.last_print_step = step

        # Ensure all Rich output is flushed to terminal immediately
        import sys
        sys.stdout.flush()

    # ─── P34-EXT: Learning Metrics Panel ────────────────────────────────────

    def _print_learning_panel(self, snapshot: Dict[str, Any], step: int) -> None:
        """
        Print a compact learning dashboard panel every N steps.

        Shows: discovery totals/deltas, novelty rate, stagnation, anti-repeat,
        milestones, model mix, evidence gate, cost efficiency.
        """
        # ── Row 1: Discovery summary ──
        tp = snapshot.get("total_ports", 0)
        ts = snapshot.get("total_services", 0)
        tc = snapshot.get("total_creds", 0)
        tsh = snapshot.get("total_shells", 0)
        tpaths = snapshot.get("total_paths", 0)

        np_ = snapshot.get("new_ports", 0)
        ns = snapshot.get("new_services", 0)
        nc = snapshot.get("new_creds", 0)
        nsh = snapshot.get("new_shells", 0)

        def _delta(n: int) -> str:
            return f"[green]+{n}[/green]" if n > 0 else "[dim]+0[/dim]"

        disc_line = (
            f"🔓 Ports: {tp} ({_delta(np_)}) │ "
            f"⚙️  Services: {ts} ({_delta(ns)}) │ "
            f"🔑 Creds: {tc} ({_delta(nc)}) │ "
            f"💀 Shells: {tsh} ({_delta(nsh)}) │ "
            f"🌐 Paths: {tpaths}"
        )

        # ── Row 2: Learning quality ──
        novelty = snapshot.get("novelty_rate", 0.0)
        stag = snapshot.get("stagnation_steps", 0)
        ar_total = snapshot.get("anti_repeat_total", 0)
        total_cmds = snapshot.get("total_commands", 0)
        unique_tmpls = snapshot.get("unique_templates", 0)
        phase_changes = snapshot.get("phase_changes", 0)

        nov_color = "green" if novelty > 0.6 else "yellow" if novelty > 0.3 else "red"
        stag_color = "green" if stag < 3 else "yellow" if stag < 6 else "red"

        quality_line = (
            f"[{nov_color}]📈 Novelty: {novelty:.0%}[/{nov_color}] "
            f"({unique_tmpls}/{total_cmds} unique) │ "
            f"[{stag_color}]⏳ Stagnation: {stag} steps[/{stag_color}] │ "
            f"🔁 Anti-Repeat: {ar_total} │ "
            f"🔀 Phase Changes: {phase_changes}"
        )

        # ── Row 3: Window metrics (if present) ──
        window = snapshot.get("window", {})
        window_line = ""
        if window:
            wd = window.get("discoveries_delta", 0)
            wnov = window.get("novelty_rate", 0.0)
            war = window.get("anti_repeat_rate", 0.0)
            wstag = window.get("stagnation_avg", 0.0)
            wpt = window.get("phase_thrash", 0)
            wcpd = window.get("cost_per_discovery", 0.0)
            ws = window.get("window_size", 5)
            wd_color = "green" if wd > 0 else "red"
            window_line = (
                f"[dim]Window({ws}):[/dim] "
                f"[{wd_color}]Δdisc={wd}[/{wd_color}] │ "
                f"nov={wnov:.0%} │ AR={war:.0%} │ "
                f"stag_avg={wstag:.1f} │ thrash={wpt} │ "
                f"$/disc=${wcpd:.4f}"
            )

        # ── Row 4: Milestones (compact) ──
        milestones = snapshot.get("milestones", {})
        ms_parts = []
        ms_icons = {
            "first_port": "🔓", "first_service": "⚙️",
            "first_creds": "🔑", "first_foothold": "💀",
            "user_flag": "🚩", "root_flag": "🏴",
        }
        for key, icon in ms_icons.items():
            val = milestones.get(key, -1)
            if val >= 0:
                ms_parts.append(f"[green]{icon} {key}@s{val}[/green]")
        milestone_line = " │ ".join(ms_parts) if ms_parts else "[dim]No milestones yet[/dim]"

        # ── Row 5: Model mix (compact) ──
        mix = snapshot.get("model_mix", {})
        mix_calls = mix.get("calls", {})
        mix_cost = mix.get("total_cost", 0.0)
        cache_rate = mix.get("cache_hit_rate", 0.0)
        mix_line = (
            f"🧠 codex:{mix_calls.get('codex', 0)} "
            f"mini:{mix_calls.get('mini', 0)} "
            f"nano:{mix_calls.get('nano', 0)} │ "
            f"💰 ${mix_cost:.4f} │ "
            f"📦 cache: {cache_rate:.0%}"
        )

        # Assemble panel
        lines = [disc_line, quality_line]
        if window_line:
            lines.append(window_line)
        lines.append(milestone_line)
        lines.append(mix_line)

        console.print(Panel(
            "\n".join(lines),
            title=f"[bold bright_green]📊 Learning Dashboard (step {step + 1})[/bold bright_green]",
            border_style="bright_green",
            box=box.ROUNDED,
            expand=True,
            padding=(0, 2),
        ))

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
            title="[bold bright_white]Agent Performance[/bold bright_white]",
            show_header=True, header_style="bold bright_white on dark_blue",
            border_style="bright_magenta", box=box.ROUNDED, padding=(0, 2),
            expand=True,
        )
        table.add_column("Agent", style="bold", ratio=2, no_wrap=True)
        table.add_column("Phase", ratio=2, no_wrap=True)
        table.add_column("Cmds", ratio=1, justify="right")
        table.add_column("Uniq", ratio=1, justify="right")
        table.add_column("Ep Reward", ratio=2, justify="right")
        table.add_column("Conf", ratio=1, justify="center")
        table.add_column("Mentor", ratio=1, justify="center")
        table.add_column("Top Source", ratio=2)

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
    # P35: COHERENCE PANEL — postcard + contradiction + scores
    # =========================================================================
    def print_coherence_panel(
        self,
        coherence_result: Optional[Any] = None,
        step: int = 0,
    ) -> None:
        """Render a compact coherence panel after each step.

        Shows:
          - State postcard (1-line)
          - Contradiction status (RED DESYNC banner if detected)
          - Coherence / novelty / repeat-risk scores
        """
        if coherence_result is None:
            return

        from rich.panel import Panel
        from rich.text import Text
        from rich.columns import Columns

        lines: list = []

        # Postcard
        postcard = getattr(coherence_result.summary, "postcard", "")
        if postcard:
            lines.append(f"[bold cyan]📮[/bold cyan] {postcard}")

        # Contradiction banner
        cont = coherence_result.contradiction
        if cont.contradiction_detected:
            sev = cont.severity.upper()
            sev_color = {"HIGH": "bold red", "MED": "yellow", "LOW": "dim"}.get(sev, "dim")
            lines.append(f"[{sev_color}]⚠ DESYNC ({sev})[/{sev_color}]: {cont.fix_hint}")
            for c in cont.contradictions[:2]:
                lines.append(f"  [red]•[/red] {c}")

        # Scores
        sc = coherence_result.score
        coh_bar = "█" * int(sc.coherence_score * 10) + "░" * (10 - int(sc.coherence_score * 10))
        nov_bar = "█" * int(sc.novelty_score * 10) + "░" * (10 - int(sc.novelty_score * 10))
        rr_color = "red" if sc.repeat_risk > 0.6 else "yellow" if sc.repeat_risk > 0.3 else "green"
        lines.append(
            f"Coherence [cyan]{coh_bar}[/cyan] {sc.coherence_score:.0%}  "
            f"Novelty [green]{nov_bar}[/green] {sc.novelty_score:.0%}  "
            f"RepeatRisk [{rr_color}]{sc.repeat_risk:.0%}[/{rr_color}]"
        )

        border_style = "red" if cont.contradiction_detected else "dim cyan"
        panel = Panel(
            "\n".join(lines),
            title=f"[bold]P35 Coherence │ Step {step}[/bold]",
            border_style=border_style,
            padding=(0, 1),
        )
        console.print(panel)

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
        per_coach_ppo: Optional[Dict[str, Dict[str, float]]] = None,
        ddqn_metrics: Optional[Dict[str, Any]] = None,
        decision_sources: Optional[Dict[str, int]] = None,
        discovery_board: Optional[Dict[str, Any]] = None,
        gpt_cost_summary: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.current_episode = episode
        self.episode_rewards.append(total_reward)
        if highest_phase:
            self.episode_phases.append(highest_phase)

        console.print()
        console.rule(f"[bold bright_green]━━━ Episode {episode} Complete ━━━[/bold bright_green]", style="bright_green")

        # ── METRICS TABLE with deltas ────────────────────────────────
        table = Table(
            title=f"[bold bright_white]📋 Episode {episode} Summary[/bold bright_white]",
            show_header=True, header_style="bold bright_white on dark_blue",
            border_style="bright_green", box=box.ROUNDED,
            padding=(0, 2),
        )
        table.add_column("Metric", style="bold cyan", min_width=20)
        table.add_column("Value", justify="right", min_width=14)
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
        for src, cnt in sorted(all_sources.items(), key=lambda x: -x[1]):
            pct = cnt / total_decisions * 100
            src_icon = SOURCE_STYLES.get(src[:8], ("", "dim"))[0]
            src_parts.append(f"{src_icon}{src}:{cnt}({pct:.0f}%)")
        if src_parts:
            # Show top sources in Value column, rest in Trend column
            table.add_row("Decision Mix", " ".join(src_parts[:3]),
                          " ".join(src_parts[3:]) if len(src_parts) > 3 else "")

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

        # Phase 23: GPT Cost Summary
        if gpt_cost_summary:
            _cum = gpt_cost_summary.get("cumulative_usd", 0.0)
            _ep = gpt_cost_summary.get("episode_usd", 0.0)
            _cost_clr = "red" if _cum > 1.0 else "yellow" if _cum > 0.25 else "green"
            models_info = gpt_cost_summary.get("models", {})
            model_parts = []
            for mname, mdata in models_info.items():
                short = mname.replace("gpt-", "").replace("-codex", "c")
                model_parts.append(f"{short}:{mdata.get('requests', 0)}r/{mdata.get('tokens', 0):,}t")
            table.add_row(
                "💳 GPT Cost",
                f"[{_cost_clr}]${_cum:.4f}[/{_cost_clr}]",
                f"ep: ${_ep:.4f}  {' '.join(model_parts)}"
            )

        console.print(table)

        # ── PHASE 10.2: UNIFIED ALGORITHM PANELS ─────────────────────
        # Side-by-side algorithm visualization panels
        algo_panels = []

        # PPO Panel (always show if we have any PPO data)
        ppo_panel_data = ppo_metrics
        if ppo_panel_data or per_coach_ppo:
            algo_panels.append(self._build_ppo_panel(ppo_panel_data, per_coach_ppo))

        # Phase 37: GPT↔RL Integration Panel
        _llm_bridge_data = kwargs.get("llm_bridge_snapshot")
        if _llm_bridge_data:
            algo_panels.append(self._build_llm_bridge_panel(_llm_bridge_data))

        # DDQN Panel
        if ddqn_metrics:
            algo_panels.append(self._build_ddqn_panel(ddqn_metrics))

        # Print algorithm panels side-by-side (2 per row)
        if algo_panels:
            if len(algo_panels) >= 2:
                console.print(Columns(algo_panels[:2], equal=True, expand=True))
                if len(algo_panels) > 2:
                    console.print(Columns(algo_panels[2:], equal=True, expand=True))
            else:
                console.print(algo_panels[0])

        # Decision Pipeline + Discovery Board side-by-side
        bottom_panels = []
        # Build decision sources from param or from tracked agent_stats
        eff_sources = decision_sources
        if not eff_sources:
            eff_sources = {}
            for stats in self.agent_stats.values():
                for src, cnt in stats.get("decision_sources", {}).items():
                    eff_sources[src] = eff_sources.get(src, 0) + cnt
        if eff_sources:
            bottom_panels.append(self._build_decision_pipeline_panel(eff_sources))

        # Discovery board from param or from episode discoveries
        eff_disc = discovery_board
        if not eff_disc and self.episode_discoveries:
            eff_disc = dict(self.episode_discoveries)
        if eff_disc:
            bottom_panels.append(self._build_discovery_board_panel(eff_disc))

        if bottom_panels:
            if len(bottom_panels) >= 2:
                console.print(Columns(bottom_panels[:2], equal=True, expand=True))
            elif bottom_panels:
                console.print(bottom_panels[0])

        # ── PHASE TIMELINE ───────────────────────────────────────────
        if self.phase_timeline:
            parts = []
            for _step_num, pname in self.phase_timeline:
                ic = PHASE_ICONS.get(pname, "")
                parts.append(f"[bold]{ic} {pname}[/bold] (s{_step_num})")
            console.print(Panel(
                " → ".join(parts),
                title="[bold bright_blue]📈 Phase Timeline[/bold bright_blue]",
                border_style="bright_blue", box=box.ROUNDED, padding=(0, 2),
            ))

        # ── KILL CHAIN PROGRESS BAR (Phase 6.7) ─────────────────────
        phases_reached = {pname for _, pname in self.phase_timeline}
        self.print_kill_chain_bar(highest_phase, phases_reached)
        self.print_cost_ticker(self.tokens_total)

        # ── REWARD CHART ─────────────────────────────────────────────
        if len(self.episode_rewards) >= 2:
            console.print(Panel(
                self._ascii_reward_chart(),
                title="[bold bright_yellow]📈 Reward Trend[/bold bright_yellow]",
                border_style="bright_yellow", box=box.ROUNDED, padding=(0, 2),
            ))

        # ── DISCOVERY SUMMARY ────────────────────────────────────────
        if self.episode_discoveries:
            parts = []
            for dtype, items in sorted(self.episode_discoveries.items()):
                istr = ", ".join(str(i) for i in sorted(items))
                parts.append(f"[bold]{dtype}[/bold]: {istr}")
            console.print(Panel(
                "\n".join(parts),
                title="[bold bright_green]✨ Discoveries This Episode[/bold bright_green]",
                border_style="bright_green", box=box.ROUNDED, padding=(0, 2),
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
        console.rule("[bold bright_green]🏁 Training Complete[/bold bright_green]", style="bright_green")

        # ── MAIN METRICS TABLE ───────────────────────────────────────
        table = Table(title=f"[bold bright_white]📊 Run: {run_id}[/bold bright_white]",
                      show_header=True, header_style="bold bright_white on dark_blue",
                      box=box.ROUNDED, border_style="bright_green",
                      padding=(0, 2))
        table.add_column("Metric", style="bold", width=22)
        table.add_column("Value", justify="right", width=18)
        table.add_column("Trend", width=30)
        table.add_row("Total Episodes", str(total_episodes), "")
        table.add_row("Time", f"{total_time:.1f}s ({total_time/max(total_episodes,1):.1f}s/ep)", "")
        avg_r = final_metrics.get('avg_reward_recent', 0)
        best_r = max(self.episode_rewards) if self.episode_rewards else 0
        table.add_row("Avg Reward", f"{avg_r:+.2f}",
                       sparkline(self.episode_rewards, 20) if self.episode_rewards else "")
        table.add_row("Best Reward", f"{best_r:+.1f}", "")
        table.add_row("Tokens", f"{self.tokens_total:,}", "")

        # Learning quality metrics
        if self.episode_rewards and len(self.episode_rewards) >= 2:
            first_half = self.episode_rewards[:len(self.episode_rewards)//2]
            second_half = self.episode_rewards[len(self.episode_rewards)//2:]
            improvement = (
                (sum(second_half)/len(second_half)) - (sum(first_half)/len(first_half))
                if first_half else 0
            )
            imp_color = "green" if improvement > 0 else "red" if improvement < 0 else "dim"
            table.add_row("Learning Δ", f"[{imp_color}]{improvement:+.1f}[/{imp_color}]",
                           f"1st→2nd half reward improvement")

        # Skill library
        if self.skill_library_size:
            table.add_row("Skill Library", str(self.skill_library_size), "persistent skills")

        console.print(table)

        # ── ALGORITHM TREND ANALYSIS ─────────────────────────────────
        if self.ppo_loss_history or self.ddqn_history or self.decision_source_history:
            algo_lines = []

            # PPO loss trends
            if self.ppo_loss_history:
                algo_lines.append("[bold underline]PPO Training Curves:[/bold underline]")
                for coach, losses in sorted(self.ppo_loss_history.items()):
                    icon = AGENT_ICONS.get(coach, ("🤖", "", "dim"))[0]
                    loss_list = list(losses)
                    vloss_list = list(self.ppo_vloss_history.get(coach, []))
                    ent_list = list(self.ppo_entropy_history.get(coach, []))
                    pi_spark = sparkline(loss_list, 15) if loss_list else ""
                    v_spark = sparkline(vloss_list, 15) if vloss_list else ""
                    h_spark = sparkline(ent_list, 15) if ent_list else ""
                    short = coach.replace("Agent", "")
                    algo_lines.append(f"  {icon} [bold]{short}[/bold]")
                    if loss_list:
                        algo_lines.append(f"    π loss: {loss_list[-1]:.5f} {pi_spark}")
                    if vloss_list:
                        algo_lines.append(f"    V loss: {vloss_list[-1]:.4f}  {v_spark}")
                    if ent_list:
                        algo_lines.append(f"    Entropy: {ent_list[-1]:.4f} {h_spark}")

            # Decision source evolution
            if self.decision_source_history:
                algo_lines.append("")
                algo_lines.append("[bold underline]Decision Source Evolution:[/bold underline]")
                # Aggregate across episodes
                for src in ["ppo", "playbook", "registry", "anti_repeat", "mentor", "codex_meta"]:
                    vals = [d.get(src, 0) for d in self.decision_source_history]
                    if any(v > 0 for v in vals):
                        src_icon = SOURCE_STYLES.get(src, ("❓", "dim"))[0]
                        spark = sparkline(list(map(float, vals)), 15)
                        total_uses = sum(vals)
                        algo_lines.append(
                            f"  {src_icon} {src:<12} total:{total_uses:4d}  {spark}"
                        )

            if algo_lines:
                console.print(Panel(
                    "\n".join(algo_lines),
                    title="[bold bright_blue]🧬  Algorithm Performance Across Training[/bold bright_blue]",
                    border_style="bright_blue",
                    box=box.ROUNDED,
                    padding=(0, 2),
                ))

        # ── REWARD CHART ─────────────────────────────────────────────
        if self.episode_rewards:
            console.print(Panel(self._ascii_reward_chart(),
                                title="[bold bright_yellow]📈 Final Reward Trend[/bold bright_yellow]",
                                border_style="bright_yellow", box=box.ROUNDED,
                                padding=(0, 2)))

        # ── TOKENS + COST ────────────────────────────────────────────
        if self.tokens_by_agent:
            tt = Table(title="[bold bright_white]💳 Tokens by Agent[/bold bright_white]",
                      box=box.ROUNDED, border_style="bright_cyan",
                      header_style="bold bright_white on dark_blue",
                      padding=(0, 2))
            tt.add_column("Agent", style="bold")
            tt.add_column("Tokens", justify="right")
            tt.add_column("Share", justify="right")
            total_tok = sum(self.tokens_by_agent.values()) or 1
            for ag, tok in sorted(self.tokens_by_agent.items(), key=lambda x: -x[1]):
                pct = tok / total_tok * 100
                icon = AGENT_ICONS.get(ag, ("🤖", "", ""))[0]
                tt.add_row(f"{icon} {ag}", f"{tok:,}", f"{pct:.0f}%")
            console.print(tt)

        # ── ACTIVE ALGORITHMS SUMMARY ────────────────────────────────
        active_algos = [name for name, active in self.algo_active.items() if active]
        if active_algos:
            algo_str = " │ ".join(
                f"{ALGO_ICONS.get(a, ('🔧', 'dim', ''))[0]} {a}"
                for a in active_algos
            )
            console.print(f"  [bold]Active Algorithms:[/bold] {algo_str}")

        self.print_cost_ticker(self.tokens_total)
        console.print()
        console.rule("[bold bright_green]✨ ARIASKA Training Session Complete ✨[/bold bright_green]", style="bright_green")
        console.print()

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
        # Phase 10.0: Reset KG/LLM episode counters
        self.kg_queries_episode = 0
        self.kg_hits_episode = 0
        self.llm_calls_episode = 0
        self.llm_tokens_episode = 0
        self.venice_calls_episode = 0

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
[bold bright_red]     █████╗ ██████╗ ██╗ █████╗ ███████╗██╗  ██╗ █████╗ [/bold bright_red]
[bold red]    ██╔══██╗██╔══██╗██║██╔══██╗██╔════╝██║ ██╔╝██╔══██╗[/bold red]
[bold bright_red]    ███████║██████╔╝██║███████║███████╗█████╔╝ ███████║[/bold bright_red]
[bold red]    ██╔══██║██╔══██╗██║██╔══██║╚════██║██╔═██╗ ██╔══██║[/bold red]
[bold bright_red]    ██║  ██║██║  ██║██║██║  ██║███████║██║  ██╗██║  ██║[/bold bright_red]
[bold red]    ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝[/bold red]

[bold bright_cyan]    ⚡  Autonomous Multi-Agent Reinforcement Learning  ⚡[/bold bright_cyan]
[bold bright_white]    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold bright_white]
[dim italic]    5 Agents  ·  PPO Actor-Critic v3.0  ·  8-Phase Kill Chain  ·  Live Pentesting[/dim italic]
[dim italic]    107K Knowledge Corpus  ·  Evidence-Gated Execution  ·  Hybrid GPT Pipeline[/dim italic]
"""
        console.print(Panel(
            logo,
            border_style="bright_red",
            box=box.DOUBLE,
            padding=(0, 4),
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
