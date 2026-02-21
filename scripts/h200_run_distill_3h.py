"""H200 Online Distillation — 3+ hour PPO training with local mentor.

Orchestrates online distillation on a rented H200 NVL 140GB GPU:
- Uses a local vLLM mentor (OpenAI-compat) for teacher signals
- Falls back to OpenAI only for verified hard cases
- Anneals mentor involvement: heavy→medium→autonomous
- Produces checkpoints, traces, and run reports

Usage:
    python -m scripts.h200_run_distill_3h --seed 42 --max-hours 3 --checkpoint-every 10m
    python -m scripts.h200_run_distill_3h --eval-only --episodes 50 --no-mentor
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Load .env before anything reads env vars ────────────────────────────
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path, override=False)
except ImportError:
    pass  # dotenv optional — env vars can be set externally

logger = logging.getLogger("ariaska.h200_distill")

# ── Rich output (project invariant: never bare print) ───────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
except ImportError:
    raise SystemExit("ERROR: 'rich' is required. Install with: pip install rich")

console = Console(force_terminal=True)

# ── TensorBoard (optional — logs if available) ──────────────────────────
try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TENSORBOARD = True
except ImportError:
    _HAS_TENSORBOARD = False
    SummaryWriter = None  # type: ignore[misc,assignment]

TENSORBOARD_DIR = Path("runs/h200_distill")

# ---------------------------------------------------------------------------
# Sparkline + Dashboard Chars
# ---------------------------------------------------------------------------

SPARK_CHARS = "▁▂▃▄▅▆▇█"
PHASE_ICONS = {
    "RECON": "🔍", "ENUMERATION": "📋", "EXPLOITATION": "💥",
    "PRIVILEGE_ESCALATION": "👑", "LATERAL_MOVEMENT": "🔀",
    "POST_EXPLOITATION": "🏴", "EXFILTRATION": "📤", "CLOSEOUT": "🧹",
}
SOURCE_ICONS = {
    "local": ("🏠", "green"), "codex": ("🧠", "bright_magenta"),
    "ppo": ("🤖", "cyan"), "none": ("⚡", "dim"),
}


def _sparkline(values: List[float], width: int = 20) -> str:
    """Generate an ASCII sparkline from a list of values."""
    if not values:
        return ""
    vals = values[-width:]
    mn, mx = min(vals), max(vals)
    rng = mx - mn if mx != mn else 1.0
    return "".join(
        SPARK_CHARS[max(0, min(int((v - mn) / rng * (len(SPARK_CHARS) - 1)), len(SPARK_CHARS) - 1))]
        for v in vals
    )


def _progress_bar(value: float, total: float, width: int = 20, fill: str = "█", empty: str = "░") -> str:
    """Render a compact progress bar."""
    pct = min(1.0, max(0.0, value / total)) if total > 0 else 0.0
    filled = int(pct * width)
    return fill * filled + empty * (width - filled)


# ---------------------------------------------------------------------------
# DistillLiveDashboard — per-step live UI/UX tracking
# ---------------------------------------------------------------------------

class DistillLiveDashboard:
    """Live training dashboard for H200 distillation runs.

    Provides per-step compact lines, per-episode rich summaries,
    reward sparklines, phase progress, codex budget tracking,
    and discovery board — all in real-time Rich output.
    """

    def __init__(self, max_steps_per_ep: int = 150, max_episodes: int = 500) -> None:
        self.max_steps_per_ep = max_steps_per_ep
        self.max_episodes = max_episodes

        # Step reward history (per episode, reset on episode start)
        self._ep_rewards: List[float] = []
        # Global reward history (across episodes, for sparkline)
        self._episode_reward_history: List[float] = []

        # Discovery tracking per episode
        self._ep_discoveries: Dict[str, int] = {}
        self._ep_discovery_total: int = 0

        # Command family tracking per episode
        self._ep_cmd_families: Dict[str, int] = {}
        self._ep_unique_cmds: set = set()

        # Phase tracking
        self._ep_max_phase: str = "RECON"

        # Step counter
        self._step_count: int = 0
        self._mentor_steps: int = 0
        self._codex_steps: int = 0
        self._override_steps: int = 0

        # Enabled flag (can disable for benchmarking)
        self.enabled: bool = True
        # Verbosity: 'full' shows every step, 'compact' shows every 5th, 'quiet' suppresses
        self.verbosity: str = os.environ.get("DISTILL_DASHBOARD", "full")

    def episode_start(self, episode_id: int, anneal_snapshot: Optional[Dict[str, Any]] = None) -> None:
        """Reset per-episode state and print episode header."""
        self._ep_rewards = []
        self._ep_discoveries = {}
        self._ep_discovery_total = 0
        self._ep_cmd_families = {}
        self._ep_unique_cmds = set()
        self._ep_max_phase = "RECON"
        self._step_count = 0
        self._mentor_steps = 0
        self._codex_steps = 0
        self._override_steps = 0

        if not self.enabled or self.verbosity == "quiet":
            return

        anneal_phase = anneal_snapshot.get("phase", "?") if anneal_snapshot else "?"
        anneal_pct = (anneal_snapshot.get("progress", 0) * 100) if anneal_snapshot else 0
        bc = anneal_snapshot.get("bc_coef", 0) if anneal_snapshot else 0
        kl = anneal_snapshot.get("kl_coef", 0) if anneal_snapshot else 0
        alpha = anneal_snapshot.get("prior_alpha", 0) if anneal_snapshot else 0

        console.rule(
            f"[bold bright_white]Episode {episode_id:04d}[/bold bright_white]  │  "
            f"anneal=[bright_yellow]{anneal_phase}({anneal_pct:.1f}%)[/bright_yellow]  │  "
            f"BC={bc:.4f}  KL={kl:.4f}  α={alpha:.3f}",
            style="bright_blue",
        )

    def step(
        self,
        step_i: int,
        command: str,
        reward: float,
        phase: str,
        mentor_source: str,
        confidence: float,
        teacher_overrode: bool,
        discoveries: List[str],
        cmd_family: str,
        codex_budget_remaining: Optional[float] = None,
    ) -> None:
        """Record and display one training step."""
        self._step_count += 1
        self._ep_rewards.append(reward)
        self._ep_unique_cmds.add(command)
        self._ep_cmd_families[cmd_family] = self._ep_cmd_families.get(cmd_family, 0) + 1

        if mentor_source not in ("none", ""):
            self._mentor_steps += 1
        if mentor_source == "codex":
            self._codex_steps += 1
        if teacher_overrode:
            self._override_steps += 1

        # Track discoveries
        for d in discoveries:
            d_type = d.split(":")[0] if ":" in d else "misc"
            self._ep_discoveries[d_type] = self._ep_discoveries.get(d_type, 0) + 1
            self._ep_discovery_total += 1

        # Phase progression
        if phase in PHASES:
            pidx = PHASES.index(phase)
            if pidx > PHASES.index(self._ep_max_phase):
                self._ep_max_phase = phase

        if not self.enabled or self.verbosity == "quiet":
            return

        # Compact mode: only show every 5th step
        if self.verbosity == "compact" and step_i % 5 != 0:
            return

        # ── Per-step compact line ────────────────────────────────
        phase_icon = PHASE_ICONS.get(phase, "❓")
        src_icon, src_color = SOURCE_ICONS.get(mentor_source, SOURCE_ICONS["none"])
        r_color = "green" if reward > 0 else "red" if reward < 0 else "dim"
        override_tag = " [bold yellow]⇈[/bold yellow]" if teacher_overrode else ""

        # Truncate command for display
        cmd_display = command[:60] + "…" if len(command) > 60 else command

        # Confidence display
        conf_bar = _progress_bar(confidence, 1.0, width=5)
        conf_color = "green" if confidence >= 0.6 else "yellow" if confidence >= 0.3 else "red"

        # Discovery flash
        disc_flash = ""
        if discoveries:
            disc_flash = f" [bold bright_green]★+{len(discoveries)}[/bold bright_green]"

        # Codex budget flash (only if codex was used this step)
        codex_tag = ""
        if mentor_source == "codex" and codex_budget_remaining is not None:
            codex_tag = f" [bright_magenta]💰${codex_budget_remaining:.2f}[/bright_magenta]"

        # Reward sparkline (per-episode)
        spark = _sparkline(self._ep_rewards, width=10)

        console.print(
            f"  [dim]{step_i + 1:3d}[/dim] "
            f"{phase_icon} "
            f"[{src_color}]{src_icon}[/{src_color}]{override_tag} "
            f"[{conf_color}]{conf_bar}[/{conf_color}] "
            f"[{r_color}]{reward:+6.1f}[/{r_color}] "
            f"[white]{cmd_display}[/white]"
            f"{disc_flash}{codex_tag}"
            f"  [dim]{spark}[/dim]"
        )

    def episode_end(
        self,
        episode_id: int,
        result: "EpisodeResult",
        metrics: "RunMetrics",
        anneal: Optional["AnnealController"] = None,
        codex: Optional["CodexMentorClient"] = None,
        update_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Print rich episode summary with all tracked data."""
        self._episode_reward_history.append(result.total_reward)

        if not self.enabled or self.verbosity == "quiet":
            return

        # ── Episode Summary Table ────────────────────────────────
        ep_table = Table(
            title=f"Episode {episode_id:04d} Summary",
            show_header=False,
            box=box.SIMPLE_HEAVY,
            padding=(0, 1),
            min_width=72,
            border_style="bright_blue",
        )
        ep_table.add_column("key", style="cyan", width=22)
        ep_table.add_column("val", style="white")

        # Reward with sparkline
        avg_recent = metrics.avg_reward(window=10)
        ep_spark = _sparkline(self._episode_reward_history, width=15)
        r_color = "green" if result.total_reward > 0 else "red"
        ep_table.add_row(
            "Reward",
            f"[bold {r_color}]{result.total_reward:+.2f}[/bold {r_color}]  "
            f"avg10={avg_recent:+.2f}  {ep_spark}",
        )

        # Steps + speed
        ep_table.add_row("Steps", f"{result.steps} / {self.max_steps_per_ep}")

        # Phase reached with progress bar
        phase_idx = PHASES.index(result.phase_reached) if result.phase_reached in PHASES else 0
        phase_bar = _progress_bar(phase_idx + 1, len(PHASES), width=16)
        phase_icon = PHASE_ICONS.get(result.phase_reached, "❓")
        ep_table.add_row(
            "Phase",
            f"{phase_icon} [magenta]{result.phase_reached}[/magenta]  {phase_bar}",
        )

        # Mentor stats
        mentor_pct = 100 * self._mentor_steps / max(result.steps, 1)
        ep_table.add_row(
            "Mentor",
            f"{self._mentor_steps}/{result.steps} ({mentor_pct:.0f}%)  "
            f"overrides={self._override_steps}  codex={self._codex_steps}",
        )

        # Anneal state
        if anneal:
            ep_table.add_row(
                "Anneal",
                f"{anneal.phase_name} ({anneal.progress * 100:.1f}%)  "
                f"BC={anneal.bc_coef():.4f}  KL={anneal.kl_coef():.4f}  "
                f"α={anneal.prior_alpha():.3f}",
            )

        # Command diversity
        diversity = len(self._ep_unique_cmds) / max(result.steps, 1)
        top_fams = sorted(self._ep_cmd_families.items(), key=lambda x: x[1], reverse=True)[:6]
        fam_str = "  ".join(f"{k}={v}" for k, v in top_fams)
        ep_table.add_row(
            "Commands",
            f"unique={len(self._ep_unique_cmds)} diversity={diversity:.0%}  {fam_str}",
        )

        # Discoveries
        if self._ep_discoveries:
            disc_str = "  ".join(f"{k}={v}" for k, v in sorted(self._ep_discoveries.items()))
            ep_table.add_row("Discoveries", f"+{self._ep_discovery_total}  {disc_str}")
        else:
            ep_table.add_row("Discoveries", "[dim]none[/dim]")

        # Codex budget
        if codex and codex.available:
            remaining = codex.policy.budget_remaining
            spent = metrics.codex_cost_usd
            budget_bar = _progress_bar(remaining, codex.policy.total_budget, width=12)
            budget_color = "green" if remaining > 3.0 else "yellow" if remaining > 1.0 else "red"
            ep_table.add_row(
                "Codex budget",
                f"[{budget_color}]${remaining:.2f}[/{budget_color}] / "
                f"${codex.policy.total_budget:.2f}  {budget_bar}  "
                f"calls={metrics.codex_calls}  spent=${spent:.3f}",
            )

        # PPO update metrics
        if update_metrics:
            ploss = update_metrics.get("policy_loss", 0)
            vloss = update_metrics.get("value_loss", 0)
            entropy = update_metrics.get("entropy", 0)
            ep_table.add_row(
                "PPO update",
                f"π_loss={ploss:.4f}  v_loss={vloss:.4f}  entropy={entropy:.4f}",
            )

        # Echo bans
        if metrics.echo_fallback_count > 0:
            ep_table.add_row(
                "Echo bans",
                f"[red]{metrics.echo_fallback_count}[/red] (total)",
            )

        # Global totals
        ep_table.add_row(
            "Totals",
            f"ep={metrics.episodes_completed}  steps={metrics.total_steps}  "
            f"disc={metrics.discoveries_total}  ckpts={len(metrics.checkpoints_saved)}",
        )

        console.print(ep_table)
        console.print()  # breathing room


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATE_DIM = 512
ACTION_DIM = 5
REWARD_MIN = -15.0
REWARD_MAX = 100.0
CHECKPOINT_DIR = Path("models/distilled")
RESULTS_DIR = Path("results/h200_distill")
TRACES_DIR = Path("traces/h200_distill")
DATA_DIR = Path("data/distill_prep")

# Mentor anneal schedule (pct of runtime → behaviour)
ANNEAL_HEAVY_END = 0.30    # 0–30%: query mentor every step
ANNEAL_MEDIUM_END = 0.70   # 30–70%: every 3 steps
ANNEAL_LIGHT_STEPS = 10    # 70–100%: every N steps

# Distillation coefficients (decay with anneal)
BC_COEF_MAX = 0.15
BC_COEF_MIN = 0.01
KL_COEF_MAX = 0.15
KL_COEF_MIN = 0.005
RANKING_COEF = 0.05

# Default episode config
DEFAULT_MAX_STEPS_PER_EPISODE = 150
DEFAULT_EPISODES = 500
DEFAULT_MAX_HOURS = 3.0
DEFAULT_CHECKPOINT_INTERVAL_SEC = 600  # 10 minutes

# Anneal enforcement tolerance (±10%)
ANNEAL_TOLERANCE = 0.10
ANNEAL_WINDOW_SEC = 300  # 5-minute windows

# Phase ordering (mirrors core)
PHASES = [
    "RECON", "ENUMERATION", "EXPLOITATION", "PRIVILEGE_ESCALATION",
    "LATERAL_MOVEMENT", "POST_EXPLOITATION", "EXFILTRATION", "CLOSEOUT",
]

PHASE_REWARDS = {
    "RECON": 0.0, "ENUMERATION": 5.0, "EXPLOITATION": 15.0,
    "PRIVILEGE_ESCALATION": 30.0, "LATERAL_MOVEMENT": 45.0,
    "POST_EXPLOITATION": 60.0, "EXFILTRATION": 75.0, "CLOSEOUT": 90.0,
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MentorConfig:
    """Local mentor endpoint configuration."""
    base_url: str = "http://127.0.0.1:8192/v1"
    model: str = "openai/gpt-oss-120b"
    timeout: float = 45.0
    max_tokens: int = 1024  # reasoning models use internal tokens; need headroom
    temperature: float = 0.2
    enabled: bool = True

    @classmethod
    def from_env(cls) -> MentorConfig:
        """Build from environment variables."""
        return cls(
            base_url=os.environ.get("ARIASKA_LOCAL_LLM_BASE_URL", cls.base_url),
            model=os.environ.get("ARIASKA_LOCAL_LLM_MODEL", cls.model),
            timeout=float(os.environ.get("MENTOR_TIMEOUT", "45")),
            enabled=os.environ.get("FF_LOCAL_LLM", "1") == "1",
        )


@dataclass
class RunMetrics:
    """Accumulated metrics for the entire run."""
    episodes_completed: int = 0
    total_steps: int = 0
    total_rewards: List[float] = field(default_factory=list)
    phase_reached: Dict[str, int] = field(default_factory=dict)
    mentor_calls_total: int = 0
    mentor_calls_by_phase: Dict[str, int] = field(default_factory=dict)
    openai_fallback_calls: int = 0
    command_families_used: Dict[str, int] = field(default_factory=dict)
    checkpoints_saved: List[str] = field(default_factory=list)
    ppo_updates: int = 0
    anneal_history: List[Dict[str, Any]] = field(default_factory=list)
    discoveries_total: int = 0
    wall_start: float = 0.0
    wall_end: float = 0.0
    echo_fallback_count: int = 0
    mentor_json_failures: int = 0
    teacher_override_count: int = 0
    codex_calls: int = 0
    codex_tokens: int = 0
    codex_cost_usd: float = 0.0
    codex_budget_usd: float = 5.60
    codex_successes: int = 0
    codex_escalation_reasons: Dict[str, int] = field(default_factory=dict)

    @property
    def wall_hours(self) -> float:
        return (self.wall_end - self.wall_start) / 3600.0 if self.wall_end else 0.0

    def avg_reward(self, window: Optional[int] = None) -> float:
        rews = self.total_rewards if window is None else self.total_rewards[-window:]
        return float(np.mean(rews)) if rews else 0.0


@dataclass
class EpisodeResult:
    """Single episode outcome."""
    episode_id: int
    steps: int
    total_reward: float
    phase_reached: str
    discoveries: int
    mentor_calls: int
    ppo_updates: int
    commands_used: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Mentor Client (HTTP, no openai import)
# ---------------------------------------------------------------------------

class LocalMentorClient:
    """Queries the local vLLM mentor via its OpenAI-compatible HTTP API.

    Uses ``requests`` — no ``import openai`` per project invariant.
    """

    def __init__(self, config: MentorConfig):
        self.config = config
        self._session: Any = None
        self._available = False
        self._call_count = 0
        self._total_tokens = 0

    def _ensure_session(self) -> Any:
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({"Content-Type": "application/json"})
        return self._session

    def health_check(self) -> bool:
        """Return True if the mentor endpoint is reachable."""
        if not self.config.enabled:
            return False
        try:
            sess = self._ensure_session()
            resp = sess.get(
                f"{self.config.base_url}/models",
                timeout=5.0,
            )
            self._available = resp.status_code == 200
        except Exception:
            self._available = False
        return self._available

    def query_teacher_action(
        self,
        phase: str,
        step: int,
        discoveries: List[str],
        recent_commands: List[str],
        state_summary: str,
    ) -> Optional[Dict[str, Any]]:
        """Ask the mentor for a teacher action + distribution.

        Returns:
            dict with keys: command, reasoning, confidence, action_probs,
            template_name, command_family.  None on failure.
        """
        if not self.config.enabled or not self._available:
            return None

        prompt = self._build_prompt(phase, step, discoveries, recent_commands, state_summary)

        try:
            sess = self._ensure_session()
            payload = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "response_format": {"type": "json_object"},
            }
            resp = sess.post(
                f"{self.config.base_url}/chat/completions",
                json=payload,
                timeout=self.config.timeout,
            )
            if resp.status_code != 200:
                logger.warning("Mentor returned HTTP %d", resp.status_code)
                return None

            data = resp.json()
            self._call_count += 1
            usage = data.get("usage", {})
            self._total_tokens += usage.get("total_tokens", 0)

            content = data["choices"][0]["message"].get("content")
            # Reasoning models may put output in 'reasoning' if content is None
            if not content:
                reasoning_text = data["choices"][0]["message"].get("reasoning", "")
                if reasoning_text:
                    # Try to extract JSON from reasoning as last resort
                    import re
                    json_match = re.search(r'\{[^{}]*"command"[^{}]*\}', reasoning_text)
                    if json_match:
                        content = json_match.group(0)
                    else:
                        logger.debug("Mentor returned null content, no JSON in reasoning")
                        return None
                else:
                    logger.debug("Mentor returned null content")
                    return None
            # Strip markdown fences if present
            content = content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(
                    l for l in lines if not l.strip().startswith("```")
                )
            parsed = json.loads(content)
            result = self._validate_response(parsed)
            if result is not None:
                return result
            # Retry once with explicit reminder
            logger.debug("Mentor response invalid, retrying with reminder")
            retry_payload = dict(payload)
            retry_payload["messages"] = payload["messages"] + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": (
                    "Invalid response. Return ONLY valid JSON with keys: "
                    "command, teacher_action, confidence, action_probs, phase, "
                    "rationale_tokens, template_name, command_family. "
                    "command must be a REAL shell command, not echo."
                )},
            ]
            resp2 = sess.post(
                f"{self.config.base_url}/chat/completions",
                json=retry_payload,
                timeout=self.config.timeout,
            )
            if resp2.status_code == 200:
                content2 = resp2.json()["choices"][0]["message"].get("content", "") or ""
                content2 = content2.strip()
                if content2.startswith("```"):
                    lines2 = content2.split("\n")
                    content2 = "\n".join(
                        l for l in lines2 if not l.strip().startswith("```")
                    )
                parsed2 = json.loads(content2)
                return self._validate_response(parsed2)
            return None

        except Exception as exc:
            logger.debug("Mentor query failed: %s", exc)
            return None

    def _system_prompt(self) -> str:
        return (
            "You are an expert penetration-testing mentor for an RL agent.\n"
            "Return JSON only. No markdown. No prose. No explanation outside JSON.\n"
            "If you add ANY text outside the JSON object, the response is REJECTED.\n\n"
            "STRICT OUTPUT CONTRACT (every key required):\n"
            '{\n'
            '  "teacher_action": <int 0-4>,\n'
            '  "confidence": <float 0.0-1.0>,\n'
            '  "phase": "<recon|enumeration|exploitation|privesc|lateral|post_exploit|exfil|closeout>",\n'
            '  "rationale_tokens": "<max 30 tokens>",\n'
            '  "command": "<exact real shell command>",\n'
            '  "action_probs": [<5 floats summing to 1.0>],\n'
            '  "template_name": "<tool_name>",\n'
            '  "command_family": "<nmap|gobuster|nikto|searchsploit|hydra|...>"\n'
            '}\n\n'
            "RULES:\n"
            "- command MUST be a real executable shell command, never 'echo'.\n"
            "- teacher_action: 0=recon_scan, 1=enum_probe, 2=exploit_attempt, "
            "3=privesc_action, 4=post_exfil.\n"
            "- No extra keys. No markdown fences. No text before/after the JSON.\n"
            "- {target} = target IP placeholder."
        )

    def _build_prompt(
        self, phase: str, step: int,
        discoveries: List[str], recent_commands: List[str],
        state_summary: str,
    ) -> str:
        disc_str = ", ".join(discoveries[-10:]) if discoveries else "none"
        cmds_str = "\n".join(f"  - {c}" for c in recent_commands[-5:]) if recent_commands else "  none"
        return (
            f"Phase: {phase}\nStep: {step}\n"
            f"State: {state_summary}\n"
            f"Recent discoveries: {disc_str}\n"
            f"Recent commands:\n{cmds_str}\n\n"
            "What command should the agent execute next?"
        )

    @staticmethod
    def _validate_response(parsed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate and normalize mentor response per strict JSON contract.

        Hard rejects: missing keys, echo commands, non-numeric confidence.
        """
        required_keys = {"command", "confidence", "teacher_action"}
        missing = required_keys - set(parsed.keys())
        if missing:
            logger.debug("Mentor response missing required keys: %s", missing)
            return None
        # Reject echo / empty commands
        cmd = str(parsed["command"]).strip()
        if not cmd or cmd.startswith("echo ") or cmd == "echo":
            logger.debug("Mentor returned echo/empty command — rejected")
            return None
        parsed["command"] = cmd
        # Reject non-numeric confidence
        try:
            parsed["confidence"] = float(parsed["confidence"])
        except (ValueError, TypeError):
            logger.debug("Mentor confidence not numeric: %s", parsed.get("confidence"))
            return None
        # Normalize action_probs
        probs = parsed.get("action_probs", [0.2] * 5)
        if not isinstance(probs, list) or len(probs) != 5:
            probs = [0.2] * 5
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        parsed["action_probs"] = probs
        # Normalize teacher_action
        ta = parsed.get("teacher_action")
        if ta is not None:
            try:
                ta = int(ta)
                if not 0 <= ta <= 4:
                    ta = int(np.argmax(probs))
            except (ValueError, TypeError):
                ta = int(np.argmax(probs))
            parsed["teacher_action"] = ta
        else:
            parsed["teacher_action"] = int(np.argmax(probs))
        parsed.setdefault("rationale_tokens", "")
        parsed.setdefault("reasoning", parsed.get("rationale_tokens", ""))
        parsed.setdefault("template_name", "")
        parsed.setdefault("command_family", cmd.split()[0] if cmd else "manual")
        parsed["confidence"] = float(np.clip(parsed["confidence"], 0.0, 1.0))
        return parsed


# ---------------------------------------------------------------------------
# OpenAI Codex Mentor — $5.60 budget, front-loaded then anneals
# ---------------------------------------------------------------------------

# Codex pricing estimates (per 1K tokens)
_CODEX_INPUT_COST_PER_1K = 0.012   # gpt-5.2-codex input
_CODEX_OUTPUT_COST_PER_1K = 0.036  # gpt-5.2-codex output
_CODEX_BUDGET_USD = 5.60
_CODEX_MAX_CALLS = 800  # hard cap regardless of budget


class CodexEscalationPolicy:
    """Decides when to escalate to OpenAI codex based on anneal + budget.

    Front-loaded: use more codex early, taper to near-zero.
    Schedule (by training progress):
        0–20%:  every 3 steps when local mentor fails/low-conf
        20–50%: every 5 steps, only on hard cases
        50–80%: every 10 steps, critical stalls only
        80–100%: disabled
    """

    def __init__(self, budget_usd: float = _CODEX_BUDGET_USD) -> None:
        self.budget_remaining = budget_usd
        self.total_budget = budget_usd
        self.calls = 0
        self._step_counter = 0

    def should_escalate(
        self,
        progress: float,
        step_in_episode: int,
        local_failed: bool,
        local_confidence: float,
        reward_stagnant: bool,
    ) -> Tuple[bool, str]:
        """Return (should_call, reason)."""
        self._step_counter += 1

        # Hard caps
        if self.budget_remaining <= 0.01:
            return False, "budget_exhausted"
        if self.calls >= _CODEX_MAX_CALLS:
            return False, "max_calls_reached"

        # Phase 80–100%: disabled
        if progress >= 0.80:
            return False, "late_phase_disabled"

        # Phase 50–80%: every 10th step, only critical stalls
        if progress >= 0.50:
            if not reward_stagnant:
                return False, "no_stall_late"
            if self._step_counter % 10 != 0:
                return False, "step_skip_light"
            return True, "critical_stall"

        # Phase 20–50%: every 5th step, hard cases
        if progress >= 0.20:
            is_hard = local_failed or local_confidence < 0.3
            if not is_hard and not reward_stagnant:
                return False, "not_hard_mid"
            if self._step_counter % 5 != 0:
                return False, "step_skip_medium"
            return True, "hard_case_mid"

        # Phase 0–20%: every 3rd step when local mentor is weak
        if self._step_counter % 3 != 0:
            return False, "step_skip_heavy"
        if local_failed:
            return True, "local_failed_early"
        if local_confidence < 0.5:
            return True, "low_conf_early"
        # Even if local succeeded, use codex occasionally for diversity
        if self._step_counter % 9 == 0:
            return True, "diversity_early"
        return False, "local_sufficient"

    def record_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Record a codex call cost. Returns cost in USD."""
        cost = (input_tokens / 1000.0) * _CODEX_INPUT_COST_PER_1K + \
               (output_tokens / 1000.0) * _CODEX_OUTPUT_COST_PER_1K
        self.budget_remaining -= cost
        self.calls += 1
        return cost


class CodexMentorClient:
    """Calls OpenAI gpt-5.2-codex via HTTP for hard-case teacher signals.

    Uses ``requests`` — no ``import openai`` per project invariant.
    Budget-capped at $5.60.
    """

    def __init__(self) -> None:
        self._session: Any = None
        self._api_key: Optional[str] = os.environ.get("OPENAI_API_KEY")
        self._base_url = "https://api.openai.com/v1"
        self._model = "gpt-5.2-codex"
        self._available = self._api_key is not None and len(self._api_key) > 10
        self.policy = CodexEscalationPolicy()

    @property
    def available(self) -> bool:
        return self._available

    def _ensure_session(self) -> Any:
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            })
        return self._session

    def query(
        self,
        phase: str,
        step: int,
        discoveries: List[str],
        recent_commands: List[str],
        state_summary: str,
        local_response: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], int, int]:
        """Query codex for teacher signal.

        gpt-5.2-codex uses the Responses API (``/v1/responses``), NOT the
        chat completions endpoint.  The payload and response structure differ:
        - ``instructions`` (system) + ``input`` (user) instead of ``messages``
        - ``max_output_tokens`` instead of ``max_tokens``
        - response lives in ``output[].content[].text``
        - usage keys: ``input_tokens`` / ``output_tokens``

        Returns: (response_dict_or_None, input_tokens, output_tokens)
        """
        if not self._available:
            return None, 0, 0

        disc_str = ", ".join(discoveries[-8:]) if discoveries else "none"
        cmds_str = "\n".join(f"  - {c}" for c in recent_commands[-5:]) if recent_commands else "  none"

        # If local response exists, include it for codex to refine
        local_ctx = ""
        if local_response:
            local_ctx = (
                f"\nLocal mentor suggested: {local_response.get('command', '?')} "
                f"(conf={local_response.get('confidence', '?')}). "
                "Evaluate and improve if needed."
            )

        user_input = (
            f"Phase: {phase}\nStep: {step}\n"
            f"State: {state_summary}\n"
            f"Recent discoveries: {disc_str}\n"
            f"Recent commands:\n{cmds_str}"
            f"{local_ctx}\n\n"
            "What command should the agent execute next? Return JSON only."
        )

        instructions = (
            "You are gpt-5.2-codex acting as a senior pentesting mentor for an RL agent.\n"
            "Return JSON only. No markdown. No prose.\n\n"
            "OUTPUT CONTRACT (every key required):\n"
            '{\n'
            '  "teacher_action": <int 0-4>,\n'
            '  "confidence": <float 0.0-1.0>,\n'
            '  "phase": "<recon|enumeration|exploitation|privesc|...>",\n'
            '  "rationale_tokens": "<max 30 tokens>",\n'
            '  "command": "<exact real shell command>",\n'
            '  "action_probs": [<5 floats summing to 1.0>],\n'
            '  "template_name": "<tool_name>",\n'
            '  "command_family": "<nmap|gobuster|...>"\n'
            '}\n\n'
            "- command MUST be a real executable shell command, never 'echo'.\n"
            "- teacher_action: 0=recon_scan, 1=enum_probe, 2=exploit_attempt, "
            "3=privesc_action, 4=post_exfil.\n"
            "- {target} = target IP placeholder."
        )

        try:
            sess = self._ensure_session()
            # Responses API — codex uses internal reasoning tokens so we give
            # a generous output budget (reasoning tokens are hidden but count).
            payload = {
                "model": self._model,
                "instructions": instructions,
                "input": user_input,
                "max_output_tokens": 3000,  # codex uses ~1K reasoning + ~200 visible
            }
            resp = sess.post(
                f"{self._base_url}/responses",
                json=payload,
                timeout=45.0,
            )
            if resp.status_code != 200:
                logger.warning("Codex HTTP %d", resp.status_code)
                return None, 0, 0

            data = resp.json()
            usage = data.get("usage", {})
            in_tok = usage.get("input_tokens", 0)
            out_tok = usage.get("output_tokens", 0)

            # Extract text from Responses API output structure
            content = ""
            for item in data.get("output", []):
                if item.get("type") == "message":
                    for part in item.get("content", []):
                        if part.get("type") == "output_text":
                            content = part.get("text", "").strip()
                            break
                    if content:
                        break

            if not content:
                logger.debug("Codex returned empty content")
                return None, in_tok, out_tok

            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(l for l in lines if not l.strip().startswith("```"))

            parsed = json.loads(content)
            result = LocalMentorClient._validate_response(parsed)
            return result, in_tok, out_tok

        except Exception as exc:
            logger.debug("Codex query failed: %s", exc)
            return None, 0, 0


# ---------------------------------------------------------------------------
# Anneal Controller
# ---------------------------------------------------------------------------

class AnnealController:
    """Controls mentor involvement decay over training runtime."""

    def __init__(self, total_duration_sec: float):
        self._total = max(total_duration_sec, 1.0)
        self._start = time.monotonic()

    @property
    def progress(self) -> float:
        """0.0 → 1.0 progress through training runtime."""
        elapsed = time.monotonic() - self._start
        return min(1.0, elapsed / self._total)

    @property
    def phase_name(self) -> str:
        p = self.progress
        if p < ANNEAL_HEAVY_END:
            return "heavy"
        elif p < ANNEAL_MEDIUM_END:
            return "medium"
        return "light"

    def should_query_mentor(self, step_in_episode: int) -> bool:
        """Determine whether to query the mentor this step."""
        p = self.progress
        if p < ANNEAL_HEAVY_END:
            return True  # Every step
        elif p < ANNEAL_MEDIUM_END:
            return step_in_episode % 3 == 0
        else:
            return step_in_episode % ANNEAL_LIGHT_STEPS == 0

    def bc_coef(self) -> float:
        """Current BC distillation coefficient."""
        return BC_COEF_MAX - (BC_COEF_MAX - BC_COEF_MIN) * self.progress

    def kl_coef(self) -> float:
        """Current KL teacher coefficient."""
        return KL_COEF_MAX - (KL_COEF_MAX - KL_COEF_MIN) * self.progress

    def prior_alpha(self) -> float:
        """LLM prior injection weight."""
        return max(0.02, 0.50 * (1.0 - self.progress))

    def snapshot(self) -> Dict[str, Any]:
        return {
            "progress": round(self.progress, 4),
            "phase": self.phase_name,
            "bc_coef": round(self.bc_coef(), 5),
            "kl_coef": round(self.kl_coef(), 5),
            "prior_alpha": round(self.prior_alpha(), 4),
        }


# ---------------------------------------------------------------------------
# Anneal Enforcer — monitors mentor call rate vs expected schedule
# ---------------------------------------------------------------------------

class AnnealEnforcer:
    """Tracks mentor call rate per 5-minute window and warns on deviation.

    Expected rates: heavy=~100%, medium=~33%, light=~10%.
    Tolerance: ±10%.
    """

    EXPECTED: Dict[str, float] = {"heavy": 1.0, "medium": 0.333, "light": 0.10}

    def __init__(self) -> None:
        self._windows: Dict[int, Dict[str, int]] = {}
        self._start = time.monotonic()

    def _window_id(self) -> int:
        return int((time.monotonic() - self._start) // ANNEAL_WINDOW_SEC)

    def record_step(self, mentor_queried: bool) -> None:
        """Record a step in the current 5-minute window."""
        wid = self._window_id()
        if wid not in self._windows:
            self._windows[wid] = {"total": 0, "mentor": 0}
        self._windows[wid]["total"] += 1
        if mentor_queried:
            self._windows[wid]["mentor"] += 1

    def check_deviation(self, anneal_phase: str) -> Optional[str]:
        """Check if current window deviates >±10% from expected.

        Returns warning message if deviation detected, else None.
        Also returns adjustment hint for gating correction.
        """
        wid = self._window_id()
        w = self._windows.get(wid)
        if w is None or w["total"] < 20:  # need at least 20 steps to judge
            return None
        actual_rate = w["mentor"] / w["total"]
        expected_rate = self.EXPECTED.get(anneal_phase, 0.5)
        deviation = actual_rate - expected_rate
        if abs(deviation) > ANNEAL_TOLERANCE:
            direction = "HIGH" if deviation > 0 else "LOW"
            msg = (
                f"ANNEAL DEVIATION: window={wid} phase={anneal_phase} "
                f"expected={expected_rate:.0%} actual={actual_rate:.0%} "
                f"deviation={deviation:+.0%} ({direction})"
            )
            logger.warning(msg)
            self._correction = deviation  # store for gating adjustment
            return msg
        self._correction = 0.0
        return None

    @property
    def needs_correction(self) -> float:
        """Signed deviation for gating adjustment. >0 means too many calls."""
        return getattr(self, "_correction", 0.0)

    def summary(self) -> List[Dict[str, Any]]:
        """Return per-window summary."""
        result = []
        for wid in sorted(self._windows):
            w = self._windows[wid]
            rate = w["mentor"] / w["total"] if w["total"] > 0 else 0.0
            result.append({"window": wid, "total": w["total"], "mentor": w["mentor"], "rate": round(rate, 3)})
        return result


# ---------------------------------------------------------------------------
# Distillation Runner
# ---------------------------------------------------------------------------

class H200DistillationRunner:
    """Main distillation orchestrator.

    Runs PPO training episodes with mentor-guided teacher signals.
    Does NOT modify core/ — uses public APIs only.
    """

    def __init__(
        self,
        seed: int = 42,
        max_hours: float = DEFAULT_MAX_HOURS,
        max_episodes: int = DEFAULT_EPISODES,
        max_steps_per_episode: int = DEFAULT_MAX_STEPS_PER_EPISODE,
        checkpoint_interval_sec: float = DEFAULT_CHECKPOINT_INTERVAL_SEC,
        eval_only: bool = False,
        no_mentor: bool = False,
        resume_from: Optional[str] = None,
        device: str = "auto",
    ):
        self.seed = seed
        self.max_hours = max_hours
        self.max_episodes = max_episodes
        self.max_steps_per_ep = max_steps_per_episode
        self.checkpoint_interval = checkpoint_interval_sec
        self.eval_only = eval_only
        self.no_mentor = no_mentor
        self.resume_from = resume_from
        self.metrics = RunMetrics()

        # Device selection
        import torch
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Seed everything
        self._seed_all(seed)

        # Lazy-init (set in .run())
        self._ppo: Any = None
        self._env: Any = None
        self._mentor: Optional[LocalMentorClient] = None
        self._codex: Optional[CodexMentorClient] = None
        self._anneal: Optional[AnnealController] = None
        self._anneal_enforcer: Optional[AnnealEnforcer] = None
        self._last_checkpoint_time: float = 0.0
        self._trace_path: Optional[Path] = None
        self._run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._reward_history: List[float] = []  # per-step for stagnation detect

        # Live dashboard
        self._dashboard = DistillLiveDashboard(
            max_steps_per_ep=max_steps_per_episode,
            max_episodes=max_episodes,
        )

        # TensorBoard writer (lazy-init in _init_dirs)
        self._tb: Optional[Any] = None

    # ── Initialization ───────────────────────────────────────────

    def _seed_all(self, seed: int) -> None:
        import torch
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _init_ppo(self) -> None:
        """Initialize PPO agent with distillation channels enabled."""
        from core.algorithms.ppo_agent import PPOAgent, PPOConfig

        config = PPOConfig(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            use_bc_loss=True,
            bc_loss_coef=BC_COEF_MAX,
            use_kl_teacher_loss=True,
            kl_teacher_coef=KL_COEF_MAX,
            use_ranking_loss=True,
            ranking_loss_coef=RANKING_COEF,
        )
        self._ppo = PPOAgent(config=config, device=str(self.device))

        if self.resume_from:
            logger.info("Resuming from checkpoint: %s", self.resume_from)
            self._ppo.load(self.resume_from)

        logger.info(
            "PPO initialized on %s (state_dim=%d, action_dim=%d, "
            "bc=%.3f, kl=%.3f, ranking=%.3f)",
            self.device, STATE_DIM, ACTION_DIM,
            config.bc_loss_coef, config.kl_teacher_coef, config.ranking_loss_coef,
        )

    def _init_env(self) -> None:
        """Initialize CyberEnvironment."""
        from core.environment.cyber_environment import CyberEnvironment
        self._env = CyberEnvironment()
        logger.info("CyberEnvironment initialized")

    def _init_mentor(self) -> None:
        """Initialize local mentor client."""
        if self.no_mentor:
            logger.info("Mentor disabled (--no-mentor)")
            return
        config = MentorConfig.from_env()
        self._mentor = LocalMentorClient(config)
        if self._mentor.health_check():
            console.print(Panel(
                f"[green]Mentor online[/green]: {config.model}\n"
                f"Endpoint: {config.base_url}",
                title="Local Mentor",
            ))
        else:
            console.print("[yellow]Warning: Mentor endpoint unreachable — running without mentor[/yellow]")
            self._mentor = None

    def _init_codex(self) -> None:
        """Initialize OpenAI codex escalation mentor ($5.60 budget)."""
        if self.no_mentor:
            return
        self._codex = CodexMentorClient()
        if self._codex.available:
            console.print(Panel(
                f"[green]Codex online[/green]: gpt-5.2-codex\n"
                f"Budget: ${_CODEX_BUDGET_USD:.2f} | Max calls: {_CODEX_MAX_CALLS}\n"
                f"Strategy: front-loaded \u2192 anneal to zero by 80%",
                title="OpenAI Codex Escalation",
            ))
        else:
            console.print("[yellow]Codex unavailable (no OPENAI_API_KEY) \u2014 local mentor only[/yellow]")
            self._codex = None

    def _init_dirs(self) -> None:
        """Create output directories."""
        for d in [CHECKPOINT_DIR, RESULTS_DIR, TRACES_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        self._trace_path = TRACES_DIR / f"h200_distill_{self._run_id}.jsonl"

        # TensorBoard
        if _HAS_TENSORBOARD:
            tb_dir = TENSORBOARD_DIR / self._run_id
            tb_dir.mkdir(parents=True, exist_ok=True)
            self._tb = SummaryWriter(log_dir=str(tb_dir), flush_secs=30)  # type: ignore[misc]
            console.print(Panel(
                f"[green]TensorBoard logging enabled[/green]\n"
                f"Log dir: {tb_dir}\n"
                f"View: tensorboard --logdir {TENSORBOARD_DIR} --bind_all",
                title="📊 TensorBoard",
            ))
        else:
            console.print("[dim]TensorBoard not available (pip install tensorboard)[/dim]")

    # ── State encoding helper ────────────────────────────────────

    def _encode_state(self, state: Dict[str, Any], step: int = 0) -> Any:
        """Encode env state to tensor."""
        import torch
        from core.models.state_encoder import encode_state
        return encode_state(state, self.device, current_step=step, max_steps=self.max_steps_per_ep)

    # ── Command action mapper ────────────────────────────────────

    def _action_to_command(self, action_idx: int, state: Dict[str, Any]) -> str:
        """Map PPO action index to a command string."""
        phase = state.get("phase", "RECON")
        target = state.get("target_ip", "172.28.128.3")
        # Use CommandActionMapper if available
        try:
            from core.algorithms.command_action_mapper import CommandActionMapper
            role = state.get("role", "attacker")
            mapper = CommandActionMapper(role=role)
            tpl = mapper.action_to_command(action_idx)
            if tpl is not None:
                cmd = tpl.template.replace("{target}", target)
                return cmd
        except Exception:
            pass

        # Phase-aware fallback commands — REAL executable commands only
        fallback_map: Dict[int, Dict[str, str]] = {
            0: {  # recon_scan
                "RECON": f"nmap -sV -sC -p- -T4 {target}",
                "ENUMERATION": f"nmap -sV --script=vuln {target}",
                "EXPLOITATION": f"nmap -sV -p 80,443,8080 {target}",
                "PRIVILEGE_ESCALATION": f"nmap -sU --top-ports 20 {target}",
            },
            1: {  # enum_probe
                "RECON": f"whatweb http://{target}",
                "ENUMERATION": f"gobuster dir -u http://{target} -w /usr/share/wordlists/dirb/common.txt",
                "EXPLOITATION": f"nikto -h http://{target}",
                "PRIVILEGE_ESCALATION": f"enum4linux -a {target}",
            },
            2: {  # exploit_attempt
                "RECON": "searchsploit --nmap /tmp/nmap_scan.xml",
                "ENUMERATION": "searchsploit -w apache 2.4",
                "EXPLOITATION": f"msfconsole -q -x 'search type:exploit {target}; exit'",
                "PRIVILEGE_ESCALATION": "sudo -l",
            },
            3: {  # privesc_action
                "RECON": f"curl -s http://{target}/robots.txt",
                "ENUMERATION": f"dirb http://{target}",
                "EXPLOITATION": "find / -perm -4000 -type f 2>/dev/null | head -20",
                "PRIVILEGE_ESCALATION": "cat /etc/passwd",
                "POST_EXPLOITATION": "cat /etc/shadow",
            },
            4: {  # post_exfil
                "RECON": f"curl -s http://{target}/ | head -50",
                "ENUMERATION": f"wget -q http://{target}/index.html -O /tmp/index.html",
                "EXPLOITATION": "whoami && id && hostname",
                "EXFILTRATION": "tar czf /tmp/loot.tar.gz /home 2>/dev/null",
                "CLOSEOUT": "history -c && exit",
            },
        }
        phase_map = fallback_map.get(action_idx, {})
        # Try exact match, then any available command for this action
        if phase in phase_map:
            return phase_map[phase]
        # Fallback: use the first available command for this action_idx
        if phase_map:
            return next(iter(phase_map.values()))
        # Last resort: real command instead of echo placeholder
        return f"nmap -sV -p 80 {target}"

    # ── Echo-ban fallback: real phase-appropriate command ────────

    def _phase_safe_fallback(self, state: Dict[str, Any]) -> str:
        """Return a real command for the current phase.  Never echo."""
        phase = state.get("phase", "RECON")
        target = state.get("target_ip", "172.28.128.3")
        _fallbacks: Dict[str, List[str]] = {
            "RECON": [
                f"nmap -sV -sC -T4 {target}",
                f"ping -c 2 {target}",
                f"nmap -sn {target}/24",
            ],
            "ENUMERATION": [
                f"nmap -sV --script=vuln -p- {target}",
                f"gobuster dir -u http://{target} -w /usr/share/wordlists/dirb/common.txt",
                f"nikto -h http://{target}",
                f"whatweb http://{target}",
            ],
            "EXPLOITATION": [
                f"searchsploit -w --nmap /tmp/nmap_scan.xml",
                f"msfconsole -q -x 'search type:exploit {target}; exit'",
                f"hydra -L /usr/share/wordlists/metasploit/unix_users.txt -P /usr/share/wordlists/rockyou.txt {target} ssh",
            ],
            "PRIVILEGE_ESCALATION": [
                "sudo -l",
                "find / -perm -4000 -type f 2>/dev/null | head -20",
                "cat /etc/crontab",
                "uname -a",
            ],
            "POST_EXPLOITATION": [
                "cat /etc/shadow",
                "cat /etc/passwd",
                "ls -la /root/",
                "whoami && id && hostname",
            ],
            "EXFILTRATION": [
                "tar czf /tmp/loot.tar.gz /home 2>/dev/null",
                "find / -name '*.txt' -o -name '*.conf' 2>/dev/null | head -20",
            ],
        }
        choices = _fallbacks.get(phase, _fallbacks["RECON"])
        # Rotate based on step count to add variety
        idx = self.metrics.echo_fallback_count % len(choices)
        return choices[idx]

    # ── Teacher signal extraction ────────────────────────────────

    def _is_reward_stagnant(self, window: int = 30) -> bool:
        """Check if recent rewards show stagnation (no improvement)."""
        if len(self._reward_history) < window:
            return False
        recent = self._reward_history[-window:]
        # Stagnant if mean of last window is <= mean of the window before
        if len(self._reward_history) >= 2 * window:
            prev = self._reward_history[-(2 * window):-window]
            return float(np.mean(recent)) <= float(np.mean(prev))
        return float(np.mean(recent)) < -1.0  # absolute stagnation threshold

    def _get_teacher_signal(
        self,
        state: Dict[str, Any],
        step: int,
        recent_commands: List[str],
        discoveries: List[str],
    ) -> Tuple[Optional[Any], Optional[int], Optional[Dict[str, Any]]]:
        """Query local mentor first, then escalate to codex if needed.

        Codex escalation triggers:
        - Local mentor failed (None response) or JSON invalid
        - Local mentor confidence < threshold
        - Reward stagnation detected
        - Early training diversity injection

        Returns:
            (teacher_distribution_tensor, teacher_action_idx, raw_response)
        """
        import torch

        if self._mentor is None and self._codex is None:
            return None, None, None
        if self._anneal and not self._anneal.should_query_mentor(step):
            return None, None, None

        phase = state.get("phase", "RECON")
        state_summary = (
            f"ports={len(state.get('discovered_ports', []))}, "
            f"services={len(state.get('discovered_services', []))}, "
            f"creds={len(state.get('credentials', []))}, "
            f"shells={len(state.get('shells', []))}"
        )

        # ── Stage 1: Local mentor ────────────────────────────────
        local_response: Optional[Dict[str, Any]] = None
        local_failed = True
        local_confidence = 0.0

        if self._mentor is not None:
            local_response = self._mentor.query_teacher_action(
                phase=phase,
                step=step,
                discoveries=discoveries,
                recent_commands=recent_commands,
                state_summary=state_summary,
            )
            if local_response is not None:
                local_failed = False
                local_confidence = float(local_response.get("confidence", 0.0))
            else:
                self.metrics.mentor_json_failures += 1

        # ── Stage 2: Codex escalation (if warranted) ─────────────
        response = local_response  # default: use local
        codex_used = False
        codex_reason = "not_needed"

        if self._codex is not None and self._codex.available:
            progress = self._anneal.progress if self._anneal else 0.0
            reward_stagnant = self._is_reward_stagnant()

            should_escalate, codex_reason = self._codex.policy.should_escalate(
                progress=progress,
                step_in_episode=step,
                local_failed=local_failed,
                local_confidence=local_confidence,
                reward_stagnant=reward_stagnant,
            )

            if should_escalate:
                codex_resp, in_tok, out_tok = self._codex.query(
                    phase=phase,
                    step=step,
                    discoveries=discoveries,
                    recent_commands=recent_commands,
                    state_summary=state_summary,
                    local_response=local_response,
                )
                if codex_resp is not None:
                    cost = self._codex.policy.record_cost(in_tok, out_tok)
                    self.metrics.codex_calls += 1
                    self.metrics.codex_tokens += in_tok + out_tok
                    self.metrics.codex_cost_usd += cost
                    self.metrics.codex_successes += 1
                    self.metrics.codex_escalation_reasons[codex_reason] = (
                        self.metrics.codex_escalation_reasons.get(codex_reason, 0) + 1
                    )
                    codex_used = True

                    # Codex overrides local if:
                    # a) local failed entirely, OR
                    # b) codex has higher confidence
                    codex_conf = float(codex_resp.get("confidence", 0.0))
                    if local_failed or codex_conf > local_confidence:
                        response = codex_resp
                        logger.info(
                            "CODEX ESCALATION at step %d: reason=%s, "
                            "codex_conf=%.2f vs local_conf=%.2f, cost=$%.4f "
                            "(remaining=$%.2f)",
                            step, codex_reason, codex_conf, local_confidence,
                            cost, self._codex.policy.budget_remaining,
                        )
                    self.metrics.openai_fallback_calls += 1
                else:
                    # Codex call failed — still use local if available
                    cost = self._codex.policy.record_cost(in_tok, out_tok)
                    self.metrics.codex_calls += 1
                    self.metrics.codex_tokens += in_tok + out_tok
                    self.metrics.codex_cost_usd += cost

        if response is None:
            return None, None, None

        # Convert action_probs to teacher distribution tensor
        probs = response.get("action_probs", [0.2] * 5)
        teacher_dist = torch.tensor(probs, dtype=torch.float32, device=self.device)

        # Best action from teacher
        teacher_action = int(torch.argmax(teacher_dist).item())

        # Tag response with source for tracing
        response["_source"] = "codex" if codex_used else "local"
        response["_codex_reason"] = codex_reason

        # Track metrics
        self.metrics.mentor_calls_total += 1
        phase_key = state.get("phase", "UNKNOWN")
        self.metrics.mentor_calls_by_phase[phase_key] = (
            self.metrics.mentor_calls_by_phase.get(phase_key, 0) + 1
        )

        return teacher_dist, teacher_action, response

    # ── Checkpoint ───────────────────────────────────────────────

    def _maybe_checkpoint(self, episode: int, force: bool = False) -> Optional[str]:
        """Save checkpoint if interval elapsed or force=True."""
        now = time.monotonic()
        if not force and (now - self._last_checkpoint_time) < self.checkpoint_interval:
            return None

        tag = f"h200_{self._run_id}_ep{episode:04d}"
        ckpt_path = CHECKPOINT_DIR / f"{tag}.pt"
        self._ppo.save(str(ckpt_path))
        self._last_checkpoint_time = now
        self.metrics.checkpoints_saved.append(str(ckpt_path))
        logger.info("Checkpoint saved: %s", ckpt_path)
        return str(ckpt_path)

    # ── Trace logging ────────────────────────────────────────────

    def _log_trace(self, record: Dict[str, Any]) -> None:
        """Append a JSONL trace record."""
        if self._trace_path is None:
            return
        try:
            with open(self._trace_path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            pass

    # ── Single episode ───────────────────────────────────────────

    def _run_episode(self, episode_id: int) -> EpisodeResult:
        """Run one training episode with optional mentor distillation."""
        import torch

        state = self._env.reset()
        if isinstance(state, tuple):
            state = state[0] if isinstance(state[0], dict) else {"phase": "RECON"}
        if not isinstance(state, dict):
            state = self._env.get_global_state()

        episode_reward = 0.0
        discoveries: List[str] = []
        recent_commands: List[str] = []
        mentor_calls = 0
        commands_used: List[str] = []
        max_phase = "RECON"
        ppo_updates_ep = 0

        self._log_trace({
            "kind": "episode_start",
            "episode": episode_id,
            "timestamp": time.time(),
            "anneal": self._anneal.snapshot() if self._anneal else {},
        })

        # Dashboard: episode start
        self._dashboard.episode_start(
            episode_id,
            anneal_snapshot=self._anneal.snapshot() if self._anneal else None,
        )

        for step_i in range(self.max_steps_per_ep):
            state_tensor = self._encode_state(state, step_i)

            # Get teacher signal (may be None based on anneal schedule)
            teacher_dist, teacher_action, mentor_raw = self._get_teacher_signal(
                state, step_i, recent_commands, discoveries,
            )
            if teacher_dist is not None:
                mentor_calls += 1

            # PPO selects action (with optional LLM prior from teacher)
            prior_alpha = self._anneal.prior_alpha() if self._anneal else 0.0
            llm_prior = teacher_dist if teacher_dist is not None else None

            action_idx, log_prob, value = self._ppo.select_action(
                state_tensor,
                training=not self.eval_only,
                llm_prior=llm_prior,
                prior_alpha=prior_alpha if llm_prior is not None else 0.0,
            )

            # ── STEERING: Teacher action OVERRIDES env action ────────
            # When mentor is queried successfully, its command IS the
            # executed command.  PPO still stores (student_action,
            # teacher_action, teacher_conf) so distill losses train
            # properly.  This nukes "echo-fu" at the source.
            command = self._action_to_command(action_idx, state)
            teacher_overrode = False

            if mentor_raw and not self.eval_only:
                mentor_cmd = str(mentor_raw.get("command", "")).strip()
                target = state.get("target_ip", "172.28.128.3")
                # Fill {target} in mentor command
                if mentor_cmd:
                    mentor_cmd = mentor_cmd.replace("{target}", target)
                if mentor_cmd and not mentor_cmd.startswith("echo"):
                    command = mentor_cmd
                    teacher_overrode = True
                    self.metrics.teacher_override_count += 1

            # ── STEERING: Ban echo execution ─────────────────────
            # If after all mapping the command still starts with "echo",
            # replace it with a real phase-appropriate command + penalty.
            if command.startswith("echo ") or command == "echo":
                self.metrics.echo_fallback_count += 1
                logger.warning(
                    "ECHO BAN: step %d replaced echo command (count=%d)",
                    step_i, self.metrics.echo_fallback_count,
                )
                # Replace with a real phase-appropriate command
                command = self._phase_safe_fallback(state)

            # Step environment
            next_state, reward, done, info = self._env.step(command)
            if not isinstance(next_state, dict):
                next_state = self._env.get_global_state()

            reward = float(np.clip(reward, REWARD_MIN, REWARD_MAX))
            episode_reward += reward
            self._reward_history.append(reward)  # for stagnation detection

            # Track phase progression
            current_phase = next_state.get("phase", state.get("phase", "RECON"))
            if current_phase in PHASES:
                pidx = PHASES.index(current_phase)
                if pidx > PHASES.index(max_phase):
                    max_phase = current_phase

            # Track discoveries from info
            step_discoveries = info.get("discoveries", [])
            if isinstance(step_discoveries, list):
                for d in step_discoveries:
                    disc_str = str(d) if not isinstance(d, str) else d
                    discoveries.append(disc_str)
                    self.metrics.discoveries_total += 1

            # Track commands
            recent_commands.append(command)
            commands_used.append(command)
            cmd_family = self._extract_family(command)
            self.metrics.command_families_used[cmd_family] = (
                self.metrics.command_families_used.get(cmd_family, 0) + 1
            )

            # Store transition in PPO buffer (with teacher signals)
            if not self.eval_only:
                self._ppo.store_transition(
                    state_tensor,
                    action_idx,
                    log_prob,
                    reward,
                    value,
                    done,
                    teacher_distribution=teacher_dist,
                    teacher_action=teacher_action,
                )

            # Log step trace (enriched with steering + codex flags)
            mentor_was_queried = teacher_dist is not None
            mentor_source = mentor_raw.get("_source", "none") if mentor_raw else "none"
            codex_reason = mentor_raw.get("_codex_reason", "") if mentor_raw else ""
            self._log_trace({
                "kind": "step",
                "episode": episode_id,
                "step": step_i,
                "action_idx": action_idx,
                "command": command,
                "reward": reward,
                "phase": current_phase,
                "mentor_queried": mentor_was_queried,
                "mentor_source": mentor_source,
                "codex_reason": codex_reason,
                "teacher_action": teacher_action,
                "teacher_overrode": teacher_overrode,
                "echo_banned": self.metrics.echo_fallback_count,
                "cmd_family": cmd_family,
                "done": done,
                "anneal_stage": self._anneal.phase_name if self._anneal else "?",
                "codex_budget_remaining": round(self._codex.policy.budget_remaining, 2) if self._codex else 0.0,
            })

            # Anneal enforcement: record step, check deviation, adjust gating
            if self._anneal_enforcer:
                self._anneal_enforcer.record_step(mentor_was_queried)
                if step_i % 50 == 49:  # check every 50 steps
                    anneal_phase = self._anneal.phase_name if self._anneal else "heavy"
                    warn = self._anneal_enforcer.check_deviation(anneal_phase)
                    if warn:
                        console.print(f"  [yellow]⚠ {warn}[/yellow]")

            # ── Dashboard: per-step live display ─────────────────
            confidence = float(mentor_raw.get("confidence", 0.0)) if mentor_raw else 0.0
            disc_strs = [str(d) for d in step_discoveries] if isinstance(step_discoveries, list) else []
            codex_remaining = self._codex.policy.budget_remaining if self._codex else None
            self._dashboard.step(
                step_i=step_i,
                command=command,
                reward=reward,
                phase=current_phase,
                mentor_source=mentor_source,
                confidence=confidence,
                teacher_overrode=teacher_overrode,
                discoveries=disc_strs,
                cmd_family=cmd_family,
                codex_budget_remaining=codex_remaining if mentor_source == "codex" else None,
            )

            state = next_state

            if done:
                break

        # PPO update at episode end
        update_metrics: Dict[str, float] = {}
        if not self.eval_only and self._ppo.buffer.size > 0:
            # Dynamically update distillation coefficients from anneal
            if self._anneal:
                self._ppo.config.bc_loss_coef = self._anneal.bc_coef()
                self._ppo.config.kl_teacher_coef = self._anneal.kl_coef()

            last_val = value if not done else 0.0
            update_metrics = self._ppo.update(last_value=last_val)
            ppo_updates_ep = 1
            self.metrics.ppo_updates += 1

            # Log PPO update metrics to TensorBoard
            self._tb_log_ppo_update(episode_id, update_metrics)

        # Track phase reached
        self.metrics.phase_reached[max_phase] = self.metrics.phase_reached.get(max_phase, 0) + 1

        # Anneal snapshot
        if self._anneal:
            snap = self._anneal.snapshot()
            snap["episode"] = episode_id
            snap["mentor_calls"] = mentor_calls
            self.metrics.anneal_history.append(snap)

        self._log_trace({
            "kind": "episode_end",
            "episode": episode_id,
            "total_reward": episode_reward,
            "steps": step_i + 1,
            "max_phase": max_phase,
            "phase_reached": max_phase,
            "anneal_stage": self._anneal.phase_name if self._anneal else "?",
            "anneal_progress": round(self._anneal.progress, 4) if self._anneal else 0.0,
            "mentor_calls": mentor_calls,
            "teacher_overrides": self.metrics.teacher_override_count,
            "echo_bans_total": self.metrics.echo_fallback_count,
            "discoveries": len(discoveries),
            "update_metrics": update_metrics,
            "cmd_families": dict(collections.Counter(
                self._extract_family(c) for c in commands_used
            )),
        })

        ep_result = EpisodeResult(
            episode_id=episode_id,
            steps=step_i + 1,
            total_reward=episode_reward,
            phase_reached=max_phase,
            discoveries=len(discoveries),
            mentor_calls=mentor_calls,
            ppo_updates=ppo_updates_ep,
            commands_used=commands_used,
        )

        # Dashboard: episode end summary
        self._dashboard.episode_end(
            episode_id=episode_id,
            result=ep_result,
            metrics=self.metrics,
            anneal=self._anneal,
            codex=self._codex,
            update_metrics=update_metrics,
        )

        return ep_result

    @staticmethod
    def _extract_family(command: str) -> str:
        """Extract command family from command string."""
        cmd = command.strip().split()[0] if command.strip() else "unknown"
        # Strip path prefixes
        cmd = cmd.rsplit("/", 1)[-1]
        return cmd

    # ── TensorBoard logging helpers ──────────────────────────────

    def _tb_log_episode(
        self, ep: int, result: EpisodeResult, avg_recent: float, ep_elapsed: float,
    ) -> None:
        """Log per-episode metrics to TensorBoard."""
        if self._tb is None:
            return
        tb = self._tb
        global_step = self.metrics.total_steps

        # ── Reward scalars ───────────────────────────────────────
        tb.add_scalar("reward/episode", result.total_reward, ep)
        tb.add_scalar("reward/avg_10", avg_recent, ep)
        tb.add_scalar("reward/avg_all", self.metrics.avg_reward(), ep)

        # ── Episode structure ────────────────────────────────────
        tb.add_scalar("episode/steps", result.steps, ep)
        tb.add_scalar("episode/time_sec", ep_elapsed, ep)
        phase_idx = PHASES.index(result.phase_reached) if result.phase_reached in PHASES else 0
        tb.add_scalar("episode/phase_reached_idx", phase_idx, ep)
        tb.add_scalar("episode/discoveries", result.discoveries, ep)

        # ── Mentor / distillation ────────────────────────────────
        mentor_rate = result.mentor_calls / max(result.steps, 1)
        tb.add_scalar("mentor/calls_this_ep", result.mentor_calls, ep)
        tb.add_scalar("mentor/rate", mentor_rate, ep)
        tb.add_scalar("mentor/teacher_overrides_total", self.metrics.teacher_override_count, ep)
        tb.add_scalar("mentor/echo_bans_total", self.metrics.echo_fallback_count, ep)
        tb.add_scalar("mentor/json_failures_total", self.metrics.mentor_json_failures, ep)

        # ── Anneal schedule ──────────────────────────────────────
        if self._anneal:
            tb.add_scalar("anneal/progress", self._anneal.progress, ep)
            tb.add_scalar("anneal/bc_coef", self._anneal.bc_coef(), ep)
            tb.add_scalar("anneal/kl_coef", self._anneal.kl_coef(), ep)
            tb.add_scalar("anneal/prior_alpha", self._anneal.prior_alpha(), ep)
            # Phase as numeric: heavy=0, medium=1, light=2
            phase_map = {"heavy": 0, "medium": 1, "light": 2}
            tb.add_scalar("anneal/phase_num", phase_map.get(self._anneal.phase_name, 0), ep)

        # ── Codex budget ─────────────────────────────────────────
        if self._codex and self._codex.available:
            tb.add_scalar("codex/budget_remaining", self._codex.policy.budget_remaining, ep)
            tb.add_scalar("codex/calls_total", self.metrics.codex_calls, ep)
            tb.add_scalar("codex/cost_usd", self.metrics.codex_cost_usd, ep)

        # ── Command diversity ────────────────────────────────────
        unique_cmds = len(set(result.commands_used))
        tb.add_scalar("diversity/unique_cmds", unique_cmds, ep)
        tb.add_scalar("diversity/ratio", unique_cmds / max(result.steps, 1), ep)
        tb.add_scalar("diversity/families_total", len(self.metrics.command_families_used), ep)

        # ── Throughput ───────────────────────────────────────────
        elapsed_h = (time.time() - self.metrics.wall_start) / 3600.0
        tb.add_scalar("throughput/steps_per_sec", self.metrics.total_steps / max(elapsed_h * 3600, 1), ep)
        tb.add_scalar("throughput/episodes_per_hour", self.metrics.episodes_completed / max(elapsed_h, 0.001), ep)

    def _tb_log_ppo_update(self, ep: int, update_metrics: Dict[str, float]) -> None:
        """Log PPO update metrics to TensorBoard."""
        if self._tb is None or not update_metrics:
            return
        tb = self._tb
        for key, val in update_metrics.items():
            tb.add_scalar(f"ppo/{key}", val, ep)

    def _print_mini_dashboard(self, ep: int) -> None:
        """Print a compact dashboard every N episodes."""
        m = self.metrics
        elapsed_h = (time.time() - m.wall_start) / 3600
        remaining_h = max(0.0, self.max_hours - elapsed_h)
        mentor_rate = m.mentor_calls_total / max(m.total_steps, 1)

        dash = Table(title=f"\U0001f4ca Dashboard @ Ep {ep}", show_lines=True, min_width=60)
        dash.add_column("Metric", style="cyan")
        dash.add_column("Value", style="green")
        dash.add_row("Episodes", str(m.episodes_completed))
        dash.add_row("Wall time", f"{elapsed_h:.2f}h / {self.max_hours:.1f}h  (remaining: {remaining_h:.2f}h)")
        dash.add_row("Steps", str(m.total_steps))
        dash.add_row("Steps/sec", f"{m.total_steps / max(elapsed_h * 3600, 1):.1f}")
        dash.add_row("Avg reward (all)", f"{m.avg_reward():+.2f}")
        dash.add_row("Avg reward (last 10)", f"{m.avg_reward(window=10):+.2f}")
        dash.add_row("Mentor rate", f"{mentor_rate:.1%}")
        dash.add_row("Teacher overrides", str(m.teacher_override_count))
        dash.add_row("Echo bans", str(m.echo_fallback_count))
        dash.add_row("Discoveries", str(m.discoveries_total))
        dash.add_row("Checkpoints", str(len(m.checkpoints_saved)))

        # Phase distribution
        phase_str = "  ".join(f"{p[:4]}={m.phase_reached.get(p, 0)}" for p in PHASES if m.phase_reached.get(p, 0) > 0)
        dash.add_row("Phases", phase_str or "none")

        # Top command families
        sorted_fams = sorted(m.command_families_used.items(), key=lambda x: x[1], reverse=True)[:8]
        fam_str = "  ".join(f"{k}={v}" for k, v in sorted_fams)
        dash.add_row("Top cmd families", fam_str)

        # Anneal enforcement windows
        if self._anneal_enforcer:
            windows = self._anneal_enforcer.summary()
            if windows:
                last_w = windows[-1]
                dash.add_row("Anneal window", f"#{last_w['window']} rate={last_w['rate']:.0%} ({last_w['mentor']}/{last_w['total']})")

        # Codex stats
        if self._codex:
            dash.add_row("Codex calls", str(m.codex_calls))
            dash.add_row("Codex cost", f"${m.codex_cost_usd:.3f} / ${m.codex_budget_usd:.2f}")
            dash.add_row("Codex budget left", f"${self._codex.policy.budget_remaining:.2f}")

        console.print(dash)
        console.print()  # blank line for readability

    # ── Main run loop ────────────────────────────────────────────

    def run(self) -> RunMetrics:
        """Execute the full distillation run."""
        self._init_dirs()
        self._init_ppo()
        self._init_env()
        self._init_mentor()
        self._init_codex()

        total_seconds = self.max_hours * 3600
        self._anneal = AnnealController(total_seconds)
        self._anneal_enforcer = AnnealEnforcer()
        self._last_checkpoint_time = time.monotonic()
        self.metrics.wall_start = time.time()

        mode = "[cyan]EVAL[/cyan]" if self.eval_only else "[green]TRAIN[/green]"
        mentor_model = self._mentor.config.model if self._mentor else "DISABLED"
        console.print(Panel(
            f"Mode: {mode}\n"
            f"Device: {self.device}\n"
            f"Max hours: {self.max_hours}\n"
            f"Max episodes: {self.max_episodes}\n"
            f"Seed: {self.seed}\n"
            f"Mentor: {'[green]ON[/green]' if self._mentor else '[red]OFF[/red]'}\n"
            f"Mentor model: {mentor_model}\n"
            f"Steering: teacher_override=ON, echo_ban=ON, anneal_enforce=ON\n"
            f"Codex: {'[green]ON[/green] $' + f'{_CODEX_BUDGET_USD:.2f} budget' if self._codex else '[dim]OFF[/dim]'}\n"
            f"Run ID: {self._run_id}",
            title="\U0001f9e0 H200 Distillation Run",
        ))

        deadline = time.monotonic() + total_seconds

        for ep in range(self.max_episodes):
            # Time check
            if time.monotonic() >= deadline:
                console.print("[yellow]\u23f0 Time limit reached[/yellow]")
                break

            ep_start = time.monotonic()
            result = self._run_episode(ep)
            ep_elapsed = time.monotonic() - ep_start
            self.metrics.episodes_completed += 1
            self.metrics.total_steps += result.steps
            self.metrics.total_rewards.append(result.total_reward)

            anneal_phase = self._anneal.phase_name if self._anneal else "?"
            anneal_pct = (self._anneal.progress * 100) if self._anneal else 0
            avg_recent = self.metrics.avg_reward(window=10)

            # ── TensorBoard: per-episode logging ─────────────────
            self._tb_log_episode(ep, result, avg_recent, ep_elapsed)

            # Periodic checkpoint
            ckpt = self._maybe_checkpoint(ep)
            if ckpt:
                console.print(f"  [yellow]\U0001f4be Checkpoint:[/yellow] {ckpt}")

            # Every 5 episodes: print a mini-dashboard
            if (ep + 1) % 5 == 0:
                self._print_mini_dashboard(ep)

            logger.info(
                "Ep %d | R=%+.2f | avg10=%+.2f | phase=%s | mentor=%d | "
                "t_override=%d | echo_ban=%d | anneal=%s(%.1f%%)",
                ep, result.total_reward, avg_recent,
                result.phase_reached, result.mentor_calls,
                self.metrics.teacher_override_count,
                self.metrics.echo_fallback_count,
                anneal_phase, anneal_pct,
            )

        # Final checkpoint
        self._maybe_checkpoint(self.metrics.episodes_completed, force=True)
        self.metrics.wall_end = time.time()

        # Close TensorBoard writer
        if self._tb:
            self._tb.close()
            console.print("[dim]TensorBoard writer closed[/dim]")

        # Save run report
        self._save_report()
        self._print_report()

        return self.metrics

    # ── Reporting ────────────────────────────────────────────────

    def _save_report(self) -> None:
        """Save JSON run report."""
        report_path = RESULTS_DIR / f"h200_report_{self._run_id}.json"
        report = {
            "run_id": self._run_id,
            "seed": self.seed,
            "device": str(self.device),
            "eval_only": self.eval_only,
            "max_hours": self.max_hours,
            "wall_hours": self.metrics.wall_hours,
            "episodes_completed": self.metrics.episodes_completed,
            "total_steps": self.metrics.total_steps,
            "avg_reward_all": self.metrics.avg_reward(),
            "avg_reward_first_10": self.metrics.avg_reward(window=None) if len(self.metrics.total_rewards) < 10 else float(np.mean(self.metrics.total_rewards[:10])),
            "avg_reward_last_10": self.metrics.avg_reward(window=10),
            "phase_reached_distribution": self.metrics.phase_reached,
            "command_families_used": dict(sorted(
                self.metrics.command_families_used.items(),
                key=lambda x: x[1], reverse=True,
            )[:20]),
            "mentor_calls_total": self.metrics.mentor_calls_total,
            "mentor_calls_by_phase": self.metrics.mentor_calls_by_phase,
            "openai_fallback_calls": self.metrics.openai_fallback_calls,
            "teacher_override_count": self.metrics.teacher_override_count,
            "echo_fallback_count": self.metrics.echo_fallback_count,
            "mentor_json_failures": self.metrics.mentor_json_failures,
            "ppo_updates": self.metrics.ppo_updates,
            "discoveries_total": self.metrics.discoveries_total,
            "checkpoints": self.metrics.checkpoints_saved,
            "anneal_history_sample": self.metrics.anneal_history[::max(1, len(self.metrics.anneal_history) // 10)],
            "anneal_enforcement": self._anneal_enforcer.summary() if self._anneal_enforcer else [],
            "codex_calls": self.metrics.codex_calls,
            "codex_tokens": self.metrics.codex_tokens,
            "codex_cost_usd": self.metrics.codex_cost_usd,
            "codex_escalation_reasons": self.metrics.codex_escalation_reasons,
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Run report saved: %s", report_path)

    def _print_report(self) -> None:
        """Print rich terminal report."""
        m = self.metrics

        # ── ASCII Architecture Diagram ───────────────────────────
        diagram = (
            "┌──────────────────────────────────────────────────────────────────┐\n"
            "│              H200 DISTILLATION ARCHITECTURE (steered)           │\n"
            "├──────────────────────────────────────────────────────────────────┤\n"
            "│                                                                  │\n"
            "│   ┌────────────────┐       ┌──────────────────────┐             │\n"
            "│   │  vLLM Mentor   │──────▶│   PPO Training Loop  │             │\n"
            "│   │  (port 8192)   │ JSON  │  ┌────────────────┐  │             │\n"
            "│   │  gpt-oss-120b  │ only  │  │ Actor-Critic   │  │             │\n"
            "│   │  MXFP4 (70%GPU)│──────▶│  │ BC+KL+Ranking  │  │             │\n"
            "│   └────────────────┘ teach │  └────────────────┘  │             │\n"
            "│         ▲          override│  ┌────────────────┐  │             │\n"
            "│         │ fallback  ┌──────│  │ CyberEnviron   │  │             │\n"
            "│   ┌─────┴────────┐  │      │  │ (kill chain)   │  │             │\n"
            "│   │  OpenAI API  │  │      │  └────────────────┘  │             │\n"
            "│   │ (hard cases) │  │      └──────────┬───────────┘             │\n"
            "│   └──────────────┘  │                 │                         │\n"
            "│                     │ STEERING         │                         │\n"
            "│   ┌─────────────────▼──────────────┐  │ checkpoints             │\n"
            "│   │ • teacher_override = ON         │  │                         │\n"
            "│   │ • echo_ban = ON (→ real cmd)    │  │                         │\n"
            "│   │ • anneal_enforce = ON (±10%)    │  │                         │\n"
            "│   │ • JSON-only mentor contract     │  │                         │\n"
            "│   └────────────────────────────────┘  │                         │\n"
            "│         ┌─────────────────────────────┼──────────────┐          │\n"
            "│         ▼                             ▼              ▼          │\n"
            "│   ┌───────────┐               ┌────────────┐  ┌──────────┐     │\n"
            "│   │  rsync    │─── pull ──▶   │  Local git  │  │ auto-pull│     │\n"
            "│   │ (10m loop)│               │  commit+push│  │ to laptop│     │\n"
            "│   └───────────┘               └────────────┘  └──────────┘     │\n"
            "└──────────────────────────────────────────────────────────────────┘\n"
        )
        console.print(Panel(diagram, title="Architecture", border_style="blue"))

        # ── Results Table ────────────────────────────────────────
        table = Table(title="Distillation Run Results", show_lines=True)
        table.add_column("Metric", style="cyan", min_width=30)
        table.add_column("Value", style="green", min_width=25)

        table.add_row("Episodes completed", str(m.episodes_completed))
        table.add_row("Total steps", str(m.total_steps))
        table.add_row("Wall time (hours)", f"{m.wall_hours:.2f}")
        table.add_row("───────────", "───────────")

        # Reward breakdown
        n = len(m.total_rewards)
        third = max(1, n // 3)
        if n >= 3:
            r_start = float(np.mean(m.total_rewards[:third]))
            r_mid = float(np.mean(m.total_rewards[third:2*third]))
            r_end = float(np.mean(m.total_rewards[2*third:]))
        else:
            r_start = r_mid = r_end = m.avg_reward()
        table.add_row("Avg reward (start)", f"{r_start:+.2f}")
        table.add_row("Avg reward (mid)", f"{r_mid:+.2f}")
        table.add_row("Avg reward (end)", f"{r_end:+.2f}")
        table.add_row("Avg reward (all)", f"{m.avg_reward():+.2f}")
        table.add_row("───────────", "───────────")

        # Phase distribution
        for phase in PHASES:
            count = m.phase_reached.get(phase, 0)
            if count > 0:
                pct = 100.0 * count / max(1, m.episodes_completed)
                table.add_row(f"Phase: {phase}", f"{count} ({pct:.1f}%)")

        table.add_row("───────────", "───────────")

        # Command families (top 10)
        sorted_fams = sorted(m.command_families_used.items(), key=lambda x: x[1], reverse=True)[:10]
        fam_str = ", ".join(f"{k}={v}" for k, v in sorted_fams)
        table.add_row("Command families (top 10)", fam_str)
        table.add_row("Unique families", str(len(m.command_families_used)))

        table.add_row("───────────", "───────────")

        # Mentor stats
        table.add_row("Mentor calls total", str(m.mentor_calls_total))
        table.add_row("Teacher overrides", str(m.teacher_override_count))
        table.add_row("Echo bans", str(m.echo_fallback_count))
        table.add_row("Mentor JSON failures", str(m.mentor_json_failures))
        table.add_row("OpenAI fallback calls", str(m.openai_fallback_calls))
        mentor_phase_str = ", ".join(f"{k}={v}" for k, v in m.mentor_calls_by_phase.items())
        table.add_row("Mentor by phase", mentor_phase_str or "none")

        # Codex stats
        if m.codex_calls > 0:
            table.add_row("───────────", "───────────")
            table.add_row("Codex calls", str(m.codex_calls))
            table.add_row("Codex tokens", str(m.codex_tokens))
            table.add_row("Codex cost", f"${m.codex_cost_usd:.3f} / ${m.codex_budget_usd:.2f}")
            table.add_row("Codex successes", str(m.codex_successes))
            esc_str = ", ".join(f"{k}={v}" for k, v in m.codex_escalation_reasons.items())
            table.add_row("Codex escalation reasons", esc_str or "none")

        table.add_row("───────────", "───────────")

        # Training stats
        table.add_row("PPO updates", str(m.ppo_updates))
        table.add_row("Discoveries", str(m.discoveries_total))
        table.add_row("Checkpoints saved", str(len(m.checkpoints_saved)))

        # Checkpoint list
        for cp in m.checkpoints_saved[-5:]:
            cp_path = Path(cp)
            size_mb = cp_path.stat().st_size / (1024 * 1024) if cp_path.exists() else 0
            table.add_row(f"  {cp_path.name}", f"{size_mb:.1f} MB")

        console.print(table)

        # ── Anneal Curve ─────────────────────────────────────────
        if m.anneal_history:
            anneal_table = Table(title="Anneal Curve (sampled)", show_lines=True)
            anneal_table.add_column("Episode", style="cyan")
            anneal_table.add_column("Progress", style="yellow")
            anneal_table.add_column("Phase", style="magenta")
            anneal_table.add_column("BC Coef", style="green")
            anneal_table.add_column("KL Coef", style="green")
            anneal_table.add_column("Prior α", style="green")
            anneal_table.add_column("Mentor Calls", style="blue")

            sample_step = max(1, len(m.anneal_history) // 10)
            for entry in m.anneal_history[::sample_step]:
                anneal_table.add_row(
                    str(entry.get("episode", "?")),
                    f"{entry.get('progress', 0) * 100:.1f}%",
                    str(entry.get("phase", "?")),
                    f"{entry.get('bc_coef', 0):.5f}",
                    f"{entry.get('kl_coef', 0):.5f}",
                    f"{entry.get('prior_alpha', 0):.4f}",
                    str(entry.get("mentor_calls", 0)),
                )
            console.print(anneal_table)

        # ── What Changed ─────────────────────────────────────────
        changes_table = Table(title="What Changed", show_lines=True)
        changes_table.add_column("Item", style="cyan", min_width=35)
        changes_table.add_column("Status", style="green")

        changes_table.add_row("scripts/h200_run_distill_3h.py", "CREATED — distillation orchestrator")
        changes_table.add_row("scripts/h200_gpu_bootstrap.sh", "CREATED — GPU setup automation")
        changes_table.add_row("pull_from_gpu.sh", "UPDATED — local auto-pull + git push")
        changes_table.add_row(".gitignore", "UPDATED — allow models/distilled/*.pt")
        changes_table.add_row("Steering: teacher_override", "ON — mentor command overrides PPO action")
        changes_table.add_row("Steering: echo_ban", "ON — echo commands replaced with real cmds")
        changes_table.add_row("Steering: anneal_enforce", "ON — ±10% rate deviation warnings")
        changes_table.add_row("Steering: JSON-only contract", "ON — strict schema, retry once")
        changes_table.add_row("Mentor model", "gpt-oss-120b (MXFP4) or Qwen2.5-72B AWQ")
        changes_table.add_row("Verbosity", "Per-episode rich table + mini-dashboard every 5 ep")
        changes_table.add_row("OpenAI gpt-5.2-codex", "$5.60 budget, front-loaded anneal")
        changes_table.add_row("core/ modifications", "NONE — all new code in scripts/")
        changes_table.add_row("PPO hyperparameters", "UNCHANGED")

        console.print(changes_table)

        # ── Next Actions ─────────────────────────────────────────
        next_actions = (
            "1. Review checkpoints in models/distilled/\n"
            "2. Run eval: python -m scripts.h200_run_distill_3h --eval-only --episodes 50 --no-mentor\n"
            "3. Compare reward curves (start vs end)\n"
            "4. If reward improved: git pull on laptop, integrate best checkpoint\n"
            "5. Run full test suite: pytest tests/ -x -q\n"
            "6. Plan Phase 42: integrate distilled weights into live pipeline"
        )
        console.print(Panel(next_actions, title="Next Actions", border_style="green"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_time_interval(s: str) -> float:
    """Parse time interval like '10m', '30s', '1h' to seconds."""
    s = s.strip().lower()
    if s.endswith("m"):
        return float(s[:-1]) * 60
    elif s.endswith("h"):
        return float(s[:-1]) * 3600
    elif s.endswith("s"):
        return float(s[:-1])
    return float(s)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="H200 Online Distillation — 3+ hour PPO training with local mentor",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--max-hours", type=float, default=DEFAULT_MAX_HOURS, help="Max runtime in hours")
    parser.add_argument("--max-episodes", type=int, default=DEFAULT_EPISODES, help="Max episodes")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS_PER_EPISODE, help="Max steps per episode")
    parser.add_argument("--checkpoint-every", type=str, default="10m", help="Checkpoint interval (e.g., 10m, 30s, 1h)")
    parser.add_argument("--eval-only", action="store_true", help="Evaluation mode (no training)")
    parser.add_argument("--no-mentor", action="store_true", help="Disable mentor queries")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--device", type=str, default="auto", help="torch device (auto/cuda/cpu)")
    parser.add_argument("--episodes", type=int, default=None, help="Alias for --max-episodes (eval mode)")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.episodes is not None:
        args.max_episodes = args.episodes

    checkpoint_sec = _parse_time_interval(args.checkpoint_every)

    runner = H200DistillationRunner(
        seed=args.seed,
        max_hours=args.max_hours,
        max_episodes=args.max_episodes,
        max_steps_per_episode=args.max_steps,
        checkpoint_interval_sec=checkpoint_sec,
        eval_only=args.eval_only,
        no_mentor=args.no_mentor,
        resume_from=args.resume,
        device=args.device,
    )

    metrics = runner.run()

    # Exit code based on success
    if metrics.episodes_completed > 0:
        console.print(f"\n[green]✓ Distillation complete: {metrics.episodes_completed} episodes in {metrics.wall_hours:.2f}h[/green]")
        sys.exit(0)
    else:
        console.print("\n[red]✗ No episodes completed[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
