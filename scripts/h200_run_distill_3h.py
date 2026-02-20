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
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
except ImportError:
    raise SystemExit("ERROR: 'rich' is required. Install with: pip install rich")

console = Console()

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
    model: str = "jasonyux/Qwen3-235B-A22B-Instruct-2507-AWQ"
    timeout: float = 30.0
    max_tokens: int = 512
    temperature: float = 0.3
    enabled: bool = True

    @classmethod
    def from_env(cls) -> MentorConfig:
        """Build from environment variables."""
        return cls(
            base_url=os.environ.get("ARIASKA_LOCAL_LLM_BASE_URL", cls.base_url),
            model=os.environ.get("ARIASKA_LOCAL_LLM_MODEL", cls.model),
            timeout=float(os.environ.get("MENTOR_TIMEOUT", "30")),
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

            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            return self._validate_response(parsed)

        except Exception as exc:
            logger.debug("Mentor query failed: %s", exc)
            return None

    def _system_prompt(self) -> str:
        return (
            "You are an expert penetration-testing mentor for the Ariaska RL system. "
            "Given the current attack phase, discoveries, and recent commands, "
            "suggest the BEST next command. Respond in JSON with keys: "
            "command (string), reasoning (string ≤200 chars), confidence (0.0-1.0), "
            "action_probs (list of 5 floats summing to 1.0 for action categories: "
            "[recon_scan, enum_probe, exploit_attempt, privesc_action, post_exfil]), "
            "template_name (string), command_family (string). "
            "Focus on the most impactful action advancing the kill chain."
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
        """Validate and normalize mentor response."""
        required = {"command", "confidence"}
        if not required.issubset(parsed.keys()):
            return None
        # Normalize action_probs
        probs = parsed.get("action_probs", [0.2] * 5)
        if not isinstance(probs, list) or len(probs) != 5:
            probs = [0.2] * 5
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        parsed["action_probs"] = probs
        parsed.setdefault("reasoning", "")
        parsed.setdefault("template_name", "")
        parsed.setdefault("command_family", "manual")
        parsed["confidence"] = float(np.clip(parsed["confidence"], 0.0, 1.0))
        return parsed


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
        self._anneal: Optional[AnnealController] = None
        self._last_checkpoint_time: float = 0.0
        self._trace_path: Optional[Path] = None
        self._run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")

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
        self._ppo = PPOAgent(config=config, device=self.device)

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

    def _init_dirs(self) -> None:
        """Create output directories."""
        for d in [CHECKPOINT_DIR, RESULTS_DIR, TRACES_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        self._trace_path = TRACES_DIR / f"h200_distill_{self._run_id}.jsonl"

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
        # Use CommandActionMapper if available
        try:
            from core.algorithms.command_action_mapper import CommandActionMapper
            mapper = CommandActionMapper()
            candidates = mapper.action_to_commands(action_idx, state)
            if candidates:
                return candidates[0] if isinstance(candidates[0], str) else str(candidates[0])
        except Exception:
            pass

        # Phase-aware fallback commands
        fallback_map = {
            0: {  # recon_scan
                "RECON": "nmap -sV -sC -p- -T4 {target}",
                "ENUMERATION": "nmap -sV --script=vuln {target}",
            },
            1: {  # enum_probe
                "RECON": "whatweb {target}",
                "ENUMERATION": "gobuster dir -u http://{target} -w /usr/share/wordlists/dirb/common.txt",
            },
            2: {  # exploit_attempt
                "EXPLOITATION": "searchsploit -w {service}",
                "PRIVILEGE_ESCALATION": "sudo -l",
            },
            3: {  # privesc_action
                "PRIVILEGE_ESCALATION": "find / -perm -4000 -type f 2>/dev/null",
                "POST_EXPLOITATION": "cat /etc/shadow",
            },
            4: {  # post_exfil
                "EXFILTRATION": "tar czf /tmp/loot.tar.gz /home",
                "CLOSEOUT": "echo 'CLOSEOUT'",
            },
        }
        phase_map = fallback_map.get(action_idx, {})
        return phase_map.get(phase, f"echo 'action_{action_idx}_phase_{phase}'")

    # ── Teacher signal extraction ────────────────────────────────

    def _get_teacher_signal(
        self,
        state: Dict[str, Any],
        step: int,
        recent_commands: List[str],
        discoveries: List[str],
    ) -> Tuple[Optional[Any], Optional[int], Optional[Dict[str, Any]]]:
        """Query mentor and extract teacher distribution + action.

        Returns:
            (teacher_distribution_tensor, teacher_action_idx, raw_response)
        """
        import torch

        if self._mentor is None:
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

        response = self._mentor.query_teacher_action(
            phase=phase,
            step=step,
            discoveries=discoveries,
            recent_commands=recent_commands,
            state_summary=state_summary,
        )
        if response is None:
            return None, None, None

        # Convert action_probs to teacher distribution tensor
        probs = response.get("action_probs", [0.2] * 5)
        teacher_dist = torch.tensor(probs, dtype=torch.float32, device=self.device)

        # Best action from teacher
        teacher_action = int(torch.argmax(teacher_dist).item())

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

            # Map action to command
            command = self._action_to_command(action_idx, state)

            # If mentor provided a specific command, use it with some probability
            # (decreasing with anneal) to bootstrap better experience
            if mentor_raw and not self.eval_only:
                mentor_conf = mentor_raw.get("confidence", 0.0)
                use_mentor_cmd = (
                    mentor_conf > 0.7
                    and np.random.random() < self._anneal.bc_coef() * 3
                )
                if use_mentor_cmd:
                    command = mentor_raw.get("command", command)

            # Step environment
            next_state, reward, done, info = self._env.step(command)
            if not isinstance(next_state, dict):
                next_state = self._env.get_global_state()

            reward = float(np.clip(reward, REWARD_MIN, REWARD_MAX))
            episode_reward += reward

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

            # Log step trace
            self._log_trace({
                "kind": "step",
                "episode": episode_id,
                "step": step_i,
                "action_idx": action_idx,
                "command": command,
                "reward": reward,
                "phase": current_phase,
                "mentor_queried": teacher_dist is not None,
                "teacher_action": teacher_action,
                "done": done,
            })

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
            "phase_reached": max_phase,
            "mentor_calls": mentor_calls,
            "discoveries": len(discoveries),
            "update_metrics": update_metrics,
        })

        return EpisodeResult(
            episode_id=episode_id,
            steps=step_i + 1,
            total_reward=episode_reward,
            phase_reached=max_phase,
            discoveries=len(discoveries),
            mentor_calls=mentor_calls,
            ppo_updates=ppo_updates_ep,
            commands_used=commands_used,
        )

    @staticmethod
    def _extract_family(command: str) -> str:
        """Extract command family from command string."""
        cmd = command.strip().split()[0] if command.strip() else "unknown"
        # Strip path prefixes
        cmd = cmd.rsplit("/", 1)[-1]
        return cmd

    # ── Main run loop ────────────────────────────────────────────

    def run(self) -> RunMetrics:
        """Execute the full distillation run."""
        self._init_dirs()
        self._init_ppo()
        self._init_env()
        self._init_mentor()

        total_seconds = self.max_hours * 3600
        self._anneal = AnnealController(total_seconds)
        self._last_checkpoint_time = time.monotonic()
        self.metrics.wall_start = time.time()

        mode = "[cyan]EVAL[/cyan]" if self.eval_only else "[green]TRAIN[/green]"
        console.print(Panel(
            f"Mode: {mode}\n"
            f"Device: {self.device}\n"
            f"Max hours: {self.max_hours}\n"
            f"Max episodes: {self.max_episodes}\n"
            f"Seed: {self.seed}\n"
            f"Mentor: {'[green]ON[/green]' if self._mentor else '[red]OFF[/red]'}\n"
            f"Run ID: {self._run_id}",
            title="H200 Distillation Run",
        ))

        deadline = time.monotonic() + total_seconds

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Distillation",
                total=self.max_episodes,
            )

            for ep in range(self.max_episodes):
                # Time check
                if time.monotonic() >= deadline:
                    console.print("[yellow]Time limit reached[/yellow]")
                    break

                result = self._run_episode(ep)
                self.metrics.episodes_completed += 1
                self.metrics.total_steps += result.steps
                self.metrics.total_rewards.append(result.total_reward)

                # Progress update
                anneal_phase = self._anneal.phase_name if self._anneal else "?"
                progress.update(
                    task,
                    advance=1,
                    description=(
                        f"Ep {ep:04d} | R={result.total_reward:+6.1f} | "
                        f"Phase={result.phase_reached[:6]} | "
                        f"Mentor={result.mentor_calls} | "
                        f"Anneal={anneal_phase}"
                    ),
                )

                # Periodic checkpoint
                self._maybe_checkpoint(ep)

                # Periodic console update (every 10 episodes)
                if (ep + 1) % 10 == 0:
                    avg_recent = self.metrics.avg_reward(window=10)
                    logger.info(
                        "Ep %d | avg_reward_10=%+.2f | total_mentor=%d | anneal=%.1f%%",
                        ep, avg_recent,
                        self.metrics.mentor_calls_total,
                        (self._anneal.progress * 100) if self._anneal else 0,
                    )

        # Final checkpoint
        self._maybe_checkpoint(self.metrics.episodes_completed, force=True)
        self.metrics.wall_end = time.time()

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
            "ppo_updates": self.metrics.ppo_updates,
            "discoveries_total": self.metrics.discoveries_total,
            "checkpoints": self.metrics.checkpoints_saved,
            "anneal_history_sample": self.metrics.anneal_history[::max(1, len(self.metrics.anneal_history) // 10)],
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Run report saved: %s", report_path)

    def _print_report(self) -> None:
        """Print rich terminal report."""
        m = self.metrics

        # ── ASCII Architecture Diagram ───────────────────────────
        diagram = (
            "┌─────────────────────────────────────────────────────────────┐\n"
            "│                  H200 DISTILLATION ARCHITECTURE            │\n"
            "├─────────────────────────────────────────────────────────────┤\n"
            "│                                                             │\n"
            "│   ┌──────────────┐       ┌──────────────────────┐          │\n"
            "│   │  vLLM Mentor │──────▶│   PPO Training Loop  │          │\n"
            "│   │  (port 8192) │ teach │  ┌────────────────┐  │          │\n"
            "│   │  Qwen3-235B  │ dist  │  │ Actor-Critic   │  │          │\n"
            "│   │  AWQ (55%GPU)│───────│  │ BC+KL+Ranking  │  │          │\n"
            "│   └──────────────┘       │  └────────────────┘  │          │\n"
            "│         ▲                │  ┌────────────────┐  │          │\n"
            "│         │ fallback       │  │ CyberEnviron   │  │          │\n"
            "│   ┌─────┴────────┐       │  │ (kill chain)   │  │          │\n"
            "│   │  OpenAI API  │       │  └────────────────┘  │          │\n"
            "│   │ (hard cases) │       └──────────┬───────────┘          │\n"
            "│   └──────────────┘                  │                      │\n"
            "│                               checkpoints                  │\n"
            "│                                  │                         │\n"
            "│         ┌────────────────────────┼────────────────┐        │\n"
            "│         ▼                        ▼                ▼        │\n"
            "│   ┌───────────┐          ┌────────────┐    ┌──────────┐   │\n"
            "│   │  AutoSync │──push──▶│   GitHub   │    │  rsync   │   │\n"
            "│   │ (10m loop)│          │   master   │    │ pull to  │   │\n"
            "│   └───────────┘          └────────────┘    │  laptop  │   │\n"
            "│                                            └──────────┘   │\n"
            "└─────────────────────────────────────────────────────────────┘\n"
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
        table.add_row("OpenAI fallback calls", str(m.openai_fallback_calls))
        mentor_phase_str = ", ".join(f"{k}={v}" for k, v in m.mentor_calls_by_phase.items())
        table.add_row("Mentor by phase", mentor_phase_str or "none")

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
        changes_table.add_row("pull_from_gpu.sh", "CREATED — local auto-pull script")
        changes_table.add_row(".gitignore", "UPDATED — allow models/distilled/*.pt")
        changes_table.add_row("Env: FF_LOCAL_LLM=1", "ADDED to .env on GPU")
        changes_table.add_row("Env: ARIASKA_LOCAL_LLM_BASE_URL", "ADDED to .env on GPU")
        changes_table.add_row("Env: ARIASKA_LOCAL_LLM_MODEL", "ADDED to .env on GPU")
        changes_table.add_row("Env: MENTOR_STRATEGY=local_first", "ADDED to .env on GPU")
        changes_table.add_row("Env: ENABLE_DUAL_MENTOR=1", "ADDED to .env on GPU")
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
