# Ariaska_RL — Copilot System Prompt

> **Read this entire file before generating ANY code for this project.**

---

## ⚠️ CLAUDE CODE GATE — READ BEFORE ANYTHING ELSE

**You are NOT working alone. Claude Code (`claude` CLI) is your senior partner.**

| Situation | You MUST |
|-----------|----------|
| Starting ANY new task | `cc_sync.sh start` → THEN write your TODO list with `cc_sync.sh todos` |
| Touching 2+ files | Fire `claude` audit async BEFORE writing one line |
| Planning a new module/class | Get Claude architecture audit first |
| Unsure about ANYTHING in the codebase | `cc_sync.sh audit` — never assume, never guess |
| Reviewing your own implementation plan | Have Claude review it before you execute |
| Completing each TODO step | `cc_sync.sh checkpoint "<step N done: summary>"` |
| Context window filling up | `cc_sync.sh checkpoint --ask-claude "<done so far, remaining TODOs>"` IMMEDIATELY |
| About to re-read a file you already tried | STOP — you're in a loop. Call `cc_sync.sh audit` |
| Going in circles / unsure what to do next | `cc_sync.sh checkpoint --ask-claude "<stuck on X because Y>"` |

```bash
# Full workflow — no exceptions:
./scripts/cc_sync.sh start "<describe your task>"
./scripts/cc_sync.sh todos "1. step one\n2. step two\n3. step three"
# ... implement step 1 ...
./scripts/cc_sync.sh checkpoint "step 1 done: <what you did>"
# ... implement step 2 ...
./scripts/cc_sync.sh checkpoint "step 2 done: <what you did>"
./scripts/cc_sync.sh done "<full summary>"

# Mid-task: context getting full or stuck?
./scripts/cc_sync.sh checkpoint --ask-claude "completed: X. remaining: Y, Z. stuck because: <reason>"
```

**Violating this gate = wasted context + broken invariants + invariant-violating commits caught by pre-commit hook.**

### Anti-Hallucination Rules

These are HARD RULES. Violating them causes the circular loops and incomplete work the user has complained about.

1. **NEVER assume what a function/class does without reading it.** If you haven't read it this session, `cc_sync.sh audit "read and explain <path>:<line>"`.
2. **NEVER implement based on guessed file structure.** If your context is too full to read, `checkpoint --ask-claude`.
3. **NEVER re-implement something you already wrote.** Check the session and your recent edits first.
4. **NEVER stop mid-task without calling `checkpoint`.** If you must stop (context full, uncertain), record exactly where you are so Claude can guide the next step.
5. **If you've tried the same fix twice and it still doesn't work** — STOP. `cc_sync.sh audit "tried X twice, still failing because Y"`.

The `cc_sync.sh` script manages the full lifecycle. Claude Code reads `.claude/CLAUDE.md` for its instructions. Both agents share `.github/CC_SESSION.md` as the live conversation.

See the full workflow: jump to **Claude Code Integration — Dual-Agent Workflow** section below.

---

## Identity

**Ariaska_RL** — Autonomous multi-agent reinforcement learning system for live authorized penetration testing. 5 specialized agents, tiered GPT-5.2 hybrid decision pipeline, PPO Actor-Critic v3.0 primary RL, 107K v2 knowledge corpus, ~154K lines of Python across 340 modules.

**Author:** Filip Volf
**Python:** 3.11+ (developed on 3.13)
**Entry:** `ariaska_cli.py` → `SmartOrchestrator.run_training()`
**Tests:** 2,938 collected, 2,938 passing (165 test files)
**Current Phase:** Phase 53+ — Local LLM Migration + Accelerated LLM→RL Handoff

---

## ARCHITECTURAL INVARIANTS — Read First

These are non-negotiable. Violation of any invariant breaks the system.

1. **ALL LLM calls go through `GPTManager`** — never `import openai` directly
2. **Lazy imports inside methods** — this project has deep circular dependency chains
3. **Single `GPTManager` instance** — inject via constructor, never instantiate new ones
4. **`rich` for output** — `Console()`, `Panel()`, `Table()` — never bare `print()`
5. **`logging.getLogger("ariaska.<module>")`** — for all debug/info
6. **`FakeGPTManager`** in tests — deterministic, no API calls
7. **`StubToolRunner`** in tests — tracks commands without execution
8. **`ARIASKA_DRY_RUN=1`** — env var for safe test execution
9. **STATE_DIM = 512** — hardcoded everywhere, change requires full network rebuild
10. **Never modify PPO core** without explicit instruction — PPO is finely tuned (R80+)
11. **BudgetManagerV2 clamps** — 1,148,850 max, 574,425 min (50% floor). Do not change without instruction.
12. **Evidence Gate default = `enforce`** — `FF_STRICT_EXPLOIT_GATE` must stay `enforce` unless explicitly changed
13. **MicroChain escalation threshold = 0.40** — tunable via `MC_ESCALATE_THRESHOLD` env var only
14. **Git policy: `master` branch only** — no feature branches, no PRs
15. **Test baseline: 2,938 tests** — all must pass after any change
16. **Claude Code is REQUIRED for multi-file work** — fire `claude` CLI audit before touching 2+ files, before planning new modules, before any refactor. Context window is not an excuse to skip this.

---

## PROHIBITED ACTIONS

- **Never `import openai` directly** — always go through `GPTManager`
- **Never create a new `GPTManager()` instance** — inject the existing one
- **Never top-level import between core modules** — use lazy imports
- **Never hardcode API keys** — `.env` + `python-dotenv`
- **Never execute real commands in tests** — `StubToolRunner`
- **Never modify `STATE_DIM`** without rebuilding all network architectures
- **Never use `print()`** — use `rich.Console()` or `logging`
- **Never create feature branches** — `master` only
- **Never modify PPO hyperparameters** without explicit instruction
- **Never change the BudgetManagerV2 constants** (_TOTAL_BUDGET, _MIN_BUDGET) without instruction

---

## File Map — Where Things Live

```
ariaska_cli.py                              # CLI entry: smart-train, replay, watchdog, status, help
core/
├── agents/                                 # 5 agents (all implement AgentInterface + MemorySyncInterface)
│   ├── red_agent.py                        # RedAgent — offensive exploitation
│   ├── blue_agent.py                       # BlueAgent — defensive/reactive
│   ├── scout_agent.py                      # ScoutAgent — recon/discovery
│   ├── shadow_agent.py                     # ShadowAgent — stealth/alert prevention
│   ├── orion_agent.py                      # OrionAgent — strategic coordination
│   ├── enhanced_agent_base.py              # EnhancedAgentBase mixin
│   └── redagent_brain.py                   # RedAgent brain (extended cognition)
├── algorithms/
│   ├── ppo_agent.py              (1735L)   # PPOAgent — primary RL, Actor-Critic v3.0 (R68-R80)
│   ├── command_action_mapper.py            # PPO action_idx <-> CommandTemplate mapping
│   ├── replay_buffer.py                    # PER with dedup
│   ├── sac_agent.py                        # Soft Actor-Critic (alternative)
│   ├── ddqn_macro.py                       # Double DQN macro-actions
│   ├── rnd_curiosity.py                    # RND intrinsic motivation
│   └── cognition_node.py                   # Cognitive architecture node
├── commands/
│   ├── command_registry.py       (4560L)   # 294 CommandTemplates by AttackPhase
│   ├── command_enrichment.py               # Enrich commands with knowledge
│   └── learned_commands.py                 # Runtime-discovered command store
├── environment/
│   ├── cyber_environment.py      (2879L)   # CyberEnvironment — kill chain state machine
│   ├── metasploitable_handler.py           # MS2/MS3 Docker integration
│   └── context_detector.py                 # Environment context detection
├── execution/
│   ├── parser_broker.py          (291L)    # 4-stage parser: regex→SOP→Venice→GPT
│   ├── output_parser.py                    # Smart output parsing
│   ├── discovery_event.py                  # DiscoveryEvent schema
│   ├── live_executor.py                    # Real command execution
│   ├── sandboxed_executor.py               # Sandboxed execution
│   └── step_parse_cache.py                 # Parser result caching
├── interfaces/
│   ├── agent_interface.py                  # AgentInterface — required contract
│   └── memory_sync_interface.py            # MemorySyncInterface — cross-agent memory
├── knowledge/
│   ├── knowledge_candidate_v2.py (330L)    # KnowledgeCandidate dataclass (v2 schema, 14 nested)
│   ├── knowledge_query.py                  # Query interface for knowledge indices
│   ├── knowledge_packs.py       (2468L)    # Pre-built knowledge packs
│   ├── target_profiler.py       (1422L)    # Target profile resolution
│   ├── kg_manager.py                       # Knowledge graph manager
│   ├── pentesting_playbooks.py             # Playbook -> PlaybookStep chains
│   ├── ms2_exploit_graph.py                # Metasploitable 2 exploit graph
│   └── ms3_exploit_graph.py                # Metasploitable 3 exploit graph
├── llm/
│   ├── micro_chain.py            (781L)    # Phase 27: MicroChain nano→mini→nano 3-stage scoring
│   ├── phase_guided_llm.py       (818L)    # Phase 34: PhaseGuidedLLM self-guidance + distillation
│   ├── llm_policy_bridge.py      (774L)    # Phase 37: Level 5 GPT↔RL bridge (prior, features, teacher, anneal)
│   ├── mentor_trace.py           (210L)    # Phase 30: MentorTrace for structured distillation
│   ├── smart_mentor.py          (1366L)    # SmartMentor + DualMentor (GPT + Venice)
│   ├── reward_calculator.py      (906L)    # SmartRewardCalculator — shaped rewards
│   ├── budget_manager.py         (605L)    # Phase 17+38: BudgetManagerV2 with dynamic scaling
│   ├── cloud_roles.py                      # Cloud role definitions for LLM
│   ├── venice_reasoning.py                 # Venice AI reasoning integration
│   ├── codex_personas.py                   # LLM persona definitions
│   ├── reflective_cortex.py               # Self-reflective LLM reasoning
│   ├── local_llm_provider.py              # Phase 43: Local LLM server lifecycle (llama-cpp/vLLM)
│   └── model_router.py                    # Phase 43/44: Tier-based routing (local vs OpenAI)
├── cortex/
│   ├── executive_cortex.py                 # Executive decision making
│   └── tactical_cortex.py                  # Tactical decision making (7-rule quality gate)
├── memory/
│   ├── hybrid_memory.py                    # Hybrid memory (short+long term)
│   ├── enhanced_memory_sync.py             # Cross-agent memory fusion (threaded)
│   ├── unified_cognitive_bus.py            # Unified cognitive event bus
│   ├── campaign_memory.py                  # Campaign-level persistent memory
│   └── chroma_memory_store.py              # ChromaDB vector memory
├── models/
│   ├── state_encoder.py          (698L)    # 512-dim state encoder
│   ├── policy_net.py                       # Dueling DQN PolicyNet (legacy)
│   ├── advanced_networks.py                # Attention + residual + NoisyLinear
│   └── value_net.py                        # Value network
├── multiagent/
│   ├── agent_manager.py                    # Creates and coordinates all 5 agents
│   ├── agents.py                           # Factory: get_all_agents(), create_agent()
│   ├── memory_router.py                    # PER with SumTree, SQLite persistence
│   └── strategic_directive.py              # Cross-agent directive protocol
├── orchestration/
│   ├── smart_orchestrator.py    (7614L)    # Main training loop — THE BRAIN
│   └── orchestrator.py                     # Base orchestrator
├── training/
│   ├── smart_coach.py           (8403L)    # Per-agent hybrid decision pipeline (LARGEST FILE)
│   ├── mentor_policy.py                    # Annealing mentor call rates
│   ├── mentor_controller.py                # MentorController + MentorEngagement + MentorTier
│   └── ariaska_trainer.py                  # High-level training config
├── reasoning/
│   └── teacher_trace.py          (238L)    # TeacherTrace + BCBuffer (capacity 2000, dedup)
├── replay/
│   └── episode_replayer.py       (268L)    # Phase 31: JSONL trace replay
├── tracing/
│   ├── episode_trace.py          (700L)    # Episode trace writer
│   ├── event_bus.py              (403L)    # EventBus: pub/sub step/agent events
│   └── jsonl_logger.py           (293L)    # JSONL structured logging
├── postmortem/
│   ├── orion_postmortem.py                 # GPT end-of-run analysis
│   └── skill_library.py                    # Persistent SkillCard store
├── observability/
│   └── live_dashboard.py        (2436L)    # LiveDashboard v5.0 — Rich terminal UI
├── ops/                                    # Phase 38-39: Operational subsystems (5,653L total)
│   ├── ops_hub.py                (524L)    # OpsHub — central pre-flight + post-step orchestration
│   ├── orion_rethink.py          (578L)    # Phase 39: Orion deep-rethink escalation
│   ├── trust_weights.py          (357L)    # Phase 39: Trust weight annealing per agent
│   ├── debug_trace.py            (345L)    # Phase 39: Debug trace instrumentation
│   ├── phase_invariants.py       (302L)    # Phase 38: Phase hardening + shell validation
│   ├── ops_state_encoder.py      (349L)    # Phase 38: Ops-aware state encoder extensions
│   ├── engagement_metrics.py     (269L)    # Phase 38: Token flex + engagement metrics
│   ├── discovery_trust.py        (231L)    # Phase 38: Discovery trust engine
│   ├── token_flex.py             (254L)    # Phase 38: Dynamic token budget flex
│   ├── exploit_confidence.py     (240L)    # Exploit confidence scoring
│   ├── exploit_cooldown.py       (200L)    # Exploit cooldown management
│   ├── domain_manager.py         (245L)    # Multi-domain management
│   ├── hosts_manager.py          (188L)    # Target hosts tracking
│   ├── shell_validator.py        (236L)    # Shell session validation
│   ├── command_lockout.py        (198L)    # Phase 38: Anti-repeat + cooldown
│   ├── execution_classifier.py   (193L)    # Execution result classification
│   ├── sudo_handler.py           (260L)    # Sudo privilege handling
│   ├── tool_installer.py         (317L)    # Runtime tool dependency installation
│   └── ops_dashboard_panels.py   (234L)    # Ops dashboard panels for LiveDashboard
├── state/                                  # Phase 35: State management (953L total)
│   ├── canonical_state.py        (208L)    # Single source of truth state
│   ├── coherence_chain.py        (540L)    # 4-step nano coherence micro-chain
│   └── live_trace.py             (205L)    # Append-only JSONL live trace
├── testing/
│   ├── fake_gpt_manager.py                 # FakeGPTManager — deterministic mock
│   └── tool_runner.py                      # StubToolRunner / RealToolRunner + ToolResult
├── telemetry/                              # Unified trace, event telemetry
├── analytics/                              # Coherence, replay analysis, scan randomizer
├── logging/                                # JSONL structured logging
├── gpt_manager.py               (1494L)   # Centralized LLM gateway — ALL LLM calls here
├── runtime_flags.py                        # Global runtime flags (offline, dry_run, etc.)
├── feature_flags.py              (384L)    # 76 feature flags, 3 profiles, env-var overrides
└── ui/                                     # Textual TUI dashboard

data/
├── knowledge_candidates_v2/                # 107,933 entries across 18 JSONL files (~195MB)
├── knowledge_indices/                      # 11 prebuilt indices (~110MB)
├── knowledge_retriever.py                  # Query and retrieve knowledge
├── knowledge_ingestion.py                  # Ingest new knowledge
└── knowledge_loader.py                     # ChromaDB + SentenceTransformer ingestion

scripts/
├── build_knowledge_indices.py              # Rebuild indices from v2 corpus
├── fill_htb_gaps.py                        # Fill HTB readiness gaps
├── dedup_v2_corpus.py                      # Deduplicate corpus
├── htb_readiness.py                        # HTB readiness assessment
└── tool_dependency_audit.py                # Audit tool dependencies

tests/                                      # 165 test files, 2,938 tests total
```

---

## The Five Agents

| Agent | Role | Domain |
|-------|------|--------|
| **ScoutAgent** | `recon` | Network discovery, port scanning, service fingerprinting, version detection |
| **RedAgent** | `offensive` | Exploitation, privesc, exfil. Primary PPO-trained. DQN+GPT hybrid, emergency fallbacks |
| **BlueAgent** | `defensive` | Honeypots, credential resets, firewall, alert management. Reactive |
| **ShadowAgent** | `stealth` | Alert monitoring, scan timing, detection avoidance, action overrides |
| **OrionAgent** | `strategic` | Coordination, strategic reviews, cross-agent directives, phase transitions |

**Activation order** (phase-dependent, in `SmartOrchestrator._should_activate()`):
- RECON: Scout -> Shadow -> Orion -> Red -> Blue
- EXPLOITATION: Red -> Shadow -> Scout -> Orion -> Blue
- EXFILTRATION: Red -> Shadow -> Orion -> Scout -> Blue

**Interface contract** — every agent MUST implement:
```python
from core.interfaces.agent_interface import AgentInterface
from core.interfaces.memory_sync_interface import MemorySyncInterface

class NewAgent(AgentInterface, MemorySyncInterface):
    @property
    def agent_id(self) -> str: ...
    @property
    def role(self) -> str: ...
    def act(self, state: Dict[str, Any]) -> Dict[str, Any]: ...
    def learn(self, state, action, reward, next_state, done) -> float: ...
    def simulate_step(self, episode, step, shared_context) -> Dict[str, Any]: ...
    def sync_memory(self) -> bool: ...
    def reset(self) -> None: ...
```

**Adding a new agent checklist:**
1. Create class in `core/agents/` implementing both interfaces
2. Register in `core/multiagent/agents.py` (`get_all_agents()` + `create_agent()`)
3. Add to `AgentManager._initialize_agents()`
4. Configure activation in `SmartOrchestrator._should_activate()`
5. Create `SmartCoach` with role config in `SmartOrchestrator._init_smart_coaches()`
6. Add role-specific alternatives in `SmartCoach` anti-repeat pool
7. Register emergency fallbacks if the agent takes autonomous actions

---

## Decision Pipeline (SmartCoach)

Each agent's `SmartCoach` (8,403L) selects commands via **4-stage hybrid pipeline**:

```
1. PLAYBOOK    ->  Curriculum-guided chain (anneals 60% -> 10%)
2. PPO         ->  RL policy network (action_dim=5 via CommandActionMapper)
3. REGISTRY    ->  Precondition-matched from CommandRegistry (294 templates)
4. GPT MENTOR  ->  SmartMentor / DualMentor LLM (rate-limited, annealed)
```

**Intelligence layers (applied within/alongside pipeline):**
- **MicroChain** (P27): 3-stage nano->mini->nano scoring with codex escalation
- **PhaseGuidedLLM** (P34): structured JSON guidance + distillation packets
- **Evidence Gate** (P27): exploit command validation (enforce mode by default)
- **TacticalCortex**: 7-rule quality gate before execution
- **Anti-Repeat Guard**: exact + prefix repeat blocking with per-role replacement pool

**Decision source tracking** via `SmartDecisionResult.source`:
`"playbook"` | `"ppo"` | `"registry"` | `"mentor"` | `"dual_mentor"` | `"micro_chain"` | `"phase_guided"` | `"anti_repeat"` | `"fallback"`

---

## PPO — Primary RL Algorithm

**File:** `core/algorithms/ppo_agent.py` (1,543L)

```python
PPOConfig:
    state_dim = 512                    # Must match state_encoder
    action_dim = 5                     # Maps via CommandActionMapper
    hidden_dims = [512, 512, 256]
    clip_epsilon = 0.2                 # Adaptive: 0.15-0.25
    gamma = 0.99
    gae_lambda = 0.97
    learning_rate = 3e-4 -> 1e-5       # KL-adaptive annealing
    epochs_per_update = 4
    minibatch_size = 16
    max_grad_norm = 0.5
    rollout_size = 256
```

**R68-R80 advanced features:**
- Phase-gated actor heads (HRL-lite, 3 groups: recon/exploit/post-exploit)
- Self-Imitation Learning (SIL, 500-entry buffer, coef=0.25)
- Symlog value compression (DreamerV3-style)
- Cosine entropy schedule with rebound
- Prioritized advantage sampling
- Gradient accumulation (2x effective batch)
- KL-adaptive learning rate (target_kl=0.01)
- Per-phase advantage whitening
- EMA target network (tau=0.995) + value-surprise intrinsic bonus (coef=0.3)
- Dual-horizon GAE (lambda=0.97 long + lambda=0.70 short, blend=0.65)
- Spectral normalization on critic
- Soft advantage clipping (tanh-based)
- Auxiliary phase prediction head (coef=0.1)
- Adaptive clip scheduling from rolling clip_fraction

**PPO flow in SmartCoach:**
1. `ppo.select_action(state_tensor)` -> action_idx, log_prob, value
2. `CommandActionMapper.action_to_commands(idx, state)` -> candidates
3. Anti-repeat check -> store in `_ppo_pending`
4. After step -> `ppo.store_transition(s, a, log_prob, reward, value, done)`
5. Episode end -> `ppo.update(last_value)` with GAE-lambda

---

## State Encoder

**File:** `core/models/state_encoder.py` (438L) — produces **512-dim** tensor.

| Dims | Content |
|------|---------|
| 0-11 | Phase one-hot + progress (12) |
| 12-26 | State flags: ports_discovered, shell_obtained, etc. (15) |
| 27-46 | Top 20 port presence indicators (20) |
| 47-58 | Service type presence (12) |
| 59-70 | Numeric: detection_risk, blue_team_alert, etc. (12) |
| 71-80 | Action history encoding (10) |
| 81-85 | LLM/Mentor features (5) |
| 86-90 | Temporal features (5) |
| 91-511 | Reserved zero-padded (421) |

**Rule:** Always use `STATE_DIM = 512` for network input dims.

---

## MicroChain (Phase 27)

**File:** `core/llm/micro_chain.py` (610L)

3-stage iterative LLM scoring chain:

| Stage | Model | Input | Output |
|-------|-------|-------|--------|
| 1 (nano) | gpt-5.2-nano | Phase, discoveries, history | Tactical classification |
| 2 (mini) | gpt-5.2-mini | Classification + context | Candidate commands (JSON, max 3) |
| 3 (nano) | gpt-5.2-nano | Candidates + evidence | Scores: phase_fit, evidence_support, novelty |

**Constants:**
- `ESCALATION_THRESHOLD`: 0.40 (env: `MC_ESCALATE_THRESHOLD`)
- `STAGNATION_ESCALATION_STEPS`: 9
- `STAGNATION_ESCALATION_SCORE`: 0.55
- `_MAX_CANDIDATES`: 3

**Ablation:** `MC_NANO_ABLATION=1` bypasses nano stages 1+3 for A/B testing.

**Output:** `MicroChainResult` with selected `MicroChainCandidate` (command, template_name, reasoning, score, phase_fit, evidence_support, novelty).

---

## PhaseGuidedLLM (Phase 34)

**File:** `core/llm/phase_guided_llm.py` (714L)

Produces structured JSON guidance:
- Phase decision with evidence-driven stay/move conditions
- 3-6 candidate next actions as template picks
- Anomaly probes for stagnation/flag-hunting
- Distillation packet (MentorTrace target)

**Constants:**
- `_CODEX_ESCALATION_CONFIDENCE`: 0.45
- `_STALL_THRESHOLD`: 8
- `_MIN_CANDIDATES`: 3 / `_MAX_CANDIDATES`: 6
- `_PHASE_TAG`: "P34"

**Model routing:** gpt-5.2-mini default; codex escalation when confidence < 0.45, contradictions detected, or semantic stall >= 8 steps.

---

## Distillation Pipeline

```
Mentor call -> MentorTrace (P30) -> TeacherTrace (P14) -> BCSample -> BCBuffer
                                                                          |
                                                                    PPO BC loss

MicroChain + PhaseGuide + Mentor ─┐
                                  ├──> LLMPolicyBridge (P37)
         state + maturity signals ┘         │
                                            ├── action prior → PPO logits
                                            ├── teacher dist → KL loss
                                            ├── LLM features → enhanced state
                                            └── anneal alpha → all weights decay
```

**MentorTrace** (`core/llm/mentor_trace.py`, 210L):
- Decision: command, reasoning (<=512 chars), confidence, alternatives (max 3)
- Context: phase, step, episode, discoveries_at_call, stagnation_steps
- State: 512-dim state_vector for BC
- Quality: actual_reward, produced_discovery, mentor_was_correct

**TeacherTrace** (`core/reasoning/teacher_trace.py`, 238L):
- Teacher: action_idx, command, template, rationale, confidence
- Student: PPO action_idx, command, log_prob, confidence
- Divergence: |teacher - student|
- Post-hoc: student_would_have_succeeded

**BCBuffer** (capacity: 2000, dedup by state_hash+action):
- `store(trace, decay_factor)` -> BCSample
- `sample(batch_size)` -> List[BCSample]
- Thread-safe, dedup hash tracking

---

## Token Budget (Phase 17 + 36)

**File:** `core/llm/budget_manager.py` (605L)

| Constant | Value | Notes |
|----------|-------|-------|
| `_TOTAL_BUDGET` | 1,148,850 | ~$3.83/episode ceiling |
| `_MIN_BUDGET` | 574,425 | 50% floor |
| `_BURST_POOL_RATIO` | 0.12 | 12% of max as burst reserve |
| `_BURST_STEP_CAP_RATIO` | 0.03 | 3% per-step burst limit |
| `_BURST_COOLDOWN_STEPS` | 5 | Min steps between bursts |
| `_BURST_TIERS` | mini, codex | Only these tiers get bursts |

**Tier budgets (Phase 36):**

| Tier | Model(s) | Budget | Share |
|------|----------|--------|-------|
| codex | gpt-5.2-codex | 199,800 | 20% |
| full | gpt-5.2 | 199,800 | 20% |
| mini | gpt-5.2-mini, gpt-5-mini | 299,700 | 30% |
| nano | gpt-5.2-nano, gpt-5-nano | 299,700 | 30% |

**Dynamic scaling:** budget_scale = max(MIN_SCALE, 1.0 - maturity_signal)
where maturity_signal = 0.4*success_rate + 0.3*skill_coverage + 0.2*discovery_efficiency + 0.1*(1-stagnation_rate)

**ROI tags** (15 valid tags): improves_hypothesis_accuracy, reduces_steps_to_foothold, classification, verification, tactical_advice, reward_shaping, etc.

---

## Evidence Gate (Phase 27)

Located in `SmartCoach` (line ~3335). Validates exploit-phase commands.

**Mode** (via `FF_STRICT_EXPLOIT_GATE`, default `enforce`):

| Mode | Behavior |
|------|----------|
| `off` | Gate disabled |
| `log` | Log rejects without blocking; sets `evidence_gate_result = "log_reject"` |
| `enforce` | Block command, fall back to registry; sets `evidence_gate_result = "enforce_reject"` |

On enforce reject, `result.reasoning` is set to `"evidence_gate_enforce: <reasons>"`.

---

## Reward System

**File:** `core/llm/reward_calculator.py` (906L)

Reward range: **[-15.0, +100.0]** (Phase 38 raised ceiling 50→100 for proper gradient).

**Discovery bonuses:**

| Type | Reward |
|------|--------|
| open_port | 2.5 |
| service | 5.0 |
| version | 6.5 |
| user/username | 8.0 |
| hash | 16.0 |
| credential | 20.0 |
| password | 26.0 |
| shell | 40.0 |
| user_flag | 50.0 |
| root_flag | 50.0 |
| root_shell | 80.0 |

**Phase rewards:** RECON=0.0, ENUM=5.0, EXPLOIT=15.0, PRIVESC=30.0, LATERAL=45.0, POST=60.0, EXFIL=75.0, CLOSEOUT=90.0

---

## CyberEnvironment

**File:** `core/environment/cyber_environment.py` (2,879L)

Kill chain phases:
```
RECON -> ENUMERATION -> EXPLOITATION -> PRIVILEGE_ESCALATION ->
LATERAL_MOVEMENT -> POST_EXPLOITATION -> EXFILTRATION -> CLOSEOUT
```

Key methods: `reset()`, `step(action)` -> (state, reward, done, info), `get_global_state()`.

---

## SmartOrchestrator — THE BRAIN

**File:** `core/orchestration/smart_orchestrator.py` (7,614L)

```
run_training(max_steps, target_ip)         # Continuous engagement loop
  |-- Per-step: activate agents in phase order
  |-- _generate_simulated_output(command)  # 95+ command prefix handlers
  |-- _parse_output_for_discoveries(output) # Regex extraction
  |-- _compute_episode_metrics()           # Reward-invariant + reward metrics
```

**Discovery Board** (shared cross-agent state):
```python
discovery_board = {
    "ports": set(), "services": set(), "credentials": set(),
    "vulns": set(), "shells": set(), "users": set(),
    "web_paths": set(), "phase": "RECON", "flags_set": set(),
}
```

**Config:** `SmartOrchestratorConfig` — continuous engagement mode (single long episode), live target, 5 agents, adaptive mentor rates (min=0.92, max=1.0).

---

## GPTManager — LLM Gateway

**File:** `core/gpt_manager.py` (1,494L) — **ALL LLM calls go through this.**

```python
# CORRECT
from core.gpt_manager import GPTManager
gpt = GPTManager()
result = gpt.gpt_request(prompt, task_type="tactical", agent_id="RedAgent")

# WRONG — NEVER DO THIS
import openai
response = openai.chat.completions.create(...)
```

**Features:** Response caching by state fingerprint, per-episode token budgets (`can_make_request()`), auto offline mode when `OPENAI_API_KEY` missing, rate limiting, retry with exponential backoff, model routing by role/task.

**Shared pattern — ALWAYS inject, never create new:**
```python
self.gpt_manager = GPTManager()
self.red_agent = RedAgent(gpt_manager=self.gpt_manager)
self.blue_agent = BlueAgent(gpt_manager=self.gpt_manager)
```

---

## LLM Routing Rules

| Task | Model | Tier |
|------|-------|------|
| Tactical reasoning | gpt-5.2-codex | codex |
| Strategic planning | gpt-5.2-codex | codex |
| Postmortem analysis | gpt-5.2-codex | codex |
| Parsing/verification | gpt-5.2 | full |
| Playbook selection | gpt-5.2-mini | mini |
| Structured extraction | gpt-5.2-mini | mini |
| MicroChain stages 1+3 | gpt-5.2-nano | nano |
| Classification | gpt-5.2-nano | nano |

No gpt-4 models remain. All routing is gpt-5.x.

---

## Feature Flags

**File:** `core/feature_flags.py` (384L) — 76 flags with env-var overrides (prefix `FF_`).

**Profiles** (auto-detected by `resolve_profile()`):

| Profile | Condition | Effect |
|---------|-----------|--------|
| CLOUD | `OPENAI_API_KEY` present | 5 LLM role flags ON |
| DETERMINISTIC | pytest / `ARIASKA_DRY_RUN=1` | All LLM flags OFF |
| OFFLINE | No API key | All LLM flags OFF |

**Critical flags:**

| Flag | Default | Invariant |
|------|---------|-----------|
| `FF_USE_MICRO_CHAIN` | ON | MicroChain 3-stage scoring |
| `FF_STRICT_EXPLOIT_GATE` | enforce | Evidence gate mode |
| `FF_BUDGET_MANAGER_V2` | ON | Dynamic token budgets |
| `FF_STRICT_PHASE_LADDER` | ON | Phase ordering enforcement |
| `FF_BC_LOSS` | ON | BC loss from TeacherTrace |
| `FF_TEACHER_TRACE` | ON | Distillation pipeline |
| `FF_PARALLEL_AGENTS` | ON | Multi-agent parallel activation |
| `FF_NEUROMODULATORS` | ON | Biologically-inspired control |
| `FF_EVIDENCE_GRAPH` | ON | Evidence graph reasoning |
| `FF_HYPOTHESIS_ENGINE` | ON | Hypothesis engine |
| `FF_OPS_HUB` | ON | P38: OpsHub pre-flight + post-step orchestration |
| `FF_OPS_PREFLIGHT` | ON | P38: OpsHub pre-flight authority |
| `FF_DISCOVERY_TRUST` | ON | P38: Discovery trust engine |
| `FF_PHASE_INVARIANTS` | ON | P38: Phase hardening + shell validation |
| `FF_COMMAND_LOCKOUT` | ON | P38: Anti-repeat + cooldown |
| `FF_ENGAGEMENT_METRICS` | ON | P38: Token flex + engagement metrics |
| `FF_TOKEN_FLEX` | ON | P38: Dynamic token budget flex |
| `FF_OPS_STATE_ENCODER` | ON | P38: Ops-aware state encoder extensions |
| `FF_EXPLOIT_CONFIDENCE` | ON | P38: Exploit confidence scoring |
| `FF_EXPLOIT_COOLDOWN` | ON | P38: Exploit cooldown management |
| `FF_SHELL_VALIDATOR` | ON | P38: Shell session validation |
| `FF_ORION_RETHINK` | ON | P39: Orion deep-rethink escalation |
| `FF_TRUST_WEIGHTS` | ON | P39: Trust weight annealing per agent |
| `FF_CAP_GATE` | ON | P39: CAP regression gate |
| `FF_DEBUG_TRACE` | ON | P39: Debug trace instrumentation |

---

## Command Registry

**File:** `core/commands/command_registry.py` (4,560L) — 294 commands.

```python
register(CommandTemplate(
    name="my_tool",
    template="mytool --target {target} --opt {option}",
    description="What it does",
    phase=AttackPhase.ENUMERATION,
    required_params=["target"],
    optional_params={"option": "default"},
    preconditions={"ports_discovered"},
    success_indicators=["SUCCESS", "Found"],
    typical_reward=3.0,
    tags={"recon", "web"},
))
```

**Adding commands checklist:**
1. Match preconditions to `CyberEnvironment` state flags
2. Add to relevant `Playbook` in `pentesting_playbooks.py`
3. Add simulated output handler in `SmartOrchestrator._generate_simulated_output()`
4. Update `DISCOVERY_BONUSES` in `reward_calculator.py` if new discovery type

---

## Import Rules — CRITICAL

This project has deep circular dependencies. Follow strictly:

```python
# CORRECT: Lazy import inside methods
def _init_agents(self):
    from core.agents.red_agent import RedAgent
    self.red_agent = RedAgent(...)

# CORRECT: TYPE_CHECKING guard for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.gpt_manager import GPTManager

# CORRECT: Lazy-load with caching
_cached = None
def _lazy_load():
    global _cached
    if _cached is None:
        from core.algorithms.ppo_agent import PPOAgent
        _cached = PPOAgent
    return _cached

# WRONG: NEVER top-level imports between core modules
from core.algorithms.ppo_agent import PPOAgent  # <- WILL CAUSE CIRCULAR IMPORT
```

---

## Testing

**Framework:** pytest, 2,938 tests across 165 files. All passing.

```bash
make test              # Full suite
make test-fast         # Skip integration
make test-cap          # Cap regression harness
pytest tests/ -v --tb=short
```

**Test patterns:**
```python
import pytest, os
from core.testing import FakeGPTManager, StubToolRunner, get_tool_runner

class TestMyFeature:
    @pytest.fixture(autouse=True)
    def setup(self):
        os.environ['ARIASKA_DRY_RUN'] = '1'
        self.gpt = FakeGPTManager(seed=42)
        self.tool_runner = get_tool_runner(testing=True)

    def test_something(self):
        from core.agents.red_agent import RedAgent  # Lazy import!
        agent = RedAgent(gpt_manager=self.gpt, verbosity="quiet")
        result = agent.act({"phase": "RECON"})
        assert result is not None
```

**Test utilities:**
- `FakeGPTManager(seed=N)` — deterministic responses, token tracking, request history
- `StubToolRunner` — never executes, tracks commands
- `RealToolRunner` — RFC1918 allowlist, blocked commands list, IP validation
- `ToolResult` — dataclass with stdout, stderr, return_code, timed_out

---

## Runtime & Configuration

**Environment variables:**

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Required for live LLM; absence -> offline mode |
| `ARIASKA_DRY_RUN` | `1` -> prevent real command execution |
| `MC_ESCALATE_THRESHOLD` | MicroChain escalation threshold (default: 0.40) |
| `MC_NANO_ABLATION` | `1` -> bypass nano stages in MicroChain |
| `PYTHONPATH` | Should include project root |
| `ARIASKA_LOCAL_MODEL_PATH` | Path to local GGUF/AWQ model file |
| `ARIASKA_LOCAL_LLM_PORT` | Local LLM server port (default: 8192) |
| `ARIASKA_LOCAL_LLM_HOST` | Local LLM server host (default: 127.0.0.1) |
| `ARIASKA_LOCAL_BACKEND` | Backend: `llama-cpp` or `vllm` |
| `FF_LOCAL_LLM` | Enable local LLM provider (auto-detected) |
| `FF_LOCAL_LLM_OFFLOAD_ALL` | Route ALL tiers to local GPU (full offline) |
| `FF_LOCAL_LLM_OFFLOAD_NANO` | Route nano tier to local GPU |
| `FF_LOCAL_LLM_OFFLOAD_MINI` | Route mini tier to local GPU |

**Runtime flags** (`core/runtime_flags.py`):
```python
from core.runtime_flags import get_runtime_flags
flags = get_runtime_flags()  # .offline, .enable_llm, .require_llm, .dry_run
```

**CLI commands:**
```bash
python ariaska_cli.py smart-train --target IP --steps 500 [--ctf] [--seed N]
python ariaska_cli.py watchdog --target IP --max-attempts 10
python ariaska_cli.py replay <trace_file> [--verbose]
python ariaska_cli.py status
python ariaska_cli.py help
```

---

## Knowledge System (v2)

**107,933 entries** across 18 JSONL partitions in `data/knowledge_candidates_v2/`.

**Schema** (`core/knowledge/knowledge_candidate_v2.py`, 14 nested dataclasses):
```python
@dataclass
class KnowledgeCandidate:
    candidate_id: str
    title: str
    taxonomy: Taxonomy            # service_archetype, phase_fit, killchain_step, tags
    evidence_gate: EvidenceGate   # evidence_requirements, confidence_tier
    execution: Execution          # tool, original_command, command_templates, parameters
    references: References        # cves, urls, mitre
    quality: QualityMetrics
    governance: Governance        # origin, ingestion_date
    source: SourceInfo
```

**11 indices** in `data/knowledge_indices/`: by_port (417), by_cve (25K), by_service (193), by_phase (8), by_tag (71K), by_template (205), by_platform (79), by_killchain (6), by_vuln_family (19), by_exploit_archetype (13), by_origin (20).

**Query:** `data/knowledge_retriever.py`, `core/knowledge/knowledge_query.py`
**Rebuild:** `python scripts/build_knowledge_indices.py`

---

## Reward-Invariant Metrics

These measure REAL learning quality independent of reward scaling:
- `unique_commands` — distinct commands used per episode
- `diversity_ratio` — unique_commands / total_steps
- `total_discoveries` — genuinely new discoveries
- `step_at_first_exploit` — speed to exploitation
- `completion_bonus_applied` — whether EXFILTRATION reached

Always validate changes with these, not raw reward numbers.

---

## Safety & Lab Scope

- **Live execution only** — all commands against real authorized targets
- **RFC1918 validation** — `RealToolRunner` only allows private IPs
- **Sandboxed executor** — additional safety layer
- **Never hardcode API keys** — `.env` + `python-dotenv`
- **Never execute real commands in tests** — `StubToolRunner`
- **Deterministic mode** — `--seed` for reproducibility
- **Dry run** — `ARIASKA_DRY_RUN=1` prevents real execution
- **Ethics mode** — training / assessment / demo profiles
- **Input sanitization** — all LLM outputs sanitized before command use

---

## Code Generation Checklist

When generating code for this project:

- [ ] Inject `gpt_manager` — never create new `GPTManager` instances
- [ ] Use `rich` for output — `Console()`, `Panel()`, `Table()`
- [ ] Follow lazy import pattern — avoid circular deps
- [ ] Include offline fallback paths for LLM-dependent features
- [ ] Add `FakeGPTManager` support for testing
- [ ] Use `@dataclass` for config objects
- [ ] Log with `logging.getLogger("ariaska.<module>")`
- [ ] Handle `torch.device` — check CUDA, default CPU
- [ ] Respect phase system — commands must match `AttackPhase` + `preconditions`
- [ ] Write tests with `FakeGPTManager` + `StubToolRunner`
- [ ] Keep 2,938-test baseline passing
- [ ] Use reward-invariant metrics for validation
- [ ] Type hints on all function signatures
- [ ] Google-style docstrings on public APIs
- [ ] Never duplicate existing functionality — check modules first
- [ ] Respect BudgetManagerV2 tier routing
- [ ] Evidence Gate must remain in enforce mode by default

---

## Key Hyperparameters

| Parameter | Value | Location |
|-----------|-------|----------|
| State dim | 512 | `state_encoder.py` |
| Action dim | 5 | `PPOConfig` |
| PPO clip | 0.2 (adaptive 0.15-0.25) | `PPOConfig` |
| PPO LR | 3e-4 -> 1e-5 | `PPOConfig` |
| GAE lambda | 0.97 (dual: +0.70 short) | `PPOConfig` |
| Discount gamma | 0.99 | `PPOConfig` |
| SIL coef | 0.25 (SIL buffer: 500) | `PPOConfig` |
| Minibatch size | 16 | `PPOConfig` |
| Mentor anneal | 60% -> 10% | `MentorPolicy` |
| Total budget | 1,148,850 tokens | `BudgetManagerV2` |
| Min budget | 574,425 tokens | `BudgetManagerV2` |
| MicroChain threshold | 0.40 | `micro_chain.py` |
| PhaseGuide codex threshold | 0.45 | `phase_guided_llm.py` |
| Reward range | [-15.0, +100.0] | `reward_calculator.py` |
| BCBuffer capacity | 2000 | `teacher_trace.py` |
| LLM feature dim | 256 | `llm_policy_bridge.py` |
| Enhanced state dim | 768 (512+256) | `state_encoder.py` |
| Prior alpha init | 0.50 → 0.02 (cosine) | `llm_policy_bridge.py` |
| KL teacher coef | 0.15 → 0.01 | `llm_policy_bridge.py` |
| Ranking loss coef | 0.05 | `PPOConfig` |
| Value reg coef | 0.10 | `PPOConfig` |
| Anneal steps | 3000 | `LLMPolicyBridge` |

---

## Phase History (Recent)

| Phase | Focus |
|-------|-------|
| 27 | MicroChain (nano->mini->nano), Evidence Gate, intelligence pipeline |
| 28 | Postmortem refinements, skill library persistence |
| 29 | Event bus decoupling, trace writer integration |
| 30 | MentorTrace: structured mentor->apprentice transfer metadata |
| 31 | EpisodeReplayer: JSONL trace replay via CLI |
| 32 | MicroChain tuning: env-var configurable escalation threshold |
| 33 | Budget burst pool, candidate cap, dynamic burst cooldown |
| 34 | PhaseGuidedLLM: structured guidance + distillation packets |
| 35 | CanonicalState single source of truth, CoherenceChain 4-step nano validation, LiveTrace append-only JSONL, port parsing hardening |
| 36 | Codex-primary model routing, fast-learn metrics, +23% token headroom, dashboard improvements |
| 37 | Level 5 GPT↔RL Neural Integration: LLMPolicyBridge, KL teacher distillation, ranking margin loss, value reg, teacher anneal, dashboard panel |
| 38 | OpsHub authority (19 modules, 5,653L), discovery trust, phase invariants, command lockout, engagement metrics, reward ceiling 50→100, budget +15% |
| 39 | Trust Weights + Orion Deep-Rethink escalation, CAP regression gate, debug trace instrumentation |
| 40-42 | 25 new RL algorithms, self-play, world model, progressive nets, PBT, HRL options |
| 43 | Local LLM Provider (GPU acceleration), model router, llama-cpp-python/vLLM backend |
| 44 | Full-local routing: FF_LOCAL_LLM_OFFLOAD_ALL, offline operation mode |
| 45-49 | Knowledge corpus expansion, agent improvements, training refinements |
| 50 | Anti-repeat stagnation-aware thresholds, flag capture whitelist, gap-aware alternatives |
| 51-52 | Additional algorithm tuning, Reptile meta-learning default ON |
| 53 | Accelerated LLM→RL handoff: PRIOR_ALPHA 0.25, KL_TEACHER 0.08, MATURITY_FAST_DECAY 0.4 |

---
