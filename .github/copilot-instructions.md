# Ariaska_RL — Copilot System Prompt

> **Read this entire file before generating ANY code for this project.**

## Identity

**Ariaska_RL** — Autonomous multi-agent reinforcement learning system for cybersecurity simulation and live penetration testing. 5 specialized agents, GPT-hybrid decision pipeline, PPO primary RL, 107K v2 knowledge corpus, 105K lines of Python across 130+ modules.

**Author:** Filip Volf
**Python:** 3.11+ (developed on 3.13.7)
**Entry:** `ariaska_cli.py` → `SmartOrchestrator.run_training()`
**Tests:** 423 collected, 423 passing
**Current Phase:** Post-Phase 10.1 — Stabilization complete

---

## CRITICAL RULES — Read First

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

---

## File Map — Where Things Live

```
ariaska_cli.py                              # CLI entry: smart-train, status, help
core/
├── agents/                                 # 5 agents (all implement AgentInterface + MemorySyncInterface)
│   ├── red_agent.py              (2204L)   # RedAgent — offensive exploitation
│   ├── blue_agent.py                       # BlueAgent — defensive/reactive
│   ├── scout_agent.py                      # ScoutAgent — recon/discovery
│   ├── shadow_agent.py           (711L)    # ShadowAgent — stealth/alert prevention
│   ├── orion_agent.py                      # OrionAgent — strategic coordination
│   ├── enhanced_agent_base.py              # EnhancedAgentBase mixin
│   └── redagent_brain.py                   # RedAgent brain (extended cognition)
├── algorithms/
│   ├── ppo_agent.py              (1454L)   # PPOAgent — primary RL (R80 features)
│   ├── command_action_mapper.py            # PPO action_idx ↔ CommandTemplate mapping
│   ├── replay_buffer.py                    # PER with dedup
│   ├── sac_agent.py                        # Soft Actor-Critic (alternative)
│   ├── ddqn_macro.py                       # Double DQN macro-actions
│   ├── rnd_curiosity.py                    # RND intrinsic motivation
│   └── cognition_node.py                   # Cognitive architecture node
├── commands/
│   ├── command_registry.py       (3507L)   # 144+ CommandTemplates by AttackPhase
│   ├── command_enrichment.py               # Enrich commands with knowledge
│   └── learned_commands.py                 # Runtime-discovered command store
├── environment/
│   ├── cyber_environment.py      (2854L)   # CyberEnvironment — kill chain simulation
│   ├── metasploitable_handler.py           # MS2/MS3 Docker integration
│   └── context_detector.py                 # Environment context detection
├── execution/
│   ├── parser_broker.py          (291L)    # 4-stage parser: regex→SOP→Venice→GPT
│   ├── output_parser.py          (599L)    # Smart output parsing
│   ├── discovery_event.py        (159L)    # DiscoveryEvent schema
│   ├── live_executor.py          (410L)    # Real command execution
│   ├── sandboxed_executor.py               # Sandboxed execution
│   └── step_parse_cache.py                 # Parser result caching
├── interfaces/
│   ├── agent_interface.py                  # AgentInterface — required contract
│   └── memory_sync_interface.py            # MemorySyncInterface — cross-agent memory
├── knowledge/
│   ├── knowledge_candidate_v2.py (329L)    # KnowledgeCandidate dataclass (v2 schema, 14 nested dataclasses)
│   ├── knowledge_query.py        (168L)    # Query interface for knowledge indices
│   ├── knowledge_packs.py       (2468L)    # Pre-built knowledge packs
│   ├── target_profiler.py       (1422L)    # Target profile resolution
│   ├── kg_manager.py            (1015L)    # Knowledge graph manager
│   ├── pentesting_playbooks.py  (1013L)    # Playbook → PlaybookStep chains
│   ├── ms2_exploit_graph.py                # Metasploitable 2 exploit graph
│   ├── ms3_exploit_graph.py                # Metasploitable 3 exploit graph
│   └── massive_knowledge_seed.py           # Bulk knowledge seeding
├── llm/
│   ├── smart_mentor.py          (1386L)    # SmartMentor + DualMentor (GPT + Venice)
│   ├── reward_calculator.py      (796L)    # SmartRewardCalculator — shaped rewards
│   ├── cloud_roles.py            (367L)    # Cloud role definitions for LLM
│   ├── venice_reasoning.py       (462L)    # Venice AI reasoning integration
│   ├── codex_personas.py                   # LLM persona definitions
│   └── reflective_cortex.py                # Self-reflective LLM reasoning
├── cortex/
│   ├── executive_cortex.py       (739L)    # Executive decision making
│   └── tactical_cortex.py        (760L)    # Tactical decision making
├── memory/
│   ├── hybrid_memory.py                    # Hybrid memory (short+long term)
│   ├── enhanced_memory_sync.py             # Cross-agent memory fusion (threaded)
│   ├── unified_cognitive_bus.py            # Unified cognitive event bus
│   ├── campaign_memory.py                  # Campaign-level persistent memory
│   └── chroma_memory_store.py              # ChromaDB vector memory
├── models/
│   ├── state_encoder.py          (438L)    # 512-dim state encoder
│   ├── policy_net.py                       # Dueling DQN PolicyNet (legacy)
│   ├── advanced_networks.py                # Attention + residual + NoisyLinear
│   └── value_net.py                        # Value network
├── multiagent/
│   ├── agent_manager.py                    # Creates and coordinates all 5 agents
│   ├── agents.py                           # Factory: get_all_agents(), create_agent()
│   ├── memory_router.py                    # PER with SumTree, SQLite persistence
│   └── strategic_directive.py              # Cross-agent directive protocol
├── orchestration/
│   ├── smart_orchestrator.py    (5220L)    # Main training loop — THE BRAIN
│   └── orchestrator.py                     # Base orchestrator
├── training/
│   ├── smart_coach.py           (6542L)    # Per-agent hybrid decision pipeline (LARGEST FILE)
│   ├── mentor_policy.py                    # Annealing mentor call rates
│   └── ariaska_trainer.py                  # High-level training config
├── postmortem/
│   ├── orion_postmortem.py                 # GPT end-of-run analysis
│   └── skill_library.py                    # Persistent SkillCard store
├── testing/
│   ├── fake_gpt_manager.py                 # FakeGPTManager — deterministic mock
│   └── tool_runner.py                      # StubToolRunner / RealToolRunner + ToolResult
├── analytics/                              # Coherence, replay analysis, scan randomizer
├── logging/                                # JSONL structured logging
├── telemetry/                              # Event telemetry
├── observability/                          # LiveDashboard
├── gpt_manager.py               (1175L)   # Centralized LLM gateway — ALL LLM calls here
├── runtime_flags.py                        # Global runtime flags (offline, dry_run, etc.)
├── feature_flags.py                        # Feature flag system
└── ui/                                     # Textual TUI dashboard

data/
├── knowledge_candidates_v2/                # 107,933 entries across 18 JSONL files (~195MB)
├── knowledge_indices/                      # 11 prebuilt indices (~110MB)
├── knowledge_retriever.py                  # Query and retrieve knowledge
├── knowledge_ingestion.py                  # Ingest new knowledge
├── knowledge_loader.py                     # ChromaDB + SentenceTransformer ingestion
└── htb_extractions/                        # HTB machine profiles (JSON)

scripts/
├── build_knowledge_indices.py              # Rebuild indices from v2 corpus
├── fill_htb_gaps.py                        # Fill HTB readiness gaps
├── dedup_v2_corpus.py                      # Deduplicate corpus
├── htb_readiness.py                        # HTB readiness assessment
└── tool_dependency_audit.py                # Audit tool dependencies

tests/                                      # 18 test files, 423 tests total
├── test_phase0_invariants.py               # Foundation: GPTManager, agent init
├── test_phase2_invariants.py               # Metasploitable, sandboxed execution
├── test_phase3_invariants.py               # State encoder, PPO, playbooks, registry
├── test_smart_integration.py               # Full SmartOrchestrator integration
├── test_training_smoke.py                  # End-to-end training pipeline
├── test_phase62_systems.py                 # Phase 6.2 systems
├── test_phase63_components.py              # Phase 6.3 components
├── test_phase93_components.py              # Phase 9.3 components
├── test_phase95_correctness.py             # Phase 9.5 correctness
├── test_phase97_telemetry.py               # Telemetry validation
├── test_cloud_roles.py                     # Cloud role tests
└── test_testing_utilities.py               # Testing infra self-tests
```

---

## The Five Agents

| Agent | Role | Domain |
|-------|------|--------|
| **RedAgent** | `offensive` | Exploitation, privesc, exfil. Primary PPO-trained. DQN+GPT hybrid, emergency fallbacks. |
| **BlueAgent** | `defensive` | Honeypots, credential resets, firewall, alert management. Reactive. |
| **ScoutAgent** | `recon` | Network discovery, port scanning, service fingerprinting, version detection. |
| **ShadowAgent** | `stealth` | Alert monitoring, scan timing, detection avoidance, action overrides. |
| **OrionAgent** | `strategic` | Coordination, strategic reviews, cross-agent directives, phase transitions. |

**Activation order** (phase-dependent, in `SmartOrchestrator._should_activate()`):
- RECON: Scout → Shadow → Orion → Red → Blue
- EXPLOITATION: Red → Shadow → Scout → Orion → Blue
- EXFILTRATION: Red → Shadow → Orion → Scout → Blue

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

Each agent's `SmartCoach` selects commands via **4-stage hybrid pipeline** (`core/training/smart_coach.py`, 6542L):

```
1. PLAYBOOK    →  Curriculum-guided chain (anneals 60% → 10%)
2. PPO         →  RL policy network (action_dim=5 via CommandActionMapper)
3. REGISTRY    →  Precondition-matched from CommandRegistry (144+ templates)
4. GPT MENTOR  →  SmartMentor / DualMentor LLM (rate-limited, annealed)
```

**Anti-Repeat Guard** (fires after pipeline, before execution):
- Blocks exact command repeats (≥1 occurrence in episode)
- Blocks prefix repeats (≥3 same-tool uses in episode)
- Replaces with random alternative from per-role pool

**Decision source tracking** via `SmartDecisionResult.source`:
`"playbook"` | `"ppo"` | `"registry"` | `"mentor"` | `"dual_mentor"` | `"anti_repeat"` | `"fallback"`

---

## PPO — Primary RL Algorithm

**File:** `core/algorithms/ppo_agent.py` (1454L)

```python
PPOConfig:
    state_dim = 512                    # Must match state_encoder
    action_dim = 5                     # Maps via CommandActionMapper
    hidden_dims = [512, 512, 256]
    clip_epsilon = 0.2                 # Adaptive: 0.15–0.25
    gamma = 0.99
    gae_lambda = 0.97
    learning_rate = 3e-4 → 1e-5       # KL-adaptive annealing
    epochs_per_update = 4
    minibatch_size = 16
    rollout_size = 256
```

**R68–R80 advanced features:**
- Phase-gated actor heads (HRL-lite, 3 groups: recon/exploit/post-exploit)
- Self-Imitation Learning (SIL, 500-entry buffer)
- Symlog value compression (DreamerV3-style)
- Cosine entropy schedule with rebound
- Prioritized advantage sampling
- Gradient accumulation (2× effective batch)
- KL-adaptive learning rate
- Per-phase advantage whitening
- EMA target network (τ=0.995) + value-surprise intrinsic bonus
- Dual-horizon GAE (λ=0.97 long + λ=0.70 short, blend=0.65)
- Spectral normalization on critic
- Soft advantage clipping (tanh-based)
- Auxiliary phase prediction head
- Adaptive clip scheduling from rolling clip_fraction

**PPO flow in SmartCoach:**
1. `ppo.select_action(state_tensor)` → action_idx, log_prob, value
2. `CommandActionMapper.action_to_commands(idx, state)` → candidates
3. Anti-repeat check → store in `_ppo_pending`
4. After step → `ppo.store_transition(s, a, log_prob, reward, value, done)`
5. Episode end → `ppo.update(last_value)` with GAE-λ

---

## State Encoder

**File:** `core/models/state_encoder.py` (438L) — produces **512-dim** tensor.

| Dims | Content |
|------|---------|
| 0–11 | Phase one-hot + progress (12) |
| 12–26 | State flags: ports_discovered, shell_obtained, etc. (15) |
| 27–46 | Top 20 port presence indicators (20) |
| 47–58 | Service type presence (12) |
| 59–70 | Numeric: detection_risk, blue_team_alert, etc. (12) |
| 71–80 | Action history encoding (10) |
| 81–85 | LLM/Mentor features (5) |
| 86–90 | Temporal features (5) |
| 91–511 | Reserved zero-padded (421) |

**Rule:** Always use `STATE_DIM = 512` for network input dims.

---

## CyberEnvironment

**File:** `core/environment/cyber_environment.py` (2854L)

Kill chain phases:
```
RECON → ENUMERATION → EXPLOITATION → PRIVILEGE_ESCALATION →
LATERAL_MOVEMENT → POST_EXPLOITATION → EXFILTRATION
```

Key methods: `reset()`, `step(action)` → (state, reward, done, info), `get_global_state()`.

Scenarios: `dynamic`, `simulation`, `metasploitable2`.

---

## SmartOrchestrator — THE BRAIN

**File:** `core/orchestration/smart_orchestrator.py` (5220L)

```
run_training(episodes, target_ip)          # Outer episode loop
├── run_episode(episode_id, episode_number)  # Per-episode: env reset, coordination, metrics
│   ├── _run_step(episode_id, step, state)    # Activates agents in phase order
│   ├── _generate_simulated_output(command)    # 95+ command prefix handlers
│   ├── _parse_output_for_discoveries(output)  # Regex extraction
│   └── _compute_episode_metrics()             # Reward-invariant + reward metrics
```

**Discovery Board** (shared cross-agent state, reset per episode):
```python
discovery_board = {
    "ports": set(), "services": set(), "credentials": set(),
    "vulns": set(), "shells": set(), "users": set(),
    "web_paths": set(), "phase": "RECON", "flags_set": set(),
}
```

**Config:** `SmartOrchestratorConfig` — 40 steps/episode, target Metasploitable3 ---> `172.28.0.11`, Metasploitable2 ---> `172.28.0.10`, mentor budget 30%, 5 agents.

---

## Reward System

**File:** `core/llm/reward_calculator.py` (796L)

| Component | Value |
|-----------|-------|
| REWARD_MULTIPLIER | 1.0 (honest) |
| progress_bonus/step | 1.0 |
| redundancy_penalty | soft, max 0.5 |
| Reward floor | -5.0 |

**Discovery bonuses:** open_port=2.5, service=5.0, version=6.5, credential=20.0, password=26.0, shell=50.0, root_shell=130.0, flag=200.0

**Phase rewards:** RECON=1.0 escalating to EXFILTRATION=250.0

---

## Knowledge System (v2)

**107,933 entries** across 18 JSONL partitions in `data/knowledge_candidates_v2/`.

Top files by entry count:
| File | Entries |
|------|---------|
| exploitdb.jsonl | 46,491 |
| cves.jsonl | 25,467 |
| commands.jsonl | 24,342 |
| wordlists.jsonl | 5,814 |
| + 14 more | 5,819 |

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

**11 indices** in `data/knowledge_indices/`: by_port (417 keys), by_cve (25K), by_service (193), by_phase (8), by_tag (71K), by_template (205), by_platform (79), by_killchain (6), by_vuln_family (19), by_exploit_archetype (13), by_origin (20).

**Query:** `data/knowledge_retriever.py`, `core/knowledge/knowledge_query.py`
**Rebuild indices:** `python scripts/build_knowledge_indices.py`

---

## Execution Pipeline

### Parser Broker (4-stage)
**File:** `core/execution/parser_broker.py` (291L)

```
Stage 1: Regex   → Fast pattern matching (ports, services, creds)
Stage 2: SOP     → Standard Operating Procedure rules
Stage 3: Venice  → Venice AI for ambiguous output
Stage 4: GPT     → Final GPT classification for edge cases
```

Produces `DiscoveryEvent` objects (`core/execution/discovery_event.py`).

### Live Executor
**File:** `core/execution/live_executor.py` (410L) — Real command execution against targets.

### Sandboxed Executor
**File:** `core/execution/sandboxed_executor.py` — Safe execution wrapper.

---

## GPTManager — LLM Gateway

**File:** `core/gpt_manager.py` (1175L) — **ALL LLM calls go through this.**

```python
# ✅ CORRECT
from core.gpt_manager import GPTManager
gpt = GPTManager()
result = gpt.gpt_request(prompt, task_type="tactical", agent_id="RedAgent")

# ❌ WRONG
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

## Command Registry

**File:** `core/commands/command_registry.py` (3507L) — 144+ commands.

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
4. Update `DISCOVERY_BONUSES` if new discovery type

---

## Testing

**Framework:** pytest, 423 tests across 18 files. All passing.

```bash
make test              # Full suite
make smoke             # Quick 3-episode smoke
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

## Import Rules — CRITICAL

This project has deep circular dependencies. Follow strictly:

```python
# ✅ Lazy import inside methods
def _init_agents(self):
    from core.agents.red_agent import RedAgent
    self.red_agent = RedAgent(...)

# ✅ TYPE_CHECKING guard for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.gpt_manager import GPTManager

# ✅ Lazy-load with caching
_cached = None
def _lazy_load():
    global _cached
    if _cached is None:
        from core.algorithms.ppo_agent import PPOAgent
        _cached = PPOAgent
    return _cached

# ❌ NEVER top-level imports between core modules
```

---

## Runtime & Configuration

**Environment variables:**
| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Required for live LLM; absence → offline mode |
| `ARIASKA_DRY_RUN` | `1` → prevent real command execution |
| `PYTHONPATH` | Should include project root |

**Runtime flags** (`core/runtime_flags.py`):
```python
from core.runtime_flags import get_runtime_flags
flags = get_runtime_flags()  # .offline, .enable_llm, .require_llm, .dry_run
```

**Feature flags** (`core/feature_flags.py`):
Feature-gated functionality for safe rollout of new capabilities.

**CLI:**
```bash
python ariaska_cli.py smart-train --episodes 100 --steps 40 --seed 42 --env sim
python ariaska_cli.py smart-train --env msf --target 192.168.56.101
python ariaska_cli.py status
```

**Makefile:** `make venv`, `make train`, `make train-quick`, `make train-msf`, `make smoke`, `make test`, `make last`, `make clean`

---

## Key Hyperparameters

| Parameter | Value | Location |
|-----------|-------|----------|
| State dim | 512 | `state_encoder.py` |
| Action dim | 5 | `PPOConfig` |
| PPO clip | 0.2 (adaptive 0.15–0.25) | `PPOConfig` |
| PPO LR | 3e-4 → 1e-5 | `PPOConfig` |
| GAE λ | 0.97 (dual: +0.70 short) | `PPOConfig` |
| Discount γ | 0.99 | `PPOConfig` |
| Steps/episode | 40 | `SmartOrchestratorConfig` |
| Mentor anneal | 60% → 10% | `MentorPolicy` |
| Mentor budget | 30% of steps | `SmartOrchestratorConfig` |
| Reward multiplier | 1.0 | `SmartRewardCalculator` |
| Reward floor | -5.0 | `RewardBreakdown` |

---

## Metasploitable 2 — Primary Target

| Port | Service | Vulnerability | Exploit Path |
|------|---------|---------------|--------------|
| 21 | vsftpd 2.3.4 | Backdoor | `exploit/unix/ftp/vsftpd_234_backdoor` → root |
| 22 | OpenSSH 4.7p1 | Weak creds | `msfadmin:msfadmin` |
| 139/445 | Samba 3.0.20 | CVE-2007-2447 | `exploit/multi/samba/usermap_script` → root |
| 1524 | ingreslock | Backdoor | `telnet <target> 1524` → instant root |
| 3306 | MySQL 5.0.51a | No root pw | `mysql -h <target> -u root` |
| 5432 | PostgreSQL 8.3.0 | Default creds | `postgres:postgres` → RCE |
| 6667 | UnrealIRCd | Backdoor | `exploit/unix/irc/unreal_ircd_3281_backdoor` → root |
| 8180 | Tomcat | Default creds | `tomcat:tomcat` → WAR deploy → shell |
| 512-514 | rexec/rlogin/rsh | No auth | Direct remote command execution |
| 2049 | NFS | World-readable | Mount root FS, plant SSH keys |
| 5900 | VNC | Password: password | `vncviewer <target>` → desktop |

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
- [ ] Keep 423-test baseline passing
- [ ] Use reward-invariant metrics for validation
- [ ] Type hints on all function signatures
- [ ] Google-style docstrings on public APIs
- [ ] Never duplicate existing functionality — check modules first

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

## Security & Safety

- **Never hardcode API keys** — `.env` + `python-dotenv`
- **Never execute real commands in tests** — `StubToolRunner`
- **RFC1918 validation** — `RealToolRunner` only allows private IPs
- **Input sanitization** — all LLM outputs sanitized before command use
- **Deterministic mode** — `--seed` for reproducibility
- **Dry run** — `ARIASKA_DRY_RUN=1` prevents real execution
- **Sandboxed executor** — additional safety layer for live runs

---

## Common Workflows

### Tuning Rewards
1. Adjust `PHASE_REWARDS` / `DISCOVERY_BONUSES` in `reward_calculator.py`
2. Tune redundancy penalty growth rate
3. Validate with 10-episode run, check reward-invariant metrics

### Modifying State Encoder
1. Edit `state_encoder.py` — add features after existing sections
2. Keep `STATE_DIM = 512` unless changing ALL networks
3. Update `test_phase3_invariants.py` dimension checks

### Adding Simulated Output
1. Add command prefix handler in `SmartOrchestrator._generate_simulated_output()`
2. Include discoverable artifacts (ports, services, creds) in output
3. Ensure `_parse_output_for_discoveries()` regex patterns extract them
4. Test with `make smoke`

### Rebuilding Knowledge Indices
```bash
python scripts/build_knowledge_indices.py
```

### Full Validation
```bash
make test          # 423/423 passing
make smoke         # Quick 3-episode smoke
make train-quick   # 10 episodes with metrics
```
