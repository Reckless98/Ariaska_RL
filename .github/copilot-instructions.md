# GitHub Copilot Instructions for Ariaska_RL

## Project Identity

**Ariaska_RL** is an autonomous multi-agent reinforcement learning system for cybersecurity simulation and live penetration testing. It orchestrates 5 specialized agents (Red, Blue, Scout, Shadow, Orion) through a GPT-powered hybrid decision pipeline using PPO as the primary RL algorithm, a 512-dimensional state encoder, a 144+ command registry organized by attack phase, and pentesting playbooks for curriculum-guided learning.

**Author:** Filip Volf
**Python:** 3.11+ (3.8 features minimum)
**Entry Point:** `ariaska_cli.py` → `SmartOrchestrator.run_training()`
**Primary Target:** Metasploitable 2 (simulation mode and live Docker)
**Current Phase:** Phase 5.2 — Agent intelligence enhancement, MS2-specific knowledge injection, PPO training loop fixes

---

## Architecture Overview

```
Ariaska_RL/
├── ariaska_cli.py                       # Unified CLI entry point (smart-train, status, help)
├── ariaska.py                           # Launcher shim → ariaska_cli.main()
├── streamlit_app.py                     # Streamlit dashboard for training visualization
├── core/
│   ├── agents/                          # 5 specialized agent implementations
│   │   ├── red_agent.py                 # RedAgent — CyberOffense (extends EnhancedAgentBase + MemorySyncInterface)
│   │   ├── blue_agent.py               # BlueAgent — CyberDefense (AgentInterface + MemorySyncInterface)
│   │   ├── scout_agent.py              # ScoutAgent — Recon/Discovery
│   │   ├── shadow_agent.py             # ShadowAgent — Stealth/Alert Prevention
│   │   └── orion_agent.py              # OrionAgent — Strategic Coordination
│   ├── algorithms/
│   │   ├── ppo_agent.py                # PPOAgent — Primary RL (GAE-λ, clipped surrogate, actor-critic)
│   │   ├── command_action_mapper.py    # CommandActionMapper — Maps PPO actions ↔ registry commands
│   │   └── replay_buffer.py           # Prioritized experience replay with deduplication
│   ├── commands/
│   │   ├── command_registry.py         # 144+ pentesting CommandTemplates by AttackPhase
│   │   └── learned_commands.py         # Runtime-learned command store with success tracking
│   ├── environment/
│   │   ├── cyber_environment.py        # CyberEnvironment — Cyber kill chain simulation
│   │   └── metasploitable_handler.py   # Metasploitable 2 VM integration for live training
│   ├── interfaces/
│   │   ├── agent_interface.py          # AgentInterface — Required contract for all agents
│   │   └── memory_sync_interface.py    # MemorySyncInterface — Cross-agent memory protocol
│   ├── knowledge/
│   │   └── pentesting_playbooks.py     # Ordered attack chains (Playbook → PlaybookStep)
│   ├── llm/
│   │   ├── reward_calculator.py        # SmartRewardCalculator — Phase progression + novelty + discovery rewards
│   │   └── smart_mentor.py             # SmartMentor + DualMentor (GPT + Venice) for agent guidance
│   ├── memory/
│   │   ├── enhanced_memory_router.py   # Enhanced memory with agent sync
│   │   └── enhanced_memory_sync.py     # Cross-agent memory fusion (threaded)
│   ├── models/
│   │   ├── state_encoder.py            # 512-dim rich state encoder (90+ meaningful features)
│   │   ├── policy_net.py               # Dueling DQN PolicyNet (legacy)
│   │   └── advanced_networks.py        # AdvancedPolicyNetwork (attention + residual + NoisyLinear)
│   ├── multiagent/
│   │   ├── agent_manager.py            # AgentManager — Creates and coordinates all 5 agents
│   │   ├── agents.py                   # Factory: get_all_agents(), create_agent()
│   │   └── memory_router.py            # MemoryRouter — PER with SumTree, SQLite persistence
│   ├── orchestration/
│   │   └── smart_orchestrator.py       # SmartOrchestrator — Main training loop, agent coordination, simulated output
│   ├── training/
│   │   ├── smart_coach.py              # SmartCoach — Per-agent hybrid decision pipeline (Playbook→PPO→Registry→Mentor)
│   │   ├── mentor_policy.py            # MentorPolicy — Annealing mentor call rates
│   │   ├── ariaska_trainer.py          # AriaskaTrainer — High-level training config/runner
│   │   └── enhanced_unified_trainer.py # CybersecurityEnvironment (alternative trainer)
│   ├── postmortem/
│   │   ├── orion_postmortem.py         # OrionPostmortem — GPT-5.2 end-of-run analysis
│   │   └── skill_library.py            # SkillLibrary — Persistent skill cards from postmortems
│   ├── gpt_manager.py                  # GPTManager — Centralized LLM gateway with role-based routing
│   ├── llm_utils.py                    # Convenience wrappers for GPTManager
│   ├── runtime_flags.py                # Global runtime flags (offline, enable_llm, dry_run)
│   ├── testing/
│   │   ├── fake_gpt_manager.py         # FakeGPTManager — Deterministic mock for tests
│   │   └── tool_runner.py              # StubToolRunner / RealToolRunner with RFC1918 validation
│   ├── observability/                  # LiveDashboard for real-time training visibility
│   ├── tracing/                        # Execution tracing and audit logging
│   └── visualization/                  # Training visualization utilities
├── config/
│   ├── environment.json                # Environment configuration
│   └── llm_router.json                 # LLM model routing configuration
├── data/
│   ├── knowledge_loader.py             # ChromaDB + SentenceTransformer knowledge ingestion
│   └── knowledge_sources/              # Knowledge base documents
├── tests/                              # pytest suite (199 tests, all passing as of Phase 5.1)
├── postmortems/                        # OrionPostmortem analysis outputs (JSON)
├── logs/                               # Execution audit logs (JSONL)
├── traces/                             # Execution traces
├── Makefile                            # Build targets: train, test, smoke, venv
├── Dockerfile                          # Python 3.11-slim container
├── docker-compose.yml                  # Main docker compose
├── docker-compose.metasploitable.yml   # Metasploitable 2 docker setup
└── requirements.txt
```

---

## The Five Agents

All agents implement `AgentInterface` and `MemorySyncInterface` from `core/interfaces/`. Each has a distinct role with exclusive command domain.

| Agent | Role | Key Responsibilities |
|-------|------|---------------------|
| **RedAgent** | `offensive` | Exploitation, privilege escalation, data exfiltration. Primary PPO-trained attack agent. Has DQN + GPT hybrid and emergency fallbacks. |
| **BlueAgent** | `defensive` | Defense: honeypot deployment, credential resets, firewall, alert management. Currently reactive. |
| **ScoutAgent** | `recon` | Network discovery, port scanning, service fingerprinting, version detection. First to act in early phases. |
| **ShadowAgent** | `stealth` | Alert score monitoring, scan timing, detection avoidance, action overrides when risk is high. |
| **OrionAgent** | `strategic` | High-level coordination, strategic reviews, cross-agent directive issuance, phase transition decisions. |

### Agent Execution Order (Phase-Optimized)

Agents activate in phase-dependent order configured in `SmartOrchestrator._should_activate()`:
- **RECON phase:** Scout → Shadow → Orion → Red → Blue
- **EXPLOITATION phase:** Red → Shadow → Scout → Orion → Blue
- **EXFILTRATION phase:** Red → Shadow → Orion → Scout → Blue

### Agent Interface Contract

Every agent must implement:
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

### Adding a New Agent

1. Create class in `core/agents/your_agent.py` implementing both interfaces
2. Register in `core/multiagent/agents.py` (`get_all_agents()` and `create_agent()`)
3. Add to `AgentManager._initialize_agents()` in `core/multiagent/agent_manager.py`
4. Configure activation schedule in `SmartOrchestrator._should_activate()`
5. Create a `SmartCoach` with appropriate role config in `SmartOrchestrator._init_smart_coaches()`
6. Add role-specific alternative commands in `SmartCoach` anti-repeat pool
7. Register emergency fallbacks in the agent class if it takes autonomous actions

---

## Core Decision Pipeline (SmartCoach)

Each agent's `SmartCoach` selects commands through a **4-stage hybrid pipeline** (in `core/training/smart_coach.py`):

```
1. PLAYBOOK   →  Curriculum-guided chain (anneals 60% → 10% over episodes)
2. PPO         →  RL policy network (action_dim=5 mapped to commands via CommandActionMapper)
3. REGISTRY    →  Precondition-matched commands from CommandRegistry (144+ templates)
4. GPT MENTOR  →  SmartMentor / DualMentor LLM call (rate-limited, annealed)
```

**Critical: Anti-Repeat Guard** (after pipeline, before execution):
- Blocks any exact command repeat (≥1 occurrence in episode history)
- Blocks prefix repeats (≥3 same-tool uses in episode)
- Replaces with random alternative from per-role pool (~20 commands each)
- **KNOWN ISSUE (Phase 5.2 fix needed):** When anti-repeat fires, `_ppo_pending = None` silently discards PPO trajectory — PPO NEVER learns to stop proposing repeats, creating a vicious cycle. Fix: store trajectory with negative reward (-5.0) instead of nullifying.

### Decision Source Tracking

Every `SmartDecisionResult` has a `source` field tracking which pipeline stage produced it:
- `"playbook"` — Curriculum-guided
- `"ppo"` — RL policy network
- `"registry"` — Command registry precondition match
- `"mentor"` / `"dual_mentor"` — GPT/Venice LLM call
- `"anti_repeat"` — Forced alternative (anti-repeat guard fired)
- `"fallback"` — Emergency fallback

---

## PPO Algorithm (`core/algorithms/ppo_agent.py`)

The primary RL algorithm with actor-critic architecture:

```python
@dataclass
class PPOConfig:
    state_dim: int = 512            # Must match state_encoder output
    action_dim: int = 5             # Discrete actions mapped via CommandActionMapper
    hidden_dims: List[int] = [512, 512, 256]
    clip_epsilon: float = 0.2       # Surrogate clipping
    gamma: float = 0.99             # Discount factor
    gae_lambda: float = 0.95        # GAE lambda
    learning_rate: float = 3e-4     # Adam LR
    lr_min: float = 1e-5            # Minimum LR after annealing
    epochs_per_update: int = 4      # PPO epochs per rollout
    minibatch_size: int = 64
    entropy_coef: float = 0.01      # Exploration bonus
    rollout_size: int = 256         # Steps before update
    max_grad_norm: float = 0.5      # Gradient clipping
```

**PPO Flow in SmartCoach:**
1. `ppo.select_action(state_tensor)` → action index, log_prob, value
2. `CommandActionMapper.action_to_commands(action_idx, state)` → candidate commands
3. If anti-repeat doesn't block → store transition in `_ppo_pending`
4. After step result → `ppo.store_transition(state, action, log_prob, reward, value, done)`
5. End of episode → `ppo.update(last_value)` with GAE-λ advantage estimation

### PPO Terminal Rewards (Phase 5.1)

At episode end, `SmartCoach.end_episode_ppo()` injects a completion bonus into the final PPO trajectory step based on the highest phase reached:
- EXFILTRATION: +350.0
- POST_EXPLOITATION: +175.0
- LATERAL_MOVEMENT: +75.0
- PRIVILEGE_ESCALATION: +30.0
- EXPLOITATION: +15.0

---

## State Encoder (`core/models/state_encoder.py`)

Produces a **512-dimensional** tensor from environment state dicts. First ~91 dimensions carry meaningful signal:

| Dims | Content |
|------|---------|
| 0-11 | Phase one-hot + progress (12 dims) |
| 12-26 | State flags: ports_discovered, shell_obtained, etc. (15 dims) |
| 27-46 | Top 20 port presence indicators (20 dims) |
| 47-58 | Service type presence (12 dims) |
| 59-70 | Numeric features: detection_risk, blue_team_alert, etc. (12 dims) |
| 71-80 | Action history encoding (10 dims) |
| 81-85 | LLM/Mentor features (5 dims) |
| 86-90 | Temporal features (5 dims) |
| 91-511 | Reserved (zero-padded) |

Always use `STATE_DIM = 512` for network input dimensions.

---

## CyberEnvironment (`core/environment/cyber_environment.py`)

Simulates a cybersecurity engagement through the cyber kill chain:

```
RECON → ENUMERATION → EXPLOITATION → PRIVILEGE_ESCALATION →
LATERAL_MOVEMENT → POST_EXPLOITATION → EXFILTRATION
```

**Key methods:**
- `reset()` → returns initial state dict
- `step(action)` → returns `(state, reward, done, info)`
- `get_global_state()` → canonical state dict consumed by all agents

Phase transitions driven by `phase_progress` counters. Three scenarios: `dynamic`, `simulation`, `metasploitable2`.

**Simulated Output Generator:** `SmartOrchestrator._generate_simulated_output()` contains 95+ command prefix handlers that produce realistic pentesting output for simulation mode. Each handler recognizes the command prefix and returns domain-appropriate output including discoverable artifacts (ports, services, credentials, etc.).

---

## SmartOrchestrator (`core/orchestration/smart_orchestrator.py`)

The main training coordinator (2360+ lines):

1. `run_training(episodes, target_ip)` — outer loop over episodes
2. `run_episode(episode_id, episode_number)` — per-episode: env reset, agent coordination, metric collection
3. `_run_step(episode_id, step, state)` — activates agents in phase-optimal order
4. `_generate_simulated_output(command)` — produces simulated tool output
5. `_parse_output_for_discoveries(output, command)` — regex extraction of ports, services, credentials, shells
6. `_compute_episode_metrics()` — reward-invariant + reward metrics for reporting

### Discovery Board

Cross-agent shared state (reset per episode):
```python
self.discovery_board = {
    "ports": set(), "services": set(), "credentials": set(),
    "vulns": set(), "shells": set(), "users": set(),
    "web_paths": set(), "phase": "RECON", "flags_set": set(),
}
```

**KNOWN ISSUE (Phase 5.2 fix needed):** Each agent's `SmartRewardCalculator` has an independent `discoveries` set. Same port/service discovered by 5 agents gets 5× the reward bonus. Fix: shared `_episode_shared_discoveries` set passed to `record_result()`.

---

## Command Registry (`core/commands/command_registry.py`)

144+ real pentesting commands as `CommandTemplate` dataclasses organized by `AttackPhase`:

```python
register(CommandTemplate(
    name="my_new_tool",
    template="mytool --target {target} --option {option}",
    description="Description of what the tool does.",
    phase=AttackPhase.ENUMERATION,
    required_params=["target"],
    optional_params={"option": "default_value"},
    preconditions={"ports_discovered"},
    success_indicators=["SUCCESS", "Found"],
    typical_reward=3.0,
    tags={"recon", "web"},
))
```

Phases: `RECON`, `ENUMERATION`, `EXPLOITATION`, `PRIVILEGE_ESCALATION`, `LATERAL_MOVEMENT`, `POST_EXPLOITATION`, `EXFILTRATION`

When adding commands:
1. Ensure preconditions match existing state flags in `CyberEnvironment`
2. Add to relevant `Playbook` in `core/knowledge/pentesting_playbooks.py` if part of a chain
3. Add simulated output handler in `SmartOrchestrator._generate_simulated_output()` for simulation mode
4. Update `SmartRewardCalculator.DISCOVERY_BONUSES` if introducing new discovery types

---

## Reward System (`core/llm/reward_calculator.py`)

`SmartRewardCalculator` provides shaped rewards:

| Component | Value | Description |
|-----------|-------|-------------|
| REWARD_MULTIPLIER | 2.5 | Scales positive rewards only |
| progress_bonus | 1.0/step | Honest scaling (Phase 5.1: was 12.0) |
| phase_advance_multiplier | 4.0 | Bonus for reaching new phases |
| novelty_bonus | 5.0 × 1.5 | First use of a new command template |
| redundancy_penalty | 3.0 × n (soft cap 30) | Linear penalty, uncapped (Phase 5.1) |
| Floor | -5.0 | Per-step minimum to prevent PPO value divergence |

**Discovery Bonuses (selected):**
- `open_port`: 2.5, `service`: 5.0, `version`: 6.5
- `credential`: 20.0, `password`: 26.0
- `shell`: 50.0, `root_shell`: 130.0, `flag`: 200.0

**KNOWN ISSUE (Phase 5.2 fix needed):** No ceiling cap — max single-step can reach ~655 (flag + root_shell + phase advance). This asymmetry (floor -5.0 vs ceiling ~655) can destabilize PPO value function. Fix: add `max(min(raw_total, 100.0), -5.0)`.

---

## GPTManager — Centralized LLM Gateway

**ALL LLM calls MUST go through `GPTManager`** (`core/gpt_manager.py`). Never call OpenAI directly.

```python
# ✅ CORRECT
from core.gpt_manager import GPTManager
gpt = GPTManager()
result = gpt.gpt_request(prompt, task_type="tactical", agent_id="RedAgent")

# ❌ WRONG — direct API call
import openai
response = openai.chat.completions.create(...)
```

**Model Routing (Phase 5.1 centralized):**

| Role / Task | Model | Fallback |
|-------------|-------|----------|
| Red, Orion, tactical, strategic, reasoning | `gpt-5-mini` | `gpt-4o-mini` |
| Scout, Shadow, Blue, analysis, classification | `gpt-5-nano` | `gpt-4o-mini` |
| Postmortem (deep end-of-run analysis) | `gpt-5.2` | `gpt-5-mini` |
| Default / General | `gpt-5-mini` | `gpt-4o-mini` |
| Universal fallback | — | `gpt-4o-mini` |

**Key features:**
- Response caching by state fingerprint
- Per-episode token budgets with `can_make_request()` checks
- Automatic offline mode when `OPENAI_API_KEY` is missing
- Rate limiting and retry with exponential backoff
- `FakeGPTManager` in `core/testing/` for deterministic tests

---

## SmartMentor (`core/llm/smart_mentor.py`)

Provides GPT-powered guidance when the SmartCoach pipeline escalates to LLM:

- `SmartMentor`: Single-model mentor with system prompt containing pentesting knowledge
- `DualMentor`: Dual-model mentor (GPT primary + Venice secondary) for consensus
- `AttackContext`: Rich state representation for LLM prompts (phase, discoveries, history, flags)

The system prompt in `_build_system_prompt()` contains the mentor's domain knowledge. This is a critical injection point for scenario-specific knowledge (e.g., Metasploitable 2 exploitation paths).

---

## Memory System

### MemoryRouter (`core/multiagent/memory_router.py`)
- Prioritized Experience Replay with **SumTree** for O(log n) sampling
- Per-agent buffers with deduplication via state-action hashing
- SQLite persistence for durability
- Strategic directive logging between agents
- Token usage tracking per agent

### EnhancedMemorySync (`core/memory/enhanced_memory_sync.py`)
- Thread-safe cross-agent memory fusion
- Relevance-based insight sharing
- Periodic background synchronization

---

## Postmortem System (`core/postmortem/`)

End-of-run analysis using GPT-5.2 (feature-flagged):

- **OrionPostmortem**: Analyzes episode transcripts, identifies strategic errors, generates improvement instructions
- **SkillLibrary**: Persistent storage of `SkillCard` objects extracted from postmortem analysis. Skills are reused in future episodes to avoid repeating mistakes.
- **PostmortemResult**: Structured output with analysis, skill cards, and memory operations
- Output saved to `postmortems/` as JSON files

---

## Testing

### Framework & Patterns
- **pytest** with class-based and function-based tests
- **199 tests** all passing as of Phase 5.1 (commit `7ff2ad5`)
- Mock GPT with `FakeGPTManager` — deterministic, no API calls
- Mock tools with `StubToolRunner` — tracks commands without execution
- Use `ARIASKA_DRY_RUN=1` env var for safe testing
- Use `offline=True, enable_llm=False` in `TrainingConfig` for tests

### Test Files

| Test File | Scope |
|-----------|-------|
| `test_phase0_invariants.py` | Foundation: GPTManager, agent init, determinism |
| `test_phase2_invariants.py` | Metasploitable layer, sandboxed execution |
| `test_phase3_invariants.py` | State encoder, PPO, playbooks, command registry |
| `test_smart_integration.py` | Full SmartOrchestrator integration |
| `test_training_smoke.py` | End-to-end training pipeline |
| `test_multiagent_training_smoke.py` | Multi-agent coordination |
| `test_role_separation.py` | Agent role domain exclusivity |
| `test_cli_behavior.py` | CLI argument parsing and behavior |
| `test_ariaska_systems.py` | System-level integration |
| `test_testing_utilities.py` | Testing infrastructure itself |
| `test_online_requires_key.py` | API key requirement validation |
| `test_online_makes_mentor_calls.py` | Online mentor call verification |

### Running Tests
```bash
make test                    # Run full pytest suite (199 tests)
make smoke                   # Quick 3-episode smoke test
pytest tests/ -v --tb=short  # Verbose with short tracebacks
pytest tests/test_smart_integration.py -v  # Run specific test file
```

### Writing Tests
```python
import pytest
import os
from unittest.mock import MagicMock, patch
from core.testing import FakeGPTManager, StubToolRunner, get_tool_runner

class TestMyFeature:
    @pytest.fixture(autouse=True)
    def setup(self):
        os.environ['ARIASKA_DRY_RUN'] = '1'
        self.gpt = FakeGPTManager(seed=42)
        self.tool_runner = get_tool_runner(testing=True)

    def test_agent_action(self):
        from core.agents.red_agent import RedAgent
        agent = RedAgent(gpt_manager=self.gpt, verbosity="quiet")
        # Test agent behavior with deterministic GPT
```

---

## Code Standards

### Python Style
- **PEP 8** with type hints on all function signatures
- Google-style docstrings on all public functions and classes
- Descriptive variable names (e.g., `episode_reward` not `ep_r`)
- `rich` library for console output (`Console()`, `Panel()`, `Table()` — not `print()`)
- `logging.getLogger("ariaska.<module>")` for all debug/info logging

### Error Handling
- Wrap agent initialization in try/except with meaningful messages
- Validate state dimensions before neural network forward passes
- Handle environment termination gracefully (check `self.done`)
- All GPT calls must handle offline/failure modes

### Import Patterns — CRITICAL

This project has complex circular dependency chains. Follow these patterns strictly:

```python
# ✅ Lazy import pattern (used throughout)
def _init_agents(self):
    from core.agents.red_agent import RedAgent  # Import inside method
    self.red_agent = RedAgent(...)

# ✅ TYPE_CHECKING guard for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.gpt_manager import GPTManager

# ✅ Lazy-load with caching (see _lazy_ppo() in smart_coach.py)
_cached_class = None
def _lazy_load():
    global _cached_class
    if _cached_class is None:
        from core.algorithms.ppo_agent import PPOAgent
        _cached_class = PPOAgent
    return _cached_class

# ❌ NEVER do top-level imports of core modules that create circular deps
```

### Shared GPTManager Pattern
Never create multiple `GPTManager` instances. Inject the shared one:
```python
# In AgentManager or SmartOrchestrator:
self.gpt_manager = GPTManager()
self.red_agent = RedAgent(gpt_manager=self.gpt_manager)
self.blue_agent = BlueAgent(gpt_manager=self.gpt_manager)
# All coaches get the same gpt_manager instance
```

---

## Configuration & Runtime

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Required for live LLM calls; absence triggers offline mode |
| `ARIASKA_DRY_RUN` | Set to `1` to prevent real command execution |
| `PYTHONPATH` | Should include project root |

### Runtime Flags (`core/runtime_flags.py`)
```python
from core.runtime_flags import get_runtime_flags
flags = get_runtime_flags()
# flags.offline, flags.enable_llm, flags.require_llm, flags.dry_run
```

### CLI Commands
```bash
python ariaska_cli.py smart-train --episodes 100 --steps 120 --seed 42 --env sim
python ariaska_cli.py smart-train --env msf --target 192.168.56.101    # Live Metasploitable
python ariaska_cli.py status                                            # System diagnostics
```

### Makefile Targets
```bash
make venv          # Create virtualenv and install dependencies
make train         # 100 episodes, 120 steps, simulated
make train-quick   # 10 episodes, 120 steps
make train-msf     # Live Metasploitable 2 training
make smoke         # 3 episodes, 10 steps (quick validation)
make test          # Run pytest (199 tests)
make last          # View last training run
make clean         # Clean temporary files
```

---

## Key Hyperparameters

| Parameter | Value | Location |
|-----------|-------|----------|
| State dimension | 512 | `core/models/state_encoder.py` |
| Action dimension | 5 | `PPOConfig` |
| PPO clip epsilon | 0.2 | `PPOConfig` |
| PPO learning rate | 3e-4 | `PPOConfig` |
| GAE lambda | 0.95 | `PPOConfig` |
| Discount gamma | 0.99 | `PPOConfig` |
| Max steps per episode | 120 | `SmartOrchestratorConfig` |
| Reward multiplier | 2.5 | `SmartRewardCalculator` |
| Progress bonus/step | 1.0 | `SmartRewardCalculator` |
| Phase advance multiplier | 4.0 | `SmartRewardCalculator` |
| Novelty weight | 1.5 | `SmartRewardCalculator` |
| Redundancy penalty | 3.0 × n (soft cap 30) | `SmartRewardCalculator` |
| Reward floor | -5.0 | `RewardBreakdown.calculate_total()` |
| Mentor anneal | 60% → 10% | `MentorPolicy` |
| Replay buffer | 10K (MemoryRouter) / 100K (ReplayBuffer) | Various |

---

## Metasploitable 2 — Target Knowledge

The primary training target. Key vulnerable services (for knowledge injection into mentors and agents):

| Port | Service | Vulnerability | Exploitation Path |
|------|---------|---------------|-------------------|
| 21 | vsftpd 2.3.4 | Backdoor | `exploit/unix/ftp/vsftpd_234_backdoor` → root shell |
| 22 | OpenSSH 4.7p1 | Weak creds | `msfadmin:msfadmin` default credentials |
| 23 | Telnet | Default creds | `msfadmin:msfadmin` login |
| 25 | Postfix SMTP | Open relay | User enumeration via VRFY |
| 80 | Apache 2.2.8 | PHP + DVWA | Web app attacks, SQL injection |
| 139/445 | Samba 3.0.20 | CVE-2007-2447 | `exploit/multi/samba/usermap_script` → root |
| 512-514 | rexec/rlogin/rsh | No auth | Direct remote command execution |
| 1099 | Java RMI | RCE | `exploit/multi/misc/java_rmi_server` |
| 1524 | ingreslock | Backdoor | `telnet <target> 1524` → instant root shell |
| 2049 | NFS | World-readable | Mount root filesystem, plant SSH keys |
| 3306 | MySQL 5.0.51a | No root password | `mysql -h <target> -u root` |
| 5432 | PostgreSQL 8.3.0 | Default creds | `postgres:postgres` + `COPY ... FROM PROGRAM` → RCE |
| 5900 | VNC | Password: password | `vncviewer <target>` → desktop access |
| 6667 | UnrealIRCd | Backdoor | `exploit/unix/irc/unreal_ircd_3281_backdoor` → root |
| 8180 | Tomcat | Default creds | `tomcat:tomcat` manager → WAR deploy → shell |

These paths should be injected into SmartMentor system prompts and agent emergency fallbacks for MS2-targeted training.

---

## Development Phases & Current Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 0-3 | ✅ Complete | Foundation, interfaces, PPO, state encoder, command registry, playbooks |
| Phase 4 | ✅ Complete | Multi-agent coordination, reward shaping, anti-repeat, stuck detection |
| Phase 5.0 | ✅ Complete | Smart orchestrator, 100-episode validation |
| Phase 5.1 | ✅ Complete | Honest reward scaling, model centralization (gpt-5-mini), reward-invariant metrics. 199/199 tests, +10,009.5 avg reward, 100% EXFILTRATION, 92.2% skill-based. Commit `7ff2ad5`. |
| Phase 5.2 | 🔄 In Progress | PPO anti-repeat fix, discovery dedup, parser hardening, reward cap, MS2 knowledge injection, agent intelligence enhancement |

### Phase 5.2 Known Issues to Fix

1. **PPO Silent Disconnect**: Anti-repeat fires ~70% of decisions, `_ppo_pending = None` discards trajectory — PPO never learns
2. **Discovery Double-Counting**: 5 agents × independent reward calculators = 5× bonus per discovery
3. **[SIM] Parser Precedence Bug**: `if not output or output.startswith("[SIM]") and len(output) < 30:` — missing parentheses
4. **Reward Asymmetry**: Max ~655 vs floor -5.0 destabilizes PPO value function
5. **Generic Mentor Knowledge**: SmartMentor system prompt lacks MS2-specific exploitation paths
6. **Generic Agent Fallbacks**: RedAgent emergency fallbacks don't include MS2-specific high-value commands

### Reward-Invariant Metrics (Phase 5.1)

These metrics measure REAL learning quality independent of reward scaling:
- `unique_commands`: Number of distinct commands used in episode
- `diversity_ratio`: unique_commands / total_steps
- `total_discoveries`: Count of genuinely new discoveries
- `step_at_first_exploit`: How quickly agent reaches exploitation
- `completion_bonus_applied`: Whether EXFILTRATION was reached

---

## Common Development Tasks

### Tuning Reward Shaping
1. Adjust `PHASE_REWARDS` and `DISCOVERY_BONUSES` in `core/llm/reward_calculator.py`
2. Modify `REWARD_MULTIPLIER` (currently 2.5) for overall scaling
3. Tune redundancy penalty growth rate (currently 3.0 × usage_count)
4. Adjust `progress_bonus_per_step` (currently 1.0 — keep honest)
5. Always validate with 10-episode training runs and check reward-invariant metrics

### Modifying the State Encoder
1. Edit `core/models/state_encoder.py` — add new features after existing sections
2. Update the docstring architecture map at top of file
3. Keep `STATE_DIM = 512` unless changing ALL network architectures
4. Update `tests/test_phase3_invariants.py` with new dimension checks

### Adding Simulated Output for New Commands
1. Add command prefix handler in `SmartOrchestrator._generate_simulated_output()`
2. Include discoverable artifacts (ports, services, credentials) in the output
3. Ensure `_parse_output_for_discoveries()` regex patterns can extract them
4. Test with `make smoke` to verify the full pipeline

### Training Validation Workflow
1. Run `make test` → expect 199/199 passing
2. Run `make train-quick` → 10 episodes, check metrics
3. Verify reward-invariant metrics (diversity_ratio > 0.3, discoveries > 0)
4. Check decision source distribution (want PPO > 30%, anti_repeat < 40%)
5. Verify phase progression (should reach EXFILTRATION in most episodes)

---

## Security & Safety

- **Never hardcode API keys** — use `.env` file and `python-dotenv`
- **Never execute real commands in tests** — use `StubToolRunner`
- **RFC1918 validation** — `RealToolRunner` only allows targeting private IP ranges
- **Sandboxed execution** — live tool execution goes through validated runners
- **Input sanitization** — all LLM outputs are sanitized before use as commands
- **Deterministic mode** — set `--seed` for reproducible experiments
- **Dry run mode** — `ARIASKA_DRY_RUN=1` prevents any real command execution

---

## Code Generation Rules

When generating code for this project:

1. **Always inject `gpt_manager`** — never create new GPTManager instances inside agents
2. **Use `rich` for output** — `Console()`, `Panel()`, `Table()`, not `print()`
3. **Follow the lazy import pattern** to avoid circular dependencies (see Import Patterns above)
4. **Include offline fallback paths** for any LLM-dependent feature
5. **Add `FakeGPTManager` support** in any new component that uses LLM
6. **Use `@dataclass`** for configuration objects (see `PPOConfig`, `SmartOrchestratorConfig`)
7. **Log with `logging.getLogger("ariaska.<module>")`** — not print statements
8. **Handle `torch.device`** properly — check `cuda` availability, default to `cpu`
9. **Respect the phase system** — commands must match `AttackPhase` and satisfy `preconditions`
10. **Write tests** — any new module needs corresponding tests in `tests/`
11. **Never create files that duplicate existing functionality** — check existing modules first
12. **Keep reward signals honest** — no artificial inflation of training metrics
13. **Preserve the 199-test baseline** — all changes must keep tests passing
14. **Use reward-invariant metrics** to validate real learning quality
15. **When modifying the decision pipeline** — trace the full flow: SmartCoach.decide() → anti-repeat → record_result() → PPO store_transition → end_episode_ppo()