<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/OpenAI-GPT--5.2--Hybrid-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI">
  <img src="https://img.shields.io/badge/Tests-1186%20Passing-brightgreen?style=for-the-badge" alt="Tests">
  <img src="https://img.shields.io/badge/Phase-34-blueviolet?style=for-the-badge" alt="Phase 34">
  <img src="https://img.shields.io/badge/License-Non--Commercial-red?style=for-the-badge" alt="License">
</p>

<br>

```
     █████╗ ██████╗ ██╗ █████╗ ███████╗██╗  ██╗ █████╗
    ██╔══██╗██╔══██╗██║██╔══██╗██╔════╝██║ ██╔╝██╔══██╗
    ███████║██████╔╝██║███████║███████╗█████╔╝ ███████║
    ██╔══██║██╔══██╗██║██╔══██║╚════██║██╔═██╗ ██╔══██║
    ██║  ██║██║  ██║██║██║  ██║███████║██║  ██╗██║  ██║
    ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝
            ██████╗ ██╗
            ██╔══██╗██║           Autonomous Multi-Agent
            ██████╔╝██║           Reinforcement Learning for
            ██╔══██╗██║           Live Penetration Testing
            ██║  ██║███████╗
            ╚═╝  ╚═╝╚══════╝      by Filip Volf
```

<p align="center">
  <strong>5 Agents · MicroChain + PhaseGuide Intelligence · PPO Actor-Critic v3.0 · 107K Knowledge Entries · 1,186 Tests</strong>
</p>

---

## Executive Summary

Ariaska_RL is an autonomous multi-agent reinforcement learning system purpose-built for live authorized penetration testing and CTF engagements. It deploys five specialized agents — Scout, Red, Blue, Shadow, Orion — coordinated by a SmartOrchestrator through a tiered GPT-5.2 hybrid decision pipeline, with PPO Actor-Critic as the primary RL algorithm.

The system operates in live-only mode against real targets (Metasploitable 2/3, HackTheBox). There is no simulated environment. All command execution happens through a sandboxed live executor with RFC1918 validation and IP allowlisting.

| Metric | Value |
|--------|-------|
| Core modules | 207 |
| Core SLOC | ~107K Python |
| Test suite | 1,186 tests across 68 files |
| Knowledge corpus | 107,933 entries (v2 schema, 18 JSONL partitions) |
| Knowledge indices | 11 prebuilt indices (~110MB) |
| Current phase | Phase 34 — PhaseGuidedLLM + Structured Distillation |
| Author | Filip Volf |

---

## System Architecture

```
ariaska_cli.py
  └─ SmartOrchestrator (6,882L)
       ├─ SmartCoach (8,197L) ×5 — one per agent
       │    ├─ 4-stage decision pipeline: Playbook → PPO → Registry → GPT Mentor
       │    ├─ MicroChain (P27): nano→mini→nano 3-stage scoring chain
       │    ├─ PhaseGuidedLLM (P34): structured guidance + distillation packets
       │    ├─ Evidence Gate (P27): exploit validation (off / log / enforce)
       │    ├─ Anti-repeat guard + role exclusivity
       │    └─ TacticalCortex quality gate (7 rule categories)
       │
       ├─ GPTManager (1,439L) — centralized LLM gateway
       │    ├─ BudgetManagerV2 (P17+36): 999K max / 499.5K min floor
       │    ├─ 4-tier routing: codex / full / mini / nano
       │    ├─ Response caching by state fingerprint
       │    └─ ROI tag tracking + burst pool (12%)
       │
       ├─ PPOAgent (1,543L) — Actor-Critic v3.0 (R68–R80)
       │    ├─ 512-dim state, 5-dim action
       │    ├─ Phase-gated actor heads (HRL-lite)
       │    ├─ Self-Imitation Learning (SIL)
       │    ├─ Symlog value compression + cosine entropy schedule
       │    ├─ Dual-horizon GAE (λ=0.97 + λ=0.70, blend=0.65)
       │    └─ KL-adaptive LR + spectral norm critic
       │
       ├─ 5 Agents (AgentInterface + MemorySyncInterface)
       │    ├─ ScoutAgent  — recon, port scanning, service fingerprinting
       │    ├─ RedAgent    — exploitation, privesc, shell acquisition
       │    ├─ BlueAgent   — defensive, honeypots, credential resets
       │    ├─ ShadowAgent — stealth, alert monitoring, detection avoidance
       │    └─ OrionAgent  — strategic coordination, phase transitions
       │
       ├─ CyberEnvironment (2,854L) — kill chain state machine
       ├─ LiveDashboard v5.0 (2,284L) — Rich terminal UI
       ├─ EventBus — pub/sub step and agent event routing
       └─ TraceWriter — JSONL structured event logging
```

---

## Intelligence Pipeline

### MicroChain (Phase 27)

3-stage iterative LLM scoring chain for command selection:

| Stage | Model | Function |
|-------|-------|----------|
| 1 (nano) | gpt-5.2-nano | Classify tactical situation |
| 2 (mini) | gpt-5.2-mini | Generate candidate commands (JSON, max 3) |
| 3 (nano) | gpt-5.2-nano | Score candidates: phase_fit / evidence_support / novelty |

Codex escalation triggers when top score < `ESCALATION_THRESHOLD` (default 0.40, tunable via `MC_ESCALATE_THRESHOLD` env var) or stagnation >= 9 steps with scores < 0.55.

Ablation toggle: `MC_NANO_ABLATION=1` bypasses nano stages 1+3, replacing with heuristic classify/score for A/B testing.

### PhaseGuidedLLM (Phase 34)

Structured JSON guidance for agents:
- Phase decision with evidence-driven stay/move conditions
- 3–6 candidate next actions as template picks
- Anomaly probes for stagnation/flag-hunting
- Distillation packet (MentorTrace target) for apprentice policy training

Model routing: gpt-5.2-mini default; codex escalation when confidence < 0.45, contradictions detected, or semantic stall >= 8 steps.

### MentorTrace (Phase 30)

Structured mentor→apprentice transfer metadata extending TeacherTrace:
- Decision rationale + confidence + alternatives
- 16-dim summary feature extraction for state encoder Section 16
- Bridge to BCBuffer via `to_teacher_trace().to_bc_sample()`

### Distillation Pipeline

```
MentorTrace → TeacherTrace → BCSample → BCBuffer (capacity: 2000, dedup)
                                              ↓
                                         PPO BC loss (weight × confidence × decay)
```

Every mentor call produces a TeacherTrace containing teacher decision, student parallel prediction, divergence measurement, and post-hoc success marking. The BCBuffer maintains thread-safe dedup by (state_hash, action) pairs.

### Evidence Gate (Phase 27)

Validates exploit-phase commands have supporting evidence before execution. Three modes controlled by `FF_STRICT_EXPLOIT_GATE`:

| Mode | Behavior |
|------|----------|
| `off` | Gate disabled |
| `log` | Log rejects without blocking |
| `enforce` | Block exploit commands lacking evidence, fall back to registry |

---

## Token Budget Strategy (Phase 17 + 36)

BudgetManagerV2 manages per-episode and per-model-tier token allocation:

| Parameter | Value |
|-----------|-------|
| Total budget | 999,000 tokens (~$3.33/episode ceiling) |
| Min budget floor | 499,500 tokens (50%) |
| Burst pool | 12% of max budget as reserve |
| Burst step cap | 3% of max budget per step |
| Burst cooldown | 5 steps between bursts |

Dynamic scaling: budget scales from $3.33 → $0.50 based on learning maturity signal (`0.4 × avg_success_rate + 0.3 × skill_coverage + 0.2 × discovery_efficiency + 0.1 × (1 - stagnation_rate)`).

### Tier Allocation (Phase 36)

| Tier | Model | Budget | Share | Uses |
|------|-------|--------|-------|------|
| codex | gpt-5.2-codex | 199,800 | 20% | Tactical/strategic reasoning, postmortem, analysis |
| full | gpt-5.2 | 199,800 | 20% | Parsing, interpretation, verification |
| mini | gpt-5.2-mini | 299,700 | 30% | Playbook selection, structured extraction |
| nano | gpt-5.2-nano | 299,700 | 30% | Classification, MicroChain stages 1+3, reserve |

---

## Agent Roles

| Agent | Role | Domain |
|-------|------|--------|
| **ScoutAgent** | `recon` | Network discovery, port scanning, service fingerprinting, version detection |
| **RedAgent** | `offensive` | Exploitation, privilege escalation, shell acquisition. Primary PPO-trained. DQN+GPT hybrid with emergency fallbacks |
| **BlueAgent** | `defensive` | Honeypots, credential resets, firewall management, alert handling |
| **ShadowAgent** | `stealth` | Alert monitoring, scan timing, detection avoidance, action overrides |
| **OrionAgent** | `strategic` | Cross-agent coordination, strategic reviews, directives, phase transitions |

**Phase-dependent activation order** (configured in `SmartOrchestrator._should_activate()`):
- RECON: Scout → Shadow → Orion → Red → Blue
- EXPLOITATION: Red → Shadow → Scout → Orion → Blue
- EXFILTRATION: Red → Shadow → Orion → Scout → Blue

---

## Kill Chain Phases

```
RECON → ENUMERATION → EXPLOITATION → PRIVILEGE_ESCALATION →
LATERAL_MOVEMENT → POST_EXPLOITATION → EXFILTRATION → CLOSEOUT
```

Phase progression is enforced by `FF_STRICT_PHASE_LADDER` (default: ON). Commands must match their `AttackPhase` preconditions against the current `CyberEnvironment` state.

---

## Reward System

**File:** `core/llm/reward_calculator.py` (859L)

Symmetric reward range: **[-15.0, +50.0]** (3.3:1 ratio). Positive discoveries are rarer, so they carry higher magnitude.

### Discovery Bonuses

| Discovery | Reward |
|-----------|--------|
| open_port | 2.5 |
| service | 5.0 |
| version | 6.5 |
| credential | 20.0 |
| password | 26.0 |
| shell | 40.0 |
| user_flag | 50.0 |
| root_flag | 50.0 |
| root_shell | 80.0 |

### Phase Progression Rewards

| Phase | Reward |
|-------|--------|
| RECON | 0.0 |
| ENUMERATION | 5.0 |
| EXPLOITATION | 15.0 |
| PRIVILEGE_ESCALATION | 30.0 |
| LATERAL_MOVEMENT | 45.0 |
| POST_EXPLOITATION | 60.0 |
| EXFILTRATION | 75.0 |
| CLOSEOUT | 90.0 |

---

## Replay Mode (Phase 31)

Replay past engagements from JSONL event traces:

```bash
python ariaska_cli.py replay traces/events_20260218_142627.jsonl --verbose
```

Renders step-by-step Rich panels showing agent actions, decision sources, discoveries, reward breakdowns, and phase transitions.

---

## Observability

### LiveDashboard v5.0

Real-time Rich terminal UI showing:
- Per-agent commands with output, reasoning, and discoveries
- ASCII sparkline reward trends
- Phase progression timeline with kill chain bar
- Decision pipeline visualization (4-stage)
- Per-coach PPO training metrics
- Teaching point annotations + budget pressure indicators
- Discovery board heatmap

### JSONL Structured Logging

Event traces written to `traces/` for post-hoc analysis. Consumed by EpisodeReplayer and postmortem tools.

### EventBus

Pub/sub event routing for decoupled step, agent, and episode events. Subscribers include TraceWriter, LiveDashboard, and analytic collectors.

---

## Feature Flags

51 feature flags in `core/feature_flags.py` with environment variable overrides (prefix `FF_`).

### Profiles (auto-detected by `resolve_profile()`)

| Profile | Condition | LLM Flags |
|---------|-----------|-----------|
| CLOUD | `OPENAI_API_KEY` present | All 5 LLM role flags ON |
| DETERMINISTIC | Running under pytest / `ARIASKA_DRY_RUN=1` | All LLM flags OFF |
| OFFLINE | No API key | All LLM flags OFF |

### Key Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `FF_USE_MICRO_CHAIN` | ON | MicroChain 3-stage scoring |
| `FF_STRICT_EXPLOIT_GATE` | enforce | Evidence gate mode (off/log/enforce) |
| `FF_BUDGET_MANAGER_V2` | ON | Dynamic per-episode token budgets |
| `FF_STRICT_PHASE_LADDER` | ON | Phase ordering enforcement |
| `FF_BC_LOSS` | ON | Behavioral cloning loss from TeacherTrace |
| `FF_TEACHER_TRACE` | ON | TeacherTrace distillation pipeline |
| `FF_PARALLEL_AGENTS` | ON | Multi-agent parallel activation |
| `FF_NEUROMODULATORS` | ON | Biologically-inspired exploration control |

---

## Performance Metrics (Reward-Invariant)

These measure real learning quality independent of reward scaling:

| Metric | Description |
|--------|-------------|
| `unique_commands` | Distinct commands used per episode |
| `diversity_ratio` | unique_commands / total_steps |
| `total_discoveries` | Genuinely new discoveries per episode |
| `step_at_first_exploit` | Speed to exploitation phase |
| `completion_bonus_applied` | Whether EXFILTRATION was reached |

Always validate changes with these metrics, not raw reward numbers.

---

## Running Ariaska

### Prerequisites

- Python 3.11+ (developed on 3.13)
- PyTorch 2.0+
- OpenAI API key (for live LLM; offline mode without)
- Target machine (Metasploitable 2/3 or HTB)

### Setup

```bash
git clone <repo>
cd Ariaska_RL
make venv
cp .env.example .env  # Add OPENAI_API_KEY
```

### CLI Commands

```bash
# Live engagement (continuous until flag capture or --steps limit)
python ariaska_cli.py smart-train --target 10.129.1.54 --steps 500

# CTF mode (auto-close on both user + root flags)
python ariaska_cli.py smart-train --target 10.129.1.54 --ctf --steps 500

# Watchdog mode (persistent: run, monitor, debug, relaunch until both flags)
python ariaska_cli.py watchdog --target 10.129.1.54 --max-attempts 10

# Replay a past engagement
python ariaska_cli.py replay traces/events_*.jsonl --verbose

# System status
python ariaska_cli.py status
```

### Makefile Targets

```bash
make test              # Full test suite (1,186 tests)
make test-fast         # Skip integration tests
make smoke             # 3 episodes, fast validation
make train             # MS3 LIVE, 100 episodes
make train-quick       # MS3 LIVE, 10 episodes
make train-htb TARGET=10.129.x.x  # HTB target
make overnight         # Progressive 300 episodes
make status            # System diagnostics
make last              # View last training run
```

---

## Testing

**Framework:** pytest, 1,186 tests across 68 files.

```bash
make test                              # Full suite
pytest tests/ -v --tb=short            # Verbose with short tracebacks
pytest tests/test_phase0_invariants.py # Single file
```

### Test Utilities

| Utility | Purpose |
|---------|---------|
| `FakeGPTManager(seed=N)` | Deterministic LLM responses, token tracking, request history |
| `StubToolRunner` | Never executes commands, tracks invocations |
| `RealToolRunner` | RFC1918 allowlist, blocked command list, IP validation |

### Test Pattern

```python
import pytest, os
from core.testing import FakeGPTManager, get_tool_runner

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

---

## Safety & Lab Scope

- **Live-only** — all execution against real authorized targets
- **RFC1918 validation** — `RealToolRunner` only allows private IP ranges
- **Sandboxed executor** — additional safety layer for command execution
- **Dry run mode** — `ARIASKA_DRY_RUN=1` prevents real execution
- **No hardcoded credentials** — `.env` + `python-dotenv`
- **Deterministic mode** — `--seed` for reproducibility
- **Ethics mode** — training / assessment / demo profiles
- **Input sanitization** — all LLM outputs sanitized before command construction

---

## Phase History (27–34)

| Phase | Focus |
|-------|-------|
| 27 | MicroChain (nano→mini→nano), Evidence Gate, intelligence pipeline, reward boost |
| 28 | Postmortem refinements, skill library persistence |
| 29 | Event bus decoupling, trace writer integration |
| 30 | MentorTrace: structured mentor→apprentice transfer metadata |
| 31 | EpisodeReplayer: JSONL trace replay via CLI |
| 32 | MicroChain tuning: env-var configurable escalation threshold |
| 33 | Budget burst pool, candidate cap, dynamic burst cooldown |
| 34 | PhaseGuidedLLM: structured guidance + distillation packets, codex escalation |

---

## Knowledge System (v2)

107,933 entries across 18 JSONL partitions in `data/knowledge_candidates_v2/`.

| File | Entries |
|------|---------|
| exploitdb.jsonl | 46,491 |
| cves.jsonl | 25,467 |
| commands.jsonl | 24,342 |
| wordlists.jsonl | 5,814 |
| + 14 more | 5,819 |

11 prebuilt indices in `data/knowledge_indices/`: by_port (417 keys), by_cve (25K), by_service (193), by_phase (8), by_tag (71K), by_template (205), by_platform (79), by_killchain (6), by_vuln_family (19), by_exploit_archetype (13), by_origin (20).

Rebuild: `python scripts/build_knowledge_indices.py`

---

## License

Source Available Non-Commercial License v1.0. Copyright (c) 2024–2026 Filip Volf.
See [LICENSE](LICENSE) for full terms.
