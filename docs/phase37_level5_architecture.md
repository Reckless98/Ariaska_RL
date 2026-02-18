# Phase 37 — Level 5 GPT↔RL Neural Integration

> **Author:** Filip Volf | **Phase:** 37 | **Status:** ACTIVE

## Architecture Overview

Phase 37 implements **Level 5 GPT↔RL Integration** — the deepest possible fusion between LLM intelligence and the PPO reinforcement learning pipeline. LLM knowledge is injected directly into the neural network's:

1. **Policy logits** — via action prior vector injection
2. **State representation** — via 256-dim LLM feature concatenation
3. **Value estimation** — via confidence-augmented value targets
4. **Gradient updates** — via KL teacher distillation + ranking + value regularization losses
5. **Exploration** — via curriculum-driven dynamic shaping

All LLM influence **decays over time** via the teacher anneal schedule, allowing the policy to internalize GPT knowledge and become fully autonomous.

---

## System Architecture

```mermaid
graph TB
    subgraph "LLM Intelligence Sources"
        MC[MicroChain<br/>nano→mini→nano]
        PG[PhaseGuidedLLM<br/>P34 structured guidance]
        SM[SmartMentor<br/>DualMentor GPT+Venice]
        MT[MentorTrace<br/>P30 structured transfer]
    end

    subgraph "LLMPolicyBridge"
        direction TB
        AG[compute_guidance]
        AP[Action Prior Vector<br/>action_dim logits]
        LF[LLM Feature Vector<br/>256-dim embedding]
        TD[Teacher Distribution<br/>soft probability dist]
        AN[Anneal Schedule<br/>cosine + maturity]
        CU[Curriculum Adjustments<br/>explore rate, diversity]
    end

    subgraph "PPO Actor-Critic v3.0"
        direction TB
        SA[select_action<br/>+ llm_prior injection]
        IP[input_proj<br/>512 or 768-dim]
        SB[SharedBackbone<br/>ResidualBlock + Attention]
        AC[Actor Head<br/>logits + phase gates]
        CR[Critic Head<br/>spectral-normed]
        UP[update<br/>+ KL + ranking + vreg]
    end

    subgraph "Losses"
        LP[L_policy: PPO clip]
        LV[L_value: clipped MSE]
        LH[L_entropy: cosine schedule]
        LBC[L_BC: teacher trace]
        LKL[L_KL: teacher distillation]
        LR[L_rank: margin ranking]
        LVR[L_vreg: value regularization]
    end

    MC --> AG
    PG --> AG
    SM --> AG
    MT --> AG

    AG --> AP
    AG --> LF
    AG --> TD
    AG --> AN

    AP -->|prior_alpha · prior| SA
    LF -->|concat with state| IP
    TD -->|stored in buffer| UP
    AN -->|decays all weights| UP

    IP --> SB
    SB --> AC
    SB --> CR
    SA --> AC

    UP --> LP
    UP --> LV
    UP --> LH
    UP --> LBC
    UP --> LKL
    UP --> LR
    UP --> LVR

    style AG fill:#f9f,stroke:#333,stroke-width:2px
    style AN fill:#ff9,stroke:#333,stroke-width:2px
    style LKL fill:#f66,stroke:#333,stroke-width:1px
    style LR fill:#f66,stroke:#333,stroke-width:1px
    style LVR fill:#f66,stroke:#333,stroke-width:1px
```

---

## Data Flow

```mermaid
sequenceDiagram
    participant SC as SmartCoach
    participant BR as LLMPolicyBridge
    participant PPO as PPOAgent
    participant BUF as RolloutBuffer

    Note over SC: Each step in training loop

    SC->>BR: compute_guidance(state, mc_result, pg_result, mentor_conf)
    BR-->>SC: LLMGuidancePacket(prior, features, teacher_dist, alpha)

    SC->>PPO: select_action(state, llm_prior=prior, prior_alpha=alpha)
    Note over PPO: logits = network(state) + alpha * prior
    PPO-->>SC: action_idx, log_prob, value

    Note over SC: Execute command, observe reward

    SC->>PPO: store_transition(s, a, lp, r, v, done, teacher_dist, teacher_action)
    PPO->>BUF: buffer.add(... teacher_distribution, teacher_action)

    SC->>BR: record_step_outcome(reward, discoveries, exploit_success)
    Note over BR: Updates maturity signal for anneal

    Note over SC: End of episode

    SC->>PPO: update(last_value)
    Note over PPO: Compute L_total with KL + ranking + vreg
    PPO-->>SC: metrics dict

    SC->>BR: record_episode_end()
```

---

## Loss Equation

The total PPO loss with Level 5 auxiliary terms:

$$
L_{\text{total}} = \underbrace{\rho \cdot L_\pi}_{\text{PPO clip}} + \underbrace{c_v \cdot L_V}_{\text{value}} - \underbrace{c_H \cdot H(\pi)}_{\text{entropy}} + \underbrace{c_{\text{phase}} \cdot L_{\text{phase}}}_{\text{aux phase pred}} + \underbrace{c_{\text{bc}} \cdot L_{\text{BC}}}_{\text{teacher trace}} + \underbrace{c_{\text{kl}} \cdot L_{\text{KL}}}_{\text{teacher distill}} + \underbrace{c_{\text{rank}} \cdot L_{\text{rank}}}_{\text{margin ranking}} + \underbrace{c_{\text{vreg}} \cdot L_{\text{vreg}}}_{\text{value reg}}
$$

### Level 5 Auxiliary Losses (New in Phase 37)

**KL Teacher Distillation Loss:**
$$
L_{\text{KL}} = D_{\text{KL}}(\pi_{\text{teacher}} \| \pi_\theta) = \sum_a \pi_{\text{teacher}}(a) \log \frac{\pi_{\text{teacher}}(a)}{\pi_\theta(a)}
$$

**Ranking Margin Loss:**
$$
L_{\text{rank}} = \max\left(0, \; m - \left(f_\theta(s, a^*_T) - \max_{a \neq a^*_T} f_\theta(s, a)\right)\right)
$$
where $a^*_T$ is the teacher's preferred action and $m = 1.0$ is the margin.

**Value Regularization Loss:**
$$
L_{\text{vreg}} = \text{MSE}\left(V_\theta(s), \; V_{\text{target}}\right)
$$
where $V_{\text{target}}$ is derived from the teacher's confidence.

### Loss Coefficient Defaults

| Loss | Coefficient | Anneal | Min |
|------|------------|--------|-----|
| $c_{\text{kl}}$ | 0.15 | cosine + maturity | 0.01 |
| $c_{\text{rank}}$ | 0.05 | proportional to alpha | 0.005 |
| $c_{\text{vreg}}$ | 0.10 | proportional to alpha | 0.01 |

All coefficients decay proportionally to the teacher anneal alpha, ensuring the auxiliary losses shrink alongside the LLM influence.

---

## Teacher Anneal Schedule

```mermaid
graph LR
    subgraph "Anneal Timeline"
        S0[Step 0<br/>α=0.50<br/>50% LLM] --> S500[Step 500<br/>α≈0.40<br/>40%]
        S500 --> S1000[Step 1000<br/>α≈0.25<br/>25%]
        S1000 --> S2000[Step 2000<br/>α≈0.10<br/>10%]
        S2000 --> S3000[Step 3000<br/>α=0.02<br/>2%]
    end
```

The anneal follows a **cosine decay with maturity acceleration**:

$$
\alpha(t) = \alpha_{\min} + \frac{\alpha_{\text{init}} - \alpha_{\min}}{2} \left(1 + \cos\left(\pi \cdot \frac{t}{T}\right)\right) \cdot g(M)
$$

where:
- $t$ = current step, $T$ = total anneal steps (3000)
- $\alpha_{\text{init}} = 0.50$, $\alpha_{\min} = 0.02$
- $g(M)$ = maturity gate: accelerates decay when $M > 0.7$, boosts when struggling ($S_r < 0.2$)

**Maturity Signal:**
$$
M = 0.4 \cdot S_r + 0.3 \cdot R_v + 0.2 \cdot D_e + 0.1 \cdot E_s
$$

| Component | Symbol | Description |
|-----------|--------|-------------|
| Success Rate | $S_r$ | Rolling window positive reward fraction |
| Reward Velocity | $R_v$ | Recent reward change rate |
| Discovery Efficiency | $D_e$ | Discoveries per step |
| Exploit Success Rate | $E_s$ | Exploit command success fraction |

---

## LLM Feature Vector Layout (256-dim)

| Dims | Content | Source |
|------|---------|--------|
| 0–4 | Phase encoding | SmartCoach phase |
| 5–9 | MicroChain signals | MicroChainResult scores |
| 10–14 | Mentor signals | Mentor confidence, call rate |
| 15–19 | Anneal state | Alpha, KL coef, maturity |
| 20–24 | Exploration signals | Diversity pressure, burst |
| 25–29 | Risk estimate | Detection risk, alert level |
| 30–34 | Prior summary | Action prior statistics |
| 35–39 | Temporal features | Step/episode progress |
| 40–255 | Reserved | Future expansion |

---

## Ablation Protocol

Toggle via feature flag or environment variable:

```bash
# Disable Level 5 (ablation baseline)
FF_LLM_POLICY_BRIDGE=0 python ariaska_cli.py smart-train --target 10.129.1.54

# Enable Level 5 (default)
FF_LLM_POLICY_BRIDGE=1 python ariaska_cli.py smart-train --target 10.129.1.54
```

**Programmatic toggle:**
```python
bridge.set_enabled(False)  # → all priors zero, losses zero, features zero
```

**Ablation Delta Metrics** (compare ON vs OFF):
- Unique commands per episode
- Step-to-first-exploit
- Total discoveries
- Reward velocity
- Diversity ratio

---

## File Map

| File | Lines | Role |
|------|-------|------|
| `core/llm/llm_policy_bridge.py` | 775 | Central bridge: guidance packets, anneal, maturity |
| `core/algorithms/ppo_agent.py` | 1736 | PPO with Level 5: prior inject, KL/rank/vreg losses |
| `core/models/state_encoder.py` | 693 | State constants: STATE_DIM, LLM_FEATURE_DIM, ENHANCED_STATE_DIM |
| `core/training/smart_coach.py` | 8355 | Wiring: bridge init, guidance compute, trajectory storage |
| `core/observability/live_dashboard.py` | 2434 | GPT↔RL integration panel |
| `core/feature_flags.py` | 321 | `llm_policy_bridge` flag (FF_LLM_POLICY_BRIDGE) |
| `core/orchestration/smart_orchestrator.py` | 7060 | Dashboard snapshot wiring |
| `tests/test_p37_level5_integration.py` | 390 | 36 design acceptance tests |

---

## Configuration

### PPOConfig Level 5 Fields

| Field | Default | Description |
|-------|---------|-------------|
| `llm_feature_dim` | 0 | LLM feature dims (0=disabled, 256=Level 5) |
| `use_llm_prior` | False | Enable LLM prior injection into logits |
| `prior_alpha_init` | 0.50 | Initial LLM prior weight |
| `use_kl_teacher_loss` | False | Enable KL teacher distillation loss |
| `kl_teacher_coef` | 0.15 | KL loss coefficient |
| `use_ranking_loss` | False | Enable ranking margin loss |
| `ranking_loss_coef` | 0.05 | Ranking loss coefficient |
| `ranking_margin` | 1.0 | Margin for ranking loss |
| `use_value_reg_loss` | False | Enable value regularization loss |
| `value_reg_coef` | 0.10 | Value reg coefficient |

All Level 5 features default to **OFF** for backward compatibility. They are enabled automatically when `LLMPolicyBridge` initializes in `SmartCoach.__init__()`.
