# Pre-HTB Capability Hardening — Runbook

**Version:** 10.1  
**Author:** Phase 10.1 Automation  
**Baseline Tests:** 423 (original) + 145 (Phase 10.1) = 568+

---

## Quick Reference

```bash
# Run all tests (must pass before any HTB engagement)
make test

# Run Phase 10.1 subsystem tests only
.venv/bin/pytest tests/test_phase101_*.py -v

# Bootstrap tools for a target profile
python scripts/tools/bootstrap_tools.py --profile htb

# Quick smoke test (3 episodes simulated)
make smoke

# Full training run (10 episodes)
make train-quick
```

---

## Subsystem Activation Guide

Each Phase 10.1 subsystem is gated behind a feature flag (disabled by default
except privilege gating). Enable as needed:

### A. Privilege Gating (default: ON)

Commands requiring `sudo` or `root` are filtered unless the agent has
demonstrated privilege evidence (shell_obtained + privilege_level).

```python
from core.feature_flags import set_feature_flag
set_feature_flag("privilege_gating", True)   # default
set_feature_flag("allow_sudo", True)         # enable sudo commands
set_feature_flag("sudo_mode", "auto")        # auto/prompt/deny
```

**When to enable `allow_sudo`:** After obtaining initial shell access and
determining the target allows sudo escalation paths.

### B. Tool Registry + Live Install (default: OFF)

```python
set_feature_flag("allow_live_install", True)
```

**Limits:** 3 installs/episode, 10 installs/run.  
**Bootstrap before engagement:**

```bash
python scripts/tools/bootstrap_tools.py --profile htb --install
```

### C. Wordlist Mutation (default: OFF)

```python
set_feature_flag("wordlist_mutation", True)
```

**When to enable:** During EXPLOITATION phase when credential brute-force
is the active strategy and default wordlists failed.

### D. Port Knocking (default: OFF)

```python
set_feature_flag("port_knocking", True)
```

**When to enable:** When nmap shows filtered ports that should be open
(e.g., SSH filtered on an HTB box known to use knockd).

### E. Proxy Capture (default: OFF)

```python
set_feature_flag("proxy_capture", True)
```

**When to enable:** When targeting web applications where Burp Suite or
ZAP proxy data is available. Ingest HAR files to auto-discover endpoints,
parameters, auth tokens, and injection points.

### F. Payload Encoding (default: OFF)

```python
set_feature_flag("payload_encoding", True)
```

**When to enable:** When WAF/IDS is detected or detection_risk > 0.7.
The encoder auto-selects transforms based on delivery channel and platform.

---

## Pre-Engagement Checklist

Before starting an HTB box:

- [ ] `make test` passes (568+ tests)
- [ ] Docker containers running (if live target): `docker-compose up -d`
- [ ] Tool bootstrap complete: `python scripts/tools/bootstrap_tools.py --profile htb`
- [ ] Feature flags configured for target type
- [ ] Verify no raw shell execution outside CommandRegistry
- [ ] Seed set for reproducibility: `--seed 42`

---

## Registry-Only Execution Model

**CRITICAL INVARIANT:** No free-form shell commands execute outside the
CommandRegistry template pipeline.

All commands must flow through:

```text
SmartCoach → CommandRegistry.get_valid_commands() → CommandTemplate.render()
→ Live/Sandboxed Executor → Parser Broker
```

The acceptance test `test_no_raw_shell_outside_registry` enforces this.

---

## Acceptance Gates

### Gate 1: Test Suite Green

```bash
make test  # All 568+ tests pass
```

### Gate 2: Registry Integrity

```bash
.venv/bin/pytest tests/test_phase101_acceptance.py -v
```

Verifies:

- No duplicate command names in registry
- All commands have valid phases
- All privilege-gated commands have `requires_privilege` set
- No raw shell execution bypass
- All Phase 10.1 subsystems importable and feature-flag gated

### Gate 3: Smoke Test

```bash
make smoke  # 3 episodes complete without error
```

### Gate 4: Telemetry Integrity

```bash
.venv/bin/pytest tests/test_phase97_telemetry.py -v
```

Verifies StepEvent and EpisodeEvent schemas are consistent.

---

## Troubleshooting

### Import Errors

All Phase 10.1 modules use lazy imports. If you get circular import errors:

1. Check that you're importing inside a method, not at module top level
2. Use `TYPE_CHECKING` guard for type hints

### Feature Flag Not Taking Effect

Flags are cached per-process. If testing flag changes:

```python
from core.feature_flags import reset_feature_flags
reset_feature_flags()
set_feature_flag("flag_name", True)
```

### Tool Not Found

```python
from core.tools.tool_registry import get_tool_registry
registry = get_tool_registry()
if not registry.is_available("nmap"):
    registry.install_tool("nmap")
```

---

## Phase 10.1 Architecture Diagram

```text
SmartOrchestrator
├── SmartCoach (per agent)
│   ├── Playbook → Commands
│   ├── PPO → CommandActionMapper → Commands
│   ├── Registry → Precondition-matched Commands
│   ├── GPT Mentor → Commands
│   ├── _filter_by_privilege() ← Phase 10.1A
│   └── Anti-Repeat Guard
├── CyberEnvironment
│   └── step(command) → state, reward, done
├── ToolRegistry ← Phase 10.1B
│   └── is_available() / install_tool()
├── WordlistMutationEngine ← Phase 10.1C
│   └── generate() → wordlist file
├── KnockInferenceEngine ← Phase 10.1D
│   └── infer() → KnockSequence[]
├── WebProxyLayer ← Phase 10.1E
│   └── ingest_har() → discoveries
├── PayloadEncoder ← Phase 10.1F
│   └── encode() → EncodingResult
└── Telemetry + JSONL Logging
```
