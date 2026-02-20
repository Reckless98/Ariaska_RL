# Distillation Prep Pack

> **Phase:** Pre-GPU Preparation  
> **Version:** v1  
> **Author:** Ariaska_RL Distillation Pipeline

## Purpose

The Distillation Prep Pack generates **deterministic, validated synthetic artifacts** that feed the GPU-based distillation training pipeline later. It produces three categories of data:

1. **Synthetic Run Traces** — Realistic simulated Ariaska engagement runs in JSONL
2. **Teacher Trajectories** — Expert demonstration sequences grounded in the knowledge base
3. **Curriculum Metadata** — Weakness reports, manifests, and scenario profiles

## Architecture

```
      LOCAL LAPTOP (prep phase)
      ┌─────────────────────────────────┐
      │  generate_synthetic_traces.py   │──→ data/distill_prep/synthetic_traces/
      │  generate_teacher_trajectories  │──→ data/distill_prep/teacher_trajectories/
      │  summarize_artifacts.py         │──→ data/distill_prep/curriculum/
      │  validate_artifacts.py          │    data/distill_prep/manifest.json
      └─────────────────────────────────┘
                   │
                   │ rsync / git-lfs / git
                   ▼
      ┌─────────────────────────────────┐    ┌─────────────────────────────────┐
      │   TRAIN BOX (GPU)               │◄──►│   MENTOR BOX (vLLM on 96GB)     │
      │   • Load synthetic traces       │    │   • 72B mentor (Qwen/etc.)      │
      │   • Load teacher trajectories   │    │   • Batched inference            │
      │   • PPO + BC loss training      │    │   • Teacher-student distillation │
      │   • Env simulation + rollouts   │    │                                 │
      └─────────────────────────────────┘    └─────────────────────────────────┘
```

## File Structure

```
scripts/distill_prep/
├── __init__.py
├── trace_schema.py                   # Strict dataclass schemas
├── generate_synthetic_traces.py      # Synthetic run generator
├── generate_teacher_trajectories.py  # Expert trajectory generator
├── validate_artifacts.py             # Schema + sanity validation
└── summarize_artifacts.py            # Weakness report + manifest + CLI summary

data/distill_prep/
├── scenarios/                        # Scenario profile JSONs
│   ├── scenario_easy_00.json
│   ├── scenario_medium_00.json
│   └── ...
├── synthetic_traces/                 # Generated JSONL traces
│   ├── run_0000.jsonl
│   ├── run_0001.jsonl
│   └── ...
├── teacher_trajectories/             # Expert JSONL trajectories
│   ├── teacher_0000.jsonl
│   ├── teacher_0001.jsonl
│   └── ...
├── curriculum/
│   └── weakness_report.json          # Training gap analysis
└── manifest.json                     # Checksums + metadata

tests/distill_prep/
├── __init__.py
├── test_trace_generation.py
├── test_trajectory_generation.py
└── test_validation.py
```

## Quick Start

```bash
# Generate everything (200 traces, 100 trajectories, weakness report, manifest)
make distill-prep-generate

# Validate all artifacts
make distill-prep-validate

# Print summary tables
make distill-prep-summary
```

## Trace Format

Synthetic traces use the **episode_replayer-compatible** JSONL format:

```jsonl
{"kind":"episode_start","episode_id":"distill_run_0000","episode_num":0,"target_ip":"10.10.10.1","difficulty":"medium","distill_prep_version":"v1"}
{"kind":"step","step_num":0,"phase_before":"RECON","phase_after":"RECON","step_reward_total":7.5,"agent_records":[{"agent_name":"ScoutAgent","command":"nmap -sC -sV 10.10.10.1","command_family":"nmap","reward":7.5,"discoveries":[{"discovery_type":"PORT","value":"22"}]}],"distill_prep_version":"v1"}
{"kind":"episode_end","episode_id":"distill_run_0000","total_reward":125.3,"highest_phase":"PRIVILEGE_ESCALATION","total_steps":65,"distill_prep_version":"v1"}
```

### Key Fields per Step

| Field | Type | Description |
|-------|------|-------------|
| `kind` | str | Line discriminator: `episode_start`, `step`, `episode_end` |
| `step_num` | int | Step index within episode |
| `phase_before` / `phase_after` | str | Phase before/after this step |
| `step_reward_total` | float | Total reward for this step |
| `agent_records` | list | Per-agent decisions and outcomes |
| `distill_prep_version` | str | Always `"v1"` |

### Agent Record Fields

| Field | Type | Description |
|-------|------|-------------|
| `agent_name` | str | ScoutAgent, RedAgent, etc. |
| `command` | str | Full command string |
| `command_family` | str | Tool family (nmap, hydra, etc.) |
| `decision_source` | str | ppo, playbook, mentor, etc. |
| `reward` | float | Step reward [-15.0, +100.0] |
| `discoveries` | list | Discovery events |
| `is_wrong_move` | bool | Whether this is a teaching negative |
| `tactical_lesson` | str | Lesson for wrong moves |

## Teacher Trajectory Format

```jsonl
{"kind":"trajectory_start","trajectory_id":"teacher_0000","scenario_id":"scenario_easy_00","difficulty":"easy","distill_prep_version":"v1"}
{"kind":"teacher_step","phase":"RECON","command_family":"nmap","full_command":"nmap -sC -sV 10.10.10.1","reasoning":"Initial scan","expected_outcome":"Find ports","reward":15.0,"is_wrong_move":false,"distill_prep_version":"v1"}
{"kind":"trajectory_end","trajectory_id":"teacher_0000","total_reward":200.0,"highest_phase":"EXFILTRATION","success":true,"distill_prep_version":"v1"}
```

## Scenarios

Each scenario profile defines a target environment:

```json
{
  "scenario_id": "scenario_medium_00",
  "name": "Medium Target 0",
  "difficulty": "medium",
  "target_ip": "10.10.10.42",
  "os_type": "linux",
  "services": ["http", "ssh", "smb", "mysql", "ftp"],
  "open_ports": [22, 80, 445, 3306, 21, 139],
  "has_credentials": true,
  "has_flags": true,
  "max_steps": 75
}
```

## Weakness Report

The weakness report identifies training gaps:

```json
{
  "phase_histogram": {"RECON": 500, "ENUMERATION": 800, ...},
  "tool_family_coverage": {"nmap": 300, "gobuster": 200, ...},
  "avg_reward_by_phase": {"RECON": 3.5, "EXPLOITATION": 12.0, ...},
  "decision_source_pct": {"ppo": 35.0, "playbook": 20.0, ...},
  "weakness_areas": ["Low coverage for LATERAL_MOVEMENT"],
  "coverage_gaps": ["No examples for tool family: certipy"],
  "wrong_move_ratio": 0.18
}
```

## Validation

The validator checks every JSONL line for:
- Valid JSON syntax
- Required fields present
- Phase names in the 8-phase enum
- Command families in the registry
- Rewards within [-15.0, +100.0]
- No NaN/Inf floats
- Valid discovery types (24 types)
- Valid decision sources (9 types)

```bash
# Returns exit code 0 on success, 1 on failure
python -m scripts.distill_prep.validate_artifacts
```

## Determinism

All generators accept a `--seed` parameter. Same seed produces **byte-identical** output, enabling:
- Reproducible experiments
- CI validation
- Checksum verification via manifest

## How This Feeds the GPU Run

On the GPU training box, the distillation pipeline will:

1. **Load synthetic traces** → Parse with `episode_replayer` → Build replay buffer
2. **Load teacher trajectories** → Convert to `TeacherTrace` / `BCSample` objects
3. **Compute BC loss** from teacher demonstrations alongside PPO rollouts
4. **Use weakness report** to weight the curriculum (more training on weak phases)
5. **Validate** incoming artifacts with `validate_artifacts.py` before training

The manifest provides checksums so the GPU box can verify data integrity after transfer.

## Dependency Diagram

```
trace_schema.py          ← Pure schema, no core imports
     ↑
generate_synthetic_traces.py    ← Imports trace_schema only
     ↑
generate_teacher_trajectories.py ← Imports trace_schema only
     ↑
validate_artifacts.py           ← Imports trace_schema only
     ↑
summarize_artifacts.py          ← Imports trace_schema only

tests/distill_prep/
  test_trace_generation.py      ← Imports generators + validator
  test_trajectory_generation.py ← Imports generators + validator
  test_validation.py            ← Imports all scripts
```

No circular imports. No core module dependencies. Self-contained package.
