# Copilot ↔ Claude Code — Live Session

> This file is the shared context for the current task. Both Copilot and Claude Code read and append to it.
> Reset at the start of each new task. Do not delete mid-task.

## Task: Ariaska Fine-Tuning Pipeline — Complete Execution
**Status**: SFT running on GPU server

---

### [Claude → Copilot] HANDOFF — Ariaska Fine-Tuning Pipeline Continuation

**Date**: 2026-03-16
**Context**: Claude Opus 4.6 ran out of usage mid-pipeline. You MUST continue exactly where we left off. Read this ENTIRE document before doing ANYTHING.

---

## CRITICAL: What Is Happening Right Now

Training is running on GPU server `ssh -p 52191 root@1.208.108.242` (2x RTX 5070 Ti, vast.ai).

**Current stage**: 4B SFT (Stage 2) is actively running.
**Pipeline script**: `/root/ariaska_finetune/scripts/train_ariaska_v3.py`
**Local copy**: `/home/zer0/Projects/Ariaska_RL/ariaska_ai/scripts/train_ariaska_v3.py`

### Monitor commands:
```bash
# Check SFT step progress
ssh -o ConnectTimeout=20 -p 52191 root@1.208.108.242 'strings /root/ariaska_finetune/training_stderr.log 2>/dev/null | grep -oP "\d+/1342.*" | tail -3'

# Check metrics (loss, accuracy)
ssh -o ConnectTimeout=20 -p 52191 root@1.208.108.242 'tail -5 /root/ariaska_finetune/training_stdout.log 2>/dev/null'

# Check GPU state
ssh -o ConnectTimeout=20 -p 52191 root@1.208.108.242 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null'

# Check if SFT completed
ssh -o ConnectTimeout=20 -p 52191 root@1.208.108.242 'ls /root/ariaska_finetune/outputs/sft/final/adapter_config.json 2>/dev/null && echo SFT_DONE || echo SFT_RUNNING'

# Check if merge completed
ssh -o ConnectTimeout=20 -p 52191 root@1.208.108.242 'ls /root/ariaska_finetune/outputs/ariaska-cybersec-merged/config.json 2>/dev/null && echo MERGED || echo NOT_YET'
```

---

## FULL PIPELINE — Where We Are

| Stage | Status | Details |
|-------|--------|---------|
| **1. 4B CPT** | DONE | 125/125 steps, adapter at `/root/ariaska_finetune/outputs/cpt/final/` |
| **2. 4B SFT** | RUNNING | 1342 total steps, ~55s/step. Loss=0.31, accuracy=91.2% at step 91. ETA ~19h from start |
| **3. 4B Merge + GGUF** | AUTO | Script auto-merges after SFT, exports Q8_0 + Q6_K GGUFs |
| **4. DPO** | SKIPPED THIS RUN | Was launched with `--skip-dpo`. Run separately after SFT validation |
| **5. 9B Teacher Fine-tune** | PENDING | Same CPT->SFT pipeline on 9B. Run with `--distill` |
| **6. 9B->4B Distillation** | PENDING | Fine-tuned 9B generates 10K responses, 4B learns from them |
| **7. Final GGUF + Ollama** | PENDING | Register as `ariaska-cybersec` in Ollama |

---

## WHAT YOU MUST DO (in exact order)

### Step 1: Monitor SFT completion
- Check every 15-30 min with the monitor commands above
- SFT has 1342 total steps at ~55s/step
- When `SFT_DONE` is true, the script will auto-merge and try GGUF export
- Check logs for errors: `tail -30 /root/ariaska_finetune/training_stdout.log`

### Step 2: Verify merge and GGUF export
After SFT completes, the script automatically:
1. Merges CPT+SFT adapters into base model -> `/root/ariaska_finetune/outputs/ariaska-cybersec-merged/`
2. Exports GGUF Q8_0 and Q6_K -> `/root/ariaska_finetune/outputs/gguf/`

Check:
```bash
ssh -p 52191 root@1.208.108.242 'ls -la /root/ariaska_finetune/outputs/gguf/ 2>/dev/null'
```

If GGUF export fails (likely — needs llama.cpp), you must:
```bash
# Install llama.cpp on GPU server
ssh -p 52191 root@1.208.108.242 'pip install llama-cpp-python && git clone https://github.com/ggerganov/llama.cpp /root/llama.cpp && cd /root/llama.cpp && pip install -r requirements.txt'

# Manual GGUF export
ssh -p 52191 root@1.208.108.242 'python3 /root/llama.cpp/convert_hf_to_gguf.py /root/ariaska_finetune/outputs/ariaska-cybersec-merged --outtype q8_0 --outfile /root/ariaska_finetune/outputs/gguf/ariaska-cybersec-q8_0.gguf'
ssh -p 52191 root@1.208.108.242 'python3 /root/llama.cpp/convert_hf_to_gguf.py /root/ariaska_finetune/outputs/ariaska-cybersec-merged --outtype q6_k --outfile /root/ariaska_finetune/outputs/gguf/ariaska-cybersec-q6_k.gguf'
```

### Step 3: Download GGUF to local machine
```bash
scp -P 52191 root@1.208.108.242:/root/ariaska_finetune/outputs/gguf/ariaska-cybersec-q8_0.gguf /home/zer0/Projects/Ariaska_RL/
scp -P 52191 root@1.208.108.242:/root/ariaska_finetune/outputs/gguf/ariaska-cybersec-q6_k.gguf /home/zer0/Projects/Ariaska_RL/
```

### Step 4: Run DPO (preference alignment)
```bash
ssh -p 52191 root@1.208.108.242 'nohup python3 /root/ariaska_finetune/scripts/train_ariaska_v3.py --stage dpo > /root/ariaska_finetune/dpo_stdout.log 2> /root/ariaska_finetune/dpo_stderr.log &'
```
Monitor: `ssh -p 52191 root@1.208.108.242 'strings /root/ariaska_finetune/dpo_stderr.log 2>/dev/null | grep -oP "\d+/\d+.*" | tail -3'`

After DPO completes, re-merge and re-export GGUF:
```bash
ssh -p 52191 root@1.208.108.242 'python3 /root/ariaska_finetune/scripts/train_ariaska_v3.py --stage merge'
```

### Step 5: Run 9B fine-tuning + distillation
```bash
ssh -p 52191 root@1.208.108.242 'nohup python3 /root/ariaska_finetune/scripts/train_ariaska_v3.py --distill > /root/ariaska_finetune/distill_stdout.log 2> /root/ariaska_finetune/distill_stderr.log &'
```

This does:
- Stage 5a: Downloads Qwen3.5-9B (~18GB)
- Stage 5b: Teacher CPT (same 95K cybersec corpus)
- Stage 5c: Teacher SFT (same 34K task dataset, NEFTune enabled)
- Stage 5d: Merges teacher into single 9B
- Stage 5e: Generates 10K expert responses from fine-tuned 9B
- Stage 5f: Trains 4B student on teacher responses

After distillation, re-merge + re-export GGUF. The distilled model becomes the final `ariaska-cybersec`.

### Step 6: Register in Ollama
```bash
# On the user's local laptop
cd /home/zer0/Projects/Ariaska_RL
ollama create ariaska-cybersec -f Modelfile_ariaska_cybersec
ollama list | grep ariaska
```

The Modelfile is at `/home/zer0/Projects/Ariaska_RL/Modelfile_ariaska_cybersec` and references `./ariaska-cybersec-q8_0.gguf`.

### Step 7: Run evaluation
```bash
cd /home/zer0/Projects/Ariaska_RL
PYTHONPATH=/home/zer0/Projects/Ariaska_RL .venv/bin/python ariaska_ai/scripts/evaluate.py --baseline jaahas/qwen3.5-uncensored:4b --finetuned ariaska-cybersec
```

This evaluates both models on the V3 holdout set (9 task families) and prints a comparison.

### Step 8: Run full test suite
```bash
ARIASKA_DRY_RUN=1 PYTHONPATH=/home/zer0/Projects/Ariaska_RL .venv/bin/python -m pytest tests/ -x --tb=line -q --timeout=120 --ignore=tests/test_online_makes_mentor_calls.py --ignore=tests/test_online_requires_key.py
```
ALL 2935 tests must pass.

---

## ARCHITECTURAL INVARIANTS — DO NOT VIOLATE

1. **ALL LLM calls through GPTManager** — never `import openai` directly
2. **Lazy imports inside methods** — circular dependency chains will break everything
3. **Single GPTManager instance** — inject, never create new
4. **`rich` for output, `logging` for debug** — never `print()`
5. **`STATE_DIM = 512`** — hardcoded everywhere, do not change
6. **Never modify PPO core** without explicit instruction
7. **BudgetManagerV2 clamps**: 1,148,850 max, 574,425 min
8. **Evidence Gate default = `enforce`**
9. **Test baseline: all 2935 tests must pass after changes**

---

## CHANGES ALREADY MADE (verified, tested, 2935/2935 pass)

### 1. Model Router Collapse (`core/llm/model_router.py`)
- All 3 tiers (`FAST_MODEL`, `MEDIUM_MODEL`, `REASONING_MODEL`) now default to `ariaska-cybersec`
- Env var overrides preserved for testing

### 2. SmartMentor RAG Swap (`core/llm/smart_mentor.py` lines 535-549)
- Replaced old ChromaDB `core.knowledge.knowledge_query.build_rag_context()` with FAISS `ariaska_ai.retriever.rag_retriever.build_rag_context()`
- API: `build_rag_context(phase, evidence, recent_commands, top_k=5) -> str`

### 3. SmartCoach RAG Injection (`core/training/smart_coach.py`)
- Added FAISS RAG at 3 sites:
  - `_ask_mentor_reasoning()` ~line 1818 (top_k=3)
  - `_codex_meta_check()` ~line 2195 (top_k=3)
  - `_codex_strategic_check()` ~line 2510 (top_k=3)
- All use lazy imports, try/except, proper error handling

### 4. Evaluate Script (`ariaska_ai/scripts/evaluate.py`)
- Full V3 holdout evaluation with all 9 task families
- Scorers: JSON parse rate, phase accuracy, command quality, reasoning quality (F1)
- Side-by-side baseline vs fine-tuned comparison

### 5. Training Script (`ariaska_ai/scripts/train_ariaska_v3.py`)
- 5-stage pipeline: CPT -> SFT -> DPO -> Merge/GGUF -> Distill (9B fine-tune + distill to 4B)
- QLoRA (NF4, bf16, gradient checkpointing) on all stages
- NEFTune on SFT (alpha=5.0)
- Packing enabled with SDPA (kernels flash-attn2 crashes on Qwen3.5 hybrid attention on Blackwell GPUs)
- Knowledge distillation: fine-tunes 9B first (CPT+SFT), then distills into 4B

### 6. Ollama Modelfile (`Modelfile_ariaska_cybersec`)
- References `./ariaska-cybersec-q8_0.gguf`
- System prompt with multi-agent identity and attack phases
- Qwen chat template (`<|im_start|>/<|im_end|>`)

---

## KNOWN ISSUES — READ BEFORE DOING ANYTHING

1. **flash-attn2 DOES NOT WORK** with Qwen3.5 on Blackwell GPUs — `kernels-community/flash-attn2` causes `CUDA illegal memory access` in `chunk_gated_delta_rule`. Use SDPA only. DO NOT try to install flash-attn or use flash attention implementations.
2. **SSH to GPU server is flaky** under load — use `-o ConnectTimeout=30 -o ServerAliveInterval=15` and keep commands SHORT
3. **vast.ai GPU server will shut down** when instance expires — check instance status
4. **SFT uses only 1 GPU effectively** — device_map="auto" puts most layers on GPU 0. GPU 1 holds overflow only.
5. **tqdm progress goes to stderr** — use `strings ... | grep -oP "\d+/\d+.*"` to extract progress from binary-polluted stderr

---

## DATASET STATS

- **SFT**: 34,317 samples across 9 families (command_validate, evidence_check, next_step, phase_classification, postmortem, retrieval_reasoning, retry_or_pivot, state_summary, tool_output_parse)
- **DPO**: 1,795 preference pairs
- **CPT**: 95,935 cybersecurity corpus documents
- **Holdout**: ~4,850 samples for evaluation
- All at `/root/ariaska_finetune/dataset/v3/` on GPU server and `/home/zer0/Projects/Ariaska_RL/ariaska_ai/dataset/v3/` locally

---

## WHAT "DONE" LOOKS LIKE

1. `ariaska-cybersec` registered in Ollama on local laptop
2. Evaluation shows improvement over baseline `jaahas/qwen3.5-uncensored:4b`
3. All 2935 tests pass with the new model
4. Model router points to `ariaska-cybersec` (already done)
5. RAG retriever wired into SmartMentor + SmartCoach (already done)
6. DPO preference alignment applied
7. 9B->4B knowledge distillation applied
8. GGUF Q8_0 (primary) and Q6_K (backup) exported

**DO NOT CUT CORNERS. The user explicitly requested maximum quality. Follow every step. Do not skip DPO or distillation.**

---

## PHASE 2: SYSTEM IMPROVEMENTS (after ariaska-cybersec is deployed)

These are architectural upgrades to maximize LLM intelligence in the core Ariaska system. Execute AFTER the fine-tuned model is registered in Ollama and working.

### Priority 1: Async Queued Agent Dispatch (replace sequential execution)

**Current**: Agents run sequentially: Scout→Red→Blue→Orion→Shadow (5 serial LLM calls)
**Target**: Parallel agent reasoning with dependency-aware queue

**File**: `core/orchestration/orchestrator.py`

**Implementation**:
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Agent dependency graph (who can run in parallel)
AGENT_DEPS = {
    "ScoutAgent": [],           # No deps — runs first
    "RedAgent": ["ScoutAgent"], # Needs Scout's recon
    "BlueAgent": [],            # Independent — runs parallel with Scout
    "OrionAgent": ["RedAgent", "BlueAgent"],  # Needs attack+defense context
    "ShadowAgent": ["OrionAgent"],  # Needs Orion's critique
}

async def _run_step_async(self, state, phase):
    """Run agents with dependency-aware parallelism."""
    executor = ThreadPoolExecutor(max_workers=3)
    results = {}
    pending = set(AGENT_DEPS.keys())

    while pending:
        # Find agents whose dependencies are satisfied
        ready = [a for a in pending if all(d in results for d in AGENT_DEPS[a])]

        # Run ready agents in parallel
        tasks = [
            asyncio.get_event_loop().run_in_executor(
                executor, self._get_agent_proposal, agent, state, phase
            )
            for agent in ready
        ]
        batch_results = await asyncio.gather(*tasks)

        for agent, result in zip(ready, batch_results):
            results[agent] = result
            pending.discard(agent)

    return results
```

**Result**: Scout+Blue run in parallel (saves ~2s per step). Orion waits for Red+Blue. 3 LLM calls instead of 5 sequential.

### Priority 2: LLM-Driven Output Parsing (replace hardcoded regex)

**Current**: 15+ compiled regex patterns in `core/orchestration/output_parser.py`
**Target**: LLM parses tool output with semantic understanding

**File**: `core/orchestration/output_parser.py`

**Implementation**: Add LLM fallback after regex extraction. The LLM reasons about what discoveries mean and assigns confidence.

```python
def parse_with_llm(self, tool_output: str, tool_name: str, phase: str) -> DiscoveryResult:
    """LLM-enhanced output parsing — semantic extraction with confidence."""
    # Step 1: Fast regex extraction (keep for speed)
    regex_result = self._parse_regex(tool_output)

    # Step 2: LLM semantic analysis (only when regex finds something interesting)
    if regex_result.has_discoveries():
        from core.llm.gpt_manager import GPTManager
        prompt = f"""Analyze this {tool_name} output. Current phase: {phase}.
Extract: ports, services, credentials, vulnerabilities, attack paths.
For each discovery, rate exploitation_priority (1-5) and confidence (0-1).
Respond JSON only."""

        llm_analysis = GPTManager.instance().query(
            prompt=prompt, context=tool_output[:2000],
            task_type="tool_output_parse"
        )
        # Merge LLM insights into regex results
        regex_result.merge_llm_analysis(llm_analysis)

    return regex_result
```

**Result**: Regex stays for speed, LLM adds semantic reasoning (e.g., "MySQL on 3306 + root:empty_password = CRITICAL path to privesc").

### Priority 3: Dynamic Agent Selection (skip irrelevant agents)

**Current**: All 5 agents always run every step
**Target**: LLM decides which agents should act THIS step

**File**: `core/orchestration/orchestrator.py`

**Implementation**: Before agent dispatch, ask the LLM which agents matter:

```python
def _select_active_agents(self, state, phase, discovery_board):
    """LLM selects which agents should participate this step."""
    prompt = f"""Phase: {phase}. State: shell={'YES' if state.get('shell_obtained') else 'NO'},
    root={'YES' if state.get('root_shell_obtained') else 'NO'}.
    Recent: {self._recent_actions[-3:]}.
    Which agents should act? Choose from: Scout, Red, Blue, Orion, Shadow.
    Skip agents that have nothing useful to contribute this step.
    Respond JSON: {{"agents": ["Red", "Blue"], "reason": "..."}}"""

    result = self.gpt.query(prompt=prompt, task_type="phase_classification")
    # Parse and return active agent list
    # Fallback: all agents if parsing fails
```

**Result**: In EXPLOITATION phase, maybe only Red+Orion act. Saves 60% of LLM calls when agents would produce no-op actions.

### Priority 4: LLM Phase Guidance (replace hardcoded boolean gates)

**Current**: Phase transitions are hardcoded boolean flag checks in `core/commands/command_registry.py` lines 3781-3841
**Target**: LLM reasons about phase transitions with context

**File**: `core/commands/command_registry.py`

**Implementation**: Replace `get_phase_from_state()` with LLM-augmented version:

```python
def get_phase_from_state_llm(self, state_flags, discovery_board, command_history):
    """LLM-guided phase classification with tactical regression support."""
    # Fast path: use hardcoded gates as baseline
    baseline_phase = self._get_phase_from_state_hardcoded(state_flags)

    # LLM reasoning: should we advance, hold, or regress?
    prompt = f"""Current phase: {baseline_phase}.
    Discoveries: {self._summarize_board(discovery_board)}.
    Last 5 commands: {command_history[-5:]}.
    Should we: ADVANCE to next phase, HOLD current, or REGRESS to re-enumerate?
    Consider: Are we stuck? Did we miss enumeration? Is there a faster path?
    Respond JSON: {{"decision": "ADVANCE|HOLD|REGRESS", "target_phase": "...", "reason": "..."}}"""

    result = self.gpt.query(prompt=prompt, task_type="phase_classification")
    # Apply LLM decision with safety bounds (can't skip more than 1 phase)
```

**Result**: LLM can say "we exploited SSH but missed web services — REGRESS to ENUMERATION for port 80". No hardcoded system can do this.

### Priority 5: Batch LLM Calls (single call for multiple agents)

**Current**: 5 separate LLM calls per step
**Target**: Single batched prompt for coordinated multi-agent reasoning

**File**: `core/llm/gpt_manager.py`

**Implementation**: Add batch query method:

```python
def query_batch(self, agent_prompts: dict[str, str], task_type: str) -> dict[str, str]:
    """Single LLM call for multiple agents — coordinated reasoning."""
    combined = "You are coordinating 5 pentesting agents. For each active agent, provide their action.\n\n"
    for agent, prompt in agent_prompts.items():
        combined += f"### {agent}\n{prompt}\n\n"
    combined += "Respond with JSON: {\"Scout\": {...}, \"Red\": {...}, ...}"

    result = self.query(prompt=combined, task_type=task_type)
    # Parse per-agent responses from single JSON
    return self._split_batch_response(result, agent_prompts.keys())
```

**Result**: 1 LLM call instead of 5. 80% reduction in inference time. Better coordination because LLM sees all agents' context simultaneously.

### Priority 6: Inter-Agent Message Bus

**Current**: Agents share state via discovery_board (passive)
**Target**: Active message passing between agents

**File**: NEW `core/orchestration/agent_bus.py`

```python
@dataclass
class AgentMessage:
    sender: str
    receiver: str  # or "ALL"
    msg_type: str  # "REQUEST_RECON", "EXPLOIT_RESULT", "DEFENSE_ALERT", "STRATEGY_UPDATE"
    content: dict
    priority: int = 0

class AgentBus:
    def __init__(self):
        self._queue: list[AgentMessage] = []

    def send(self, msg: AgentMessage):
        self._queue.append(msg)

    def get_messages(self, agent: str) -> list[AgentMessage]:
        return [m for m in self._queue if m.receiver in (agent, "ALL")]

    def inject_into_prompt(self, agent: str) -> str:
        msgs = self.get_messages(agent)
        if not msgs:
            return ""
        return "\n".join(f"[{m.sender}→{agent}] {m.msg_type}: {m.content}" for m in msgs)
```

**Result**: Red can tell Scout "I need deeper enum on port 8080". Orion can tell Red "stop repeating hydra, try web exploit instead".

---

## IMPLEMENTATION ORDER FOR COPILOT

After ariaska-cybersec is deployed and working:

1. **Batch LLM calls** (Priority 5) — biggest speed win, simplest change
2. **Dynamic agent selection** (Priority 3) — second biggest speed win
3. **Async agent dispatch** (Priority 1) — parallel execution
4. **LLM output parsing** (Priority 2) — intelligence upgrade
5. **LLM phase guidance** (Priority 4) — tactical flexibility
6. **Inter-agent message bus** (Priority 6) — coordination upgrade

Each change should:
- Use lazy imports (circular dep protection)
- Go through GPTManager (never direct LLM calls)
- Have fallback to current behavior if LLM fails
- Pass all 2935 tests after implementation
- Use `task_type` parameter for model routing

---

## PHASE 56 SYSTEM IMPROVEMENTS — IMPLEMENTED (2935/2935 tests pass)

### [Claude → Copilot] Implementation Complete

**Date**: 2026-03-16
**All 6 priorities implemented and tested.** 2935/2935 tests pass.

### 1. Async Agent Dispatch (`core/orchestration/orchestrator.py`)
- Replaced sequential `for agent_name in self.AGENT_ORDER` loop with dependency-aware parallel dispatch
- `_build_dispatch_waves()` computes execution waves from `AGENT_DEPS` graph:
  - Wave 0: Scout + Blue (parallel, no deps)
  - Wave 1: Red (needs Scout recon)
  - Wave 2: Orion (needs Red + Blue)
  - Wave 3: Shadow (needs Orion)
- `_execute_wave()` uses `ThreadPoolExecutor` for parallel agents within each wave
- Single-agent waves skip thread overhead
- Fault-tolerant: failed agents return noop results instead of crashing the pipeline
- Results sorted by `AGENT_ORDER` for deterministic output

### 2. Inter-Agent Message Bus (`core/orchestration/agent_bus.py` + wired into orchestrator)
- `AgentMessageBus` with 8 message types: REQUEST_RECON, EXPLOIT_RESULT, DEFENSE_ALERT, STRATEGY_UPDATE, MEMORY_INSIGHT, PHASE_SUGGESTION, DISCOVERY, COORDINATION
- Bus messages injected into each agent's `enriched_state["agent_comms"]` via `inject_into_prompt()`
- Each agent's chosen action auto-posted as DISCOVERY to the bus for downstream agents
- Reward broadcasts posted as STRATEGY_UPDATE
- Bus cleared between episodes in `run_episode()`

### 3. Dynamic Agent Selection (`orchestrator._select_active_agents()`)
- Phase-aware defaults:
  - RECON/ENUMERATION: Scout + Red + Blue + Orion
  - EXPLOITATION: Red + Blue + Shadow + Orion
  - PRIVILEGE_ESCALATION+: All five
- High detection risk (>0.5) forces Blue + Shadow inclusion
- Agent bus PHASE_SUGGESTION/REQUEST_RECON messages can pull Scout back in
- Always includes RedAgent (executor)
- Early steps (<3) always activate all agents for exploration

### 4. LLM-Enhanced Output Parsing (`core/orchestration/output_parser.py`)
- `parse_with_llm()` supplements regex extraction with LLM semantic analysis
- LLM only called when output is substantial (>200 chars) but regex found few results (<5)
- Structured JSON extraction prompt for 13 discovery types
- `_merge_discoveries()` safely merges LLM findings into regex results with type validation
- `_parse_llm_json()` handles markdown fences and malformed JSON gracefully

### 5. Batch LLM Calls (`core/gpt_manager.py` `batch_request()`)
- `GPTManager.batch_request(queries, max_workers=3)` for parallel LLM queries
- Each query dict has `prompt`, `agent_id`, optional `task_type`, `max_tokens`, etc.
- Uses `ThreadPoolExecutor` with configurable worker count
- Single-query optimization: skips thread overhead
- 60s timeout per query with graceful fallback
- Returns results in input order

### 6. LLM Phase Guidance Enhancements (`core/llm/phase_guided_llm.py`)
- Added `agent_comms` parameter to `guide()` and `_fast_local_guide()`
- Inter-agent bus messages injected into both full and fast-path LLM prompts
- Phase decisions now account for multi-agent context (e.g., if Red reported exploit success, phase guide can confidently advance)
- Agent comms truncated to 200 chars in fast path to stay within 4B context budget

---

## HANDOFF — Claude Opus 4.6 → Copilot (2026-03-17)

### [Claude → Copilot] CURRENT STATE — What Has Been Done

**Date**: 2026-03-17

#### COMPLETED
1. **4B Fine-tuning (SFT + DPO)** — DONE on 2x RTX 5070 Ti server (`ssh -p 52191 root@1.208.108.242`)
   - CPT: 125 steps, adapter at `/root/ariaska_finetune/outputs/cpt/final/`
   - SFT: 1342 steps, loss 0.1084, accuracy 95.02%
   - DPO: 202 steps on merged SFT model
   - Both adapters merged into `/root/ariaska_finetune/outputs/ariaska-cybersec-merged/` (7.9GB safetensors)

2. **GGUF Q8_0 Export** — DONE
   - File: `/root/ariaska_finetune/outputs/gguf/ariaska-cybersec-q8_0.gguf` (4.2GB)
   - llama.cpp required a tokenizer hash patch: added `1444df51289cfa8063b96f0e62b1125440111bc79a52003ea14b6eac7016fd5f` → `"qwen35"` in `convert_hf_to_gguf.py`

3. **GGUF Q8_0 Downloading** — IN PROGRESS (~2MB/s from vast.ai → local)
   - Target: `/home/zer0/Projects/Ariaska_RL/ariaska-cybersec-q8_0.gguf`
   - Check: `ls -lh /home/zer0/Projects/Ariaska_RL/ariaska-cybersec-q8_0.gguf` — should be ~4.2GB when done

4. **Model Router** — ALREADY COLLAPSED to `ariaska-cybersec` default (`core/llm/model_router.py` line 65)

5. **RAG Retriever** — FIXED and WIRED into SmartMentor + SmartCoach
   - FAISS index: 44,432 chunks (traces + postmortems + v2/v3 datasets)
   - `ariaska_ai/retriever/rag_retriever.py` handles both old and new trace formats

6. **Modelfile** — READY at `Modelfile_ariaska_cybersec` (references `./ariaska-cybersec-q8_0.gguf`)
   - Added `/no_think` to system prompt to suppress Qwen3.5 thinking tags for structured JSON output

#### IN PROGRESS (ON GPU SERVERS — DO NOT INTERFERE)
1. **9B Teacher SFT** — RUNNING on RTX 4080S 32GB (`ssh -p 19936 root@175.155.64.231`)
   - Step ~66/1342, ~20h remaining (57s/step)
   - Loss 0.39, accuracy 89.6% — converging well
   - Script: `python3 scripts/train_ariaska_v3.py --stage distill` (runs full CPT→SFT→DPO→merge→distill pipeline)
   - Monitor: `ssh -o ConnectTimeout=20 -p 19936 root@175.155.64.231 'strings /root/ariaska_finetune/teacher_stderr.log 2>/dev/null | grep -oP "\d+/1342.*" | tail -3'`
   - Metrics: `ssh -o ConnectTimeout=20 -p 19936 root@175.155.64.231 'tail -5 /root/ariaska_finetune/teacher_stdout.log'`

2. **llama.cpp build** — BUILDING locally at `/home/zer0/llama.cpp/` (for Q6_K quantization)

---

### [Claude → Copilot] EXACT STEPS TO CONTINUE

#### Step 1: Wait for Q8_0 download to complete
```bash
# Check if download is done (should be ~4.2GB)
ls -lh /home/zer0/Projects/Ariaska_RL/ariaska-cybersec-q8_0.gguf
```

#### Step 2: Make Q6_K from Q8_0 locally
```bash
# Build llama-quantize if not already built
cd /home/zer0/llama.cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target llama-quantize -j$(nproc)

# Quantize Q8_0 → Q6_K
/home/zer0/llama.cpp/build/bin/llama-quantize \
  /home/zer0/Projects/Ariaska_RL/ariaska-cybersec-q8_0.gguf \
  /home/zer0/Projects/Ariaska_RL/ariaska-cybersec-q6_k.gguf \
  Q6_K
```

#### Step 3: Register ariaska-cybersec in Ollama (use Q8_0 — higher quality)
```bash
cd /home/zer0/Projects/Ariaska_RL
ollama create ariaska-cybersec -f Modelfile_ariaska_cybersec
ollama list | grep ariaska
```

#### Step 4: Quick smoke test
```bash
ollama run ariaska-cybersec '{"task": "phase_classification", "context": {"ports": [22, 80, 443], "services": ["ssh", "http", "https"], "shell_obtained": false}}'
```
Expected: valid JSON with `phase` field, no `<think>` tags.

#### Step 5: Run evaluation
```bash
cd /home/zer0/Projects/Ariaska_RL
PYTHONPATH=/home/zer0/Projects/Ariaska_RL .venv/bin/python ariaska_ai/scripts/evaluate.py --baseline jaahas/qwen3.5-uncensored:4b --finetuned ariaska-cybersec
```

#### Step 6: Run full test suite
```bash
ARIASKA_DRY_RUN=1 PYTHONPATH=/home/zer0/Projects/Ariaska_RL .venv/bin/python -m pytest tests/ -x --tb=line -q --timeout=120 --ignore=tests/test_online_makes_mentor_calls.py --ignore=tests/test_online_requires_key.py
```

#### Step 7: Monitor 9B Teacher (background — check every few hours)
```bash
# Progress
ssh -o ConnectTimeout=20 -p 19936 root@175.155.64.231 'strings /root/ariaska_finetune/teacher_stderr.log 2>/dev/null | grep -oP "\d+/1342.*" | tail -3'
# Metrics
ssh -o ConnectTimeout=20 -p 19936 root@175.155.64.231 'tail -5 /root/ariaska_finetune/teacher_stdout.log'
# GPU
ssh -o ConnectTimeout=20 -p 19936 root@175.155.64.231 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'
```

When 9B SFT finishes, the script auto-continues to DPO → merge → distill.
After distillation completes, download the new distilled GGUF and re-register in Ollama.

#### Step 8: When 5070 Ti can be stopped
The 5070 Ti (`ssh -p 52191 root@1.208.108.242`) is ONLY needed for the merged model safetensors (for potential upload to 4080S for distillation). **After the Q8_0 download completes, you may stop the 5070 Ti instance.** The merged safetensors will also need to be uploaded to the 4080S later — but the 4080S has only 6.8GB free (model is 7.9GB). Clean up the 4080S disk after 9B SFT+DPO completes (delete checkpoints/caches) before uploading.

Alternatively: the distillation stage in `train_ariaska_v3.py` can re-merge from the existing adapters on the 4080S. If adapters are small enough to upload (~500MB total for CPT+SFT+DPO adapters), upload those instead of the full 7.9GB merged model.

---

### SERVER DETAILS

| Server | SSH | GPU | Purpose | Status |
|--------|-----|-----|---------|--------|
| 5070 Ti (vast.ai) | `ssh -p 52191 root@1.208.108.242` | 2x RTX 5070 Ti 16GB | 4B training (DONE), GGUF export (DONE) | Can stop after download completes |
| 4080S (vast.ai) | `ssh -p 19936 root@175.155.64.231` | RTX 4080S 32GB | 9B teacher training + distillation | ACTIVE — 9B SFT running |

### MODEL DELIVERABLES — What must exist locally when done

| Model | File | Ollama Name | Modelfile | Status |
|-------|------|-------------|-----------|--------|
| 4B fine-tuned Q8_0 | `ariaska-cybersec-q8_0.gguf` | `ariaska-cybersec` | `Modelfile_ariaska_cybersec` | Downloading |
| 4B fine-tuned Q6_K | `ariaska-cybersec-q6_k.gguf` | `ariaska-cybersec-q6k` | `Modelfile_ariaska_cybersec_q6k` | Pending (quantize from Q8_0) |
| 9B teacher Q6_K | `ariaska-teacher-9b-q6_k.gguf` | `ariaska-teacher-9b` | `Modelfile_ariaska_teacher_9b` | Pending (after 9B training) |

**Both 4B versions get distillation from the 9B teacher.** After distillation, re-export both Q8_0 and Q6_K from the distilled model.

### Q6_K Quantization (after Q8_0 download completes)
```bash
/home/zer0/llama.cpp/build/bin/llama-quantize \
  /home/zer0/Projects/Ariaska_RL/ariaska-cybersec-q8_0.gguf \
  /home/zer0/Projects/Ariaska_RL/ariaska-cybersec-q6_k.gguf \
  Q6_K

# Register both in Ollama
cd /home/zer0/Projects/Ariaska_RL
ollama create ariaska-cybersec -f Modelfile_ariaska_cybersec
ollama create ariaska-cybersec-q6k -f Modelfile_ariaska_cybersec_q6k
```

### 9B Teacher GGUF (after 9B training completes on 4080S)
The 9B model is ~18GB safetensors. Export on GPU server then download:
```bash
# On 4080S: need the same hash fix in llama.cpp
ssh -p 19936 root@175.155.64.231 'cd /root/llama.cpp 2>/dev/null || git clone --depth 1 https://github.com/ggerganov/llama.cpp /root/llama.cpp'
# Add hash: 1444df51... → "qwen35" (may need different hash for 9B — check error output)
# Convert to f16 first, then quantize to Q6_K
ssh -p 19936 root@175.155.64.231 'python3 /root/llama.cpp/convert_hf_to_gguf.py /root/ariaska_finetune/outputs/teacher_merged --outtype f16 --outfile /tmp/teacher-f16.gguf'
# Then quantize (need llama-quantize built on 4080S too, or download f16 and quantize locally)
```

### CRITICAL: What NOT to do
- Do NOT kill the training process on the 4080S
- Do NOT modify `train_ariaska_v3.py` on the 4080S while training is running
- Do NOT try to install flash-attn (causes CUDA crashes on Qwen3.5 hybrid architecture)
- Do NOT change `STATE_DIM = 512` or PPO core parameters
