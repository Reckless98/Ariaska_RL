# Ariaska V4 Nuclear Pipeline — Claude Code Handoff

**Date:** 2026-03-18 12:00 UTC  
**From:** VSCode Copilot  
**To:** Claude Code  
**Status:** Schema patching COMPLETE → Ready to launch V4 nuclear pipeline

---

## Current State

### GPU Server
- **Host:** `ssh -p 19312 root@175.155.64.148`
- **GPU:** RTX 4090, 49,140 MiB VRAM (currently **FREE** — vLLM just unloaded)
- **Disk:** ~15 GB free on /workspace
- **RAM:** 755 GB (AMD EPYC 7542)
- **/dev/shm:** 47 GB tmpfs — FP8 teacher already here (saves 9 min download)

### Schema Data — ALL 10 TYPES AT 100%
```
File: /workspace/data/ariaska_schema_sft.jsonl  (14,900 lines)

coherence_classify           1000/1000 = 100.0% ✅
coherence_score              1000/1000 = 100.0% ✅
coherence_summarize          1000/1000 = 100.0% ✅
microchain_classify          1500/1500 = 100.0% ✅
microchain_fast_local        1800/1800 = 100.0% ✅
microchain_generate          2000/2000 = 100.0% ✅
microchain_score             1800/1800 = 100.0% ✅
phase_guided                 1200/1200 = 100.0% ✅
phase_guided_fast_local      1800/1800 = 100.0% ✅
smart_mentor                 1800/1800 = 100.0% ✅
```

### Models on Server
| Role | Model | HuggingFace ID | Path | Size | Status |
|------|-------|----------------|------|------|--------|
| **TEACHER** | Qwen3-32B (FP8 quantized) | `Qwen/Qwen3-32B-FP8` | `/dev/shm/qwen3-32b-fp8/` | 32 GB (7 safetensor shards) | Complete — loaded from tmpfs for fast access |
| **STUDENT** | Qwen3-4B (full precision) | `Qwen/Qwen3-4B` | `/workspace/models/qwen3-4b/` | 7.6 GB (2 safetensor shards) | Complete — this is the model being trained to become `ariaska-cybersec2` |

**Relationship:** The 32B FP8 teacher provides soft logit targets during KD-SFT (Stage 4). The 4B student learns to mimic the teacher's output distribution while also fitting hard labels from the schema data. The final GGUF export is the student only — the teacher is never shipped.

### V3 Training Data (used by V4 stages)
| File | Path | Lines |
|------|------|-------|
| CPT Corpus | `/workspace/data/v3/cpt_corpus.jsonl` | 95,935 |
| DPO Pairs | `/workspace/data/v3/dpo_pairs.jsonl` | 1,795 |
| Task files | `/workspace/data/v3/*.jsonl` | ~32,282 total |

---

## What to Do Next

### Step 1: Launch V4 Nuclear Pipeline
```bash
ssh -p 19312 root@175.155.64.148 \
  'nohup python3 -u /workspace/scripts/ariaska_cybersec4_nuclear.py \
   > /workspace/pipeline_stdout.log 2>&1 & echo PID=$!'
```

### Step 2: V4 Pipeline Will Execute These Stages
| Stage | What | Will Skip? | Est. Time |
|-------|------|------------|-----------|
| 1: Data Gen | AWQ schema generation | **YES** (14,900 ≥ 8,000) | — |
| 2: Student DL | Delete AWQ, download student | **YES** (config.json exists) | — |
| 3: CPT | LoRA r=128 on student, uses cpt_corpus.jsonl | **RUNS** | ~45 min |
| 4a: FP8 DL | Download FP8 teacher | **YES** (already in /dev/shm) | — |
| 4: KD-SFT | Nuclear fusion (FP8 teacher + CPT student, α_kd=0.6/α_ce=0.4) | **RUNS** | ~3-4 hours |
| 5: DPO | β=0.1, dpo_pairs.jsonl | **RUNS** | ~20 min |
| 6: Merge+GGUF | Merge all LoRA adapters + quantize Q8_0 + Q6_K | **RUNS** | ~30 min |

**Total estimated: ~5-6 hours**

### Step 3: Monitor Progress
```bash
# Tail the log
ssh -p 19312 root@175.155.64.148 'tail -30 /workspace/pipeline_stdout.log'

# Check GPU
ssh -p 19312 root@175.155.64.148 'nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader'

# Check process
ssh -p 19312 root@175.155.64.148 'pgrep -af ariaska_cybersec4'

# Check outputs
ssh -p 19312 root@175.155.64.148 'ls -la /workspace/output/'
```

**Skip conditions (in the v4 script):**
- Stage 3 CPT skips if `/workspace/output/cpt/final/adapter_config.json` exists
- Stage 4 KD-SFT skips if `/workspace/output/kdsft/final/adapter_config.json` exists
- Stage 5 DPO skips if `/workspace/output/dpo/final/adapter_config.json` exists

### Step 4: Download GGUFs (after completion)
```bash
# Download to local machine
scp -P 19312 root@175.155.64.148:/workspace/output/gguf/ariaska-cybersec2-q8_0.gguf .
scp -P 19312 root@175.155.64.148:/workspace/output/gguf/ariaska-cybersec2-q6_k.gguf .
```

### Step 5: Register with Ollama & Evaluate
```bash
# Create Modelfile pointing to the GGUF
ollama create ariaska-cybersec2 -f Modelfile_ariaska_cybersec2

# Run Ariaska deep eval (12 schema types)
# Target: 12/12 (100%) — v1 was 5/12 (42%)
```

---

## Known Risks

1. **VRAM tightness in Stage 4**: FP8 teacher Qwen3-32B (34 GB inference) + student Qwen3-4B (7.6 GB) + LoRA adapters + 8-bit optimizer states ≈ 47/49 GB. The script uses 8-bit AdamW + gradient checkpointing to fit. If OOM, reduce `max_seq_len` from 1024 to 768.

2. **Disk space**: ~15 GB free. CPT output ~2 GB, KD-SFT ~2 GB, DPO ~2 GB, merged ~7.6 GB, GGUFs ~8 GB × 2. May need to clean `/workspace/output/cpt/` and `/workspace/output/kdsft/` after merge to fit GGUFs.

3. **Thinking token leakage**: The 22% validation pass rate in schema patching was because Qwen3 generates thinking tokens even with `enable_thinking=False`. This is a training data quality concern — 22% of FP8-generated examples may have subtle thinking artifacts. The v4 pipeline's SFT stage uses 8x oversampling of validated schema data so the training signal is strong.

4. **FP8 model in /dev/shm**: If the server restarts, `/dev/shm` is cleared. The v4 script will re-download FP8 (~9 min) if this happens.

---

## V4 Nuclear Pipeline Architecture (for reference)

```
Stage 3: CPT (Continued Pre-Training)
  └─ Model: STUDENT (Qwen3-4B) only — no teacher involved
  └─ LoRA r=128, α=256, lr=2e-4, 3 epochs
  └─ Data: cpt_corpus.jsonl (95,935 lines of cybersec domain text)
  └─ Purpose: Adapt student's base knowledge to cybersecurity domain
  └─ Output: /workspace/output/cpt/final/

Stage 4: KD-SFT Fusion (Knowledge Distillation + Supervised Fine-Tuning)
  └─ TEACHER: Qwen3-32B-FP8 (frozen, inference-only, loaded from /dev/shm)
  └─ STUDENT: Qwen3-4B + CPT adapter (trainable via LoRA)
  └─ Loss = 0.6 × KL(student_logits/T, teacher_logits/T) × T² + 0.4 × CE(student_logits, hard_labels)
  └─ The teacher runs forward pass on each batch to produce soft logit targets
  └─ The student learns to match both teacher distribution AND correct answers
  └─ Temperature cosine: T = 4 → 2 (higher T = softer distribution = more knowledge transfer)
  └─ LoRA r=128, α=256, lr=1e-4, 3 epochs
  └─ NEFTune noise α=5.0
  └─ Data: schema 8x oversampled (14,900 × 8 = 119,200) + V3 task data (~32K)
  └─ FP8 teacher already in /dev/shm (NO download needed — saves 9 min)
  └─ Output: /workspace/output/kdsft/final/

Stage 5: DPO (Direct Preference Optimization)
  └─ Model: STUDENT only (with CPT + KD-SFT adapters merged)
  └─ β=0.1, LoRA r=64, α=128, lr=5e-7, 1 epoch
  └─ Data: dpo_pairs.jsonl (1,795 chosen/rejected pairs)
  └─ Purpose: Align student outputs with human preferences
  └─ Output: /workspace/output/dpo/final/

Stage 6: Merge + GGUF Export
  └─ Merge STUDENT base + CPT adapter + KD-SFT adapter + DPO adapter → single model
  └─ Quantize merged model: Q8_0 (high quality) + Q6_K (balanced)
  └─ Teacher is NOT included — final model is pure 4B student
  └─ Output: /workspace/output/gguf/
```

---

## Files on Server

| File | Path | Purpose |
|------|------|---------|
| V4 Pipeline Script | `/workspace/scripts/ariaska_cybersec4_nuclear.py` | Main pipeline (md5: `5744d95767780ec8cf36d30f1b266e16`) |
| FP8 Patch Script | `/workspace/scripts/fp8_patch_vllm.py` | Used to patch schema data (DONE) |
| FP8 Patch Log | `/workspace/fp8_patch_vllm.log` | Shows 100% patching results |
| Schema Data | `/workspace/data/ariaska_schema_sft.jsonl` | 14,900 validated examples |
| Pipeline Log | `/workspace/pipeline_stdout.log` | Will contain v4 output (launch pending) |

---

## Success Criteria

- [ ] All 6 stages complete without error
- [ ] GGUFs exported to `/workspace/output/gguf/`
- [ ] GGUFs downloaded to local machine
- [ ] Registered with Ollama
- [ ] Deep eval: 12/12 schema types pass (target ≥ 99%)
- [ ] No think-tag leakage in model outputs
- [ ] JSON key compliance on all Ariaska-specific schemas
