# Ariaska Cybersec 2.0 — Claude Code Handoff Brief

## What's Happening

KD-SFT (Knowledge Distillation - Supervised Fine-Tuning) training is running on a vast.ai GPU server.  
**Copilot is monitoring it every 15 minutes.** If something changes, the user will inform you.

## Server Access

```bash
ssh -p 19312 root@175.155.64.148
```

## Current Training Status

- **Pipeline script**: `/workspace/scripts/ariaska_cybersec4_nuclear.py`
- **Stage**: KD-SFT (Stage 4) — training 1 epoch, 14,900 steps
- **Started**: 2026-03-18 15:58 UTC
- **ETA completion**: ~21:30-22:00 UTC (March 18)
- **Loss at step 4000**: 0.98 (KD: 1.38, CE: 0.38) — converging well
- **GPU**: RTX 4090, 48GB/49GB used, stable
- **Speed**: ~1.35s/step
- **Config**: lr=5e-5 cosine, temp 4.0→2.0, alpha_kd=0.6, alpha_ce=0.4, LoRA r=128
- **Log file**: `/workspace/pipeline_stdout.log`

## Remaining Pipeline Stages (automated in script)

After KD-SFT completes, the script automatically runs:

1. **Stage 5: DPO** — Direct Preference Optimization using 1,795 preference pairs
   - Output: `/workspace/output/dpo/final`
   - Should take ~20-30 min
   
2. **Stage 6: Merge + GGUF Export**
   - Merges all adapters (CPT → KD-SFT → DPO) onto base Qwen3-4B
   - Exports **F16** (full precision, ~8GB) and **Q8_0** (quantized, ~4.5GB)
   - Output: `/workspace/output/gguf/ariaska-cybersec2-f16.gguf` and `ariaska-cybersec2-q8_0.gguf`

## Monitoring Commands

```bash
# Check latest training progress
ssh -p 19312 root@175.155.64.148 'tail -5 /workspace/pipeline_stdout.log'

# Check if process is alive
ssh -p 19312 root@175.155.64.148 'ps aux | grep nuclear | grep -v grep'

# Check GPU usage
ssh -p 19312 root@175.155.64.148 'nvidia-smi | grep MiB | head -2'

# Check for errors
ssh -p 19312 root@175.155.64.148 'grep -i "error\|Error\|OOM\|CUDA\|Traceback" /workspace/pipeline_stdout.log | tail -10'

# Check if DPO/GGUF stages started
ssh -p 19312 root@175.155.64.148 'grep -i "STAGE\|stage\|complete\|GGUF\|DPO" /workspace/pipeline_stdout.log | tail -15'
```

## After Pipeline Completes — Download GGUFs

```bash
# Download F16 (full precision, for further training)
scp -P 19312 root@175.155.64.148:/workspace/output/gguf/ariaska-cybersec2-f16.gguf /home/zer0/Projects/Ariaska_RL/

# Download Q8_0 (quantized, for inference)
scp -P 19312 root@175.155.64.148:/workspace/output/gguf/ariaska-cybersec2-q8_0.gguf /home/zer0/Projects/Ariaska_RL/
```

## Quality Evaluation Needed

After downloading, Claude Code should evaluate the model:

1. **Schema compliance test** — Does the model produce all required fields for each of the 10 schema types?
   - `coherence_classify`, `coherence_score`, `coherence_summarize`
   - `microchain_classify`, `microchain_fast_local`, `microchain_generate`, `microchain_score`
   - `phase_guided`, `phase_guided_fast_local`, `smart_mentor`

2. **Compare against V1** — V1 had 100% general cybersec but 42% Ariaska schema compliance
   - Target: 100% schema compliance + maintained general capability

3. **Integration test** — If schema compliance is good, test the Q8 GGUF with `ollama` or `llama-cpp-python`:
   ```bash
   # Create Modelfile
   echo 'FROM ./ariaska-cybersec2-q8_0.gguf' > Modelfile_ariaska_cybersec2
   ollama create ariaska-cybersec2 -f Modelfile_ariaska_cybersec2
   ```

## If Training Needs More Steps

If loss at end of epoch is still dropping (>0.01 per 1000 steps), consider:
1. Modify script: change `num_epochs` to 2 or add a continuation loop
2. The LR will have decayed significantly by step 14,900 — may need warmup restart
3. DPO + GGUF takes ~1h total, so there's time buffer

## Time Budget

- **Server rental**: ~11h total from start (~15:40 UTC March 18)
- **KD-SFT**: ~5.8h → done ~21:30 UTC
- **DPO**: ~20-30 min
- **Merge+GGUF**: ~30 min
- **Download**: F16 ~8GB + Q8 ~4.5GB (depends on connection speed)
- **Buffer**: ~3h for evaluation, debugging, or extra training

## Key Files on Server

| Path | Purpose |
|------|---------|
| `/workspace/scripts/ariaska_cybersec4_nuclear.py` | Pipeline script |
| `/workspace/pipeline_stdout.log` | Training log (stdout+stderr) |
| `/workspace/data/ariaska_schema_sft.jsonl` | 14,900 training examples |
| `/workspace/output/kdsft/` | KD-SFT checkpoints |
| `/workspace/output/dpo/` | DPO output (after stage 5) |
| `/workspace/output/gguf/` | Final GGUF files |
| `/workspace/output/merged/` | Merged HF model (pre-GGUF) |
| `/dev/shm/qwen3-32b-fp8/` | FP8 teacher model (32GB in RAM) |
| `/workspace/models/qwen3-4b/` | Base student model |

## Loss Trajectory So Far

| Step | Loss | KD Loss | CE Loss | Notes |
|------|------|---------|---------|-------|
| 50 | 2.878 | 3.098 | 2.548 | Starting |
| 250 | 1.949 | 2.262 | 1.482 | Fast drop |
| 750 | 1.151 | 1.638 | 0.420 | CE converging |
| 1400 | 1.059 | 1.503 | 0.394 | Plateau starting |
| 2700 | 0.983 | 1.382 | 0.383 | Below 1.0 |
| 4000 | 0.980 | 1.377 | 0.383 | Very stable |

CE loss at 0.38 = model is already producing correct tokens with very high confidence.
KD loss still declining = student matching teacher distribution more closely over time.
