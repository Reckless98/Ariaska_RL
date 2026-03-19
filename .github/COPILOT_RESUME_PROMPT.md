# Paste this into VSCode Copilot Chat (Claude Opus)

Read `.github/CC_SESSION.md` fully before doing anything. It contains the complete state of the ariaska fine-tuning pipeline.

## Your immediate tasks (in order):

### 1. Check if Q8_0 download finished
```bash
ls -lh /home/zer0/Projects/Ariaska_RL/ariaska-cybersec-q8_0.gguf
# Should be ~4.2GB. If still growing, wait.
```

### 2. Quantize Q8_0 to Q6_K locally
```bash
/home/zer0/llama.cpp/build/bin/llama-quantize \
  /home/zer0/Projects/Ariaska_RL/ariaska-cybersec-q8_0.gguf \
  /home/zer0/Projects/Ariaska_RL/ariaska-cybersec-q6_k.gguf \
  Q6_K
```

### 3. Register BOTH in Ollama
```bash
cd /home/zer0/Projects/Ariaska_RL
ollama create ariaska-cybersec -f Modelfile_ariaska_cybersec
ollama create ariaska-cybersec-q6k -f Modelfile_ariaska_cybersec_q6k
ollama list | grep ariaska
```

### 4. Smoke test both
```bash
ollama run ariaska-cybersec '{"task": "phase_classification", "context": {"ports": [22, 80], "shell_obtained": false}}'
ollama run ariaska-cybersec-q6k '{"task": "phase_classification", "context": {"ports": [22, 80], "shell_obtained": false}}'
```

### 5. Run evaluation
```bash
cd /home/zer0/Projects/Ariaska_RL
PYTHONPATH=/home/zer0/Projects/Ariaska_RL .venv/bin/python ariaska_ai/scripts/evaluate.py --baseline jaahas/qwen3.5-uncensored:4b --finetuned ariaska-cybersec
```

### 6. Run tests
```bash
ARIASKA_DRY_RUN=1 PYTHONPATH=/home/zer0/Projects/Ariaska_RL .venv/bin/python -m pytest tests/ -x --tb=line -q --timeout=120 --ignore=tests/test_online_makes_mentor_calls.py --ignore=tests/test_online_requires_key.py
```

### 7. Monitor 9B teacher on 4080S (background check)
```bash
ssh -o ConnectTimeout=20 -p 19936 root@175.155.64.231 'strings /root/ariaska_finetune/teacher_stderr.log 2>/dev/null | grep -oP "\d+/1342.*" | tail -3'
ssh -o ConnectTimeout=20 -p 19936 root@175.155.64.231 'tail -5 /root/ariaska_finetune/teacher_stdout.log'
```
9B SFT is at step ~66/1342, ~20h remaining. When it finishes, the script auto-continues to DPO, merge, and distillation.

## Context you need to know:
- The 4B model (`ariaska-cybersec`) was fine-tuned with CPT + SFT + DPO on Qwen3.5-4B
- Qwen3.5 is a HYBRID architecture (linear_attention + full_attention layers with SSM/Mamba components) — flash-attn DOES NOT WORK, use SDPA only
- GGUF export required a tokenizer hash fix: hash `1444df51289cfa8063b96f0e62b1125440111bc79a52003ea14b6eac7016fd5f` mapped to `"qwen35"` in llama.cpp `convert_hf_to_gguf.py`
- Model router at `core/llm/model_router.py` line 65 already defaults to `ariaska-cybersec` for all tiers
- RAG retriever is wired into SmartMentor + SmartCoach (FAISS, 44K chunks)
- Modelfiles exist: `Modelfile_ariaska_cybersec` (Q8_0), `Modelfile_ariaska_cybersec_q6k` (Q6_K), `Modelfile_ariaska_teacher_9b` (9B teacher Q6_K — pending)
- 5070 Ti server (`ssh -p 52191 root@1.208.108.242`) can be stopped after Q8_0 download completes
- 4080S server (`ssh -p 19936 root@175.155.64.231`) is running 9B training — DO NOT interrupt

## After 9B training finishes (tomorrow):
1. Export 9B teacher to GGUF Q6_K on 4080S (needs same hash fix in llama.cpp)
2. Download 9B teacher GGUF locally (~7GB)
3. Register as `ariaska-teacher-9b` in Ollama using `Modelfile_ariaska_teacher_9b`
4. After distillation completes: re-export distilled 4B as Q8_0 + Q6_K, re-register both
