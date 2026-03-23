#!/bin/bash
# retrain_v6.sh — Clean retrain pipeline for ariaska-cybersec4
# Uses Qwen3.5-27B (AWQ 4-bit) as teacher for reasoning enhancement
#
# Steps:
# 1. Download Qwen3.5-27B-AWQ teacher (~14GB)
# 2. Regenerate clean SFT data (fixed walkthrough miner)
# 3. Enhance shallow reasoning with teacher
# 4. SFT + DPO train on enhanced data
# 5. Merge + GGUF + upload to zer0aria/ariaska-cybersec4

set -euo pipefail

WORKSPACE="/workspace"
SCRIPTS="$WORKSPACE/scripts"
DATA="$WORKSPACE/data"
MODELS="$WORKSPACE/models"
TEACHER_MODEL="QuantTrio/Qwen3.5-27B-AWQ"
TEACHER_DIR="$MODELS/teacher"
BASE_MODEL="$MODELS/ariaska-cybersec2-fp16"  # Same base as v3
OUTPUT_REPO="zer0aria/ariaska-cybersec4"

echo "============================================================"
echo " Ariaska V6 Retrain Pipeline — Teacher-Enhanced Data"
echo "============================================================"

# ── Step 0: Clean up v5 artifacts to free disk ─────────────────────
echo "[CLEANUP] Removing v5 training artifacts..."
rm -rf "$MODELS/sft_lora" "$MODELS/dpo_lora" 2>/dev/null || true
rm -f "$DATA/htb_sft_v5.jsonl" "$DATA/htb_dpo_v5.jsonl" 2>/dev/null || true
echo "[CLEANUP] Done"

# ── Step 1: Download teacher model ────────────────────────────────
echo "[TEACHER] Downloading $TEACHER_MODEL..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$TEACHER_MODEL', local_dir='$TEACHER_DIR')
print('[TEACHER] Download complete')
"

# ── Step 2: Regenerate clean SFT data ─────────────────────────────
echo "[DATA] Regenerating clean SFT data (fixed walkthrough miner)..."
python3 "$SCRIPTS/generate_htb_data.py" \
    --output-dir "$DATA" \
    --sft-count 15000 \
    --dpo-count 4000 \
    --output-parse-count 2000 \
    --seed 42

# Verify no contamination
echo "[DATA] Auditing generated data..."
python3 -c "
import json, re
with open('$DATA/htb_sft_v5.jsonl') as f:
    lines = f.readlines()
bad = 0
for line in lines:
    ex = json.loads(line)
    if ex.get('schema_type') == 'smart_mentor_walkthrough':
        asst = json.loads(ex['messages'][-1]['content'])
        cmd = asst.get('selected_command', '')
        if not re.match(r'^[a-z][a-z0-9_]+$', cmd):
            bad += 1
print(f'Walkthrough contamination: {bad} bad examples')
if bad > 10:
    print('WARNING: Still too many contaminated examples!')
    exit(1)
print('[DATA] Clean!')
"

# ── Step 3: Enhance reasoning with teacher ────────────────────────
echo "[ENHANCE] Running teacher enhancement on shallow reasoning..."
python3 "$SCRIPTS/teacher_enhance.py" \
    --input "$DATA/htb_sft_v5.jsonl" \
    --output "$DATA/htb_sft_v6_enhanced.jsonl" \
    --model "$TEACHER_DIR"

# Verify enhancement quality
python3 -c "
import json
with open('$DATA/htb_sft_v6_enhanced.jsonl') as f:
    lines = f.readlines()
shallow = 0
for line in lines:
    ex = json.loads(line)
    st = ex.get('schema_type', '')
    if st in ('smart_mentor', 'smart_mentor_walkthrough', 'microchain_fast_local'):
        try:
            r = json.loads(ex['messages'][-1]['content'])
            if len(r.get('reasoning', r.get('reason', ''))) < 30:
                shallow += 1
        except: pass
total = len(lines)
print(f'Enhanced data: {total} examples, {shallow} still shallow ({100*shallow/max(1,total):.1f}%)')
"

# ── Step 4: Free teacher VRAM + disk, start training ──────────────
echo "[TRAIN] Freeing teacher model VRAM + disk (~14GB)..."
rm -rf "$TEACHER_DIR"
echo "[TRAIN] Teacher model deleted, disk reclaimed"

echo "[TRAIN] Starting V6 SFT+DPO training with enhanced data..."
# Use the enhanced SFT data for training
cp "$DATA/htb_sft_v6_enhanced.jsonl" "$DATA/htb_sft_v5.jsonl"

python3 "$SCRIPTS/ariaska_v5_htb_training.py" 2>&1 | tee "$WORKSPACE/v6_training.log"

echo "============================================================"
echo " V6 Retrain Complete"
echo "============================================================"
