#!/usr/bin/env bash
set -euo pipefail
# ─────────────────────────────────────────────────────────────────────────────
# Ariaska — GPU Server Fine-tuning Automation
# Runs on the REMOTE GPU server. Handles full pipeline:
#   1. Install dependencies
#   2. Download base model
#   3. Train QLoRA
#   4. Merge adapters
#   5. Export GGUF (Q4_K_M + Q5_K_M)
#   6. Signal readiness for download
# ─────────────────────────────────────────────────────────────────────────────

WORKDIR="${WORKDIR:-/root/ariaska_finetune}"
MODEL_NAME="Qwen/Qwen3-4B"
CONFIG_FILE="$WORKDIR/configs/qlora_qwen35_4b.yaml"
OUTPUT_DIR="$WORKDIR/outputs/qwen35-4b-ariaska"
GGUF_DIR="$WORKDIR/outputs/gguf"

echo "════════════════════════════════════════════════════"
echo "  Ariaska Fine-tuning Pipeline — GPU Server"
echo "════════════════════════════════════════════════════"
echo "Workdir: $WORKDIR"
echo "Model:   $MODEL_NAME"
echo ""

# ── Step 0: System check ────────────────────────────────────────────────────

echo "[0/6] System check..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || {
    echo "ERROR: nvidia-smi failed. No GPU detected."
    exit 1
}
echo ""

# ── Step 1: Install dependencies ────────────────────────────────────────────

echo "[1/6] Installing Python dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || true
pip install transformers datasets peft accelerate bitsandbytes trl
pip install sentencepiece protobuf pyyaml flash-attn --no-build-isolation 2>/dev/null || {
    echo "WARN: flash-attn install failed — will fall back to sdpa"
}

# llama.cpp for GGUF export
if [ ! -d "$WORKDIR/llama.cpp" ]; then
    echo "Cloning llama.cpp for GGUF conversion..."
    cd "$WORKDIR"
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp
    pip install -r requirements.txt 2>/dev/null || true
    # Build quantize tool
    cmake -B build -DGGML_CUDA=ON 2>/dev/null || cmake -B build
    cmake --build build --config Release -j$(nproc) --target llama-quantize 2>/dev/null || {
        echo "WARN: cmake build failed, trying make..."
        make -j$(nproc) llama-quantize 2>/dev/null || echo "WARN: quantize build failed — will try Python-only export"
    }
fi
echo "Dependencies installed."
echo ""

# ── Step 2: Download base model ─────────────────────────────────────────────

echo "[2/6] Downloading base model: $MODEL_NAME..."
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Downloading tokenizer...')
tok = AutoTokenizer.from_pretrained('$MODEL_NAME', trust_remote_code=True)
print('Downloading model...')
model = AutoModelForCausalLM.from_pretrained('$MODEL_NAME', trust_remote_code=True, torch_dtype='auto')
print('Model downloaded successfully')
print(f'Parameters: {model.num_parameters():,}')
del model
"
echo ""

# ── Step 3: QLoRA Fine-tuning ───────────────────────────────────────────────

echo "[3/6] Starting QLoRA fine-tuning..."
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"

cd "$WORKDIR"
python3 scripts/finetune.py --config configs/qlora_qwen35_4b.yaml

ADAPTER_PATH="$OUTPUT_DIR/final"
if [ ! -d "$ADAPTER_PATH" ]; then
    echo "ERROR: Training did not produce adapter at $ADAPTER_PATH"
    ls -la "$OUTPUT_DIR"/
    exit 1
fi
echo "Training complete. Adapter at: $ADAPTER_PATH"
echo ""

# ── Step 4: Merge LoRA adapter ──────────────────────────────────────────────

echo "[4/6] Merging LoRA adapter into base model..."
python3 -c "
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

adapter_path = '$ADAPTER_PATH'
output_path = '$GGUF_DIR/merged_fp16'
os.makedirs(output_path, exist_ok=True)

config = PeftConfig.from_pretrained(adapter_path)
print(f'Base model: {config.base_model_name_or_path}')

print('Loading base model in fp16...')
base = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map='cpu',
    trust_remote_code=True,
)

print('Loading and merging adapter...')
model = PeftModel.from_pretrained(base, adapter_path)
merged = model.merge_and_unload()

print(f'Saving merged model to {output_path}...')
merged.save_pretrained(output_path, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
tokenizer.save_pretrained(output_path)
print('Merge complete.')
"
echo ""

# ── Step 5: Convert to GGUF ─────────────────────────────────────────────────

echo "[5/6] Converting to GGUF..."
MERGED_DIR="$GGUF_DIR/merged_fp16"
F16_GGUF="$GGUF_DIR/ariaska-qwen35-4b-f16.gguf"

# Find convert script
CONVERT_SCRIPT=""
for path in "$WORKDIR/llama.cpp/convert_hf_to_gguf.py" "/opt/llama.cpp/convert_hf_to_gguf.py"; do
    if [ -f "$path" ]; then
        CONVERT_SCRIPT="$path"
        break
    fi
done

if [ -z "$CONVERT_SCRIPT" ]; then
    echo "ERROR: convert_hf_to_gguf.py not found"
    echo "Merged model is at: $MERGED_DIR"
    echo "Manually convert with: python convert_hf_to_gguf.py $MERGED_DIR --outfile model.gguf --outtype f16"
    exit 1
fi

echo "Converting to f16 GGUF..."
python3 "$CONVERT_SCRIPT" "$MERGED_DIR" --outfile "$F16_GGUF" --outtype f16

# Quantize
QUANTIZE_BIN=""
for path in "$WORKDIR/llama.cpp/build/bin/llama-quantize" "$WORKDIR/llama.cpp/llama-quantize"; do
    if [ -f "$path" ] && [ -x "$path" ]; then
        QUANTIZE_BIN="$path"
        break
    fi
done

if [ -n "$QUANTIZE_BIN" ]; then
    echo "Quantizing to Q4_K_M..."
    "$QUANTIZE_BIN" "$F16_GGUF" "$GGUF_DIR/ariaska-qwen35-4b-q4_k_m.gguf" Q4_K_M

    echo "Quantizing to Q5_K_M..."
    "$QUANTIZE_BIN" "$F16_GGUF" "$GGUF_DIR/ariaska-qwen35-4b-q5_k_m.gguf" Q5_K_M

    # Clean f16 intermediate
    rm -f "$F16_GGUF"
else
    echo "WARN: llama-quantize not found. Keeping f16 GGUF."
    echo "Quantize manually: llama-quantize $F16_GGUF output.gguf Q4_K_M"
fi
echo ""

# ── Step 6: Create Ollama Modelfile ──────────────────────────────────────────

echo "[6/6] Creating Ollama Modelfile..."
PRIMARY_GGUF="$GGUF_DIR/ariaska-qwen35-4b-q4_k_m.gguf"
if [ ! -f "$PRIMARY_GGUF" ]; then
    PRIMARY_GGUF="$F16_GGUF"
fi

cat > "$GGUF_DIR/Modelfile_ariaska" << 'MODELFILE'
FROM ./ariaska-qwen35-4b-q4_k_m.gguf

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 2048
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

SYSTEM """You are Ariaska, a cybersecurity AI coprocessor specialized in authorized penetration testing. You analyze engagement state, classify attack phases, recommend tactical actions, parse tool outputs, and provide structured reasoning for offensive security operations. You operate within authorized lab environments only. Be concise, precise, and always output in the requested format."""

TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{- end }}
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ .Response }}<|im_end|>"""
MODELFILE

echo ""
echo "════════════════════════════════════════════════════"
echo "  PIPELINE COMPLETE"
echo "════════════════════════════════════════════════════"
echo ""
echo "Output files:"
ls -lh "$GGUF_DIR"/*.gguf 2>/dev/null || echo "  (check $GGUF_DIR)"
echo ""
echo "Modelfile: $GGUF_DIR/Modelfile_ariaska"
echo "Adapter:   $ADAPTER_PATH"
echo "Merged:    $GGUF_DIR/merged_fp16"
echo ""
echo "To import into Ollama locally:"
echo "  scp -P 35188 root@1.193.137.38:$GGUF_DIR/ariaska-qwen35-4b-q4_k_m.gguf ."
echo "  scp -P 35188 root@1.193.137.38:$GGUF_DIR/Modelfile_ariaska ."
echo "  ollama create ariaska-cybersec -f Modelfile_ariaska"
echo ""
