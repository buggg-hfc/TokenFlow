# !/bin/bash

# Ensure the t2i root is in PYTHONPATH so llava_t2i package can be found
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

python3 llava_t2i/eval/run_llava_samples.py \
--model-path "ByteFlow-AI/TokenFlow-t2i" \
--tokenizer-path "../pretrained_ckpts/tokenflow_clipb_32k_enhanced.pt" \
--output-path "generations/" \
--cfg 7.5 \
--loop 1 \
--mixed_precision bf16 \
--batch_size 20 \
# --g_seed 0




