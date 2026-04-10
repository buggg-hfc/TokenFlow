#!/bin/bash

# -----------------------------------------------------------------
# Paths - modify these for your environment
# -----------------------------------------------------------------
export MODEL_LOG_PATH=./checkpoints/inital-try

# DATA_PATH is currently unused by ExampleDataset (which loads
# sayakpaul/coco-30-val-2014 from HuggingFace automatically).
# Set it to any non-empty string or your own data path.
export DATA_PATH="unused"

export VISION_TOWER_CKPT="/home/weight/TokenFlow/tokenflow_clipb_32k_enhanced.pt"

# Choose base LLM: Llama-2-7B (default) or TinyLlama-1.1B (lighter)
# export MODEL_PATH="TinyLlama/TinyLlama_v1.1"
export MODEL_PATH="/path/to/Llama-2-7b-hf"   # set to local path

# -----------------------------------------------------------------
# Bypass SSL certificate verification for HuggingFace dataset download
# (needed in environments with self-signed proxy certificates)
# -----------------------------------------------------------------
export CURL_CA_BUNDLE=""
export REQUESTS_CA_BUNDLE=""
export HF_DATASETS_TRUST_REMOTE_CODE=1

# Ensure llava_t2i package is importable regardless of where this script is called from
T2I_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${T2I_DIR}:${PYTHONPATH}"

# Always run from the t2i/ root so relative paths (scripts/, llava_t2i/) work correctly
cd "${T2I_DIR}"

deepspeed  \
  "${T2I_DIR}/llava_t2i/train/train_plain.py" \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $MODEL_PATH \
    --version plain_img \
    --data_path $DATA_PATH \
    --vision_tower $VISION_TOWER_CKPT \
    --mm_vision_vq_type TOKEN_FLOW \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_vq_token True \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --ignore_text_loss True \
    --bf16 True \
    --output_dir $MODEL_LOG_PATH \
    --max_grad_norm 0.5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --eval_steps 2000 \
    --save_total_limit 50 \
    --num_train_epochs 10 \
    --learning_rate 5e-4 \
    --weight_decay 0.1 \
    --warmup_steps 5000 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048 \
    --max_text_token_num 128 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to none
