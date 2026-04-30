#!/bin/bash
# -----------------------------------------------------------------
# Fine-tune the official TokenFlow-t2i pretrained model on custom data.
#
# Key differences from train_t2i.sh (training from scratch):
#   - MODEL_PATH points to the full TokenFlow checkpoint, not base LLM
#   - Lower learning rate (1e-5 instead of 5e-4)
#   - Shorter warmup (200 steps instead of 5000)
#   - Fewer epochs (typically 1-3 is enough for fine-tuning)
#
# Strategy options (uncomment one):
#   A) Full fine-tune   : default below (all params trainable)
#   B) Projector-only   : add --tune_mm_mlp_adapter_and_logits True --freeze_backbone True
#   C) QLoRA            : add --use_qlora True, switch to zero2.json
# -----------------------------------------------------------------

# Path to the official downloaded TokenFlow-t2i model
export MODEL_PATH="/home/weight/TokenFlow/TokenFlow-t2i/"

# Tokenizer (VQ vision tower) checkpoint — same as before
export VISION_TOWER_CKPT="/home/weight/TokenFlow/tokenflow_clipb_32k_enhanced.pt"

# Your own dataset directory (HuggingFace arrow format), or "unused" for COCO demo data
export DATA_PATH="/home/dataset/coco-30-val-2014"

# Output directory for checkpoints
export MODEL_LOG_PATH=./checkpoints/finetune-tokenflow

# -----------------------------------------------------------------
export CURL_CA_BUNDLE=""
export REQUESTS_CA_BUNDLE=""
export HF_DATASETS_TRUST_REMOTE_CODE=1

T2I_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${T2I_DIR}:${PYTHONPATH}"
cd "${T2I_DIR}"

CUDA_VISIBLE_DEVICES=6,7 deepspeed \
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
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --warmup_steps 200 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048 \
    --max_text_token_num 128 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to none

# -----------------------------------------------------------------
# Strategy B: Projector-only fine-tune (add these flags instead)
# -----------------------------------------------------------------
# --tune_mm_mlp_adapter_and_logits True \
# --freeze_backbone True \
# --learning_rate 1e-4 \
# --warmup_steps 100 \

# -----------------------------------------------------------------
# Strategy C: QLoRA fine-tune (replace deepspeed config + add flags)
# -----------------------------------------------------------------
# --deepspeed ./scripts/zero2.json \   # QLoRA requires ZeRO-2
# --use_qlora True \
# --lora_r 64 \
# --lora_alpha 16 \
# --lora_dropout 0.05 \
# --learning_rate 2e-4 \
