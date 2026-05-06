"""
Merge a partial adapter checkpoint (mm_projector.bin) produced by
--tune_mm_mlp_adapter_and_logits training back into the full base model
and save a self-contained model directory usable by run_llava_samples.py
and run_geneval.py.

Usage
-----
python3 t2i/scripts/export_finetuned.py \
    --base_model   /path/to/TokenFlow-t2i \
    --adapter_ckpt ./checkpoints/finetune-tokenflow-bf16/checkpoint-5625 \
    --output_dir   ./checkpoints/finetune-tokenflow-merged \
    --vision_tower /home/weight/TokenFlow/tokenflow_clipb_32k_enhanced.pt

The adapter checkpoint directory must contain:
  mm_projector.bin  – fine-tuned weights (projector / lm_head / lvl_embed / pos_1LC)
  config.json       – model config saved by the trainer
"""

import argparse
import os
import torch
import transformers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True,
                        help="Path to the full pretrained TokenFlow-t2i model")
    parser.add_argument("--adapter_ckpt", required=True,
                        help="Checkpoint directory containing mm_projector.bin + config.json")
    parser.add_argument("--output_dir", required=True,
                        help="Where to save the merged full model")
    parser.add_argument("--vision_tower", required=True,
                        help="Path to tokenflow_clipb_32k_enhanced.pt")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    # ------------------------------------------------------------------ #
    # 1. Load the base model (full weights, all parameters)
    # ------------------------------------------------------------------ #
    print(f"[1/4] Loading base model from {args.base_model} ...")
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from llava_t2i.model import LlavaLlamaForCausalLM

    model = LlavaLlamaForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        mm_vision_tower=args.vision_tower,
    )

    # ------------------------------------------------------------------ #
    # 2. Load the adapter weights
    # ------------------------------------------------------------------ #
    adapter_bin = os.path.join(args.adapter_ckpt, "mm_projector.bin")
    print(f"[2/4] Loading adapter weights from {adapter_bin} ...")
    adapter_weights = torch.load(adapter_bin, map_location="cpu")
    print(f"      Keys in adapter: {list(adapter_weights.keys())[:8]} ...")

    # ------------------------------------------------------------------ #
    # 3. Apply adapter weights (strict=False: only update matching keys)
    # ------------------------------------------------------------------ #
    print("[3/4] Merging adapter into base model ...")
    missing, unexpected = model.load_state_dict(adapter_weights, strict=False)
    if unexpected:
        print(f"  WARNING: unexpected keys (ignored): {unexpected}")
    if missing:
        # Expected: the vast majority of LLM backbone keys will be "missing"
        # from adapter_weights — that's normal, they stay as-is from base model.
        print(f"  Info: {len(missing)} keys kept from base model (not in adapter)")

    # ------------------------------------------------------------------ #
    # 4. Save the merged model
    # ------------------------------------------------------------------ #
    print(f"[4/4] Saving merged model to {args.output_dir} ...")
    os.makedirs(args.output_dir, exist_ok=True)

    # Normalize non-JSON-serializable config fields before saving.
    # VQType is an enum; store only its name string so config.json is valid.
    if hasattr(model.config, "mm_vision_vq_type"):
        vq = model.config.mm_vision_vq_type
        if not isinstance(vq, str):
            model.config.mm_vision_vq_type = vq.name if hasattr(vq, "name") else str(vq)

    model.save_pretrained(args.output_dir)

    # Copy tokenizer files from base model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model, use_fast=False
    )
    tokenizer.save_pretrained(args.output_dir)

    # Copy generation_config if present
    gen_cfg_src = os.path.join(args.base_model, "generation_config.json")
    if os.path.exists(gen_cfg_src):
        import shutil
        shutil.copy(gen_cfg_src, args.output_dir)

    print("Done. Merged model saved to:", args.output_dir)
    print()
    print("Run inference with:")
    print(f"  python llava_t2i/eval/run_llava_samples.py \\")
    print(f"    --model-path {args.output_dir} \\")
    print(f"    --tokenizer-path {args.vision_tower} \\")
    print(f"    --output-path ./generation")


if __name__ == "__main__":
    main()
