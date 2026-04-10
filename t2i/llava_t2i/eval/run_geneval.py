import argparse
import json
import os

import numpy as np
import torch
import transformers
from PIL import Image
from tqdm import tqdm

from llava_t2i.dataset.process import crop_and_encode_text_and_img
from llava_t2i.model import LlavaLlamaForCausalLM
from llava_t2i.utils import disable_torch_init

multi_step_infer_strategy = {
    1: {"topk_list": [600],        "topp_list": [0.6]},
    2: {"topk_list": [1200, 1],    "topp_list": [0.8, 0]},
    3: {"topk_list": [1200, 100, 1], "topp_list": [0.8, 0.8, 0]},
}

NEGATIVE_PROMPT = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
    "fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, "
    "signature, watermark, username, blurry."
)


def load_model(args):
    disable_torch_init()
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    ptdtype = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
        args.mixed_precision
    ]
    model = LlavaLlamaForCausalLM.from_pretrained(
        args.model_path,
        attn_implementation="eager",
        mm_vision_tower=args.tokenizer_path,
    )
    model = model.eval().to(ptdtype).cuda()
    model.get_vision_tower().to(ptdtype)
    model.config.mm_vision_vq_type = str(model.config.mm_vision_vq_type)
    assert getattr(model.config, "mm_use_vq_token", False)
    model.config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=model.config.tokenizer_model_max_length,
        padding_side="right",
        use_fast=False,
    )
    model.reinit_image_token_start_end(tokenizer)
    return model, tokenizer, ptdtype


def generate_batch(model, tokenizer, prompts, args, topk_list, topp_list):
    batch_size = len(prompts)
    prefix_text_codes = []
    for p in prompts:
        input_id, _ = crop_and_encode_text_and_img(
            tokenizer, p, image=None, max_text_token_num=128
        )
        prefix_text_codes.append(input_id)

    uncond_id, _ = crop_and_encode_text_and_img(
        tokenizer, NEGATIVE_PROMPT, image=None, max_text_token_num=128
    )
    prefix_text_codes += [uncond_id] * batch_size

    with torch.inference_mode():
        samples = model.autoregressive_infer_cfg(
            B=batch_size,
            prefix_text_codes=prefix_text_codes,
            cfg=args.cfg,
            topk_list=topk_list,
            topp_list=topp_list,
            g_seed=args.g_seed,
        )
    return samples  # list of (H, W, 3) uint8 tensors


def main(args):
    # Load GenEval metadata
    with open(args.metadata_path) as f:
        metadata = [json.loads(line) for line in f if line.strip()]

    topk_list = multi_step_infer_strategy[args.loop]["topk_list"]
    topp_list = multi_step_infer_strategy[args.loop]["topp_list"]

    model, tokenizer, _ = load_model(args)

    # GenEval expects: output_dir/{index}/{sample_index}.png
    os.makedirs(args.output_path, exist_ok=True)

    # Build flat list of (prompt, out_dir) pairs
    tasks = []
    for idx, meta in enumerate(metadata):
        prompt = meta["prompt"]
        out_dir = os.path.join(args.output_path, f"{idx:0>5}")
        os.makedirs(out_dir, exist_ok=True)
        for sample_idx in range(args.num_samples):
            out_file = os.path.join(out_dir, f"{sample_idx:0>4}.png")
            if not os.path.exists(out_file):
                tasks.append((prompt, out_file))

    print(f"Total images to generate: {len(tasks)}")

    batch_size = args.batch_size
    for i in tqdm(range(0, len(tasks), batch_size), desc="Generating"):
        batch = tasks[i : i + batch_size]
        prompts = [t[0] for t in batch]
        out_files = [t[1] for t in batch]

        samples = generate_batch(model, tokenizer, prompts, args, topk_list, topp_list)

        for img_tensor, out_file in zip(samples, out_files):
            Image.fromarray(img_tensor.numpy().astype(np.uint8)).save(out_file)

        torch.cuda.empty_cache()

    print(f"Done. Images saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="/tmp/geneval/prompts/evaluation_metadata.jsonl",
        help="Path to GenEval evaluation_metadata.jsonl",
    )
    parser.add_argument("--output-path", type=str, default="geneval_generations/")
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--loop", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=4,
                        help="Number of images to generate per prompt")
    parser.add_argument("--g_seed", type=int, default=None)
    args = parser.parse_args()
    main(args)
