"""
VLM Multi-Model Inference Script for SageMaker
================================================
Supports: Phi-3.5-vision, Qwen2-VL, LLaVA-1.6, InternVL2, DeepSeek-VL2
Usage:
    python vlm_inference.py --model phi3_v --images img1.jpg img2.jpg --question "What do you see?"
    python vlm_inference.py --model qwen2_vl --images img1.jpg --question "Describe this image"
    python vlm_inference.py --compare --images img1.jpg --question "What is this?"
"""

import argparse
import logging
import time
import json
import os
from typing import NamedTuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

import PIL.Image
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_image


# ── Shared defaults ──────────────────────────────────────────────────────────

DEFAULT_QUESTION = "What is the content of each image?"
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.2


# ── Data container ────────────────────────────────────────────────────────────

class ModelRequestData(NamedTuple):
    llm: LLM
    prompt: str
    image_data: list
    stop_token_ids: Optional[list] = None
    chat_template: Optional[str] = None


# ── Model loaders ─────────────────────────────────────────────────────────────

def load_phi3_v(question: str, images: list) -> ModelRequestData:
    """Microsoft Phi-3.5-vision-instruct — strong general-purpose VLM."""
    llm = LLM(
        model="microsoft/Phi-3.5-vision-instruct",
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": len(images)},
        mm_processor_kwargs={"num_crops": 4},
    )
    # Phi-3.5 uses numbered image tags: <|image_1|>, <|image_2|>, ...
    placeholders = "\n".join(
        f"<|image_{i}|>" for i in range(1, len(images) + 1)
    )
    prompt = f"<|user|>\n{placeholders}\n{question}<|end|>\n<|assistant|>\n"
    return ModelRequestData(
        llm=llm,
        prompt=prompt,
        image_data=images,
    )


def load_qwen2_vl(question: str, images: list) -> ModelRequestData:
    """Qwen2-VL-7B-Instruct — excellent at fine-grained visual understanding."""
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        raise ImportError("Run: pip install qwen-vl-utils")

    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    llm = LLM(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": len(images)},
    )

    placeholders = [{"type": "image", "image": img} for img in images]
    messages = [{
        "role": "user",
        "content": [*placeholders, {"type": "text", "text": question}],
    }]

    processor = AutoProcessor.from_pretrained(model_name)
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_data, _ = process_vision_info(messages)

    return ModelRequestData(
        llm=llm,
        prompt=prompt,
        image_data=image_data,
    )


def load_llava_next(question: str, images: list) -> ModelRequestData:
    """LLaVA-v1.6-Mistral-7B — widely used open-source baseline."""
    model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
    llm = LLM(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": len(images)},
    )

    # LLaVA uses a single <image> token per image
    image_tokens = "\n".join("<image>" for _ in images)
    prompt = f"[INST] {image_tokens}\n{question} [/INST]"

    return ModelRequestData(
        llm=llm,
        prompt=prompt,
        image_data=images,
    )


def load_internvl2(question: str, images: list) -> ModelRequestData:
    """InternVL2-8B — strong multilingual and document understanding VLM."""
    model_name = "OpenGVLab/InternVL2-8B"
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": len(images)},
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    # InternVL2 uses <image> tokens with a special chat format
    image_tokens = "\n".join(
        f"Image-{i}: <image>" for i in range(1, len(images) + 1)
    )
    messages = [{"role": "user", "content": f"{image_tokens}\n{question}"}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return ModelRequestData(
        llm=llm,
        prompt=prompt,
        image_data=images,
        stop_token_ids=[tokenizer.eos_token_id],
    )


def load_deepseek_vl2(question: str, images: list) -> ModelRequestData:
    """DeepSeek-VL2-Tiny — efficient DeepSeek vision model."""
    llm = LLM(
        model="deepseek-ai/deepseek-vl2-tiny",
        max_model_len=4096,
        max_num_seqs=2,
        hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
        limit_mm_per_prompt={"image": len(images)},
    )

    placeholder = "".join(
        f"image_{i}:<image>\n" for i in range(1, len(images) + 1)
    )
    prompt = f"<|User|>: {placeholder}{question}\n\n<|Assistant|>:"

    return ModelRequestData(
        llm=llm,
        prompt=prompt,
        image_data=images,
    )


# ── Model registry ────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "phi3_v":      {"loader": load_phi3_v,      "label": "Phi-3.5-Vision"},
    "qwen2_vl":    {"loader": load_qwen2_vl,     "label": "Qwen2-VL-7B"},
    "llava_next":  {"loader": load_llava_next,   "label": "LLaVA-v1.6-Mistral"},
    "internvl2":   {"loader": load_internvl2,    "label": "InternVL2-8B"},
    "deepseek_vl2":{"loader": load_deepseek_vl2, "label": "DeepSeek-VL2-Tiny"},
}


# ── Image loading helpers ─────────────────────────────────────────────────────

def load_images(image_paths_or_urls: list) -> list:
    """Load images from local paths or URLs into PIL.Image objects."""
    images = []
    for src in image_paths_or_urls:
        if src.startswith("http://") or src.startswith("https://"):
            images.append(fetch_image(src))
        else:
            path = Path(src)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {src}")
            images.append(PIL.Image.open(path).convert("RGB"))
    return images


# ── Inference runner ──────────────────────────────────────────────────────────

def run_inference(
    model_key: str,
    question: str,
    images: list,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict:
    """Run inference for a single model. Returns a result dict."""
    logger.info("Running %s on %d image(s) | q: %s", MODEL_REGISTRY[model_key]["label"], len(images), question)

    loader = MODEL_REGISTRY[model_key]["loader"]

    t0 = time.time()
    req = loader(question, images)

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop_token_ids=req.stop_token_ids or [],
    )

    outputs = req.llm.generate(
        {
            "prompt": req.prompt,
            "multi_modal_data": {"image": req.image_data},
        },
        sampling_params=sampling_params,
    )

    elapsed = time.time() - t0
    answer = outputs[0].outputs[0].text.strip()

    logger.info("%s answered in %.1fs: %s", MODEL_REGISTRY[model_key]["label"], elapsed, answer)

    return {
        "model": MODEL_REGISTRY[model_key]["label"],
        "model_key": model_key,
        "question": question,
        "answer": answer,
        "elapsed_seconds": round(elapsed, 2),
    }


# ── Compare mode ──────────────────────────────────────────────────────────────

def run_comparison(
    model_keys: list,
    question: str,
    images: list,
    max_tokens: int,
    temperature: float,
    output_file: Optional[str] = None,
):
    """Run the same question+images through multiple models and compare."""
    results = []
    for key in model_keys:
        try:
            result = run_inference(key, question, images, max_tokens, temperature)
            results.append(result)
        except Exception as e:
            logger.error("%s failed: %s", MODEL_REGISTRY[key]["label"], e)
            results.append({
                "model": MODEL_REGISTRY[key]["label"],
                "model_key": key,
                "error": str(e),
            })

    # ── Save to JSON if requested ──
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved → %s", output_file)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="VLM inference across multiple models on SageMaker"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=list(MODEL_REGISTRY.keys()),
        default="phi3_v",
        help="Which model to run (ignored when --compare is set)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run ALL models and compare their outputs",
    )
    parser.add_argument(
        "--compare-models",
        nargs="+",
        choices=list(MODEL_REGISTRY.keys()),
        default=list(MODEL_REGISTRY.keys()),
        help="Subset of models to compare (default: all)",
    )
    parser.add_argument(
        "--images", "-i",
        nargs="+",
        required=True,
        help="Local image paths or URLs (space-separated)",
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        default=DEFAULT_QUESTION,
        help="Question to ask about the image(s)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Optional JSON file to save results (e.g. results.json)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    images = load_images(args.images)

    if args.compare:
        run_comparison(
            model_keys=args.compare_models,
            question=args.question,
            images=images,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            output_file=args.output,
        )
    else:
        result = run_inference(
            model_key=args.model,
            question=args.question,
            images=images,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            logger.info("Result saved → %s", args.output)


if __name__ == "__main__":
    main()
