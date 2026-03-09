"""
VLM inference — load-once, run-many design.

Each loader returns a VLMModel instance.  The model is loaded into GPU memory
once and reused for every chunk of every video, keeping VRAM allocation stable.

MockVLMModel skips GPU loading entirely and is used when MOCK_INFERENCE=true.
"""

import logging
import time
from typing import Callable, Optional

import PIL.Image
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.2


# ── VLM model interface ───────────────────────────────────────────────────────

class VLMModel:
    """Wraps a loaded vLLM instance with a model-specific prompt builder."""

    def __init__(self, key: str, label: str, llm, build_fn: Callable):
        """
        Args:
            llm:      Loaded vLLM LLM instance.
            build_fn: Callable(question, images) → (prompt, image_data, stop_token_ids)
        """
        self.key = key
        self.label = label
        self._llm = llm
        self._build_fn = build_fn

    def run(
        self,
        question: str,
        images: list,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> tuple[str, float]:
        """Run inference on a list of PIL images. Returns (answer, elapsed_seconds)."""
        from vllm import SamplingParams

        t0 = time.time()
        prompt, image_data, stop_token_ids = self._build_fn(question, images)

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids or [],
        )

        outputs = self._llm.generate(
            {"prompt": prompt, "multi_modal_data": {"image": image_data}},
            sampling_params,
        )

        elapsed = round(time.time() - t0, 2)
        answer = outputs[0].outputs[0].text.strip()
        logger.info("%s answered in %.1fs", self.label, elapsed)
        return answer, elapsed


class MockVLMModel:
    """Fake VLM that returns placeholder text — no GPU required."""

    def __init__(self, key: str, label: str):
        self.key = key
        self.label = label

    def run(
        self,
        question: str,
        images: list,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> tuple[str, float]:
        answer = (
            f"[MOCK:{self.label}] Processed {len(images)} frame(s). "
            f"Question: '{question}'. Placeholder description of observed scene activity."
        )
        logger.info("[MOCK] %s returned placeholder answer for %d frame(s)", self.label, len(images))
        return answer, 0.05


# ── Model loaders ─────────────────────────────────────────────────────────────

def load_phi3_v(chunk_size: int) -> VLMModel:
    from vllm import LLM

    llm = LLM(
        model="microsoft/Phi-3.5-vision-instruct",
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": chunk_size},
        enforce_eager=True,  # workaround for vLLM 0.17.0 CUBLAS crash during vision encoder profiling
    )

    def build(question: str, images: list):
        placeholders = "\n".join(f"<|image_{i}|>" for i in range(1, len(images) + 1))
        prompt = f"<|user|>\n{placeholders}\n{question}<|end|>\n<|assistant|>\n"
        return prompt, images, None

    return VLMModel(key="phi3_v", label="Phi-3.5-Vision", llm=llm, build_fn=build)


def load_qwen2_vl(chunk_size: int) -> VLMModel:
    from vllm import LLM
    from transformers import AutoProcessor

    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        raise ImportError("Run: pip install qwen-vl-utils")

    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    llm = LLM(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": chunk_size},
    )
    processor = AutoProcessor.from_pretrained(model_name)

    def build(question: str, images: list):
        placeholders = [{"type": "image", "image": img} for img in images]
        messages = [{"role": "user", "content": [*placeholders, {"type": "text", "text": question}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_data, _ = process_vision_info(messages)
        return prompt, image_data, None

    return VLMModel(key="qwen2_vl", label="Qwen2-VL-7B", llm=llm, build_fn=build)


def load_llava_next(chunk_size: int) -> VLMModel:
    from vllm import LLM

    model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
    llm = LLM(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": chunk_size},
    )

    def build(question: str, images: list):
        image_tokens = "\n".join("<image>" for _ in images)
        prompt = f"[INST] {image_tokens}\n{question} [/INST]"
        return prompt, images, None

    return VLMModel(key="llava_next", label="LLaVA-v1.6-Mistral", llm=llm, build_fn=build)


def load_internvl2(chunk_size: int) -> VLMModel:
    from vllm import LLM
    from transformers import AutoTokenizer

    model_name = "OpenGVLab/InternVL2-8B"
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": chunk_size},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def build(question: str, images: list):
        image_tokens = "\n".join(f"Image-{i}: <image>" for i in range(1, len(images) + 1))
        messages = [{"role": "user", "content": f"{image_tokens}\n{question}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt, images, [tokenizer.eos_token_id]

    return VLMModel(key="internvl2", label="InternVL2-8B", llm=llm, build_fn=build)


def load_qwen35_vl(chunk_size: int) -> VLMModel:
    """Qwen3.5-0.8B — lightweight multimodal model, same vision pipeline as Qwen2-VL."""
    from vllm import LLM
    from transformers import AutoProcessor

    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        raise ImportError("Run: pip install qwen-vl-utils")

    model_name = "Qwen/Qwen3.5-0.8B"
    llm = LLM(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": chunk_size},
        enforce_eager=True,  # workaround for vLLM 0.17.0 CUBLAS crash during vision encoder profiling
    )
    processor = AutoProcessor.from_pretrained(model_name)

    def build(question: str, images: list):
        placeholders = [{"type": "image", "image": img} for img in images]
        messages = [{"role": "user", "content": [*placeholders, {"type": "text", "text": question}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_data, _ = process_vision_info(messages)
        return prompt, image_data, None

    return VLMModel(key="qwen35_vl", label="Qwen3.5-0.8B", llm=llm, build_fn=build)


def load_minicpm_v4(chunk_size: int) -> VLMModel:
    """MiniCPM-V-4 — 4.1B multimodal model (SigLIP2-400M + MiniCPM4-3B)."""
    from vllm import LLM

    model_name = "openbmb/MiniCPM-V-4"
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": chunk_size},
    )

    def build(question: str, images: list):
        placeholders = "\n".join(f"<|image_{i}|>" for i in range(1, len(images) + 1))
        prompt = (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{placeholders}\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return prompt, images, None

    return VLMModel(key="minicpm_v4", label="MiniCPM-V-4", llm=llm, build_fn=build)


def load_deepseek_vl2(chunk_size: int) -> VLMModel:
    from vllm import LLM

    llm = LLM(
        model="deepseek-ai/deepseek-vl2-tiny",
        max_model_len=4096,
        max_num_seqs=2,
        hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
        limit_mm_per_prompt={"image": chunk_size},
    )

    def build(question: str, images: list):
        placeholder = "".join(f"image_{i}:<image>\n" for i in range(1, len(images) + 1))
        prompt = f"<|User|>: {placeholder}{question}\n\n<|Assistant|>:"
        return prompt, images, None

    return VLMModel(key="deepseek_vl2", label="DeepSeek-VL2-Tiny", llm=llm, build_fn=build)


# ── Model registry ────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, dict] = {
    "phi3_v":       {"loader": load_phi3_v,       "label": "Phi-3.5-Vision"},
    "qwen2_vl":     {"loader": load_qwen2_vl,     "label": "Qwen2-VL-7B"},
    "qwen35_vl":    {"loader": load_qwen35_vl,    "label": "Qwen3.5-0.8B"},
    "minicpm_v4":   {"loader": load_minicpm_v4,   "label": "MiniCPM-V-4"},
    "llava_next":   {"loader": load_llava_next,   "label": "LLaVA-v1.6-Mistral"},
    "internvl2":    {"loader": load_internvl2,    "label": "InternVL2-8B"},
    "deepseek_vl2": {"loader": load_deepseek_vl2, "label": "DeepSeek-VL2-Tiny"},
}


# ── Factory ───────────────────────────────────────────────────────────────────

def load_model(model_key: str, chunk_size: int, mock: bool = False) -> VLMModel | MockVLMModel:
    """Load a VLM (or return a mock). Call once per model before processing videos."""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key '{model_key}'. Available: {list(MODEL_REGISTRY)}")

    info = MODEL_REGISTRY[model_key]

    if mock:
        logger.info("Loading MOCK model for %s (no GPU required)", info["label"])
        return MockVLMModel(key=model_key, label=info["label"])

    logger.info("Loading %s into GPU memory…", info["label"])
    return info["loader"](chunk_size)


# ── Image loading helper ──────────────────────────────────────────────────────

def load_images(image_paths: list[str]) -> list[PIL.Image.Image]:
    """Load local JPG/PNG paths into PIL.Image objects (RGB)."""
    images = []
    for src in image_paths:
        path = Path(src)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {src}")
        images.append(PIL.Image.open(path).convert("RGB"))
    return images
