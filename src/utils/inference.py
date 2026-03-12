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


class TransformersVLMModel:
    """VLM backed by HuggingFace Transformers (for models without native vLLM support)."""

    def __init__(self, key: str, label: str, run_fn: Callable):
        self.key = key
        self.label = label
        self._run_fn = run_fn

    def run(
        self,
        question: str,
        images: list,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> tuple[str, float]:
        t0 = time.time()
        answer = self._run_fn(question, images, max_tokens, temperature)
        elapsed = round(time.time() - t0, 2)
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
        enforce_eager=True,
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
        enforce_eager=True,
    )
    processor = AutoProcessor.from_pretrained(model_name)

    def build(question: str, images: list):
        placeholders = [{"type": "image", "image": img} for img in images]
        messages = [{"role": "user", "content": [*placeholders, {"type": "text", "text": question}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_data, _ = process_vision_info(messages)
        return prompt, image_data, None

    return VLMModel(key="qwen35_vl", label="Qwen3.5-0.8B", llm=llm, build_fn=build)


def load_qwen35_vl_2b(chunk_size: int) -> VLMModel:
    """Qwen3.5-2B — same pipeline as Qwen3.5-0.8B, larger capacity."""
    from vllm import LLM
    from transformers import AutoProcessor

    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        raise ImportError("Run: pip install qwen-vl-utils")

    model_name = "Qwen/Qwen3.5-2B"
    llm = LLM(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": chunk_size},
        enforce_eager=True,
    )
    processor = AutoProcessor.from_pretrained(model_name)

    def build(question: str, images: list):
        placeholders = [{"type": "image", "image": img} for img in images]
        messages = [{"role": "user", "content": [*placeholders, {"type": "text", "text": question}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_data, _ = process_vision_info(messages)
        return prompt, image_data, None

    return VLMModel(key="qwen35_vl_2b", label="Qwen3.5-2B", llm=llm, build_fn=build)


def load_qwen35_vl_4b(chunk_size: int) -> VLMModel:
    """Qwen3.5-4B — same pipeline as Qwen3.5-0.8B, larger capacity."""
    from vllm import LLM
    from transformers import AutoProcessor

    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        raise ImportError("Run: pip install qwen-vl-utils")

    model_name = "Qwen/Qwen3.5-4B"
    llm = LLM(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": chunk_size},
        enforce_eager=True,
    )
    processor = AutoProcessor.from_pretrained(model_name)

    def build(question: str, images: list):
        placeholders = [{"type": "image", "image": img} for img in images]
        messages = [{"role": "user", "content": [*placeholders, {"type": "text", "text": question}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_data, _ = process_vision_info(messages)
        return prompt, image_data, None

    return VLMModel(key="qwen35_vl_4b", label="Qwen3.5-4B", llm=llm, build_fn=build)


def load_qwen35_vl_9b(chunk_size: int) -> VLMModel:
    """Qwen3.5-9B — same pipeline as Qwen3.5-0.8B, larger capacity."""
    from vllm import LLM
    from transformers import AutoProcessor

    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        raise ImportError("Run: pip install qwen-vl-utils")

    model_name = "Qwen/Qwen3.5-9B"
    llm = LLM(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": chunk_size},
        enforce_eager=True,
    )
    processor = AutoProcessor.from_pretrained(model_name)

    def build(question: str, images: list):
        placeholders = [{"type": "image", "image": img} for img in images]
        messages = [{"role": "user", "content": [*placeholders, {"type": "text", "text": question}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_data, _ = process_vision_info(messages)
        return image_data, None

    return VLMModel(key="qwen35_vl_9b", label="Qwen3.5-9B", llm=llm, build_fn=build)


def load_minicpm_v4(chunk_size: int) -> VLMModel:
    """MiniCPM-V-4 — 4.1B multimodal model (SigLIP2-400M + MiniCPM4-3B)."""
    from vllm import LLM
    from transformers import AutoTokenizer

    model_name = "openbmb/MiniCPM-V-4"
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": chunk_size},
        enforce_eager=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in stop_tokens]

    def build(question: str, images: list):
        placeholders = "".join("(<image>./</image>)\n" for _ in images)
        messages = [{"role": "user", "content": f"{placeholders}{question}"}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt, images, stop_token_ids

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


def _load_fastvlm(model_name: str, key: str, label: str) -> TransformersVLMModel:
    """Shared loader for all FastVLM variants (0.5B, 1.5B, 7B).

    FastVLM uses a custom LLaVA-based architecture that requires manual image
    token handling: each <image> in the prompt is replaced by token index -200,
    and pixel values are prepared via the model's own vision tower processor.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    IMAGE_TOKEN_INDEX = -200

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    def run_fn(question: str, images: list, max_tokens: int, temperature: float) -> str:
        # One <image> placeholder per frame
        image_placeholders = "\n".join("<image>" for _ in images)
        messages = [{"role": "user", "content": f"{image_placeholders}\n{question}"}]
        rendered = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        # Split at each <image> and insert IMAGE_TOKEN_INDEX (-200)
        parts = rendered.split("<image>")
        input_ids_parts = []
        for i, part in enumerate(parts):
            ids = tok(part, return_tensors="pt", add_special_tokens=False).input_ids
            input_ids_parts.append(ids)
            if i < len(parts) - 1:
                input_ids_parts.append(torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=ids.dtype))
        input_ids = torch.cat(input_ids_parts, dim=1).to(model.device)

        # Process all frames via the vision tower's image processor
        px = model.get_vision_tower().image_processor(
            images=images, return_tensors="pt"
        )["pixel_values"].to(model.device, dtype=model.dtype)

        with torch.no_grad():
            output_ids = model.generate(
                inputs=input_ids,
                attention_mask=torch.ones_like(input_ids),
                images=px,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
            )
        generated = output_ids[:, input_ids.shape[1]:]
        return tok.decode(generated[0], skip_special_tokens=True).strip()

    return TransformersVLMModel(key=key, label=label, run_fn=run_fn)


def load_fastvlm_0b5(chunk_size: int) -> TransformersVLMModel:
    """FastVLM-0.5B — smallest FastVLM variant, Qwen2-0.5B backbone (~1 GB VRAM)."""
    return _load_fastvlm("apple/FastVLM-0.5B", "fastvlm_0b5", "FastVLM-0.5B")


def load_fastvlm(chunk_size: int) -> TransformersVLMModel:
    """FastVLM-1.5B — Apple's efficient VLM with FastViTHD vision encoder (~3 GB VRAM)."""
    return _load_fastvlm("apple/FastVLM-1.5B", "fastvlm", "FastVLM-1.5B")


def load_fastvlm_7b(chunk_size: int) -> TransformersVLMModel:
    """FastVLM-7B — largest FastVLM variant, Qwen2-7B backbone (~16 GB VRAM)."""
    return _load_fastvlm("apple/FastVLM-7B", "fastvlm_7b", "FastVLM-7B")


def load_smolvlm2(chunk_size: int) -> TransformersVLMModel:
    """SmolVLM2-2.2B-Instruct — HuggingFace compact VLM (~5 GB VRAM)."""
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq

    model_name = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    def run_fn(question: str, images: list, max_tokens: int, temperature: float) -> str:
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": question}]
        messages = [{"role": "user", "content": content}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        device = next(model.parameters()).device
        inputs = processor(text=prompt, images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
            )
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return processor.decode(generated[0], skip_special_tokens=True).strip()

    return TransformersVLMModel(key="smolvlm2", label="SmolVLM2-2.2B", run_fn=run_fn)


def load_florence2(chunk_size: int) -> TransformersVLMModel:
    """Florence-2-large-ft — Microsoft's seq2seq VLM (~2 GB VRAM).

    NOTE: Florence-2 is task-token-driven, not chat-based. It always uses the
    <DETAILED_CAPTION> task token regardless of the user prompt. Each frame in
    the chunk is captioned independently and the results are joined.
    """
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM

    model_name = "microsoft/Florence-2-large-ft"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    ).eval()

    TASK = "<DETAILED_CAPTION>"

    def run_fn(question: str, images: list, max_tokens: int, temperature: float) -> str:
        device = next(model.parameters()).device
        descriptions = []
        for img in images:
            inputs = processor(text=TASK, images=img, return_tensors="pt").to(device)
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    num_beams=3,
                )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(
                generated_text, task=TASK, image_size=(img.width, img.height)
            )
            descriptions.append(parsed.get(TASK, ""))
        return " | ".join(descriptions)

    return TransformersVLMModel(key="florence2", label="Florence-2-large", run_fn=run_fn)


def load_lfm25_vl(chunk_size: int) -> TransformersVLMModel:
    """LFM2.5-VL-1.6B — Liquid AI's 1.6B VLM (SigLIP2 + LFM2.5-1.2B, ~4 GB VRAM).

    Requires transformers>=5.1. No native vLLM support.
    """
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    model_name = "LiquidAI/LFM2.5-VL-1.6B"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    def run_fn(question: str, images: list, max_tokens: int, temperature: float) -> str:
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": question}]
        messages = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
            )
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return processor.decode(generated[0], skip_special_tokens=True).strip()

    return TransformersVLMModel(key="lfm25_vl", label="LFM2.5-VL-1.6B", run_fn=run_fn)


def load_glm46v_flash(chunk_size: int) -> VLMModel:
    """GLM-4.6V-Flash — 9B multimodal model (MIT license, 128K context)."""
    from vllm import LLM
    from transformers import AutoProcessor

    model_name = "zai-org/GLM-4.6V-Flash"
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": chunk_size},
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    def build(question: str, images: list):
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt, images, None

    return VLMModel(key="glm46v_flash", label="GLM-4.6V-Flash", llm=llm, build_fn=build)


def load_step3_vl(chunk_size: int) -> VLMModel:
    """STEP3-VL-10B — 10B multimodal model (PE-lang 1.8B + Qwen3-8B)."""
    from vllm import LLM
    from transformers import AutoProcessor

    model_name = "stepfun-ai/Step3-VL-10B"
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": chunk_size},
        hf_overrides={"vision_config": {"enable_patch": False}},
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    def build(question: str, images: list):
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt, images, None

    return VLMModel(key="step3_vl", label="STEP3-VL-10B", llm=llm, build_fn=build)


def load_minicpm_v45(chunk_size: int) -> VLMModel:
    """MiniCPM-V-4.5 — 8.7B multimodal model (SigLIP2-400M + Qwen3-8B)."""
    from vllm import LLM
    from transformers import AutoTokenizer

    model_name = "openbmb/MiniCPM-V-4_5"
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": chunk_size},
        enforce_eager=True,
        disable_mm_preprocessor_cache=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in stop_tokens]

    def build(question: str, images: list):
        placeholders = "".join("(<image>./</image>)\n" for _ in images)
        messages = [{"role": "user", "content": f"{placeholders}{question}"}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt, images, stop_token_ids

    return VLMModel(key="minicpm_v45", label="MiniCPM-V-4.5", llm=llm, build_fn=build)


def load_gemma3(chunk_size: int) -> VLMModel:
    """Gemma-3-4B-IT — Google's 4B multimodal model."""
    from vllm import LLM
    from transformers import AutoProcessor

    model_name = "google/gemma-3-4b-it"
    llm = LLM(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": chunk_size},
    )
    processor = AutoProcessor.from_pretrained(model_name)

    def build(question: str, images: list):
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt, images, [1, 106]

    return VLMModel(key="gemma3", label="Gemma-3-4B-IT", llm=llm, build_fn=build)


# ── Model registry ────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, dict] = {
    "phi3_v":       {"loader": load_phi3_v,       "label": "Phi-3.5-Vision"},
    "qwen2_vl":     {"loader": load_qwen2_vl,     "label": "Qwen2-VL-7B"},
    "qwen35_vl":    {"loader": load_qwen35_vl,    "label": "Qwen3.5-0.8B"},
    "qwen35_vl_2b": {"loader": load_qwen35_vl_2b, "label": "Qwen3.5-2B"},
    "qwen35_vl_4b": {"loader": load_qwen35_vl_4b, "label": "Qwen3.5-4B"},
    "qwen35_vl_9b": {"loader": load_qwen35_vl_9b, "label": "Qwen3.5-9B"},
    "minicpm_v4":   {"loader": load_minicpm_v4,   "label": "MiniCPM-V-4"},
    "llava_next":   {"loader": load_llava_next,   "label": "LLaVA-v1.6-Mistral"},
    "internvl2":    {"loader": load_internvl2,    "label": "InternVL2-8B"},
    "deepseek_vl2": {"loader": load_deepseek_vl2, "label": "DeepSeek-VL2-Tiny"},
    "fastvlm_0b5":  {"loader": load_fastvlm_0b5,  "label": "FastVLM-0.5B"},
    "fastvlm":      {"loader": load_fastvlm,      "label": "FastVLM-1.5B"},
    "fastvlm_7b":   {"loader": load_fastvlm_7b,   "label": "FastVLM-7B"},
    "smolvlm2":     {"loader": load_smolvlm2,     "label": "SmolVLM2-2.2B"},
    "florence2":    {"loader": load_florence2,    "label": "Florence-2-large"},
    "lfm25_vl":     {"loader": load_lfm25_vl,    "label": "LFM2.5-VL-1.6B"},
    "glm46v_flash": {"loader": load_glm46v_flash, "label": "GLM-4.6V-Flash"},
    "step3_vl":     {"loader": load_step3_vl,     "label": "STEP3-VL-10B"},
    "minicpm_v45":  {"loader": load_minicpm_v45,  "label": "MiniCPM-V-4.5"},
    "gemma3":       {"loader": load_gemma3,        "label": "Gemma-3-4B-IT"},
}


# ── Factory ───────────────────────────────────────────────────────────────────

def load_model(model_key: str, chunk_size: int, mock: bool = False) -> VLMModel | TransformersVLMModel | MockVLMModel:
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
