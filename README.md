# VLM Arena

Evaluate multiple Vision Language Models (VLMs) on video understanding tasks. The pipeline extracts frames from videos, runs them through several VLMs in parallel, and uses an LLM-as-a-judge to score each model's descriptions against ground-truth annotations.

---

## How it works

```
videos/<name>/
├── video.mp4          # input video
├── prompt.txt         # question to ask the VLMs  (optional — falls back to "Describe this scene.")
└── annotations.txt    # ground-truth event notes   (optional — skips judge if missing)
```

1. **Frame extraction** — splits each video into JPG frames at the configured FPS
2. **Chunked inference** — groups frames into chunks of 5 and runs each VLM on every chunk
3. **Judge** — aggregates all per-chunk answers into a chronological narrative, then asks a Groq-hosted LLM to score each model against the annotations
4. **Results** — saved to `results/` as JSON

---

## Project structure

```
src/
├── arena.py              # entry point — orchestrates the full pipeline
├── config.py             # settings loaded from config.txt
└── utils/
    ├── inference.py      # vLLM model loaders and inference runner
    ├── judge.py          # LLM-as-a-judge via Groq
    └── preprocessing.py  # frame extraction via OpenCV
videos/                   # place your video folders here
results/                  # output JSON files (created automatically)
config.txt                # configuration (see below)
```

---

## Setup

### 1. Instance requirements

VLMs require a GPU. Recommended SageMaker instances:

| Instance          | GPU         | VRAM   | Fits                                      |
|-------------------|-------------|--------|-------------------------------------------|
| `ml.g5.2xlarge`   | A10G (1×)   | 24 GB  | Phi-3.5-V, DeepSeek-VL2-Tiny             |
| `ml.g5.12xlarge`  | A10G (4×)   | 96 GB  | Qwen2-VL-7B, LLaVA-1.6, InternVL2-8B    |
| `ml.p4d.24xlarge` | A100 (8×)   | 320 GB | All models, large batches                 |

### 2. Install dependencies

```bash
python3 -m venv vllm-env
source vllm-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure

Create `config.txt` in the project root:

```ini
# Hugging Face token (required for gated models like Phi-3.5)
HF_TOKEN=hf_your_token_here

# Groq API key (required for the judge)
GROQ_API_KEY=gsk_your_key_here

# Frames per second to extract from each video
FPS=1

# VLM model keys to run (space-separated)
MODELS=["phi3_v", "qwen2_vl"]
```

---

## Running

```bash
cd src
python arena.py
```

The script discovers all `*.mp4` files under `videos/`, processes each one, and writes results to `results/`.

---

## Input format

Each video lives in its own subdirectory under `videos/`:

```
videos/
└── my_video/
    ├── video.mp4
    ├── prompt.txt        # e.g. "What actions are performed in this scene?"
    └── annotations.txt   # e.g. "Person picks up object. Places it on table."
```

- If `prompt.txt` is missing, the pipeline falls back to `"Describe this scene."` and logs a warning.
- If `annotations.txt` is missing, the judge step is skipped and a warning is logged.

---

## Output format

**Per-story inference** (`results/<name>_story000.json`):
```json
[
  {
    "model": "Phi-3.5-Vision",
    "model_key": "phi3_v",
    "question": "What actions are performed in this scene?",
    "answer": "A person reaches for ...",
    "elapsed_seconds": 12.4
  }
]
```

**Judge scores** (`results/<name>_judgment.json`):
```json
[
  {
    "model": "Phi-3.5-Vision",
    "score": 0.82,
    "analysis": "The model correctly identified the main action but missed ..."
  }
]
```

---

## Available models

| Key              | Model                    | VRAM  |
|------------------|--------------------------|-------|
| `phi3_v`         | Phi-3.5-vision-instruct  | ~10 GB |
| `qwen2_vl`       | Qwen2-VL-7B-Instruct     | ~18 GB |
| `llava_next`     | LLaVA-v1.6-Mistral-7B   | ~16 GB |
| `internvl2`      | InternVL2-8B             | ~18 GB |
| `deepseek_vl2`   | DeepSeek-VL2-Tiny        | ~8 GB  |

---

## Troubleshooting

**CUDA out of memory**
→ Reduce `max_model_len` in the relevant loader in `inference.py`, or use a smaller model (`deepseek_vl2` is the lightest).

**`trust_remote_code` errors**
→ Set automatically for Phi-3.5 and InternVL2. Accept the model license on Hugging Face first.

**`qwen_vl_utils` not found**
→ `pip install qwen-vl-utils`

**vLLM hangs on startup**
→ Normal — vLLM compiles CUDA kernels on first run. Wait 60–90 seconds.

**Model downloads are slow**
→ Models are cached in `~/.cache/huggingface/hub`. On SageMaker, preload to an EFS mount to avoid re-downloading across sessions.
