# VLM Arena

Evaluate multiple Vision Language Models (VLMs) on video understanding tasks. The pipeline extracts frames from videos, runs them through several VLMs, and uses an LLM-as-a-judge to score each model's descriptions against per-event ground-truth annotations.

---

## How it works

```
videos/<name>/
├── video.mp4          # input video
├── prompt.txt         # question to ask the VLMs  (optional — falls back to "Describe this scene.")
└── annotations.json   # list of annotated events with timeframes  (optional — skips judge if missing)
```

1. **Frame extraction** — splits each video into JPG frames at the configured FPS
2. **Chunked inference** — groups frames into overlapping chunks and runs each VLM on every chunk → one **Story** per chunk
3. **Per-event judging** — for each annotated event, the judge collects all stories whose timeframe overlaps the event's timeframe and asks a Groq-hosted LLM to score them → one **Judgement** per event
4. **Results** — one JSON file per video-model pair, saved to `results/<model_key>/<video_id>.json`
5. **Leaderboard** — aggregated average scores across all videos and events, saved to `results/leaderboard.json`

---

## Project structure

```
src/
├── arena.py              # entry point — orchestrates the full pipeline
├── config.py             # settings loaded from config.txt
├── schemas.py            # Pydantic models: Timeframe, Event, Story, Judgement, VideoResult
└── utils/
    ├── inference.py      # VLMModel class, model loaders, MockVLMModel
    ├── judge.py          # LLM-as-a-judge via Groq (per-event)
    └── preprocessing.py  # frame extraction, timestamp parsing, chunking
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
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure

Create `.env` in the project root (see `example.env` for all options):

```ini
HF_TOKEN=hf_your_token_here
GROQ_API_KEY=gsk_your_key_here

FPS=1
CHUNK_SIZE=5
CHUNK_OVERLAP=0

MODELS=["phi3_v", "qwen2_vl"]

# Optional
MAX_VIDEOS=10        # process only the first N videos (0 = no limit)
PASS_THRESHOLD=0.6   # judgements >= this score are "pass"; below are "fail"
```

---

## Running

```bash
python src/arena.py
```

The script discovers all `*.mp4` files under `videos/`, loads all configured models once, processes every video × model pair, and writes results to `results/`.

---

## Input format

Each video lives in its own subdirectory under `videos/`:

```
videos/
└── my_video/
    ├── video.mp4
    ├── prompt.txt         # e.g. "What actions are performed in this scene?"
    └── annotations.json   # list of events with timeframes (see below)
```

- If `prompt.txt` is missing, the pipeline falls back to `"Describe this scene."` and logs a warning.
- If `annotations.json` is missing, the judge step is skipped and a warning is logged.

### annotations.json format

A JSON array of event objects. Each event has an `event_id`, a `description`, and a `timeframe` (`"HH:MM:SS"`):

```json
[
  {
    "event_id": "evt_001",
    "description": "Person enters the room",
    "timeframe": {
      "start": "00:00:00",
      "end": "00:00:08"
    }
  },
  {
    "event_id": "evt_002",
    "description": "Person picks up the object from the table",
    "timeframe": {
      "start": "00:00:10",
      "end": "00:00:20"
    }
  }
]
```

---

## Output format

**Per video-model pair** (`results/<model_key>/<video_id>.json`):

```json
{
  "video_id": "my_video",
  "model_key": "phi3_v",
  "model_label": "Phi-3.5-Vision",
  "video_duration": "00:01:30",
  "fps": 1,
  "chunk_size": 5,
  "chunk_overlap": 0,
  "stories": [
    {
      "story_id": "story_000",
      "timeframe": { "start": "00:00:00", "end": "00:00:04" },
      "question": "What actions are performed in this scene?",
      "answer": "A person reaches for ...",
      "frame_count": 5,
      "elapsed_seconds": 12.4
    }
  ],
  "judgements": [
    {
      "event_id": "evt_001",
      "event_description": "Person enters the room",
      "event_timeframe": { "start": "00:00:00", "end": "00:00:08" },
      "story_ids": ["story_000", "story_001"],
      "score": 0.82,
      "analysis": "The model correctly identified the entry but missed ..."
    }
  ]
}
```

**Leaderboard** (`results/leaderboard.json`):

```json
[
  {
    "model_label": "Phi-3.5-Vision",
    "model_key": "phi3_v",
    "avg_score": 0.78,
    "n_judgements": 12,
    "n_passed": 9,
    "n_failed": 3,
    "success_ratio": 0.75
  }
]
```

`n_passed` / `n_failed` count judgements at or above / below `PASS_THRESHOLD`. `success_ratio` is `n_passed / n_judgements`.

---

## Available models

| Key              | Model                    | VRAM   |
|------------------|--------------------------|--------|
| `phi3_v`         | Phi-3.5-vision-instruct  | ~10 GB |
| `qwen2_vl`       | Qwen2-VL-7B-Instruct     | ~18 GB |
| `qwen35_vl`      | Qwen3.5-0.8B             | ~2 GB  |
| `minicpm_v4`     | MiniCPM-V-4              | ~10 GB |
| `llava_next`     | LLaVA-v1.6-Mistral-7B   | ~16 GB |
| `internvl2`      | InternVL2-8B             | ~18 GB |
| `deepseek_vl2`   | DeepSeek-VL2-Tiny        | ~8 GB  |

### Adding a new model

1. **Write a loader** in `src/utils/inference.py`. The loader must accept `chunk_size: int`, create a vLLM `LLM` instance, define a `build(question, images) → (prompt, image_data, stop_token_ids)` function, and return a `VLMModel`:

```python
def load_my_model(chunk_size: int) -> VLMModel:
    from vllm import LLM

    llm = LLM(
        model="org/my-model",
        trust_remote_code=True,   # if required
        max_model_len=4096,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": chunk_size},
    )

    def build(question: str, images: list):
        # format prompt with image placeholders for this model's chat template
        prompt = f"<|user|>\n<image>\n{question}<|end|>\n<|assistant|>\n"
        return prompt, images, None  # (prompt, image_data, stop_token_ids)

    return VLMModel(key="my_model", label="My Model Name", llm=llm, build_fn=build)
```

2. **Register it** by adding an entry to `MODEL_REGISTRY` in the same file:

```python
MODEL_REGISTRY: dict[str, dict] = {
    ...
    "my_model": {"loader": load_my_model, "label": "My Model Name"},
}
```

3. **Enable it** by adding the key to `MODELS` in your `.env`:

```ini
MODELS=["my_model"]
```

The prompt format varies by model family — check the model card or vLLM's [supported models](https://docs.vllm.ai/en/stable/models/supported_models/) page for the correct image tokens and chat template.

---

## Testing without a GPU

Set `MOCK_INFERENCE=true` in `.env`. Models will not be loaded; a placeholder answer is returned for every chunk so the full pipeline (chunking, judging, output writing) can be exercised locally.

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
