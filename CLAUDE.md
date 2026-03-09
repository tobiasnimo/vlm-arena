# CLAUDE.md — VLM Arena

Project context for Claude Code. Read this at the start of every session.

---

## What this project does

Benchmarks Vision Language Models (VLMs) on video understanding tasks.
Given a video and a set of annotated events (each with a timeframe), it:
1. Extracts frames from the video at a configurable FPS.
2. Groups frames into overlapping chunks and runs each VLM on every chunk → **Story**.
3. For each annotated event, judges the stories that overlap its timeframe via a Groq LLM → **Judgement**.
4. Writes one output file per video-model pair and a global leaderboard.

---

## Key files

| File | Role |
|------|------|
| `src/arena.py` | Entry point. Loads models once, iterates videos × models, saves results. |
| `src/schemas.py` | All Pydantic models: `Timeframe`, `Event`, `Story`, `Judgement`, `VideoResult`. |
| `src/config.py` | `pydantic-settings` config loaded from `config.txt`. |
| `src/utils/inference.py` | `VLMModel` (load-once), `MockVLMModel`, `load_model()` factory, model loaders. |
| `src/utils/judge.py` | `judge_event(event, stories) → Judgement` via Groq. |
| `src/utils/preprocessing.py` | `extract_frames()`, `parse_frame_timestamp()`, `make_chunks()`. |

---

## Data flow

```
videos/<video_id>/
├── video.mp4
├── prompt.txt          # optional; fallback: "Describe this scene."
└── annotations.json    # list of Event objects (event_id, description, timeframe HH:MM:SS)
```

Pipeline per video × model:
```
frames = extract_frames(video, fps)
chunks = make_chunks(frames, CHUNK_SIZE, CHUNK_OVERLAP)   # frame-count-based overlap
stories = [model.run(chunk) for chunk in chunks]          # VLM inference
judgements = [judge_event(event, stories) for event in events]  # LLM judge, per-event overlap
```

Output:
```
results/<model_key>/<video_id>.json    # VideoResult (stories + judgements)
results/leaderboard.json               # avg score per model across all judgements
```

---

## Schemas (src/schemas.py)

```python
Timeframe(start: str, end: str)        # "HH:MM:SS"; has .overlaps(other) method
Event(event_id, description, timeframe)
Story(story_id, timeframe, question, answer, frame_count, elapsed_seconds)
Judgement(event_id, event_description, event_timeframe, story_ids, score, analysis)
VideoResult(video_id, model_key, model_label, video_duration, fps,
            chunk_size, chunk_overlap, stories, judgements)
```

---

## Config keys (config.txt)

| Key | Default | Description |
|-----|---------|-------------|
| `HF_TOKEN` | — | Hugging Face token (required for Phi-3.5, InternVL2) |
| `GROQ_API_KEY` | — | Groq API key (required for the judge) |
| `FPS` | 1 | Frames per second to extract |
| `CHUNK_SIZE` | 5 | Frames per VLM inference chunk |
| `CHUNK_OVERLAP` | 0 | Overlapping frames between chunks (must be < CHUNK_SIZE) |
| `MODELS` | `[]` | List of model keys to run |
| `MOCK_INFERENCE` | false | Skip GPU loading; return placeholder answers |

---

## Inference design

- `VLMModel` wraps a loaded vLLM `LLM` instance + a model-specific `build_fn(question, images) → (prompt, image_data, stop_token_ids)`.
- Models are loaded **once** before iterating videos (not per-chunk).
- `limit_mm_per_prompt` is set to `CHUNK_SIZE` at load time.
- `MockVLMModel` has the same `.run()` interface but returns fake text instantly — safe on machines with no GPU.
- Factory: `load_model(model_key, chunk_size, mock=False)`.

---

## Frame filenames

Frames are written as `frame_HH-MM-SS-mmm.jpg`. Timestamps are parsed with:
```python
parse_frame_timestamp("frame_00-01-23-456.jpg")  # → "00:01:23"
```
Story timeframes are derived from the first and last frame in each chunk.

---

## Important: no GPU on dev machine

**Do not run `arena.py` without `MOCK_INFERENCE=true` on this machine.**
Running real VLM inference here will attempt to load multi-GB models and likely crash.
Always use `MOCK_INFERENCE=true` for local testing and schema/logic validation.

---

## Available model keys

`phi3_v`, `qwen2_vl`, `llava_next`, `internvl2`, `deepseek_vl2`
