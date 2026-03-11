"""
VLM Arena — main orchestration script.

Pipeline:
  1. Load all requested VLMs into memory (once).
  2. For each video:
       a. Extract frames at the configured FPS.
       b. Split frames into overlapping chunks.
       c. Load prompt and events from the video folder.
       d. For each loaded model:
            - Run inference on every chunk → Story list.
            - For each annotated event, judge against overlapping stories → Judgement list.
            - Save VideoResult to results/<model_key>/<video_id>.json.
  3. Aggregate scores across all results and write results/leaderboard.json.
"""

import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from config import settings
from schemas import Event, Story, Timeframe, VideoResult
from utils.inference import load_model, load_images, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE
from utils.judge import judge_event
from utils.preprocessing import extract_frames, get_video_duration, parse_frame_timestamp, make_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

FPS = settings.fps
CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap
MODELS = settings.models
MOCK = settings.mock_inference
MAX_VIDEOS = settings.max_videos
MAX_VIDEO_DURATION = settings.max_video_duration
PASS_THRESHOLD = settings.pass_threshold

RESULTS_DIR = Path(__file__).parent.parent / "results"
VIDEOS_DIR = Path(settings.videos_dir) if Path(settings.videos_dir).is_absolute() else Path(__file__).parent.parent / settings.videos_dir

DEFAULT_PROMPT = "Describe this scene."

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_events(video_dir: Path) -> list[Event]:
    """Parse annotations.json into a list of Event objects."""
    annotations_path = video_dir / "annotations.json"
    if not annotations_path.exists():
        return []
    with open(annotations_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Event.model_validate(e) for e in raw]


def load_prompt(video_dir: Path) -> str:
    prompt_path = video_dir / "prompt.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    logger.warning("prompt.txt not found in %s — using default prompt", video_dir)
    return DEFAULT_PROMPT


def seconds_to_hms(total_seconds: float) -> str:
    h = int(total_seconds // 3600)
    m = int((total_seconds % 3600) // 60)
    s = int(total_seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ── Per-video, per-model processing ──────────────────────────────────────────

def process_video(video_path: Path, model, events: list[Event], prompt: str) -> VideoResult:
    """Run inference + judging for one video × one model pair."""
    video_dir = video_path.parent
    video_id = video_dir.name

    # Frame extraction (reuse if already extracted)
    frames_dir = video_dir / "frames"
    if frames_dir.exists() and any(frames_dir.iterdir()):
        frames = sorted(str(p) for p in frames_dir.glob("*.jpg"))
        logger.info("Reusing %d cached frames from %s", len(frames), frames_dir)
    else:
        frames = extract_frames(video_path, frames_dir, FPS)

    if not frames:
        logger.warning("No frames extracted for %s — skipping", video_id)
        return VideoResult(
            video_id=video_id,
            model_key=model.key,
            model_label=model.label,
            video_duration="00:00:00",
            fps=FPS,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            stories=[],
            judgements=[],
        )

    video_duration = parse_frame_timestamp(frames[-1])
    chunks = make_chunks(frames, CHUNK_SIZE, CHUNK_OVERLAP)

    # ── VLM inference ─────────────────────────────────────────────────────────
    stories: list[Story] = []

    for i, chunk_paths in enumerate(chunks):
        story_id = f"story_{i:03d}"
        images = load_images(chunk_paths)

        try:
            answer, elapsed = model.run(
                question=prompt,
                images=images,
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=DEFAULT_TEMPERATURE,
            )
        except Exception as exc:
            logger.error("[%s] inference failed on chunk %d: %s", model.label, i, exc)
            answer = f"[ERROR] {exc}"
            elapsed = 0.0

        story = Story(
            story_id=story_id,
            timeframe=Timeframe(
                start=parse_frame_timestamp(chunk_paths[0]),
                end=parse_frame_timestamp(chunk_paths[-1]),
            ),
            question=prompt,
            answer=answer,
            frame_count=len(chunk_paths),
            elapsed_seconds=elapsed,
        )
        stories.append(story)
        logger.info("[%s] %s | %s–%s | %.2fs", model.label, story_id,
                    story.timeframe.start, story.timeframe.end, elapsed)

    # ── Judging ───────────────────────────────────────────────────────────────
    judgements = []
    if not events:
        logger.warning("No annotations.json for %s — skipping judge", video_id)
    else:
        for event in events:
            judgement = judge_event(event=event, stories=stories)
            judgements.append(judgement)

    return VideoResult(
        video_id=video_id,
        model_key=model.key,
        model_label=model.label,
        video_duration=video_duration,
        fps=FPS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        stories=stories,
        judgements=judgements,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not MODELS:
        logger.error("No models configured. Set MODELS in .env file.")
        sys.exit(1)

    if CHUNK_OVERLAP >= CHUNK_SIZE:
        logger.error("CHUNK_OVERLAP (%d) must be less than CHUNK_SIZE (%d).", CHUNK_OVERLAP, CHUNK_SIZE)
        sys.exit(1)

    videos = list(VIDEOS_DIR.glob("**/*.mp4"))
    if not videos:
        logger.warning("No .mp4 files found under %s", VIDEOS_DIR)
        sys.exit(0)

    if MAX_VIDEOS > 0:
        videos = videos[:MAX_VIDEOS]
        logger.info("MAX_VIDEOS=%d — processing %d of the available video(s)", MAX_VIDEOS, len(videos))

    if MAX_VIDEO_DURATION > 0:
        filtered = []
        for v in videos:
            try:
                duration = get_video_duration(v)
                if duration > MAX_VIDEO_DURATION:
                    logger.info("Skipping %s — duration %.1fs exceeds MAX_VIDEO_DURATION=%.1fs",
                                v.parent.name, duration, MAX_VIDEO_DURATION)
                else:
                    filtered.append(v)
            except Exception as exc:
                logger.warning("Could not read duration of %s: %s — including anyway", v, exc)
                filtered.append(v)
        videos = filtered

    logger.info("Found %d video(s). Loading %d model(s)%s…",
                len(videos), len(MODELS), " [MOCK]" if MOCK else "")

    # Load all models once before iterating videos
    loaded_models = []
    for key in MODELS:
        try:
            loaded_models.append(load_model(key, chunk_size=CHUNK_SIZE, mock=MOCK))
        except Exception as exc:
            logger.error("Failed to load model '%s': %s", key, exc)

    if not loaded_models:
        logger.error("No models could be loaded. Exiting.")
        sys.exit(1)

    # Process every video × model pair
    all_results: list[VideoResult] = []

    for video_path in tqdm(videos, desc="Videos"):
        video_dir = video_path.parent
        events = load_events(video_dir)
        prompt = load_prompt(video_dir)

        if not events:
            logger.warning("No events found for %s — judging will be skipped", video_dir.name)

        for model in loaded_models:
            logger.info("── %s × %s ──", video_dir.name, model.label)
            result = process_video(video_path, model, events, prompt)

            # Save result: results/<model_key>/<video_id>.json
            out_dir = RESULTS_DIR / model.key
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{video_dir.name}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(result.model_dump_json(indent=2))
            logger.info("Saved → %s", out_path)

            all_results.append(result)

    # ── Leaderboard ───────────────────────────────────────────────────────────
    # Load all previously saved results from disk so the leaderboard always
    # reflects every model that has ever been run, not just the current session.
    # Only include results whose video still exists in the videos directory to
    # avoid stale entries from removed videos.
    current_video_ids = {p.parent.name for p in videos}
    all_result_paths = [p for p in RESULTS_DIR.glob("*/*.json") if p.name != "leaderboard.json"]
    for path in all_result_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                result = VideoResult.model_validate_json(f.read())
            if result.video_id not in current_video_ids:
                logger.debug("Skipping stale result %s (video no longer in videos dir)", path)
                continue
            if result not in all_results:
                all_results.append(result)
        except Exception as exc:
            logger.warning("Could not read result file %s: %s", path, exc)

    scores_by_model: dict[str, list[float]] = {}
    for result in all_results:
        for j in result.judgements:
            if j.score is not None:
                scores_by_model.setdefault(result.model_label, []).append(j.score)

    leaderboard = sorted(
        [
            {
                "model_label": label,
                "model_key": next(r.model_key for r in all_results if r.model_label == label),  # stable — model_key is consistent per label
                "avg_score": round(sum(scores) / len(scores), 4),
                "n_judgements": len(scores),
                "n_passed": sum(1 for s in scores if s >= PASS_THRESHOLD),
                "n_failed": sum(1 for s in scores if s < PASS_THRESHOLD),
                "success_ratio": round(sum(1 for s in scores if s >= PASS_THRESHOLD) / len(scores), 4),
            }
            for label, scores in scores_by_model.items()
        ],
        key=lambda x: x["avg_score"],
        reverse=True,
    )

    for entry in leaderboard:
        logger.info(
            "%-30s avg score: %.4f  pass: %d  fail: %d  success ratio: %.2f%%  (n=%d)",
            entry["model_label"], entry["avg_score"],
            entry["n_passed"], entry["n_failed"],
            entry["success_ratio"] * 100, entry["n_judgements"],
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "leaderboard.json", "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2)
    logger.info("Leaderboard saved → %s", RESULTS_DIR / "leaderboard.json")


if __name__ == "__main__":
    main()
