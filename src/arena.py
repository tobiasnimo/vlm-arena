import json
import logging
from pathlib import Path

from tqdm import tqdm

from utils.inference import run_comparison, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE
from utils.preprocessing import extract_frames
from utils.judge import judge_descriptions
from config import settings

logger = logging.getLogger(__name__)

# --- Config ---

FPS = settings.fps
MODELS = settings.models

# --- Evaluation pipeline ---

def evaluate_vlms(video_path: Path) -> list[dict]:

    # --- Paths ---
    frames_path = video_path.parent / "frames"
    prompt_path = video_path.parent / "prompt.txt"
    annotations_path = video_path.parent / "annotations.txt"

    # --- Video pre-processing ---
    if frames_path.exists() and any(frames_path.iterdir()):
        frames = sorted(str(p) for p in frames_path.glob("*.jpg"))
    else:
        frames = extract_frames(video_path, frames_path, FPS)

    def chunk_list(lst, m):
        return [lst[i:i+m] for i in range(0, len(lst), m)]

    stories = chunk_list(frames, 5)

    # --- VLM inference ---
    DEFAULT_PROMPT = "Describe this scene."

    if prompt_path.exists():
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read()
    else:
        prompt = DEFAULT_PROMPT
        logger.warning("prompt.txt not found for %s — using default prompt: %r", video_path.parent, DEFAULT_PROMPT)

    Path("results").mkdir(exist_ok=True)

    # Collect results from every story so the judge sees the full video
    all_story_results = []
    for i, story in enumerate(stories):
        RESULTS_PATH = Path("results") / f"{video_path.parent.name}_story{i:03d}.json"

        story_results = run_comparison(
                model_keys=MODELS,
                question=prompt,
                images=story,
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=DEFAULT_TEMPERATURE,
                output_file=RESULTS_PATH,
        )
        all_story_results.append(story_results)

    # --- Judge ---
    if not annotations_path.exists():
        logger.warning("annotations.txt not found for %s — skipping judge", video_path.parent)
        return []

    with open(annotations_path, "r", encoding="utf-8") as f:
        annotations = f.read()

    # Aggregate all per-story answers by model (chronological order)
    model_descriptions: dict[str, list[str]] = {}
    for story_results in all_story_results:
        for result in story_results:
            if "error" in result:
                continue
            model_descriptions.setdefault(result["model"], []).append(result["answer"])

    # Combine all story descriptions into a single chronological narrative and pass it to the judge
    judgments = []
    for model_label, story_descriptions in model_descriptions.items():
        descriptions = "\n\n".join(
            f"[Chunk {i + 1}] {desc}" for i, desc in enumerate(story_descriptions)
        )
        judgment = judge_descriptions(annotations=annotations, descriptions=descriptions)
        judgment["model"] = model_label
        logger.info("[%s] Score: %s — %s", model_label, judgment["score"], judgment["analysis"])
        judgments.append(judgment)

    JUDGE_PATH = Path("results") / f"{video_path.parent.name}_judgment.json"
    with open(JUDGE_PATH, "w") as f:
        json.dump(judgments, f, indent=2)

    return judgments

# --- Run ---

VIDEOS_DIR = Path(__file__).parent.parent / "videos"
videos = list(VIDEOS_DIR.glob("**/*.mp4"))

all_judgments: list[dict] = []
for video in tqdm(videos, desc="Running VLM Arena"):
    logger.info("Processing: %s", video)
    all_judgments.extend(evaluate_vlms(video))

# --- Leaderboard ---
scores_by_model: dict[str, list[float]] = {}
for j in all_judgments:
    if j["score"] is not None:
        scores_by_model.setdefault(j["model"], []).append(j["score"])

leaderboard = sorted(
    [
        {"model": model, "avg_score": round(sum(scores) / len(scores), 4), "n_videos": len(scores)}
        for model, scores in scores_by_model.items()
    ],
    key=lambda x: x["avg_score"],
    reverse=True,
)

for entry in leaderboard:
    logger.info("%-30s avg score: %.4f  (n=%d)", entry["model"], entry["avg_score"], entry["n_videos"])

Path("results").mkdir(exist_ok=True)
with open(Path("results") / "leaderboard.json", "w") as f:
    json.dump(leaderboard, f, indent=2)