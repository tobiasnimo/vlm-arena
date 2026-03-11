import json
import sys
import PIL.Image
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from schemas import Event, Story, Timeframe


# ── Shared timeframe fixtures ─────────────────────────────────────────────────

@pytest.fixture
def tf_0_10():
    return Timeframe(start="00:00:00", end="00:00:10")

@pytest.fixture
def tf_5_15():
    return Timeframe(start="00:00:05", end="00:00:15")

@pytest.fixture
def tf_20_30():
    return Timeframe(start="00:00:20", end="00:00:30")


# ── Sample event ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_event():
    return Event(
        event_id="evt_001",
        description="Person enters the room",
        timeframe=Timeframe(start="00:00:03", end="00:00:12"),
    )


# ── Sample stories ────────────────────────────────────────────────────────────

@pytest.fixture
def sample_stories():
    """Three stories: first two overlap the sample_event, third does not."""
    return [
        Story(
            story_id="story_000",
            timeframe=Timeframe(start="00:00:00", end="00:00:04"),
            question="Describe this scene.",
            answer="An empty corridor.",
            frame_count=5,
            elapsed_seconds=1.2,
        ),
        Story(
            story_id="story_001",
            timeframe=Timeframe(start="00:00:05", end="00:00:09"),
            question="Describe this scene.",
            answer="A person walks through the door.",
            frame_count=5,
            elapsed_seconds=1.1,
        ),
        Story(
            story_id="story_002",
            timeframe=Timeframe(start="00:00:20", end="00:00:24"),
            question="Describe this scene.",
            answer="The room is empty again.",
            frame_count=5,
            elapsed_seconds=1.0,
        ),
    ]


# ── Temp video directory with fake frames ─────────────────────────────────────

@pytest.fixture
def fake_video_dir(tmp_path):
    """
    A video folder with 12 tiny JPEG frames and annotations.json.
    Frames cover 00:00:00 → 00:00:11 at 1 fps.
    No real video.mp4 is needed — frames/ already exist so extract_frames is skipped.
    """
    video_dir = tmp_path / "test_video"
    frames_dir = video_dir / "frames"
    frames_dir.mkdir(parents=True)

    for i in range(12):
        img = PIL.Image.new("RGB", (8, 8), color=(i * 20 % 255, 100, 150))
        img.save(frames_dir / f"frame_00-00-{i:02d}-000.jpg", format="JPEG")

    events = [
        {
            "event_id": "evt_001",
            "description": "Person enters room",
            "timeframe": {"start": "00:00:00", "end": "00:00:05"},
        },
        {
            "event_id": "evt_002",
            "description": "Person picks up object",
            "timeframe": {"start": "00:00:06", "end": "00:00:11"},
        },
    ]
    (video_dir / "annotations.json").write_text(json.dumps(events))
    (video_dir / "prompt.txt").write_text("Describe the scene.")

    # arena.py expects a video_path; the parent dir is what matters
    video_path = video_dir / "video.mp4"
    video_path.touch()

    return video_path
