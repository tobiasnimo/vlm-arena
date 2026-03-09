import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from schemas import Event, Judgement, Story, Timeframe
from utils.inference import MockVLMModel
import arena


# ── load_events ───────────────────────────────────────────────────────────────

class TestLoadEvents:
    def test_valid_annotations_json(self, tmp_path):
        data = [
            {"event_id": "evt_001", "description": "A happens",
             "timeframe": {"start": "00:00:00", "end": "00:00:10"}},
            {"event_id": "evt_002", "description": "B happens",
             "timeframe": {"start": "00:00:10", "end": "00:00:20"}},
        ]
        (tmp_path / "annotations.json").write_text(json.dumps(data))
        events = arena.load_events(tmp_path)
        assert len(events) == 2
        assert events[0].event_id == "evt_001"
        assert events[1].timeframe.start == "00:00:10"

    def test_missing_file_returns_empty(self, tmp_path):
        events = arena.load_events(tmp_path)
        assert events == []

    def test_empty_array_returns_empty(self, tmp_path):
        (tmp_path / "annotations.json").write_text("[]")
        assert arena.load_events(tmp_path) == []


# ── load_prompt ───────────────────────────────────────────────────────────────

class TestLoadPrompt:
    def test_reads_prompt_txt(self, tmp_path):
        (tmp_path / "prompt.txt").write_text("What is happening?")
        assert arena.load_prompt(tmp_path) == "What is happening?"

    def test_missing_returns_default(self, tmp_path):
        prompt = arena.load_prompt(tmp_path)
        assert prompt == "Describe this scene."

    def test_whitespace_stripped(self, tmp_path):
        (tmp_path / "prompt.txt").write_text("  Describe carefully.  \n")
        assert arena.load_prompt(tmp_path) == "Describe carefully."


# ── seconds_to_hms ────────────────────────────────────────────────────────────

class TestSecondsToHms:
    def test_zero(self):
        assert arena.seconds_to_hms(0) == "00:00:00"

    def test_seconds(self):
        assert arena.seconds_to_hms(45) == "00:00:45"

    def test_minutes(self):
        assert arena.seconds_to_hms(90) == "00:01:30"

    def test_hours(self):
        assert arena.seconds_to_hms(3661) == "01:01:01"


# ── process_video ─────────────────────────────────────────────────────────────

class TestProcessVideo:
    def test_produces_video_result(self, fake_video_dir):
        model = MockVLMModel(key="phi3_v", label="Phi-3.5-Vision")
        events = arena.load_events(fake_video_dir.parent)
        prompt = arena.load_prompt(fake_video_dir.parent)

        mock_judgement = Judgement(
            event_id="evt_001",
            event_description="Person enters room",
            event_timeframe=Timeframe(start="00:00:00", end="00:00:05"),
            story_ids=["story_000"],
            score=0.8,
            analysis="Good.",
        )

        with patch("arena.judge_event", return_value=mock_judgement):
            result = arena.process_video(fake_video_dir, model, events, prompt)

        assert result.video_id == "test_video"
        assert result.model_key == "phi3_v"
        assert len(result.stories) > 0
        assert len(result.judgements) == len(events)

    def test_story_ids_sequential(self, fake_video_dir):
        model = MockVLMModel(key="phi3_v", label="Phi-3.5-Vision")
        events = arena.load_events(fake_video_dir.parent)

        with patch("arena.judge_event", return_value=MagicMock(spec=Judgement)):
            result = arena.process_video(fake_video_dir, model, events, "Q")

        ids = [s.story_id for s in result.stories]
        assert ids[0] == "story_000"
        assert ids[1] == "story_001"

    def test_story_timeframes_from_frames(self, fake_video_dir):
        """Story start/end must be derived from frame filenames."""
        model = MockVLMModel(key="phi3_v", label="Phi-3.5-Vision")
        events = arena.load_events(fake_video_dir.parent)

        with patch("arena.judge_event", return_value=MagicMock(spec=Judgement)):
            result = arena.process_video(fake_video_dir, model, events, "Q")

        first_story = result.stories[0]
        assert first_story.timeframe.start == "00:00:00"
        assert first_story.timeframe.end == "00:00:04"

    def test_no_annotations_skips_judging(self, fake_video_dir):
        """Remove annotations.json — judge should not be called."""
        (fake_video_dir.parent / "annotations.json").unlink()
        model = MockVLMModel(key="phi3_v", label="Phi-3.5-Vision")

        with patch("arena.judge_event") as mock_judge:
            result = arena.process_video(fake_video_dir, model, [], "Q")
            mock_judge.assert_not_called()

        assert result.judgements == []

    def test_chunk_size_controls_frame_count(self, fake_video_dir, mocker):
        """Each story's frame_count should match the chunk size (except possibly the last)."""
        mocker.patch.object(arena, "CHUNK_SIZE", 3)
        mocker.patch.object(arena, "CHUNK_OVERLAP", 0)

        model = MockVLMModel(key="phi3_v", label="Phi-3.5-Vision")
        with patch("arena.judge_event", return_value=MagicMock(spec=Judgement)):
            result = arena.process_video(fake_video_dir, model, [], "Q")

        # All stories except possibly the last should have 3 frames
        for story in result.stories[:-1]:
            assert story.frame_count == 3

    def test_result_config_metadata(self, fake_video_dir, mocker):
        mocker.patch.object(arena, "CHUNK_SIZE", 5)
        mocker.patch.object(arena, "CHUNK_OVERLAP", 0)
        mocker.patch.object(arena, "FPS", 1)

        model = MockVLMModel(key="phi3_v", label="Phi-3.5-Vision")
        with patch("arena.judge_event", return_value=MagicMock(spec=Judgement)):
            result = arena.process_video(fake_video_dir, model, [], "Q")

        assert result.chunk_size == 5
        assert result.chunk_overlap == 0
        assert result.fps == 1
