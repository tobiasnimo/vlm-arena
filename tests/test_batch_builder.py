"""Tests for bridge.batch_builder — Batch construction from events and stories."""

from batch_builder import build_batches
from schemas import Event, Story, Timeframe


def _make_event(event_id: str, start: str, end: str, description: str = "event") -> Event:
    return Event(event_id=event_id, description=description, timeframe=Timeframe(start=start, end=end))


def _make_story(story_id: str, start: str, end: str, answer: str = "answer") -> Story:
    return Story(
        story_id=story_id,
        timeframe=Timeframe(start=start, end=end),
        question="prompt",
        answer=answer,
        frame_count=5,
        elapsed_seconds=1.0,
    )


class TestBuildBatches:
    def test_overlapping_stories_produce_one_batch_per_event(self):
        event = _make_event("e1", "00:00:05", "00:00:15", "a cat appears")
        stories = [
            _make_story("s0", "00:00:00", "00:00:10", "hello"),
            _make_story("s1", "00:00:08", "00:00:18", "world"),
        ]
        batches = build_batches([event], stories)

        assert len(batches) == 1
        assert batches[0].qa_id == "e1"
        assert batches[0].assistant == "hello world"
        assert batches[0].ground_truth_assistant == "a cat appears"

    def test_non_overlapping_stories_produce_zero_batches(self):
        event = _make_event("e1", "00:01:00", "00:01:30")
        stories = [
            _make_story("s0", "00:00:00", "00:00:10"),
            _make_story("s1", "00:00:10", "00:00:20"),
        ]
        batches = build_batches([event], stories)

        assert len(batches) == 0

    def test_partial_overlap_only_includes_overlapping_stories(self):
        event = _make_event("e1", "00:00:10", "00:00:20", "the event")
        stories = [
            _make_story("s0", "00:00:00", "00:00:05", "no overlap"),
            _make_story("s1", "00:00:05", "00:00:12", "partial"),
            _make_story("s2", "00:00:15", "00:00:25", "also partial"),
            _make_story("s3", "00:00:25", "00:00:35", "too late"),
        ]
        batches = build_batches([event], stories)

        assert len(batches) == 1
        assert batches[0].assistant == "partial also partial"

    def test_multiple_events(self):
        events = [
            _make_event("e1", "00:00:00", "00:00:10"),
            _make_event("e2", "00:00:20", "00:00:30"),
        ]
        stories = [
            _make_story("s0", "00:00:05", "00:00:08", "first"),
            _make_story("s1", "00:00:22", "00:00:28", "second"),
        ]
        batches = build_batches(events, stories)

        assert len(batches) == 2
        assert batches[0].qa_id == "e1"
        assert batches[1].qa_id == "e2"

    def test_empty_stories(self):
        event = _make_event("e1", "00:00:00", "00:00:10")
        assert build_batches([event], []) == []

    def test_empty_events(self):
        story = _make_story("s0", "00:00:00", "00:00:10")
        assert build_batches([], [story]) == []
