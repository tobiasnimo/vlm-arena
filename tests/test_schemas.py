import pytest
from pydantic import ValidationError

from schemas import Event, Judgement, Story, Timeframe, VideoResult


# ── Timeframe._to_seconds ─────────────────────────────────────────────────────

class TestTimeframeToSeconds:
    def test_zero(self):
        assert Timeframe._to_seconds("00:00:00") == 0.0

    def test_seconds_only(self):
        assert Timeframe._to_seconds("00:00:30") == 30.0

    def test_minutes(self):
        assert Timeframe._to_seconds("00:01:00") == 60.0

    def test_hours(self):
        assert Timeframe._to_seconds("01:00:00") == 3600.0

    def test_combined(self):
        assert Timeframe._to_seconds("01:02:03") == 3723.0

    def test_start_end_seconds(self):
        tf = Timeframe(start="00:00:10", end="00:01:30")
        assert tf.start_seconds() == 10.0
        assert tf.end_seconds() == 90.0


# ── Timeframe.overlaps ────────────────────────────────────────────────────────

class TestTimeframeOverlaps:
    def test_clear_overlap(self, tf_0_10, tf_5_15):
        assert tf_0_10.overlaps(tf_5_15)
        assert tf_5_15.overlaps(tf_0_10)

    def test_no_overlap(self, tf_0_10, tf_20_30):
        assert not tf_0_10.overlaps(tf_20_30)
        assert not tf_20_30.overlaps(tf_0_10)

    def test_touching_endpoints_overlap(self):
        """Shared endpoint counts as overlap."""
        tf_a = Timeframe(start="00:00:00", end="00:00:10")
        tf_b = Timeframe(start="00:00:10", end="00:00:20")
        assert tf_a.overlaps(tf_b)

    def test_contained(self):
        outer = Timeframe(start="00:00:00", end="00:01:00")
        inner = Timeframe(start="00:00:20", end="00:00:40")
        assert outer.overlaps(inner)
        assert inner.overlaps(outer)

    def test_identical(self, tf_0_10):
        assert tf_0_10.overlaps(tf_0_10)

    def test_adjacent_no_gap(self):
        a = Timeframe(start="00:00:00", end="00:00:05")
        b = Timeframe(start="00:00:06", end="00:00:10")
        assert not a.overlaps(b)


# ── Event ─────────────────────────────────────────────────────────────────────

class TestEvent:
    def test_valid(self):
        e = Event(
            event_id="evt_001",
            description="Person enters",
            timeframe=Timeframe(start="00:00:00", end="00:00:10"),
        )
        assert e.event_id == "evt_001"

    def test_missing_field_raises(self):
        with pytest.raises(ValidationError):
            Event(event_id="evt_001", description="Missing timeframe")


# ── Story ─────────────────────────────────────────────────────────────────────

class TestStory:
    def test_valid(self):
        s = Story(
            story_id="story_000",
            timeframe=Timeframe(start="00:00:00", end="00:00:04"),
            question="Describe.",
            answer="A room.",
            frame_count=5,
            elapsed_seconds=1.5,
        )
        assert s.story_id == "story_000"
        assert s.frame_count == 5

    def test_missing_field_raises(self):
        with pytest.raises(ValidationError):
            Story(story_id="story_000", question="Q", answer="A")


# ── Judgement ─────────────────────────────────────────────────────────────────

class TestJudgement:
    def test_valid_with_score(self):
        j = Judgement(
            event_id="evt_001",
            event_description="Person enters",
            event_timeframe=Timeframe(start="00:00:00", end="00:00:10"),
            story_ids=["story_000", "story_001"],
            score=0.85,
            analysis="Accurately captured.",
        )
        assert j.score == 0.85
        assert len(j.story_ids) == 2

    def test_none_score_allowed(self):
        j = Judgement(
            event_id="evt_001",
            event_description="Event",
            event_timeframe=Timeframe(start="00:00:00", end="00:00:05"),
            story_ids=[],
            score=None,
            analysis="No overlap.",
        )
        assert j.score is None


# ── VideoResult ───────────────────────────────────────────────────────────────

class TestVideoResult:
    def test_valid(self):
        vr = VideoResult(
            video_id="vid_01",
            model_key="phi3_v",
            model_label="Phi-3.5-Vision",
            video_duration="00:01:30",
            fps=1,
            chunk_size=5,
            chunk_overlap=0,
            stories=[],
            judgements=[],
        )
        assert vr.video_id == "vid_01"

    def test_serialisation_round_trip(self):
        import json

        story = Story(
            story_id="story_000",
            timeframe=Timeframe(start="00:00:00", end="00:00:04"),
            question="Q",
            answer="A",
            frame_count=5,
            elapsed_seconds=1.0,
        )
        j = Judgement(
            event_id="evt_001",
            event_description="Desc",
            event_timeframe=Timeframe(start="00:00:00", end="00:00:08"),
            story_ids=["story_000"],
            score=0.9,
            analysis="Good.",
        )
        vr = VideoResult(
            video_id="vid_01",
            model_key="phi3_v",
            model_label="Phi-3.5-Vision",
            video_duration="00:00:30",
            fps=1,
            chunk_size=5,
            chunk_overlap=0,
            stories=[story],
            judgements=[j],
        )
        parsed = json.loads(vr.model_dump_json())
        assert parsed["video_id"] == "vid_01"
        assert parsed["stories"][0]["story_id"] == "story_000"
        assert parsed["judgements"][0]["score"] == 0.9
        assert parsed["judgements"][0]["event_timeframe"]["start"] == "00:00:00"
