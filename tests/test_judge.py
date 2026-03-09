import pytest
from unittest.mock import MagicMock, patch

from schemas import Event, Judgement, Story, Timeframe
from utils.judge import judge_event


def _mock_groq_response(content: str):
    response = MagicMock()
    response.content = content
    return response


# ── judge_event — happy path ──────────────────────────────────────────────────

class TestJudgeEvent:
    def test_returns_judgement_with_score(self, sample_event, sample_stories):
        groq_json = '{"score": 0.75, "analysis": "Detected entry, missed details."}'
        with patch("utils.judge.ChatGroq") as MockGroq:
            MockGroq.return_value.invoke.return_value = _mock_groq_response(groq_json)
            result = judge_event(sample_event, sample_stories)

        assert isinstance(result, Judgement)
        assert result.score == 0.75
        assert result.event_id == sample_event.event_id
        assert result.event_description == sample_event.description

    def test_overlapping_stories_included(self, sample_event, sample_stories):
        """story_000 (00:00:00–00:00:04) and story_001 (00:00:05–00:00:09)
        both overlap event (00:00:03–00:00:12); story_002 (00:00:20–00:00:24) does not."""
        groq_json = '{"score": 0.6, "analysis": "Partial."}'
        with patch("utils.judge.ChatGroq") as MockGroq:
            MockGroq.return_value.invoke.return_value = _mock_groq_response(groq_json)
            result = judge_event(sample_event, sample_stories)

        assert "story_000" in result.story_ids
        assert "story_001" in result.story_ids
        assert "story_002" not in result.story_ids

    def test_event_timeframe_preserved(self, sample_event, sample_stories):
        groq_json = '{"score": 0.5, "analysis": "OK."}'
        with patch("utils.judge.ChatGroq") as MockGroq:
            MockGroq.return_value.invoke.return_value = _mock_groq_response(groq_json)
            result = judge_event(sample_event, sample_stories)

        assert result.event_timeframe.start == sample_event.timeframe.start
        assert result.event_timeframe.end == sample_event.timeframe.end

    def test_groq_called_once(self, sample_event, sample_stories):
        groq_json = '{"score": 0.8, "analysis": "Good."}'
        with patch("utils.judge.ChatGroq") as MockGroq:
            MockGroq.return_value.invoke.return_value = _mock_groq_response(groq_json)
            judge_event(sample_event, sample_stories)
            MockGroq.return_value.invoke.assert_called_once()


# ── judge_event — no overlapping stories ─────────────────────────────────────

class TestJudgeEventNoOverlap:
    def test_returns_judgement_with_none_score(self, sample_event):
        far_stories = [
            Story(
                story_id="story_000",
                timeframe=Timeframe(start="00:01:00", end="00:01:05"),
                question="Q",
                answer="Nothing relevant.",
                frame_count=5,
                elapsed_seconds=1.0,
            )
        ]
        with patch("utils.judge.ChatGroq") as MockGroq:
            result = judge_event(sample_event, far_stories)
            MockGroq.return_value.invoke.assert_not_called()

        assert result.score is None
        assert result.story_ids == []
        assert "No" in result.analysis

    def test_empty_stories_list(self, sample_event):
        with patch("utils.judge.ChatGroq") as MockGroq:
            result = judge_event(sample_event, [])
            MockGroq.return_value.invoke.assert_not_called()

        assert result.score is None
        assert result.story_ids == []


# ── judge_event — malformed LLM response ─────────────────────────────────────

class TestJudgeEventBadResponse:
    def test_non_json_response_score_is_none(self, sample_event, sample_stories):
        with patch("utils.judge.ChatGroq") as MockGroq:
            MockGroq.return_value.invoke.return_value = _mock_groq_response(
                "Sorry, I cannot evaluate this."
            )
            result = judge_event(sample_event, sample_stories)

        assert result.score is None
        assert "Sorry" in result.analysis

    def test_json_missing_score_key(self, sample_event, sample_stories):
        with patch("utils.judge.ChatGroq") as MockGroq:
            MockGroq.return_value.invoke.return_value = _mock_groq_response(
                '{"analysis": "Only analysis, no score."}'
            )
            result = judge_event(sample_event, sample_stories)

        assert result.score is None
        assert "Only analysis" in result.analysis
