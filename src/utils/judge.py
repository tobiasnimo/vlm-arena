"""
fair-forge Vision Similarity judge for VLM Arena.

For each annotated event, the judge evaluates all stories whose timeframe
overlaps with the event's timeframe and returns a Judgement with a 0–1 score
computed via cosine similarity between VLM descriptions and ground truth.
"""

import logging
import sys
from pathlib import Path

from fair_forge.metrics.vision import VisionSimilarity
from fair_forge.schemas import Batch

# bridge/ lives one level above src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "bridge"))
from retriever import ArenaRetriever

from schemas import Event, Judgement, Story

logger = logging.getLogger(__name__)


def judge_event(event: Event, stories: list[Story]) -> Judgement:
    """
    Score a single event against all stories that overlap with it.

    Args:
        event:   The annotated event to evaluate.
        stories: All stories for this video-model pair.

    Returns:
        A Judgement with score, analysis, and the list of story IDs used.
    """
    overlapping = [s for s in stories if event.timeframe.overlaps(s.timeframe)]

    if not overlapping:
        logger.warning("No stories overlap with event '%s' (%s → %s) — skipping judge call",
                       event.event_id, event.timeframe.start, event.timeframe.end)
        return Judgement(
            event_id=event.event_id,
            event_description=event.description,
            event_timeframe=event.timeframe,
            story_ids=[],
            score=None,
            analysis="No VLM stories overlap with this event's timeframe.",
        )

    batch_item = Batch(
        qa_id=event.event_id,
        assistant=" ".join(s.answer for s in overlapping),
        ground_truth_assistant=event.description,
        query="",
    )

    vs = VisionSimilarity(ArenaRetriever)
    session_id = event.event_id
    vs.batch(
        session_id=session_id,
        context="",
        assistant_id="vlm-arena",
        batch=[batch_item],
    )

    interaction = vs._session_data[session_id]["interactions"][0]
    score = interaction["similarity_score"]
    analysis = (
        f"Cosine similarity between VLM description and ground truth: {score:.4f}"
    )

    logger.info("[%s] event '%s' → score: %s", event.event_id, event.description[:60], score)

    return Judgement(
        event_id=event.event_id,
        event_description=event.description,
        event_timeframe=event.timeframe,
        story_ids=[s.story_id for s in overlapping],
        score=score,
        analysis=analysis,
    )
