"""
LLM-as-a-Judge for VLM Arena.

For each annotated event, the judge evaluates all stories whose timeframe
overlaps with the event's timeframe and returns a Judgement with a 0–1 score.
"""

import json
import logging

from langchain_groq import ChatGroq

from config import settings
from schemas import Event, Judgement, Story

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """\
You are an objective evaluator assessing how well a Vision Language Model (VLM) described a specific event in a video.

## Event to Evaluate
Description: {event_description}
Timeframe: {event_start} → {event_end}

## VLM Descriptions
The VLM processed the video in chunks. The following chunks overlap with the event's timeframe:
{descriptions}

## Task
Evaluate how well the VLM descriptions captured the event described above.
Focus only on whether the event was detected and accurately described — ignore content outside the event's scope.

Respond with a JSON object (and nothing else) in this exact format:
{{
  "analysis": "<2-3 sentence evaluation of what was captured correctly and what was missed>",
  "score": <float 0-1>
}}

Score guide: 0 = event completely missed, 1 = event fully and accurately captured."""


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

    descriptions = "\n\n".join(
        f"[{s.story_id} | {s.timeframe.start}–{s.timeframe.end}]\n{s.answer}"
        for s in overlapping
    )

    judge = ChatGroq(model="llama-3.1-8b-instant", api_key=settings.groq_api_key)

    prompt = JUDGE_PROMPT.format(
        event_description=event.description,
        event_start=event.timeframe.start,
        event_end=event.timeframe.end,
        descriptions=descriptions,
    )

    response = judge.invoke(prompt)
    raw = response.content.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Judge returned non-JSON response for event '%s'", event.event_id)
        parsed = {"analysis": raw, "score": None}

    score = parsed.get("score")
    analysis = parsed.get("analysis", "")

    logger.info("[%s] event '%s' → score: %s", event.event_id, event.description[:60], score)

    return Judgement(
        event_id=event.event_id,
        event_description=event.description,
        event_timeframe=event.timeframe,
        story_ids=[s.story_id for s in overlapping],
        score=score,
        analysis=analysis,
    )
