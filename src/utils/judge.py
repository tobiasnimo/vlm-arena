"""
LLM-as-a-Judge for VLM Arena
==============================
Compares VLM descriptions (aggregated across ALL video stories) against
ground-truth event annotations and returns a compatibility score per model.
"""

import json

from langchain_groq import ChatGroq

from config import settings


JUDGE_PROMPT = """\
You are an objective evaluator assessing how well a Vision Language Model (VLM) described a video.

## Ground-Truth Annotations
The following bullet points describe the key events that occur in the video:
{annotations}

## VLM Descriptions
The VLM processed the video in chronological chunks and produced the following descriptions:
{descriptions}

## Task
Compare the VLM descriptions against the ground-truth annotations.
Evaluate how well the VLM captured the events listed in the annotations.

Respond with a JSON object (and nothing else) in this exact format:
{{
  "analysis": "<2-3 sentence evaluation of what was captured correctly and what was missed>",
  "score": <float 0-1>
}}

Score guide: 0 = missed everything, 1 = all events perfectly captured."""


def judge_descriptions(
    annotations: str,
    descriptions: str,
) -> dict:
    """Judge a single model's full-video descriptions against annotations.

    Args:
        annotations: Ground-truth event annotations for the video.
        descriptions: Pre-aggregated chronological descriptions for one model.

    Returns:
        Dict with 'score' (float 0-1) and 'analysis' (str).
    """
    judge = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=settings.groq_api_key,
    )

    prompt = JUDGE_PROMPT.format(
        annotations=annotations.strip(),
        descriptions=descriptions,
    )

    response = judge.invoke(prompt)
    raw = response.content.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"analysis": raw, "score": None}

    return {
        "score": parsed.get("score"),
        "analysis": parsed.get("analysis", ""),
    }
