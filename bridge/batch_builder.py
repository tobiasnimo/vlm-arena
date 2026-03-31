"""Build fair-forge Batch objects from VLM Arena events and stories."""

from fair_forge.schemas import Batch

from schemas import Event, Story


def build_batches(events: list[Event], stories: list[Story]) -> list[Batch]:
    """For each event, find overlapping stories and return a Batch.

    A story overlaps an event when:
        story.start < event.end AND event.start < story.end

    Events with no overlapping stories are skipped (no Batch produced).
    """
    batches: list[Batch] = []
    for event in events:
        overlapping = [
            s for s in stories
            if s.timeframe.start_seconds() < event.timeframe.end_seconds()
            and event.timeframe.start_seconds() < s.timeframe.end_seconds()
        ]
        if not overlapping:
            continue
        batches.append(
            Batch(
                qa_id=event.event_id,
                assistant=" ".join(s.answer for s in overlapping),
                ground_truth_assistant=event.description,
                query="",
            )
        )
    return batches
