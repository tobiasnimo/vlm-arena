from typing import Optional
from pydantic import BaseModel


class Timeframe(BaseModel):
    start: str  # "HH:MM:SS"
    end: str    # "HH:MM:SS"

    @staticmethod
    def _to_seconds(t: str) -> float:
        h, m, s = t.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    def start_seconds(self) -> float:
        return self._to_seconds(self.start)

    def end_seconds(self) -> float:
        return self._to_seconds(self.end)

    def overlaps(self, other: "Timeframe") -> bool:
        """True if this timeframe intersects with other (inclusive endpoints)."""
        return self.start_seconds() <= other.end_seconds() and self.end_seconds() >= other.start_seconds()


class Event(BaseModel):
    event_id: str
    description: str
    timeframe: Timeframe


class Story(BaseModel):
    story_id: str          # e.g. "story_000"
    timeframe: Timeframe   # derived from first/last frame in chunk
    question: str
    answer: str
    frame_count: int       # number of frames in this chunk
    elapsed_seconds: float


class Judgement(BaseModel):
    event_id: str
    event_description: str
    event_timeframe: Timeframe
    story_ids: list[str]        # stories used as context
    score: Optional[float]      # 0–1, None if judge failed or no overlapping stories
    analysis: str


class VideoResult(BaseModel):
    video_id: str
    model_key: str
    model_label: str
    video_duration: str   # "HH:MM:SS" — timestamp of last extracted frame
    fps: int
    chunk_size: int
    chunk_overlap: int
    stories: list[Story]
    judgements: list[Judgement]
