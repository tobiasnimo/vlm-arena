"""
Extract frames from an MP4 video as JPG images.
Filename format: frame_HH-MM-SS-mmm.jpg (hours, minutes, seconds, milliseconds)
"""

import cv2
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_frames(video_path, output_dir, fps: float = None) -> list[str]:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file '{video_path}'")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps

    frame_interval = 1 if fps is None else max(1, round(video_fps / fps))
    effective_fps = video_fps / frame_interval

    logger.info(
        "Video: %s | duration: %.2fs | source FPS: %.2f | frames: %d",
        video_path, duration, video_fps, total_frames,
    )
    logger.info(
        "Extracting at %.2f FPS (every %d frame(s)) → %s",
        effective_fps, frame_interval, output_dir,
    )

    os.makedirs(str(output_dir), exist_ok=True)

    frame_paths = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            hours   = int(timestamp_ms // 3_600_000)
            minutes = int((timestamp_ms % 3_600_000) // 60_000)
            seconds = int((timestamp_ms % 60_000) // 1_000)
            millis  = int(timestamp_ms % 1_000)

            filename = f"frame_{hours:02d}-{minutes:02d}-{seconds:02d}-{millis:03d}.jpg"
            filepath = os.path.join(str(output_dir), filename)
            cv2.imwrite(filepath, frame)
            frame_paths.append(filepath)

        frame_index += 1

    cap.release()
    frame_paths.sort()
    logger.info("Saved %d frames to '%s'", len(frame_paths), output_dir)
    return frame_paths


def parse_frame_timestamp(frame_path: str) -> str:
    """
    Extract "HH:MM:SS" from a frame filename of the form frame_HH-MM-SS-mmm.jpg.

    Args:
        frame_path: Path or filename of the extracted frame.

    Returns:
        Timestamp string "HH:MM:SS".

    Raises:
        ValueError: If the filename does not match the expected pattern.
    """
    name = Path(frame_path).name
    m = re.match(r"frame_(\d{2})-(\d{2})-(\d{2})-\d{3}\.jpg", name)
    if not m:
        raise ValueError(f"Cannot parse timestamp from frame filename: '{name}'")
    h, mi, s = m.groups()
    return f"{h}:{mi}:{s}"


def make_chunks(frames: list[str], chunk_size: int, overlap: int) -> list[list[str]]:
    """
    Split a sorted list of frame paths into overlapping chunks.

    Args:
        frames:     Sorted list of frame file paths.
        chunk_size: Number of frames per chunk.
        overlap:    Number of frames shared between consecutive chunks.

    Returns:
        List of frame-path lists.
    """
    if overlap >= chunk_size:
        raise ValueError(f"CHUNK_OVERLAP ({overlap}) must be less than CHUNK_SIZE ({chunk_size})")

    step = chunk_size - overlap
    return [frames[i: i + chunk_size] for i in range(0, len(frames), step)]
