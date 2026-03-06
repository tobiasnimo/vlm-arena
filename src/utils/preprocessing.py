"""
Extract frames from an MP4 video as JPG images.
Filename format: frame_HH-MM-SS-mmm.jpg (hours, minutes, seconds, milliseconds)

Usage:
    python extract_frames.py <video.mp4> [output_dir] [--fps N]

Examples:
    python extract_frames.py video.mp4
    python extract_frames.py video.mp4 frames/
    python extract_frames.py video.mp4 frames/ --fps 1
"""

import cv2
import logging
import os
import argparse

logger = logging.getLogger(__name__)


def extract_frames(video_path: str, output_dir: str = "frames", fps: float = None):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file '{video_path}'")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps

    # How many video frames to skip between each saved frame
    frame_interval = 1 if fps is None else max(1, round(video_fps / fps))
    effective_fps = video_fps / frame_interval

    logger.info("Video: %s | duration: %.2fs | source FPS: %.2f | frames: %d", video_path, duration, video_fps, total_frames)
    logger.info("Extracting at %.2f FPS (every %d frame(s)) → %s", effective_fps, frame_interval, output_dir)

    os.makedirs(output_dir, exist_ok=True)

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
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            frame_paths.append(filepath)

        frame_index += 1

    cap.release()
    frame_paths.sort()
    logger.info("Saved %d frames to '%s'", len(frame_paths), output_dir)
    return frame_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from an MP4 video as JPG images.")
    parser.add_argument("video", help="Path to the input MP4 file")
    parser.add_argument("output_dir", nargs="?", default="frames", help="Output directory (default: frames/)")
    parser.add_argument("--fps", type=float, default=None, help="Frames per second to extract (default: every frame)")
    args = parser.parse_args()

    paths = extract_frames(args.video, args.output_dir, args.fps)
    print("\nFrame paths:")
    for p in paths:
        print(p)