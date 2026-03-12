#!/usr/bin/env python3
"""
Preprocessing pipeline: extract N random videos from each category in the crime_dataset
and organize them with prompt.txt and annotations.json files.

Requirements:
    pip install opencv-python

Usage:
    python preprocess_pipeline.py [options]

Examples:
    python preprocess_pipeline.py --num-videos 25 --max-duration 60
    python preprocess_pipeline.py --num-videos 10 --max-duration 120 --input-path /path/to/raw --outout-path /path/to/out
    python preprocess_pipeline.py --categories robbery shooting --num-videos 5
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Optional
import cv2


def get_video_duration_seconds(video_path: Path) -> Optional[float]:
    """Get video duration in seconds using OpenCV, or None on failure."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps == 0:
            return None
        return frame_count / fps
    except Exception as e:
        print(f"Error getting duration for {video_path}: {e}")
        return None


def seconds_to_hhmmss(seconds: float) -> str:
    """Convert seconds to HH:MM:SS string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def pluralize_category(category: str) -> str:
    """Convert category name to plural form."""
    plural_map = {
        "no_crime": "no crimes",
        "road_accidents": "road accidents",
        "robbery": "robberies",
        "shooting": "shootings",
    }
    return plural_map.get(category, category + "s")


def create_prompt_file(video_folder: Path, category: str) -> None:
    """Create prompt.txt file in video folder."""
    prompt_path = video_folder / "prompt.txt"
    category_plural = pluralize_category(category)
    prompt_path.write_text(f"Describe this scene. Watch out for {category_plural}.")
    print(f"  Created: {prompt_path}")


def create_annotations_file(video_folder: Path, category: str, duration_seconds: float) -> None:
    """Create annotations.json file in video folder."""
    annotations_path = video_folder / "annotations.json"
    annotations = [
        {
            "event_id": "evt_001",
            "description": category,
            "timeframe": {
                "start": "00:00:00",
                "end": seconds_to_hhmmss(duration_seconds),
            },
        }
    ]
    annotations_path.write_text(json.dumps(annotations, indent=2))
    print(f"  Created: {annotations_path}")


def copy_video_to_folder(video_path: Path, video_folder: Path) -> None:
    """Copy video file to its folder."""
    destination = video_folder / video_path.name
    shutil.copy2(video_path, destination)
    print(f"  Copied video: {destination}")


def process_category(
    category: str,
    raw_data_path: Path,
    prep_data_path: Path,
    num_videos: int,
    max_duration: Optional[float],
) -> None:
    """Process a single category: filter, select random videos, and create necessary files."""
    category_path = raw_data_path / category
    if not category_path.exists():
        print(f"Warning: category path not found: {category_path}, skipping.")
        return

    video_files = sorted(category_path.glob("*.mp4"))

    # Filter by max duration if specified
    if max_duration is not None:
        print(f"\n[{category}] Filtering {len(video_files)} videos by max duration {max_duration}s...")
        filtered = []
        for vp in video_files:
            dur = get_video_duration_seconds(vp)
            if dur is None:
                print(f"  Skipping {vp.name}: could not read duration")
                continue
            if dur <= max_duration:
                filtered.append((vp, dur))
            else:
                print(f"  Skipping {vp.name}: {dur:.1f}s > {max_duration}s")
        print(f"  {len(filtered)}/{len(video_files)} videos pass the duration filter")
    else:
        filtered = [(vp, get_video_duration_seconds(vp) or 0.0) for vp in video_files]

    # Select N random videos (0 = all)
    if num_videos and num_videos > 0:
        if len(filtered) < num_videos:
            print(f"Warning: {category} has only {len(filtered)} eligible videos, selecting all")
            selected = filtered
        else:
            selected = random.sample(filtered, num_videos)
    else:
        selected = filtered

    print(f"\nProcessing {category} ({len(selected)} videos):")

    for video_path, duration_seconds in selected:
        video_name = video_path.stem
        video_folder = prep_data_path / video_name
        video_folder.mkdir(parents=True, exist_ok=True)
        print(f"  Created folder: {video_folder}")

        copy_video_to_folder(video_path, video_folder)
        create_prompt_file(video_folder, category)
        create_annotations_file(video_folder, category, duration_seconds)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Preprocess pipeline: extract and organize crime dataset videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Path to raw data directory (default: <script_dir>/input)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Path to output preprocessed data directory (default: <script_dir>/output)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["no_crime", "road_accidents", "robbery", "shooting"],
        metavar="CATEGORY",
        help="Categories to process (default: all four)",
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        default=25,
        metavar="N",
        help="Number of random videos per category (0 = all, default: 25)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Maximum video duration in seconds; longer videos are excluded (default: no limit)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="SEED",
        help="Random seed for reproducibility (default: none)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    base_path = Path(__file__).parent
    raw_data_path = args.input_path or base_path / "input"
    prep_data_path = args.output_path or base_path / "output"

    if not raw_data_path.exists():
        print(f"Error: raw data directory not found at {raw_data_path}")
        sys.exit(1)

    if args.seed is not None:
        random.seed(args.seed)

    prep_data_path.mkdir(exist_ok=True)
    print(f"Output directory: {prep_data_path}")
    print(f"Categories: {args.categories}")
    print(f"Videos per category: {args.num_videos or 'all'}")
    print(f"Max duration: {args.max_duration}s" if args.max_duration else "Max duration: unlimited")

    for category in args.categories:
        process_category(
            category=category,
            raw_data_path=raw_data_path,
            prep_data_path=prep_data_path,
            num_videos=args.num_videos,
            max_duration=args.max_duration,
        )

    total = sum(1 for _ in prep_data_path.iterdir() if _.is_dir())
    print(f"\nDone! All videos extracted to {prep_data_path}")
    print(f"Total folders created: {total}")


if __name__ == "__main__":
    main()
