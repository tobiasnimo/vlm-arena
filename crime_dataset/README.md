# UCF-Crime Dataset

[**UCF-Crime Dataset**](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/) is a large-scale real-world surveillance video dataset containing anomaly events such as road accidents, robbery, and shooting, as well as normal (no crime) footage

I used a preprocessed **subset of this dataset as input for **vlm-arena**.

## Source Data

Raw videos were downloaded from the this [dropbox](dropbox.com/scl/fo/2aczdnx37hxvcfdo4rq4q/AOjRokSTaiKxXmgUyqdcI6k?rlkey=5bg7mxxbq46t7aujfch46dlvz&dl=0) as ZIP files.

I stored the videos under `crime_data/`, and organized them by category:

```
crime_data/
├── no_crime/
├── road_accidents/
├── robbery/
└── shooting/
```

## Preprocessing

Then, I ran the `preprocess_pipeline.py` script:

```bash
python preprocess_pipeline.py --num-videos 25 --max-duration 60 --output-path "dataset"
```

This produced 25 videos per category (max 60 seconds each) under `dataset/`.

> The script selects a random sample of videos per category filters them by maximum duration, and structures each video into its own folder with the metadata files required by vlm-arena.

### Output

```
dataset/
├── Normal_Videos_365_x264/
│   ├── Normal_Videos_365_x264.mp4 — original video file (`.mp4`)
│   ├── prompt.txt — scene description prompt fed to the VLM
│   └── annotations.json — ground truth metadata with category and video duration
├── RoadAccidents032_x264/
│   ├── RoadAccidents032_x264.mp4
│   ├── prompt.txt
│   └── annotations.json
└── ...
```

### Other options

| Flag | Default | Description |
|---|---|---|
| `--num-videos` | 25 | Videos to sample per category (0 = all) |
| `--max-duration` | none | Exclude videos longer than N seconds |
| `--categories` | all 4 | Space-separated list of categories to process |
| `--input-path` | `./input` | Path to raw input directory |
| `--output-path` | `./output` | Path to output directory |
| `--seed` | none | Random seed for reproducibility |

