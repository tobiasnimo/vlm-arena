import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
        case_sensitive=False,
    )

    # API keys
    hf_token: str = ""
    groq_api_key: str = ""

    # Frame extraction
    fps: int = 1

    # Chunking
    chunk_size: int = 5     # number of frames per chunk
    chunk_overlap: int = 0  # overlapping frames between consecutive chunks

    # Models to run (list of model keys)
    models: list[str] = []

    # Set to True to skip GPU model loading and return placeholder answers
    mock_inference: bool = False

    # Limit the number of videos to process (0 = no limit)
    max_videos: int = 0

    # Skip videos longer than this many seconds (0 = no limit)
    max_video_duration: float = 0

    # Score threshold: judgements >= this value are "pass", below are "fail"
    pass_threshold: float = 0.5

    # Override HuggingFace cache location (useful when root volume is small, e.g. SageMaker)
    hf_home: str = ""


settings = Settings()

os.environ["HF_TOKEN"] = settings.hf_token
os.environ["GROQ_API_KEY"] = settings.groq_api_key
os.environ["HF_HUB_DISABLE_XET"] = "1"  # disable xet transfer protocol (unstable on some instances)
if settings.hf_home:
    os.environ["HF_HOME"] = settings.hf_home
