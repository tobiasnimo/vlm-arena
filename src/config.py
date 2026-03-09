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


settings = Settings()
os.environ["HF_TOKEN"] = settings.hf_token
os.environ["GROQ_API_KEY"] = settings.groq_api_key
