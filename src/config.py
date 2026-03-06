import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="config.txt",
        env_file_encoding="utf-8",
        extra="allow",
        case_sensitive=False,
    )

    # API KEYS
    hf_token: str = ""
    groq_api_key: str = ""

    # Arena Settings
    fps: int = 1
    models: list[str] = []

settings = Settings()
os.environ["HF_TOKEN"] = settings.hf_token
os.environ["GROQ_API_KEY"] = settings.groq_api_key