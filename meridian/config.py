import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    litellm_model_synthesis: str = "anthropic/claude-sonnet-4-6"
    litellm_model_classification: str = "anthropic/claude-haiku-4-5-20251001"

    # Database
    database_url: str = "postgresql://meridian:meridian@localhost:5432/meridian"
    redis_url: str = "redis://localhost:6379/0"

    # Vector search
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = 1024
    retrieval_top_k: int = 8
    rerank_top_k: int = 3

    # External APIs
    newsapi_key: str = ""
    gdelt_base_url: str = "https://api.gdeltproject.org/api/v2"

    # Storage
    s3_bucket: str = ""
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    tmp_dir: str = ".tmp/meridian"
    output_dir: str = "output/meridian"

    # Eval
    ragas_eval_threshold_faithfulness: float = 0.75
    ragas_eval_threshold_relevancy: float = 0.70

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
