"""OpenAI API configuration settings."""

# Standard library imports
from functools import lru_cache
from typing import Optional

# Third-party imports
from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


load_dotenv()


class OpenAISetting(BaseSettings):
    """Application settings loaded from environment variables"""

    OPENAI_API_KEY: str = Field(description="OpenAI API Key", min_length=160)

    MODEL_NAME_EMBEDDING: str = Field(
        default="text-embedding-3-large",
        description="Model name for embeddings",
    )

    MODEL_NAME_RAG: str = Field(
        default="gpt-4o",
        description="Model name for RAG tasks",
    )

    TEMPERATURE_RAG: float = Field(
        default=0.0,
        description="Temperature setting for RAG model",
        ge=0.0,
        le=1.0,
    )

    MAX_TOKENS_RAG: Optional[int] = Field(
        default=None,
        description="Maximum tokens for RAG model responses",
        ge=1,
    )

    MAX_RETRIES_INVOKE_LLM: int = Field(
        default=3,
        description="Maximum number of retries for invoking the LLM",
        ge=1,
    )

    BASE_DELAY_RETRY_LLM: int = Field(
        default=20,
        description="Base delay in seconds for retrying LLM invocations",
        ge=1,
    )

    @field_validator("OPENAI_API_KEY")
    def check_key_starts_with_sk(cls, v: str) -> str:
        if not v.startswith("sk-"):
            raise ValueError("The OPENAI_API_KEY must start with 'sk-'.")
        return v


@lru_cache()
def get_openai_settings() -> OpenAISetting:
    """Get cached settings instance."""
    return OpenAISetting()
