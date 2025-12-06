"""LangSmith configuration settings."""

# Standard library imports
from functools import lru_cache

# Third-party imports
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings


load_dotenv()


class LangSmithSettings(BaseSettings):
    """LangSmith settings loaded from environment variables"""

    LANGSMITH_API_KEY: str = Field(
        default="",
        description="LangSmith API Key for tracing",
    )

    LANGSMITH_PROJECT: str = Field(
        default="ChatBot-RAG",
        description="LangSmith project name",
    )

    LANGSMITH_ENDPOINT: str = Field(
        default="https://api.smith.langchain.com",
        description="LangSmith API endpoint",
    )

    LANGSMITH_TRACING: bool = Field(
        default=True,
        description="Enable LangSmith tracing",
    )


@lru_cache()
def get_langsmith_settings() -> LangSmithSettings:
    """Get cached settings instance."""
    return LangSmithSettings()
