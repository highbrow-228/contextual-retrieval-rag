"""LangSmith configuration settings."""

# Standard library imports
from enum import Enum
from functools import lru_cache

# Third-party imports
from dotenv import load_dotenv
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


load_dotenv()


class LANGSMITH_TRACING_VALUES(str, Enum):
    TRUE = "true"
    FALSE = "false"


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

    LANGSMITH_TRACING: LANGSMITH_TRACING_VALUES = Field(
        default=LANGSMITH_TRACING_VALUES.FALSE,
        description="Enable LangSmith tracing",
    )

    @model_validator(mode="after")
    def validate_langsmith_settings(self):
        """Validate LangSmith settings and warn if tracing is disabled."""
        if self.LANGSMITH_TRACING == "true" and not self.LANGSMITH_API_KEY:
            raise ValueError(
                "LANGSMITH_TRACING is enabled but LANGSMITH_API_KEY is not set. "
                "Set LANGSMITH_API_KEY environment variable or disable tracing with LANGSMITH_TRACING=false"
            )
        return self


@lru_cache()
def get_langsmith_settings() -> LangSmithSettings:
    """Get cached settings instance."""
    return LangSmithSettings()
