"""API configuration settings for the chatbot server."""

# Standard library imports
from functools import lru_cache

# Third-party imports
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings


DEFAULT_HOST = "127.0.0.1"
load_dotenv()


class APIConfig(BaseSettings):
    """Configuration for API processing"""

    HOST: str = Field(default=DEFAULT_HOST, description="API host address")
    PORT: int = Field(default=8000, description="API port number")
    RELOAD: bool = Field(
        default=True, description="Enable auto-reload for the API server"
    )


@lru_cache()
def get_api_settings() -> APIConfig:
    """Get cached settings instance."""
    return APIConfig()
