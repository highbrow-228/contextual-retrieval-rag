"""Chunk size configuration settings for text splitting operations."""

# Standard library imports
from functools import lru_cache

# Third-party imports
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings


load_dotenv()


class ChunkSettings(BaseSettings):
    """Application settings loaded from environment variables"""

    CHUNK_SIZE: int = Field(default=700, description="Size of each text chunk")

    CHUNK_OVERLAP: int = Field(
        default=200, description="Overlap size between text chunks"
    )


@lru_cache()
def get_chunk_settings() -> ChunkSettings:
    """Get cached settings instance."""
    return ChunkSettings()
