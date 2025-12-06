"""Database configuration settings for ChromaDB connection and embedding models."""

# Standard library imports
from enum import Enum
from functools import lru_cache

# Third-party imports
from dotenv import load_dotenv
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


load_dotenv()


# Enum for database operation modes
class DatabaseMode(str, Enum):
    """Enum for database operation modes."""

    CLOUD = "cloud"
    SERVER = "server"


# Enum for collection names in the vector database
class CollectionName(str, Enum):
    """Enum for collection names in the vector database."""

    BASELINE_DOCUMENTS = "baseline_documents"
    ADVANCED_DOCUMENTS = "advanced_documents"
    RAG_DOCUMENT = "rag_document"
    ADDITIONAL_DOCS = "additional_docs"


# Global list of valid database modes
VALID_DATABASE_MODES = [mode.value for mode in DatabaseMode]

# Global list of valid collection names
VALID_COLLECTION_NAMES = [name.value for name in CollectionName]


class DatabaseSettings(BaseSettings):
    """Application settings loaded from environment variables"""

    CHROMA_MODE: DatabaseMode = Field(
        default=DatabaseMode.SERVER,
        description="ChromaDB mode. You can host it locally or use the cloud version",
    )

    # ChromaDB settings
    CHROMA_HOST: str = Field(default="", description="ChromaDB host address")

    CHROMA_PORT: int = Field(default=0, description="ChromaDB port number")

    CHROMA_TENANT: str = Field(
        default="", description="ChromaDB tenant identifier"
    )

    CHROMA_API_KEY: str = Field(
        default="", description="ChromaDB API key for authentication"
    )

    DATA_FORMAT: str = Field(
        default="%Y-%m-%d %H:%M:%S UTC",
        description="Datetime format for displaying dates",
    )

    @model_validator(mode="after")
    def validate_mode_settings(self):
        """Validate that required settings are present based on mode"""
        match self.CHROMA_MODE:
            case DatabaseMode.SERVER:
                if not self.CHROMA_HOST or self.CHROMA_HOST.strip() == "":
                    raise ValueError(
                        "CHROMA_HOST is required for 'server' mode"
                    )
                if not self.CHROMA_PORT or self.CHROMA_PORT <= 0:
                    raise ValueError(
                        "CHROMA_PORT must be a positive integer for 'server' mode"
                    )
            case DatabaseMode.CLOUD:
                if (
                    not self.CHROMA_API_KEY
                    or self.CHROMA_API_KEY.strip() == ""
                ):
                    raise ValueError(
                        "CHROMA_API_KEY is required for 'cloud' mode"
                    )
                if not self.CHROMA_TENANT or self.CHROMA_TENANT.strip() == "":
                    raise ValueError(
                        "CHROMA_TENANT is required for 'cloud' mode"
                    )
            case _:
                raise ValueError(
                    f"Invalid CHROMA_MODE. Must be {VALID_DATABASE_MODES}, got: {self.CHROMA_MODE}"
                )

        return self


@lru_cache()
def get_db_settings() -> DatabaseSettings:
    """Get cached settings instance."""
    return DatabaseSettings()
