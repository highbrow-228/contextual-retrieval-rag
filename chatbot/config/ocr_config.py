"""OCR configuration settings for document processing."""

# Standard library imports
from functools import lru_cache

# Third-party imports
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings


load_dotenv()


class OCRConfig(BaseSettings):
    """Configuration for OCR processing"""

    ENGINE: str = Field(default="easyocr", description="OCR engine to use")

    CUDA_VISIBLE_DEVICES: int = Field(
        default=0, description="CUDA device index to use"
    )


@lru_cache()
def get_ocr_settings() -> OCRConfig:
    """Get cached settings instance."""
    return OCRConfig()
