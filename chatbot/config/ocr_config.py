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

    CODE_ENRICHMENT: bool = Field(
        default=False, description="Enable code enrichment during OCR"
    )

    FORMULA_ENRICHMENT: bool = Field(
        default=False, description="Enable formula enrichment during OCR"
    )

    CUDA_VISIBLE_DEVICES: int = Field(
        default=0, description="CUDA device index to use"
    )

    OMP_NUM_THREADS: int = Field(
        default=8, description="Number of threads for parallel processing"
    )


@lru_cache()
def get_ocr_settings() -> OCRConfig:
    """Get cached settings instance."""
    return OCRConfig()
