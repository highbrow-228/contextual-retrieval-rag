# Standard library imports
from enum import Enum


class PipelineMode(str, Enum):
    """Enum for database operation modes."""

    BASELINE = "baseline"
    ADVANCED = "advanced"
