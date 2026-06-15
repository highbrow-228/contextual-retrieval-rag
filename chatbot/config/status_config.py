# Standard library imports
from enum import Enum


class Status(str, Enum):
    """Enum for operation status."""

    SUCCESS = "Success"
    FAILED = "Failed"
