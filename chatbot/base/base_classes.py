"""Abstract base classes for core processing components."""

# Standard library imports
from abc import ABC, abstractmethod


class BaseProcessor(ABC):
    """Abstract base class for all processing components."""

    @abstractmethod
    def process(self, *args, **kwargs):
        """Process method to be implemented by subclasses."""
        pass
