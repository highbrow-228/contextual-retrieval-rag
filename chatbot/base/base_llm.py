"""Abstract base classes for LLM core processing components."""

# Standard library imports
from abc import ABC, abstractmethod


class BaseLLMHandler(ABC):
    """Abstract base class for LLM handler implementations."""

    @abstractmethod
    def invoke_rag(
        self, user_input: str, text_chunks: str, *args, **kwargs
    ) -> str:
        """Process method to be implemented by subclasses."""
        pass

    @abstractmethod
    async def ainvoke_rag(
        self, user_input: str, text_chunks: str, *args, **kwargs
    ) -> str:
        """Asynchronously invoke the RAG pipeline with the given user input and retrieved text chunks."""
        pass
