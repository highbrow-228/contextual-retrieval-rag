"""Text embedding service using OpenAI's API for semantic representations."""

# Standard library imports
import asyncio
from typing import List

# Third-party imports
from chromadb.utils.embedding_functions import EmbeddingFunction
from langsmith import traceable
from loguru import logger
from openai import OpenAI

# Local imports
from chatbot.base.base_classes import BaseProcessor
from chatbot.models.chunk import Chunk
from chatbot.utils.embedding_utils import get_embedding_model_name


class OpenAIEmbedder(BaseProcessor, EmbeddingFunction):
    """Handles text embedding operations using OpenAI's API."""

    def __init__(self, api_key: str, embedding_model_name: str) -> None:
        self.client = OpenAI(api_key=api_key)
        self.embedding_model_name = embedding_model_name

    @traceable(
        run_type="embedding",
        name="Generate Embedding",
        metadata={
            "ls_provider": "OpenAIEmbedder",
            "model": "openai/" + get_embedding_model_name(),
        },
    )
    def process(self, text: str) -> List[float]:
        """Generate embeddings for the input text."""
        preview = text[:50] + ("..." if len(text) > 50 else "")
        logger.opt(colors=True).debug(
            f"Generating embedding for text of length <yellow>{len(text)}</yellow>. Preview: <green>{preview}</green>"
        )

        response = self.client.embeddings.create(
            input=text, model=self.embedding_model_name
        )

        embedding = response.data[0].embedding

        logger.opt(colors=True).debug(
            f"Embedding generated successfully, with vector length: <yellow>{len(embedding)}</yellow>."
        )
        return embedding

    @traceable(
        run_type="embedding",
        name="Generate Embedding Async",
        metadata={
            "ls_provider": "OpenAIEmbedder",
            "model": "openai/" + get_embedding_model_name(),
        },
    )
    async def process_async(self, text: str) -> List[float]:
        return await asyncio.to_thread(self.process, text)

    @traceable(
        run_type="embedding",
        name="Generate Batch Embeddings Async",
        metadata={
            "ls_provider": "OpenAIEmbedder",
            "model": "openai/" + get_embedding_model_name(),
        },
    )
    async def process_batch(self, texts: list[str]) -> list[list[float]]:
        response = await asyncio.to_thread(
            self.client.embeddings.create,
            input=texts,
            model=self.embedding_model_name,
        )
        return [item.embedding for item in response.data]

    @traceable(
        run_type="chain",
        name="Create Embedding for Chunks",
        metadata={
            "ls_provider": "OpenAIEmbedder",
            "model": "openai/" + get_embedding_model_name(),
        },
    )
    def create_embedding(
        self, chunks: list[Chunk], filename: str
    ) -> List[float]:
        # Embed chunks using embedding model
        for idx, chunk in enumerate(chunks):
            try:
                embeddings: list[float] = self.process(chunk.content)
                chunk.embeddings = embeddings

            except Exception as e:
                logger.opt(colors=True).error(
                    f"Error embedding chunk <yellow>{idx}</yellow> from <green>{filename}</green>: <red>{str(e)}</red>"
                )
                raise

        return chunks

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Generate embeddings for the input texts.

        Args:
            input: List of texts to embed

        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        embeddings = []
        for text in input:
            embedding = self.process(text)
            embeddings.append(embedding)
        return embeddings
