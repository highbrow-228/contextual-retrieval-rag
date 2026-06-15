"""Text embedding service using OpenAI's API for semantic representations."""

# Standard library imports
import asyncio
import time
from typing import List

# Third-party imports
from chromadb.utils.embedding_functions import EmbeddingFunction
from fastapi import UploadFile
from langsmith import traceable
from loguru import logger
from openai import AsyncOpenAI, OpenAI

# Local imports
from chatbot.base.base_classes import BaseProcessor
from chatbot.models.chunk import Chunk
from chatbot.utils.embedding_utils import get_embedding_model_name


class OpenAIEmbedder(BaseProcessor, EmbeddingFunction):
    """Handles text embedding operations using OpenAI's API."""

    def __init__(
        self, api_key: str, embedding_model_name: str, batch_size: int
    ) -> None:
        self.client: OpenAI = OpenAI(api_key=api_key)
        self.async_client: AsyncOpenAI = AsyncOpenAI(api_key=api_key)
        self.embedding_model_name = embedding_model_name
        self.batch_size = batch_size

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
        response = await self.async_client.embeddings.create(
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
        start_time = time.time()
        logger.opt(colors=True).info(
            f"Starting embedding process for <green>{len(chunks)}</green> chunks from file: <green>{filename}</green>"
        )
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

        logger.opt(colors=True).info(
            f"Embedding process completed for file: <green>{filename}</green> in <yellow>{time.time() - start_time}</yellow> seconds"
        )
        return chunks

    @traceable(
        run_type="chain",
        name="Create Embedding for Chunks Async Batch",
        metadata={
            "ls_provider": "OpenAIEmbedder",
            "model": "openai/" + get_embedding_model_name(),
        },
    )
    async def create_embedding_async(
        self, chunks: list[Chunk], file: UploadFile
    ) -> list[Chunk]:
        """Generate embeddings for chunks asynchronously in batches.

        Args:
            chunks: List of chunks to embed
            filename: Name of the file being processed

        Returns:
            List of chunks with embeddings set
        """
        start_time = time.time()
        logger.opt(colors=True).info(
            f"Starting async embedding process for <green>{len(chunks)}</green> chunks. "
            f"From file: <green>{file.filename}</green>. "
            f"Where batch size: <yellow>{self.batch_size}</yellow>."
        )

        # Process chunks in batches
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
        for batch_num, batch_idx in enumerate(
            range(0, len(chunks), self.batch_size), 1
        ):
            batch = chunks[batch_idx : batch_idx + self.batch_size]

            logger.opt(colors=True).info(
                f"Processing batch <yellow>{batch_num}/{total_batches}</yellow>"
            )

            try:
                # Extract content from chunks
                texts = [chunk.content for chunk in batch]

                # Get embeddings for the batch
                embeddings_list = await self.process_batch(texts)

                # Assign embeddings to chunks
                for chunk, embedding in zip(batch, embeddings_list):
                    chunk.embeddings = embedding

            except Exception as e:
                logger.opt(colors=True).error(
                    f"Error embedding batch starting at <yellow>{batch_idx}</yellow> from <green>{file.filename}</green>: <red>{str(e)}</red>"
                )
                raise

        logger.opt(colors=True).info(
            f"Async embedding process completed for file: <green>{file.filename}</green> in <yellow>{round(time.time() - start_time, 3)}</yellow> seconds"
        )
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
