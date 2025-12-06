"""Text chunking module for splitting documents into manageable pieces."""

# Standard library imports
import asyncio

# Third-party imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
from loguru import logger

# Local imports
from chatbot.base.base_classes import BaseProcessor
from chatbot.models.chunk import Chunk, Metadata
from chatbot.models.rag import RetrievedChunk


class Chunker(BaseProcessor):
    """Handles text chunking operations."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=False,
            strip_whitespace=True,
        )

    @traceable(
        run_type="tool",
        name="Forming Chunks",
        metadata={
            "ls_provider": "Chunker",
            "type": "sequential",
        },
    )
    async def process(self, text_to_split: str, filename: str) -> list[Chunk]:
        """Process the input text into chunks."""
        logger.opt(colors=True).debug(
            f"Starting text chunking process for file: <green>{filename}</green>"
        )

        split_text: list[str] = self.splitter.split_text(text_to_split)

        final_chunks: list[Chunk] = []

        for chunk_order, content in enumerate(split_text):
            final_chunks.append(
                Chunk(
                    content=content,
                    metadata=Metadata(
                        filename=filename, order_number_of_chunk=chunk_order
                    ),
                )
            )

        logger.opt(colors=True).debug(
            f"Text chunking completed: <yellow>{len(final_chunks)}</yellow> chunks created for file: <green>{filename}</green>"
        )
        return final_chunks

    @traceable(
        run_type="tool",
        name="Process Batch",
        metadata={
            "ls_provider": "Chunker",
            "type": "Batch",
        },
    )
    async def process_batch(
        self, documents: list[tuple[str, str]]
    ) -> list[list[Chunk]]:
        tasks = [self.process(text, filename) for text, filename in documents]

        return await asyncio.gather(*tasks)

    @traceable(
        run_type="tool",
        name="Format Retrieved Chunks",
        metadata={
            "ls_provider": "Chunker",
            "type": "sequential",
        },
    )
    def format_retrieved_chunks(
        self, retrieved_chunks: list[RetrievedChunk]
    ) -> str:
        """Format retrieved chunks into a readable string for LLM context.

        Args:
            retrieved_chunks: List of chunks retrieved from the vector database.

        Returns:
            Formatted string with chunk content and metadata separated by dashes.
        """
        formatted_chunks = []

        for rc in retrieved_chunks:
            formatted_chunks.append(
                (
                    "---\n"
                    f"Score of chunk: {rc.relevance_score:.2f}\n"
                    f"Position of chunk: {rc.position}\n"
                    f'Content of chunk """{rc.chunk.content}"""\n'
                    "---\n"
                )
            )

        return "\n\n".join(formatted_chunks)
