"""Text chunking module for splitting documents into manageable pieces."""

# Standard library imports
import asyncio

# Third-party imports
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
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

        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ]
        )

        # TODO: check semantic splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
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

        md_chunks: list[Document] = self._markdown_split(
            text_to_split, filename
        )

        final_chunks: list[Chunk] = []
        chunk_order = 0

        for chunk in md_chunks:
            # Extract markdown metadata
            if chunk.metadata:
                ((markdown_header, markdown_section),) = chunk.metadata.items()
            else:
                markdown_header, markdown_section = "", ""

            if len(chunk.page_content) > self.chunk_size:
                split = self.splitter.split_text(chunk.page_content)

                for s in split:
                    final_chunks.append(
                        Chunk(
                            content=s,
                            metadata=Metadata(
                                filename=filename,
                                order_number_of_chunk=chunk_order,
                                markdown_header=markdown_header,
                                markdown_section=markdown_section,
                            ),
                        )
                    )

                    chunk_order += 1
            else:
                final_chunks.append(
                    Chunk(
                        content=chunk.page_content,
                        metadata=Metadata(
                            filename=filename,
                            order_number_of_chunk=chunk_order,
                            markdown_header=markdown_header,
                            markdown_section=markdown_section,
                        ),
                    )
                )

                chunk_order += 1

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

    @traceable(
        run_type="tool",
        name="Markdown Split",
        metadata={
            "ls_provider": "Chunker",
            "type": "sequential",
        },
    )
    def _markdown_split(
        self, text_to_split: str, filename: str
    ) -> list[Document]:
        """Split text using markdown headers."""

        logger.opt(colors=True).debug(
            f"Starting markdown splitting for file: <green>{filename}</green>"
        )

        md_chunks: list[Document] = self.markdown_splitter.split_text(
            text_to_split
        )

        logger.opt(colors=True).debug(
            f"Markdown splitting resulted in <yellow>{len(md_chunks)}</yellow> chunks for file: <green>{filename}</green>"
        )

        return md_chunks
