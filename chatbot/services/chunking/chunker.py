"""Text chunking module for splitting documents into manageable pieces."""

# Standard library imports
import asyncio

# Third-party imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
from loguru import logger

# Local imports
from chatbot.base.base_classes import BaseProcessor
from chatbot.config.pipeline_config import PipelineMode
from chatbot.models.chunk import Chunk, Metadata
from chatbot.models.rag import RetrievedChunk
from chatbot.services.llm_handlers.openai_handler import OpenAIHandler


class Chunker(BaseProcessor):
    """Handles text chunking operations."""

    def __init__(
        self, chunk_size: int, chunk_overlap: int, llm_handler: OpenAIHandler
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm_handler = llm_handler

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=False,
            strip_whitespace=True,
        )

    async def process(
        self, text_to_split: str, filename: str, mode: PipelineMode
    ) -> list[Chunk]:
        """Process the input text into chunks using the specified mode."""
        match mode:
            case PipelineMode.BASELINE:
                return await self.process_baseline(
                    text_to_split=text_to_split, filename=filename
                )
            case PipelineMode.ADVANCED:
                return await self.process_advanced(
                    text_to_split=text_to_split, filename=filename
                )
            case _:
                raise ValueError(f"Unsupported chunking mode: {mode}")

    @traceable(
        run_type="chain",
        name="Forming Chunks",
        metadata={
            "ls_provider": "Chunker",
            "type": "sequential",
        },
    )
    async def process_baseline(
        self, text_to_split: str, filename: str
    ) -> list[Chunk]:
        """Process the input text into chunks."""
        logger.opt(colors=True).info(
            f"Starting text chunking process for file: <green>{filename}</green>"
        )

        split_text: list[str] = self.__split_text_baseline(text_to_split)

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

        logger.opt(colors=True).info(
            f"Text chunking completed: <yellow>{len(final_chunks)}</yellow> chunks created for file: <green>{filename}</green>"
        )
        return final_chunks

    @traceable(
        run_type="chain",
        name="Forming Advanced Chunks",
        metadata={
            "ls_provider": "Chunker",
            "type": "sequential",
        },
    )
    async def process_advanced(
        self, text_to_split: str, filename: str
    ) -> list[Chunk]:
        """Process the input text into chunks using an LLM-based approach."""
        logger.opt(colors=True).info(
            f"Starting advanced text chunking process for file: <green>{filename}</green>"
        )

        split_text: list[str] = self.__split_text_baseline(text_to_split)
        total_chunks = len(split_text)
        final_chunks: list[Chunk] = [None] * len(split_text)

        BATCH_SIZE = 30
        previous_summaries_window: list[str] = []

        # previous_summaries_window = deque(maxlen=3)

        logger.info(
            f"Generating context-aware descriptions for {len(split_text)} chunks..."
        )
        for batch_start in range(0, len(split_text), BATCH_SIZE):
            batch = split_text[batch_start : batch_start + BATCH_SIZE]
            summaries_snapshot = list(previous_summaries_window[-3:])

            tasks = [
                self.llm_handler.generate_chunk_description(
                    current_chunk_text=content,
                    previous_summaries=summaries_snapshot,
                    filename=filename,
                    max_concurrent_requests=BATCH_SIZE,
                )
                for content in batch
            ]

            descriptions = await asyncio.gather(
                *tasks, return_exceptions=False
            )

            for i, (content, description) in enumerate(
                zip(batch, descriptions)
            ):
                chunk_order = batch_start + i
                enriched_content = f"[CONTEXT DESCRIPTION]:\n{description}\n\n[ORIGINAL TEXT]:\n{content}"
                final_chunks[chunk_order] = Chunk(
                    content=enriched_content,
                    metadata=Metadata(
                        filename=filename,
                        order_number_of_chunk=chunk_order,
                    ),
                )
                # Оновлюємо вікно після батчу
                previous_summaries_window.append(description)
                if len(previous_summaries_window) > 3:
                    previous_summaries_window.pop(0)

            processed = batch_start + len(batch)
            logger.debug(f"Processed {processed}/{total_chunks} chunks...")

        logger.opt(colors=True).info(
            f"Advanced text chunking completed: <yellow>{len(final_chunks)}</yellow> "
            f"chunks created for file: <green>{filename}</green>"
        )
        return final_chunks
        # for chunk_order, content in enumerate(split_text):
        #     description = await self.llm_handler.generate_chunk_description(
        #         current_chunk_text=content,
        #         previous_summaries=list(previous_summaries_window),
        #         filename=filename
        #     )

        #     enriched_content = f"[CONTEXT DESCRIPTION]:\n{description}\n\n[ORIGINAL TEXT]:\n{content}"

        #     final_chunks.append(
        #         Chunk(
        #             content=enriched_content,
        #             metadata=Metadata(
        #                 filename=filename,
        #                 order_number_of_chunk=chunk_order
        #             ),
        #         )
        #     )

        #     previous_summaries_window.append(description)

        # logger.opt(colors=True).info(
        #     f"Advanced text chunking completed: <yellow>{len(final_chunks)}</yellow> chunks created for file: <green>{filename}</green>"
        # )
        # return final_chunks

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
        name="Split Text",
        metadata={
            "ls_provider": "Chunker",
            "type": "sequential",
            "mode": "Baseline",
        },
    )
    def __split_text_baseline(self, text: str) -> list[str]:
        """Split text into chunks using the configured splitter.

        Args:
            text: The input text to split.

        Returns:
            A list of text chunks.
        """
        return self.splitter.split_text(text)
