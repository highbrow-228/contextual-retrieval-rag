"""Data models for RAG (Retrieval-Augmented Generation) operations."""

# Third-party imports
from pydantic import BaseModel, Field

# Local imports
from chatbot.models.chunk import Chunk


class RetrievedChunk(BaseModel):
    """Chunk with relevance score for RAG operations."""

    chunk: Chunk = Field(
        description="The actual chunk data with content and metadata"
    )

    relevance_score: float = Field(
        default=0.0,
        description="Similarity/relevance score (0-1 or distance metric). Lower distance = higher relevance",
    )

    position: int = Field(
        default=0,
        description="Rank position in retrieval results (0-based)",
    )


class RAGQueryResult(BaseModel):
    """Structured result from a RAG query to the vector database."""

    query: str = Field(description="The original query text")

    retrieved_chunks: list[RetrievedChunk] = Field(
        default_factory=list,
        description="List of retrieved chunks sorted by relevance",
    )

    total_chunks_retrieved: int = Field(
        default=0,
        description="Total number of chunks retrieved",
    )

    collection_name: str = Field(
        description="Name of the collection that was queried",
    )
