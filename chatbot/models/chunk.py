# Standard library imports
from datetime import datetime
from typing import Dict, Optional
from uuid import uuid4

# Third-party imports
from pydantic import BaseModel, Field, field_serializer


class Metadata(BaseModel):
    filename: Optional[str] = Field(
        default=None,
        description="Name of the source file",
    )

    order_number_of_chunk: Optional[int] = Field(
        default=None,
        description="Order number of the chunk in the original document",
    )

    markdown_header: Optional[str] = Field(
        default="",
        description="Metadata extracted from markdown headers",
    )

    markdown_section: Optional[str] = Field(
        default="",
        description="Metadata extracted from markdown sections",
    )

    date_added: datetime = Field(
        default_factory=datetime.now,
        description="Date when the chunk was added",
    )

    @field_serializer("date_added")
    def serialize_date_added(self, value: datetime) -> str:
        """Serialize datetime to ISO format string."""
        return value.isoformat()

    @classmethod
    def from_dict(cls, data: Dict) -> "Metadata":
        """Create Metadata instance from dictionary (e.g., from database)."""
        if isinstance(data.get("date_added"), str):
            data["date_added"] = datetime.fromisoformat(data["date_added"])
        return cls(**data)


class Chunk(BaseModel):
    id: Optional[str] = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the chunk",
    )

    content: Optional[str] = Field(
        default=None,
        description="Content of the text chunk",
    )

    metadata: Metadata = Field(
        default_factory=Metadata,
        description="Metadata associated with the chunk",
    )

    embeddings: Optional[list[float]] = Field(
        default=None,
        description="Embedding vector for the chunk",
    )
