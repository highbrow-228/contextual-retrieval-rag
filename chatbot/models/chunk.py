# Standard library imports
from datetime import datetime, timezone
from typing import Dict, Optional
from uuid import uuid4

# Third-party imports
from pydantic import BaseModel, Field, field_serializer

# Local imports
from chatbot.globals import db_settings


class Metadata(BaseModel):
    filename: Optional[str] = Field(
        default=None,
        description="Name of the source file",
    )

    order_number_of_chunk: Optional[int] = Field(
        default=None,
        description="Order number of the chunk in the original document",
    )

    date_added: datetime | str = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Date when the chunk was added",
    )

    @field_serializer("date_added")
    def serialize_date_added(self, value: datetime | str) -> str:
        """Serialize datetime to ISO format string (UTC)."""
        # If value is already a string, return it as is
        if isinstance(value, str):
            return value
        
        # If value is a datetime without timezone info, add UTC
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        
        return value.strftime(db_settings.DATA_FORMAT)

    @classmethod
    def from_dict(cls, data: Dict) -> "Metadata":
        """Create Metadata instance from dictionary (e.g., from database)."""
        if isinstance(data.get("date_added"), str):
            try:
                data["date_added"] = datetime.fromisoformat(data["date_added"])
            except (ValueError, TypeError):
                # If parsing fails, use current time
                data["date_added"] = datetime.now(timezone.utc)
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
