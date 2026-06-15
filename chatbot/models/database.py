# Third-party imports
from pydantic import BaseModel, Field

# Local imports
from chatbot.config.db_config import CollectionName
from chatbot.config.status_config import Status


class CollectionNameBase(BaseModel):
    collection_name: CollectionName = Field(
        CollectionName.BASELINE_DOCUMENTS, description="Name of the collection"
    )


class StatusBase(BaseModel):
    status: Status = Field(..., description="Status of the operation")
    error: str | None = Field(
        None, description="Error message if the operation failed"
    )


class DeleteFromCollectionSchema(CollectionNameBase, StatusBase):
    doc_id: str = Field(..., description="ID of the document to delete")


class UpdateCollectionSchema(CollectionNameBase, StatusBase):
    updated_count: int = Field(..., description="Number of documents updated")
    ids: list[str] = Field(..., description="List of updated document IDs")


class PrepareDataSchema(BaseModel):
    ids: list[str] = Field(..., description="List of document IDs")
    documents: list[str] = Field(..., description="List of document contents")
    metadatas: list[dict] = Field(..., description="List of document metadata")
    embeddings: list[list[float]] = Field(
        ..., description="List of document embeddings"
    )


class DeleteByFilenameSchema(CollectionNameBase):
    filename: str = Field(..., description="Filename used for deletion")
    total: int = Field(
        ..., description="Total documents matching the filename"
    )
    successful: int = Field(
        ..., description="Number of successfully deleted documents"
    )
    failed: int = Field(
        ..., description="Number of documents that failed to delete"
    )
    errors: list[str] = Field(..., description="List of error messages if any")


class CreateCollectionSchema(CollectionNameBase, StatusBase):
    message: str = Field(
        ..., description="Detailed message about the operation"
    )
