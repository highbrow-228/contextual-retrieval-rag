# Third-party imports
from fastapi import APIRouter, HTTPException, Query, UploadFile
from loguru import logger

# Local imports
from chatbot.config.db_config import CollectionName
from chatbot.config.status_config import Status
from chatbot.models.chunk import Chunk
from chatbot.models.database import (
    CreateCollectionSchema,
    DeleteByFilenameSchema,
    DeleteFromCollectionSchema,
)
from chatbot.utils.log_utils import log_endpoint_execution
from chatbot.vector_db.vector_db_client import (
    DocumentDeletionError,
    DocumentNotFoundError,
)


router = APIRouter()


@router.get("/collections", response_model=list[str])
@log_endpoint_execution
async def list_collections_route() -> list[str]:
    """
    List available collection names.

    Returns
    -------
    - **list[str]**: A list of collection names available in the vector database.

    Raises
    ------
    - **HTTPException**: If an error occurs while listing collections.
    """
    # Local imports
    from chatbot.api.app import pipeline

    try:
        cols = pipeline.db_handler._list_collections()
        return [c.name if hasattr(c, "name") else str(c) for c in cols]
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error listing collections: {e}"
        )


@router.post("/create_collection/", response_model=CreateCollectionSchema)
@log_endpoint_execution
async def create_collection_route(
    collection_name: CollectionName = Query(
        ..., description="Name of the collection to create"
    )
) -> CreateCollectionSchema:
    """
    Create a new collection in the vector database.

    Parameters
    ----------
    - **collection_name** (str): Name of the collection to create.

    Returns
    -------
    - **CreateCollectionSchema**: Status of the collection creation operation.

    Raises
    ------
    - **HTTPException**: If collection creation fails.
    """
    # Local imports
    from chatbot.api.app import pipeline

    try:
        logger.opt(colors=True).info(
            f"Creating new collection: <green>{collection_name}</green>"
        )

        pipeline.db_handler._get_or_create_collection(collection_name)

        logger.opt(colors=True).info(
            f"Collection <green>{collection_name}</green> created successfully."
        )

        return CreateCollectionSchema(
            collection_name=collection_name,
            status=Status.SUCCESS,
            message=f"Collection '{collection_name}' created successfully.",
            error=None,
        )

    except Exception as e:
        logger.opt(colors=True).error(
            f"Error creating collection <green>{collection_name}</green>: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error creating collection: {str(e)}",
        )


@router.post("/process/", response_model=list[Chunk])
@log_endpoint_execution
async def process(
    collection_name: CollectionName, upload_file: UploadFile
) -> list[Chunk]:
    """
    Process a single uploaded document and add its chunks to the vector database.

    Supported file formats
    ----------------------
    - pdf: Process via Docling PDF pipeline and extract structured text (markdown)
    - txt: Read as plain text
    - md: Read as plain markdown text
    - docx: Extract text using python-docx when available
    - doc: Binary Word documents (.doc) are supported via conversion to .docx using LibreOffice

    Parameters
    ----------
    - **collection_name** (CollectionName): The target collection to store the document's chunks.
    - **upload_file** (UploadFile): The file uploaded by the user to be processed.

    Returns
    -------
    - **list[Chunk]**: A list of Chunk objects representing the processed document chunks.

    Raises
    ------
    - **HTTPException**: If processing fails due to an internal error.
    """
    # Local imports
    from chatbot.api.app import pipeline

    try:
        return await pipeline.process_document(
            upload_file, collection_name=collection_name
        )
    except ValueError as e:
        # Gracefully return a bad request when the processor raises a ValueError
        # (this commonly happens for unsupported file formats or missing system deps)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Generic server error for unexpected issues
        raise HTTPException(
            status_code=500, detail=f"Error processing file: {str(e)}"
        )


@router.post("/process_batch/", response_model=list[list[Chunk]])
@log_endpoint_execution
async def process_batch(
    collection_name: CollectionName, upload_files: list[UploadFile]
) -> list[list[Chunk]]:
    """
    Process multiple uploaded documents and add their chunks to the vector database.

    Parameters
    ----------
    - **collection_name** (CollectionName): The target collection to store uploaded chunks.
    - **upload_files** (list[UploadFile]): A list of files to be processed.

    Returns
    -------
    - **list[list[Chunk]]**: A list of lists, where each inner list contains chunks for a single uploaded file.

    Raises
    ------
    HTTPException
        If processing fails for any reason.
    """
    logger.opt(colors=True).debug(
        f"Received <yellow>{len(upload_files)}</yellow> files for process_batch. Filenames: <yellow>{[file.filename for file in upload_files]}</yellow>"
    )

    # Local imports
    from chatbot.api.app import pipeline

    try:
        return await pipeline.process_batch_documents(
            upload_files=upload_files, collection_name=collection_name
        )
    except ValueError as e:
        # Bad request if conversion error occurs (e.g., missing dependency)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing batch: {str(e)}"
        )


@router.get("/document", response_model=Chunk)
@log_endpoint_execution
async def get_single_document_route(
    collection_name: CollectionName,
    doc_id: str = Query(
        ...,
        description="Document ID to get",
        example=["450d5fe7-33a0-4344-92a3-f3e980547cd6"],
    ),
) -> Chunk:
    """
    Retrieve a single document from a collection by its ID.

    Parameters
    ----------
    - **collection_name** (CollectionName): The collection containing the document.
    - **doc_id** (str): The document ID to retrieve.

    Returns
    -------
    - **Chunk**: The chunk object representing the stored document.

    Raises
    ------
    - **HTTPException**: 404 if the document is not found, 500 if another error occurs.
    """
    # Local imports
    from chatbot.api.app import pipeline

    try:
        return pipeline.db_handler._get_single_doc(
            collection_name=collection_name, doc_id=doc_id
        )

    except DocumentNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving document: {str(e)}"
        )


@router.get("/documents", response_model=list[Chunk])
@log_endpoint_execution
async def get_all_documents_route(
    collection_name: CollectionName,
    need_embeddings: bool = Query(
        False,
        description="Whether to include embeddings in the returned documents",
    ),
) -> list[Chunk]:
    """
    Retrieve all documents (chunks) from a collection.

    Parameters
    ----------
    - **collection_name** (CollectionName): Name of the collection to query.
    - **need_embeddings** (bool, optional): Whether embeddings should be included in returned objects.

    Returns
    -------
    - **list[Chunk]**: A list of chunk objects contained in the collection.

    Raises
    ------
    - **HTTPException**: If there is an error fetching documents.
    """
    # Local imports
    from chatbot.api.app import pipeline

    try:
        return pipeline.db_handler._get_all_from_collection(
            collection_name, need_embeddings
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving documents: {e}"
        )


@router.delete("/document", response_model=DeleteFromCollectionSchema)
@log_endpoint_execution
async def delete_document_route(
    collection_name: CollectionName,
    doc_id: str = Query(
        ...,
        description="Document ID to delete",
        example="450d5fe7-33a0-4344-92a3-f3e980547cd6",
    ),
) -> DeleteFromCollectionSchema:
    """
    Delete a document from a collection by its ID.

    Parameters
    ----------
    - **collection_name** (CollectionName): Name of collection to delete from.
    - **doc_id** (str): The document id to delete.

    Returns
    -------
    - **DeleteFromCollectionSchema**: Schema containing operation result and possible error.

    Raises
    ------
    - **HTTPException**: 404 if document not found, 500 for other errors.
    """
    # Local imports
    from chatbot.api.app import pipeline

    try:
        pipeline.db_handler._delete_from_collection(
            collection_name=collection_name, doc_id=doc_id
        )
        return DeleteFromCollectionSchema(
            collection_name=collection_name,
            doc_id=doc_id,
            status=Status.SUCCESS,
            error=None,
        )
    except DocumentNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        )
    except DocumentDeletionError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting document: {str(e)}"
        )


@router.delete("/documents/batch", response_model=dict)
@log_endpoint_execution
async def delete_documents_batch_route(
    collection_name: CollectionName,
    doc_ids: list[str] = Query(
        ...,
        description="List of document IDs to delete",
        example=["450d5fe7-33a0-4344-92a3-f3e980547cd6"],
    ),
) -> dict:
    """
    Delete multiple documents from a collection by their IDs (batch deletion).

    This is faster than deleting one-by-one because it reduces repeated API overhead.

    Parameters
    ----------
    - **collection_name** (CollectionName): Name of the collection containing documents to remove.
    - **doc_ids** (list[str]): A list of document IDs to delete.

    Returns
    -------
    - **dict**: Summary of deletion results: {
            "total": int,
            "successful": int,
            "failed": int,
            "errors": list[str]
        }

    Raises
    ------
    - **HTTPException**: 400 if no IDs provided, 500 on other errors.
    """
    # Local imports
    from chatbot.api.app import pipeline

    try:
        if not doc_ids:
            raise HTTPException(
                status_code=400, detail="No document IDs provided"
            )

        logger.opt(colors=True).info(
            f"Starting batch deletion of <cyan>{len(doc_ids)}</cyan> documents from collection <yellow>{collection_name.value}</yellow>"
        )

        successful = 0
        failed = 0
        errors = []

        # Delete all documents
        for doc_id in doc_ids:
            try:
                pipeline.db_handler._delete_from_collection(
                    collection_name=collection_name, doc_id=doc_id
                )
                successful += 1
            except Exception as e:
                failed += 1
                error_msg = f"Failed to delete {doc_id}: {str(e)}"
                errors.append(error_msg)
                logger.opt(colors=True).warning(f"<red>{error_msg}</red>")

        logger.opt(colors=True).info(
            f"Batch deletion completed. Total: <cyan>{len(doc_ids)}</cyan>, Successful: <green>{successful}</green>, Failed: <red>{failed}</red>"
        )

        return {
            "total": len(doc_ids),
            "successful": successful,
            "failed": failed,
            "errors": errors,
        }
    except Exception as e:
        logger.opt(colors=True).error(
            f"Batch deletion error: <red>{str(e)}</red>"
        )
        raise HTTPException(
            status_code=500, detail=f"Error during batch deletion: {str(e)}"
        )


@router.delete("/documents/by_filename", response_model=DeleteByFilenameSchema)
@log_endpoint_execution
async def delete_documents_by_filename(
    collection_name: CollectionName,
    filename: str = Query(
        ...,
        description="Filename to delete documents by",
        example="example_document.pdf",
    ),
) -> DeleteByFilenameSchema:
    """
    Delete all documents from a collection that match the given filename.

    Parameters
    ----------
    - **collection_name** (CollectionName): The collection name to delete documents from.
    - **filename** (str): The filename to match against stored document metadata.

    Returns
    -------
    - **DeleteByFilenameSchema**: Schema describing deletion result (deleted count and status).

    Raises
    ------
    - **HTTPException**: If deletion fails.
    """
    # Local imports
    from chatbot.api.app import pipeline

    try:
        return pipeline.db_handler._delete_by_filename(
            collection_name=collection_name, filename=filename
        )
    except Exception as e:
        logger.opt(colors=True).error(
            f"Error deleting documents by filename <yellow>{filename}</yellow> from collection <green>{collection_name.value}</green>: <red>{str(e)}</red>"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting documents by filename: {str(e)}",
        )


@router.delete(
    "/all_documents", response_model=list[DeleteFromCollectionSchema]
)
@log_endpoint_execution
async def delete_collection_route(
    collection_name: CollectionName,
) -> list[DeleteFromCollectionSchema]:
    """
    Delete all documents from a collection.

    Parameters
    ----------
    - **collection_name** (CollectionName): Name of the collection to clear.

    Returns
    -------
    - **list[DeleteFromCollectionSchema]**: List where each item represents deletion status per document.

    Raises
    ------
    - **HTTPException**: If errors occur during deletion.
    """
    # Local imports
    from chatbot.api.app import pipeline

    try:
        # Get all documents and delete them
        chunks = pipeline.db_handler._get_all_from_collection(collection_name)

        if not chunks:
            logger.opt(colors=True).info(
                f"Collection <green>{collection_name.value}</green> is already empty."
            )
            return []

        # Delete each document and collect results
        deletion_results: list[DeleteFromCollectionSchema] = []
        for chunk in chunks:
            try:
                pipeline.db_handler._delete_from_collection(
                    collection_name=collection_name, doc_id=chunk.id
                )
                deletion_results.append(
                    DeleteFromCollectionSchema(
                        collection_name=collection_name,
                        doc_id=chunk.id,
                        status=Status.SUCCESS,
                        error=None,
                    )
                )
            except Exception as e:
                logger.opt(colors=True).error(
                    f"Error deleting document <yellow>{chunk.id}</yellow>: {e}"
                )
                deletion_results.append(
                    DeleteFromCollectionSchema(
                        collection_name=collection_name,
                        doc_id=chunk.id,
                        status=Status.FAILED,
                        error=str(e),
                    )
                )

        logger.opt(colors=True).info(
            f"Collection clear operation completed. Total: <cyan>{len(deletion_results)}</cyan>, Successful: <cyan>{sum(1 for r in deletion_results if r.status == Status.SUCCESS)}</cyan>"
        )

        return deletion_results

    except Exception as e:
        logger.opt(colors=True).error(
            f"Error clearing collection <green>{collection_name.value}</green>: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing collection: {str(e)}",
        )
