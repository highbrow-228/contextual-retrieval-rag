"""ChromaDB client for vector database operations and document storage."""

# Standard library imports
import logging
from collections.abc import Sequence

# Third-party imports
import chromadb
import numpy as np
from chromadb.api.models.Collection import Collection
from chromadb.api.types import (
    GetResult,
    Include,
    IncludeMetadataDocuments,
    IncludeMetadataDocumentsDistances,
    IncludeMetadataDocumentsEmbeddings,
    QueryResult,
)
from langsmith import traceable
from loguru import logger

# Local imports
from chatbot.config.db_config import (
    VALID_DATABASE_MODES,
    CollectionName,
    DatabaseMode,
)
from chatbot.models.chunk import Chunk, Metadata
from chatbot.models.database import (
    DeleteByFilenameSchema,
    PrepareDataSchema,
    UpdateCollectionSchema,
)
from chatbot.models.rag import RAGQueryResult, RetrievedChunk
from chatbot.services.embedding.openai_embedder import OpenAIEmbedder
from chatbot.vector_db.base_vector_db import BaseVectorDB


# Disable verbose logging from chromadb and its dependencies
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


class DocumentNotFoundError(Exception):
    """Raised when a document is not found in the collection."""

    pass


class DocumentDeletionError(Exception):
    """Raised when a document cannot be deleted from the collection."""

    pass


class ChromaDBClient(BaseVectorDB):
    """Handles vector database operations."""

    def __init__(
        self,
        embedder: OpenAIEmbedder,
        host: str,
        port: int,
        tenant: str,
        api_key: str,
        mode: str,
        concurent_requests_for_upload: int,
        **kwargs,
    ) -> None:
        self.embedding_fn = embedder
        self.batch_size_for_upload = concurent_requests_for_upload

        match mode:
            case DatabaseMode.CLOUD.value:
                self.client = chromadb.CloudClient(
                    api_key=api_key, tenant=tenant, **kwargs
                )
            case DatabaseMode.SERVER.value:
                self.client = chromadb.HttpClient(
                    host=host, port=port, **kwargs
                )
            case _:
                raise ValueError(
                    f'Database mode "{mode}" is not recognized. '
                    f"Valid modes are: {VALID_DATABASE_MODES}"
                )

    @traceable(
        run_type="tool",
        name="Add to Collection",
        metadata={
            "type": "sequential",
            "ls_provider": "ChromaDBClient",
        },
    )
    def process(
        self,
        collection_name: CollectionName,
        data: list[Chunk],
        filename: str = "",
    ) -> Collection:
        """Process data for vector database operations."""
        if data is None:
            raise ValueError("Data must be provided to add to the collection.")

        if not isinstance(data, Sequence) or not all(
            isinstance(item, Chunk) for item in data
        ):
            raise ValueError(
                f"Data must be a list of Chunk instances, got {type(data).__name__}"
            )

        return self._add_to_collection(collection_name, data)

    def prepare_data(self, data: list[Chunk]) -> PrepareDataSchema:
        """Prepare data for database operations."""
        return PrepareDataSchema(
            ids=[chunk.id for chunk in data],
            documents=[chunk.content for chunk in data],
            metadatas=[chunk.metadata.model_dump() for chunk in data],
            embeddings=[chunk.embeddings for chunk in data],
        )

    # Public Query Methods
    @traceable(
        run_type="retriever",
        name="Retrieve",
        metadata={
            "type": "sequential",
            "ls_provider": "ChromaDBClient",
        },
    )
    def retrieve_baseline(
        self,
        query: str,
        collection_name: CollectionName,
        n_results: int = 5,
    ) -> RAGQueryResult:
        """
        Retrieve relevant chunks for a query as a structured Pydantic model.

        Args:
            query: The query text to search for
            collection_name: Name of the collection to query
            n_results: Number of results to retrieve (default: 5)

        Returns:
            RAGQueryResult containing retrieved chunks with relevance scores
        """
        result: QueryResult = self._query_collection(
            collection_name=collection_name,
            query_text=query,
            n_results=n_results,
            include=IncludeMetadataDocumentsDistances,
        )

        # Early exit if no results
        ids = result.get("ids")
        if not ids or not ids[0]:
            logger.opt(colors=True).warning(
                f"No chunks retrieved for query: <yellow>{query}</yellow>"
            )
            return RAGQueryResult(
                query=query,
                retrieved_chunks=[],
                total_chunks_retrieved=0,
                collection_name=collection_name.value,
            )

        # Unpack result data once
        ids_list = ids[0]
        num_results = len(ids_list)
        distances_list = result.get("distances", [[]])[0]
        documents_list = result.get("documents", [[]])[0]
        metadatas_list = result.get("metadatas", [[]])[0]

        # Use list comprehension for faster chunk creation
        retrieved_chunks = [
            RetrievedChunk(
                chunk=Chunk(
                    id=ids_list[i],
                    content=documents_list[i],
                    metadata=Metadata(
                        filename=metadatas_list[i].get(
                            "filename", "Not specified"
                        ),
                        order_number_of_chunk=metadatas_list[i].get(
                            "order_number_of_chunk", "Not specified"
                        ),
                        date_added=metadatas_list[i].get("date_added"),
                    ),
                ),
                relevance_score=distances_list[i],
                position=i,
            )
            for i in range(num_results)
        ]

        logger.opt(colors=True).info(
            f"Retrieved <cyan>{num_results}</cyan> chunk(s) for query: <yellow>{query}</yellow>"
        )

        return RAGQueryResult(
            query=query,
            retrieved_chunks=retrieved_chunks,
            total_chunks_retrieved=num_results,
            collection_name=collection_name.value,
        )

    ###  Private method to add data to a collection ###
    # Collection Management
    def _get_or_create_collection(
        self, collection_name: CollectionName
    ) -> Collection:
        """Get or create a collection with the specified name."""
        try:
            collection_value = collection_name.value
        except AttributeError:
            raise ValueError(
                f"Collection name must be a CollectionName enum, got {type(collection_name).__name__}"
            )

        return self.client.get_or_create_collection(
            name=collection_value, embedding_function=self.embedding_fn
        )

    def _delete_collection(self, collection_name: CollectionName) -> None:
        """Delete a collection by name."""
        return self.client.delete_collection(collection_name)

    def _list_collections(self) -> Sequence[Collection]:
        """List all available collections."""
        return self.client.list_collections()

    # Document Operations
    def _add_to_collection(
        self,
        collection_name: CollectionName,
        data: list[Chunk],
        filename: str = "",
    ) -> None:
        """Add data to a collection with batch size limit.

        ChromaDB has a maximum batch size of 100, so we split larger batches.
        """

        collection: Collection = self._get_or_create_collection(
            collection_name
        )

        # Split data into batches of max 100
        total_chunks = len(data)
        for i in range(0, total_chunks, self.batch_size_for_upload):
            batch = data[i : i + self.batch_size_for_upload]
            prepared_data: PrepareDataSchema = self.prepare_data(batch)

            collection.add(
                ids=prepared_data.ids,
                documents=prepared_data.documents,
                metadatas=prepared_data.metadatas,
                embeddings=prepared_data.embeddings,
            )

            logger.opt(colors=True).info(
                f"Added batch <cyan>{i // self.batch_size_for_upload + 1}</cyan> "
                f"(<yellow>{len(batch)} chunks</yellow>) to collection "
                f"<green>{collection_name.value}</green>"
            )

        logger.opt(colors=True).info(
            f"Completed adding <cyan>{total_chunks}</cyan> chunks to collection <green>{collection_name.value}</green> filename: <yellow>{filename}</yellow>"
        )

        return collection

    def _delete_from_collection(
        self, collection_name: CollectionName, doc_id: str
    ) -> None:
        """Delete a document from the collection by its ID.

        Args:
            collection_name: Name of the collection
            doc_id: Document ID to delete

        Raises:
            DocumentNotFoundError: If document doesn't exist
            DocumentDeletionError: If deletion fails
        """
        try:
            collection: Collection = self._get_or_create_collection(
                collection_name
            )

            # First check if document exists
            result: GetResult = collection.get(ids=[doc_id])
            if not result or not result.get("ids"):
                logger.opt(colors=True).warning(
                    f"Document with ID <yellow>{doc_id}</yellow> not found in <green>{collection_name.value}</green>."
                )
                raise DocumentNotFoundError(
                    f"Document with ID '{doc_id}' not found in collection '{collection_name.value}'"
                )

            # Delete the document
            collection.delete(ids=[doc_id])

            logger.opt(colors=True).info(
                f"Successfully deleted document <yellow>{doc_id}</yellow> from collection <green>{collection_name.value}</green>."
            )

        except DocumentNotFoundError:
            raise
        except Exception as e:
            logger.opt(colors=True).error(
                f"Error deleting document <yellow>{doc_id}</yellow> from <green>{collection_name.value}</green>: {e}"
            )
            raise DocumentDeletionError(
                f"Failed to delete document '{doc_id}' from collection '{collection_name.value}': {str(e)}"
            )

    def _delete_batch_from_collection(
        self, collection_name: CollectionName, doc_ids: list[str]
    ) -> DeleteByFilenameSchema:
        """Delete a batch of documents from the collection by their IDs.

        Args:
            collection_name: Name of the collection
            doc_ids: List of document IDs to delete

        Returns:
            Dictionary with deletion statistics:
            - total: Total documents to delete
            - successful: Number of successfully deleted documents
            - failed: Number of documents that failed to delete
            - errors: List of errors if any occurred

        Raises:
            DocumentDeletionError: If batch deletion operation fails
        """
        MAX_BATCH_SIZE = 900

        if not doc_ids:
            logger.opt(colors=True).warning(
                f"No document IDs provided for deletion from collection <green>{collection_name.value}</green>."
            )
            return DeleteByFilenameSchema(
                collection_name=collection_name,
                filename="batch_delete",
                total=0,
                successful=0,
                failed=0,
                errors=[],
            )

        try:
            collection: Collection = self._get_or_create_collection(
                collection_name
            )

            total_docs = len(doc_ids)
            successful_deletes = 0
            failed_deletes = 0
            errors = []

            # Split into batches of max 900
            for batch_idx, i in enumerate(
                range(0, total_docs, MAX_BATCH_SIZE)
            ):
                batch = doc_ids[i : i + MAX_BATCH_SIZE]

                try:
                    collection.delete(ids=batch)
                    successful_deletes += len(batch)

                    logger.opt(colors=True).info(
                        f"Deleted batch <cyan>{batch_idx + 1}</cyan> ({len(batch)} document(s)) from collection <green>{collection_name.value}</green>"
                    )
                except Exception as e:
                    failed_deletes += len(batch)
                    error_msg = (
                        f"Failed to delete batch {batch_idx + 1}: {str(e)}"
                    )
                    errors.append(error_msg)
                    logger.opt(colors=True).error(
                        f"Error deleting batch <cyan>{batch_idx + 1}</cyan> from <green>{collection_name.value}</green>: <red>{str(e)}</red>"
                    )

            logger.opt(colors=True).info(
                f"Batch deletion completed from collection <green>{collection_name.value}</green>: "
                f"<green>{successful_deletes}</green> successful, <red>{failed_deletes}</red> failed"
            )

            return DeleteByFilenameSchema(
                collection_name=collection_name,
                filename="batch_delete",
                total=total_docs,
                successful=successful_deletes,
                failed=failed_deletes,
                errors=errors,
            )

        except Exception as e:
            logger.opt(colors=True).error(
                f"Error deleting batch of documents from <green>{collection_name.value}</green>: <red>{str(e)}</red>"
            )
            raise DocumentDeletionError(
                f"Failed to delete batch of documents from collection '{collection_name.value}': {str(e)}"
            )

    def _delete_by_filename(
        self, collection_name: CollectionName, filename: str
    ) -> DeleteByFilenameSchema:
        """Delete all documents with the specified filename from the collection.

        Args:
            collection_name: Name of the collection
            filename: Filename to match in metadata

        Returns:
            Dictionary with deletion statistics:
            - total: Total documents with matching filename
            - successful: Number of successfully deleted documents
            - failed: Number of documents that failed to delete
            - errors: List of errors if any occurred

        Raises:
            DocumentDeletionError: If deletion operation fails
        """
        try:
            collection: Collection = self._get_or_create_collection(
                collection_name
            )

            # Get all documents from collection
            result: GetResult = collection.get(
                include=IncludeMetadataDocuments
            )

            if not result or not result.get("ids"):
                logger.opt(colors=True).info(
                    f"No documents found in collection <green>{collection_name.value}</green>."
                )
                return DeleteByFilenameSchema(
                    collection_name=collection_name,
                    filename=filename,
                    total=0,
                    successful=0,
                    failed=0,
                    errors=[],
                )

            # Find all document IDs with matching filename
            ids_to_delete = []
            metadatas = result.get("metadatas", [])

            for i, meta in enumerate(metadatas):
                if meta.get("filename") == filename:
                    ids_to_delete.append(result["ids"][i])

            if not ids_to_delete:
                logger.opt(colors=True).warning(
                    f"No documents with filename <yellow>{filename}</yellow> found in <green>{collection_name.value}</green>."
                )
                return DeleteByFilenameSchema(
                    collection_name=collection_name,
                    filename=filename,
                    total=0,
                    successful=0,
                    failed=0,
                    errors=[],
                )

            # Delete all matching documents
            successful_deletes = 0
            failed_deletes = 0
            errors = []

            for doc_id in ids_to_delete:
                try:
                    collection.delete(ids=[doc_id])
                    successful_deletes += 1
                except Exception as e:
                    failed_deletes += 1
                    error_msg = (
                        f"Failed to delete document '{doc_id}': {str(e)}"
                    )
                    errors.append(error_msg)
                    logger.opt(colors=True).error(
                        f"Error deleting document <yellow>{doc_id}</yellow>: {e}"
                    )

            logger.opt(colors=True).info(
                f"Deleted <cyan>{successful_deletes}</cyan> document(s) "
                f"Filename <yellow>{filename}</yellow> "
                f"From collection <green>{collection_name.value}</green>. "
                f"Failed: <red>{failed_deletes}</red>."
            )

            return DeleteByFilenameSchema(
                collection_name=collection_name,
                filename=filename,
                total=len(ids_to_delete),
                successful=successful_deletes,
                failed=failed_deletes,
                errors=errors,
            )

        except Exception as e:
            logger.opt(colors=True).error(
                f"Error deleting documents by filename <yellow>{filename}</yellow> from <green>{collection_name.value}</green>: {e}"
            )
            raise DocumentDeletionError(
                f"Failed to delete documents with filename '{filename}' from collection '{collection_name.value}': {str(e)}"
            )

    def _update_in_collection(
        self, collection_name: CollectionName, data: list[Chunk]
    ) -> UpdateCollectionSchema:
        """Update documents in the collection."""
        if data is None:
            raise ValueError("Data must be provided to update the collection.")

        if not isinstance(data, Sequence) or not all(
            isinstance(item, Chunk) for item in data
        ):
            raise ValueError(
                f"Data must be a list of Chunk instances, got {type(data).__name__}"
            )

        collection: Collection = self._get_or_create_collection(
            collection_name
        )

        prepared_data: PrepareDataSchema = self.prepare_data(data)

        try:
            collection.upsert(
                ids=prepared_data.ids,
                documents=prepared_data.documents,
                metadatas=prepared_data.metadatas,
                embeddings=prepared_data.embeddings,
            )

            logger.info(
                f'Successfully updated {len(prepared_data.ids)} document(s) in "{collection_name}"'
            )

            return UpdateCollectionSchema(
                status="Success",
                collection_name=collection_name,
                updated_count=len(prepared_data.ids),
                ids=prepared_data.ids,
            )
        except Exception as e:
            logger.error(
                f'Error updating documents in "{collection_name}": {e}'
            )
            return UpdateCollectionSchema(
                status="Failed",
                collection_name=collection_name,
                ids=prepared_data.ids,
                error=str(e),
            )

    # Query Operations
    def _query_collection(
        self,
        collection_name: CollectionName,
        query_text: str,
        n_results: int = 5,
        include: Include = IncludeMetadataDocumentsEmbeddings,
    ) -> QueryResult:
        """Query the collection with the given text."""
        collection: Collection = self._get_or_create_collection(
            collection_name
        )

        return collection.query(
            query_embeddings=self.embedding_fn.process(query_text),
            n_results=n_results,
            include=include,
        )

    def _get_single_doc(
        self, collection_name: CollectionName, doc_id: str
    ) -> Chunk:
        """Retrieve a single document by its ID from the collection.

        Args:
            collection_name: Name of the collection
            doc_id: Document ID to retrieve

        Returns:
            Chunk object with document data

        Raises:
            DocumentNotFoundError: If document is not found
        """
        try:
            collection: Collection = self._get_or_create_collection(
                collection_name
            )

            result: GetResult = collection.get(
                ids=[doc_id], include=IncludeMetadataDocumentsEmbeddings
            )

            if not result or not result.get("ids"):
                logger.opt(colors=True).warning(
                    f"Document with ID <yellow>{doc_id}</yellow> not found in <green>{collection_name.value}</green>."
                )
                raise DocumentNotFoundError(
                    f"Document with ID '{doc_id}' not found in collection '{collection_name.value}'"
                )

            logger.opt(colors=True).info(
                f"Retrieved document <yellow>{doc_id}</yellow> from collection <green>{collection_name.value}</green>."
            )

            # Get metadata safely
            metadata_dict = (
                result["metadatas"][0] if result.get("metadatas") else {}
            )
            metadata = Metadata.from_dict(metadata_dict)

            # Get embeddings safely
            embeddings = self.__extract_embeddings(result.get("embeddings"))

            return Chunk(
                id=doc_id,
                content=(
                    result["documents"][0] if result.get("documents") else None
                ),
                metadata=metadata,
                embeddings=embeddings,
            )

        except DocumentNotFoundError:
            raise
        except Exception as e:
            logger.opt(colors=True).error(
                f"Error retrieving document <yellow>{doc_id}</yellow> from <green>{collection_name.value}</green>: {e}"
            )
            raise

    def _get_all_from_collection(
        self, collection_name: CollectionName, need_embeddings: bool = True
    ) -> list[Chunk]:
        """Retrieve all documents from the collection.

        Args:
            collection_name: Name of the collection

        Returns:
            List of Chunk objects from the collection (empty list if no documents)
        """
        try:
            collection: Collection = self._get_or_create_collection(
                collection_name
            )

            if need_embeddings:
                result: GetResult = collection.get(
                    include=IncludeMetadataDocumentsEmbeddings
                )
            else:
                result: GetResult = collection.get(
                    include=IncludeMetadataDocuments
                )

            # If no documents found, return empty list
            if not result or not result.get("ids"):
                logger.opt(colors=True).info(
                    f"No documents found in collection <green>{collection_name.value}</green>."
                )
                return []

            chunks: list[Chunk] = self.__convert_get_result_to_chunks(result)

            logger.opt(colors=True).info(
                f"Retrieved <cyan>{len(chunks)}</cyan> document(s) from collection <green>{collection_name.value}</green>."
            )

            return chunks

        except Exception as e:
            logger.opt(colors=True).error(
                f"Error retrieving documents from <green>{collection_name.value}</green>: {e}"
            )
            raise

    ### Private helper methods ###
    def __extract_embeddings(
        self, embeddings_list: list | None
    ) -> list[float] | None:
        """Extract embeddings from query result.

        Args:
            embeddings_list: List of embeddings from Chroma result

        Returns:
            First embedding vector or None if not available
        """
        if (
            embeddings_list is not None
            and isinstance(embeddings_list, (list, np.ndarray))
            and len(embeddings_list) > 0
        ):
            first = embeddings_list[0]
            if isinstance(first, np.ndarray):
                return first.tolist()
            return first
        return None

    def __convert_get_result_to_chunks(self, result: GetResult) -> list[Chunk]:
        """Convert Chroma GetResult to list of Chunk objects.

        Args:
            result: GetResult from Chroma collection.get()

        Returns:
            List of Chunk objects

        Raises:
            ValueError: If result is empty or malformed
        """
        if not result or not result.get("ids"):
            raise ValueError("No documents found in result")

        chunks = []
        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])
        embeddings_list = result.get("embeddings", [])
        has_embeddings = (
            embeddings_list is not None and len(embeddings_list) > 0
        )

        for i, doc_id in enumerate(ids):
            metadata_dict = metadatas[i] if i < len(metadatas) else {}
            metadata = Metadata.from_dict(metadata_dict)

            # Extract embeddings only if available
            embeddings = None
            if has_embeddings and i < len(embeddings_list):
                embeddings = self.__extract_embeddings([embeddings_list[i]])
                if embeddings is None:
                    logger.opt(colors=True).debug(
                        f"Failed to extract embeddings for document <yellow>{doc_id}</yellow>"
                    )

            chunks.append(
                Chunk(
                    id=doc_id,
                    content=documents[i] if i < len(documents) else None,
                    metadata=metadata,
                    embeddings=embeddings,
                )
            )

        return chunks
