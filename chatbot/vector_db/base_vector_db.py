"""Abstract base class for vector database implementations."""

# Standard library imports
from abc import ABC, abstractmethod
from typing import Any, Dict, List

# Local imports
from chatbot.base.base_classes import BaseProcessor


class BaseVectorDB(BaseProcessor, ABC):
    """Abstract base class for vector database operations."""

    # Collection Management
    @abstractmethod
    def _get_or_create_collection(self, name: str) -> Any:
        """Get or create a collection with the specified name."""
        pass

    @abstractmethod
    def _delete_collection(self, name: str) -> None:
        """Delete a collection by name."""
        pass

    @abstractmethod
    def _list_collections(self) -> List[Any]:
        """List all available collections."""
        pass

    # Document Operations
    @abstractmethod
    def _add_to_collection(
        self, collection_name: str, data: Dict[str, Any]
    ) -> Any:
        """Add data to a collection."""
        pass

    @abstractmethod
    def _delete_from_collection(
        self, collection_name: str, doc_id: str
    ) -> None:
        """Delete a document from the collection by its ID."""
        pass

    @abstractmethod
    def _update_in_collection(
        self, collection_name: str, data: Dict[str, Any]
    ) -> None:
        """Update a document in the collection."""
        pass

    # Query Operations
    @abstractmethod
    def _query_collection(
        self, collection_name: str, query_texts: List[str], n_results: int = 5
    ) -> Dict[str, Any]:
        """Query the collection with the given text."""
        pass

    @abstractmethod
    def _get_single_doc(self, collection_name: str, doc_id: str) -> Any:
        """Retrieve a single document by its ID from the collection."""
        pass

    @abstractmethod
    def _get_all_from_collection(self, collection_name: str) -> Any:
        """Retrieve all documents from the collection."""
        pass
