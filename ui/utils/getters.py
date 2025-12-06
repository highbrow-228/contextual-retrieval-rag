# Third-party imports
import requests
import streamlit as st

# Local imports
from chatbot.globals import DEFAULT_API_URL_FOR_UI, TIMEOUT_PER_REQUEST
from chatbot.models.chunk import Chunk


def get_collections() -> list[str]:
    """Fetch available collections from API."""
    try:
        response = requests.get(
            f"{DEFAULT_API_URL_FOR_UI}/database/collections",
            timeout=TIMEOUT_PER_REQUEST,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to API server. Make sure the API is running:\n"
            f"- Expected API at: `{DEFAULT_API_URL_FOR_UI}`\n"
            f"- You can check swagger docs at: `{DEFAULT_API_URL_FOR_UI}/docs`\n"
        )
        return []
    except requests.exceptions.Timeout:
        st.error(
            f"Request timeout ({TIMEOUT_PER_REQUEST}s). API server is slow or not responding."
        )
        return []
    except Exception as e:
        st.error(f"Failed to fetch collections: {e}")
        return []


def get_documents(collection_name: str) -> list[Chunk]:
    """Fetch all documents from a collection."""
    try:
        response = requests.get(
            f"{DEFAULT_API_URL_FOR_UI}/database/documents",
            params={
                "collection_name": collection_name,
                "need_embeddings": False,
            },
            timeout=TIMEOUT_PER_REQUEST,
        )
        response.raise_for_status()

        # Use list comprehension instead of loop - faster
        return [Chunk(**result) for result in response.json()]
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to API server. Make sure the API is running:\n"
            f"- Expected API at: `{DEFAULT_API_URL_FOR_UI}`\n"
            f"- You can check swagger docs at: `{DEFAULT_API_URL_FOR_UI}/docs`\n"
        )
        return []
    except requests.exceptions.Timeout:
        st.error(
            f"Request timeout ({TIMEOUT_PER_REQUEST}s). API server is slow or not responding."
        )
        return []
    except Exception as e:
        st.error(f"Failed to fetch documents: {e}")
        return []


def get_collection_stats(collection_name: str) -> dict:
    """Get statistics for a collection."""
    try:
        chunks: list[Chunk] = get_documents(collection_name)

        total_symbols = 0
        for chunk in chunks:
            content = chunk.content or ""
            total_symbols += len(content.split())

        return {
            "document_count": len(chunks),
            "total_symbols": total_symbols,
            "last_updated": chunk.metadata.date_added if chunks else None,
        }

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to API server. Make sure the API is running:\n"
            f"- Expected API at: `{DEFAULT_API_URL_FOR_UI}`\n"
            f"- You can check swagger docs at: `{DEFAULT_API_URL_FOR_UI}/docs`\n"
        )
        return {}
    except Exception as e:
        st.error(f"Failed to fetch stats: {e}")
        return {}
