# Third-party imports
import requests
import streamlit as st

# Local imports
from chatbot.globals import DEFAULT_API_URL_FOR_UI, TIMEOUT_UPLOAD_REQUEST


def upload_document_to_collection(
    collection_name: str, uploaded_file
) -> dict | None:
    """
    Upload a single document to the vector database.

    Args:
        collection_name: Name of the collection to upload to
        uploaded_file: Streamlit uploaded file object

    Returns:
        Dictionary with upload result or None if failed
    """
    try:
        # Prepare file for upload
        files = {
            "upload_file": (uploaded_file.name, uploaded_file.getbuffer())
        }

        response = requests.post(
            f"{DEFAULT_API_URL_FOR_UI}/database/process/",
            params={"collection_name": collection_name},
            files=files,
            timeout=TIMEOUT_UPLOAD_REQUEST,
        )
        response.raise_for_status()

        result = response.json()
        st.success(
            f"File '{uploaded_file.name}' uploaded successfully! "
            f"({len(result)} chunk(s) created)"
        )
        return result

    except requests.exceptions.HTTPError as e:
        st.error(f"Upload failed: {e.response.json().get('detail', str(e))}")
        return None
    except requests.exceptions.Timeout:
        st.error(
            f"Upload timeout (>{TIMEOUT_UPLOAD_REQUEST}s). File may be too large or API is slow."
        )
        return None
    except Exception as e:
        st.error(f"Failed to upload document: {e}")
        return None


def upload_documents_batch_to_collection(
    collection_name: str, uploaded_files: list
) -> dict | None:
    """
    Upload multiple documents to the vector database.

    Args:
        collection_name: Name of the collection to upload to
        uploaded_files: List of Streamlit uploaded file objects

    Returns:
        Dictionary with upload results or None if failed
    """
    try:
        # Prepare files for batch upload
        files = [
            ("upload_files", (file.name, file.getbuffer()))
            for file in uploaded_files
        ]

        response = requests.post(
            f"{DEFAULT_API_URL_FOR_UI}/database/process_batch/",
            params={"collection_name": collection_name},
            files=files,
            timeout=TIMEOUT_UPLOAD_REQUEST,
        )
        response.raise_for_status()

        result = response.json()
        total_chunks = sum(len(chunks) for chunks in result)

        st.success(
            f"{len(uploaded_files)} file(s) uploaded successfully! "
            f"({total_chunks} chunk(s) created total)"
        )
        return result

    except requests.exceptions.HTTPError as e:
        st.error(
            f"Batch upload failed: {e.response.json().get('detail', str(e))}"
        )
        return None
    except requests.exceptions.Timeout:
        st.error(
            f"Upload timeout (>{TIMEOUT_UPLOAD_REQUEST}s). Files may be too large or API is slow."
        )
        return None
    except Exception as e:
        st.error(f"Failed to upload documents: {e}")
        return None
