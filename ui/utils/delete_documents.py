# Third-party imports
import requests
import streamlit as st

# Local imports
from chatbot.globals import DEFAULT_API_URL_FOR_UI, TIMEOUT_PER_REQUEST


def delete_document_by_id(
    selected_collection: str, selected_doc_id: str
) -> None:
    try:
        response = requests.delete(
            f"{DEFAULT_API_URL_FOR_UI}/database/document",
            params={
                "collection_name": selected_collection,
                "doc_id": selected_doc_id,
            },
            timeout=TIMEOUT_PER_REQUEST,
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            st.error("Document not found")
        else:
            st.error(
                f"Failed to delete document: {e.response.json().get('detail', str(e))}"
            )
    except Exception as e:
        st.error(f"Failed to delete document: {e}")


def delete_document_by_filename(
    selected_collection: str, selected_filename: str
) -> dict | None:
    try:
        response = requests.delete(
            f"{DEFAULT_API_URL_FOR_UI}/database/documents/by_filename",
            params={
                "collection_name": selected_collection,
                "filename": selected_filename,
            },
            timeout=TIMEOUT_PER_REQUEST,
        )
        response.raise_for_status()
        return response.json()

    except Exception as e:
        st.error(f"Failed to delete documents: {e}")


def delete_all_documents(selected_collection: str) -> bool:
    try:
        response = requests.delete(
            f"{DEFAULT_API_URL_FOR_UI}/database/all_documents",
            params={"collection_name": selected_collection},
            timeout=TIMEOUT_PER_REQUEST,
        )
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Failed to delete documents: {e}")
