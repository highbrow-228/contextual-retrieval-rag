"""Documents page - Manage documents in the database."""

# Standard library imports
from datetime import datetime
from typing import Any

# Third-party imports
import streamlit as st

# Local imports
from chatbot.globals import db_settings, ui_settings
from chatbot.models.chunk import Chunk
from chatbot.models.database import DeleteByFilenameSchema
from ui.utils.delete_documents import (
    delete_all_documents,
    delete_document_by_filename,
    delete_document_by_id,
)
from ui.utils.getters import (
    get_collection_stats,
    get_collections,
    get_documents,
)
from ui.utils.upload_documents import (
    upload_document_to_collection,
    upload_documents_batch_to_collection,
)


# Configure page
st.set_page_config(
    page_title=ui_settings.UI_DOCUMENT_PAGE_TITLE,
    page_icon=ui_settings.UI_DOCUMENT_PAGE_ICON,
    layout="wide",
)

st.title(ui_settings.UI_DOCUMENT_PAGE_TITLE)
st.markdown("---")

# Select collection
collections = get_collections()

if not collections:
    st.warning("No collections found. Make sure the API server is running.")
else:
    selected_collection = st.selectbox(
        "Select Collection", collections, key="collection_select"
    )

    # Get stats
    stats = get_collection_stats(selected_collection)

    # Display metrics
    if stats:
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Documents", stats.get("document_count", 0))

        with col2:
            st.metric("Total Symbols", stats.get("total_symbols", 0))

    st.divider()

    # Upload section - available even if no documents
    st.subheader("Upload Documents")

    upload_tab1, upload_tab2 = st.tabs(["Single File", "Multiple Files"])

    with upload_tab1:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "txt", "docx", "doc"],
            key="single_file_uploader",
        )

        if uploaded_file is not None:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{uploaded_file.name}")
            with col2:
                if st.button(
                    "Upload",
                    type="primary",
                    key="single_upload_btn",
                ):
                    with st.spinner(
                        text="Processing and uploading...",
                        show_time=True,
                    ):
                        result = upload_document_to_collection(
                            selected_collection, uploaded_file
                        )
                        if result is not None:
                            st.rerun()

    with upload_tab2:
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "txt", "docx", "doc"],
            accept_multiple_files=True,
            key="batch_file_uploader",
        )

        if uploaded_files:
            st.write(f"{len(uploaded_files)} file(s) selected:")
            for file in uploaded_files:
                st.caption(f"{file.name}")

            if st.button(
                "Upload All",
                type="primary",
                key="batch_upload_btn",
            ):
                with st.spinner(
                    text="Processing and uploading...",
                    show_time=True,
                ):
                    result = upload_documents_batch_to_collection(
                        selected_collection, uploaded_files
                    )
                    if result is not None:
                        st.rerun()

    st.divider()

    # Get and display documents
    documents: list[Chunk] = get_documents(selected_collection)

    if documents:
        st.subheader(f"Documents in '{selected_collection}'")

        # Create a table view
        doc_data: dict[str, Any] = []
        for doc in documents:
            date_added = doc.metadata.date_added or "N/A"
            if isinstance(date_added, str) or not isinstance(
                date_added, datetime
            ):
                try:
                    date_obj = datetime.fromisoformat(date_added)
                    date_added = date_obj.strftime(db_settings.DATA_FORMAT)
                except (ValueError, AttributeError):
                    pass

            doc_data.append(
                {
                    "ID": doc.id or "N/A",
                    "Source": doc.metadata.filename or "N/A",
                    "Date Added": doc.metadata.date_added or "N/A",
                    "Content": doc.content or "N/A",
                }
            )

        st.dataframe(
            doc_data,
            column_config={
                "ID": st.column_config.TextColumn(width="small"),
                "Source": st.column_config.TextColumn(width="small"),
                "Date Added": st.column_config.TextColumn(width="small"),
                "Content": st.column_config.TextColumn(width="large"),
            },
            hide_index=False,
        )

        st.divider()

        # Delete options
        st.subheader("Delete Documents")

        # Create tabs for different deletion options
        delete_tab1, delete_tab2, delete_tab3 = st.tabs(
            ["Delete by ID", "Delete by Filename", "Delete All"]
        )

        # Tab 1: Delete single document by ID
        with delete_tab1:
            col1, col2 = st.columns([3, 1])

            with col1:
                selected_doc_id = st.selectbox(
                    "Select document to delete",
                    options=[doc.id for doc in documents],
                    format_func=lambda doc_id: f"{doc_id}  ({next((d.metadata.filename for d in documents if d.id == doc_id), 'Unknown')})",
                    key="delete_by_id_select",
                )

            with col2:
                st.write("")  # Spacer for alignment
                st.write("")  # Spacer for alignment

                if st.button(
                    "🗑️ Delete Document",
                    type="secondary",
                    key="delete_single_btn",
                ):
                    with st.spinner(
                        text="Deleting document...", show_time=True
                    ):
                        delete_document_by_id(
                            selected_collection, selected_doc_id
                        )
                        st.success("Document deleted successfully!")
                        st.rerun()

        # Tab 2: Delete all documents by filename
        with delete_tab2:
            # Get unique filenames
            unique_filenames = sorted(
                set(
                    doc.metadata.filename
                    for doc in documents
                    if doc.metadata.filename
                )
            )

            if unique_filenames:
                col1, col2 = st.columns([3, 1])

                with col1:
                    selected_filename = st.selectbox(
                        "Select filename to delete",
                        options=unique_filenames,
                        key="delete_by_filename_select",
                    )

                with col2:
                    st.write("")  # Spacer for alignment
                    st.write("")  # Spacer for alignment

                    if st.button(
                        "🗑️ Delete by Filename",
                        type="secondary",
                        key="delete_by_filename_btn",
                    ):
                        with st.spinner(
                            text=f"Deleting documents from {selected_filename}...",
                            show_time=True,
                        ):
                            temp_result: dict | None = (
                                delete_document_by_filename(
                                    selected_collection, selected_filename
                                )
                            )

                            result: DeleteByFilenameSchema | None = (
                                DeleteByFilenameSchema(**temp_result)
                                if temp_result
                                else None
                            )

                            if isinstance(result, DeleteByFilenameSchema):
                                if result and result.successful > 0:
                                    st.success(
                                        f'Documents from "{result.filename}" deleted successfully!\n\n'
                                        f'Collection: "{result.collection_name}"\n\n'
                                        f"Delete Summary:\n"
                                        f"  - Total: {result.total}\n"
                                        f"  - Successful: {result.successful}\n"
                                        f"  - Failed: {result.failed}"
                                    )
                                elif result and result.successful == 0:
                                    st.warning(
                                        f'No documents were deleted for "{result.filename}".\n'
                                        f"Total found: {result.total}, Failed: {result.failed}"
                                        f'Collection: "{result.collection_name}"'
                                    )
                                else:
                                    st.error(
                                        "Failed to delete documents. Please try again."
                                    )
                            else:
                                st.error(
                                    "Failed to delete documents. Please try again."
                                    f"Result type: {type(result)} and value: {result}"
                                )
                            st.rerun()

        # Tab 3: Delete all documents
        with delete_tab3:
            st.warning(
                "⚠️ This action will delete ALL documents in the collection and cannot be undone!"
            )

            if st.button(
                "🗑️ Delete All Documents",
                type="secondary",
                key="delete_all_btn",
            ):
                if st.session_state.get("confirm_delete_all"):
                    with st.spinner(
                        text="Deleting all documents...", show_time=True
                    ):
                        result: bool = delete_all_documents(
                            selected_collection
                        )
                        if result:
                            st.success(
                                f"Deleted {result['successful']} document(s) from '{selected_collection}'"
                            )
                        else:
                            st.info(
                                "No documents to delete or something went wrong."
                            )
                        st.session_state.confirm_delete_all = False
                        st.rerun()
                else:
                    st.session_state.confirm_delete_all = True
                    st.warning("Click the button again to confirm deletion")

            if st.session_state.get("confirm_delete_all"):
                if st.button("Cancel", key="cancel_delete_all"):
                    st.session_state.confirm_delete_all = False
                    st.rerun()

    else:
        st.info("No documents found in this collection.")
