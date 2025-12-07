"""Type-safe session state management for Streamlit."""

# Standard library imports
from typing import TypedDict

# Third-party imports
import streamlit as st

# Local imports
from chatbot.config.pipeline_config import PipelineMode


class ChatSessionState(TypedDict):
    """Type definition for chat page session state."""

    messages: list
    rag_mode: PipelineMode


def init_session_state() -> ChatSessionState:
    """Initialize and return typed session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "rag_mode" not in st.session_state:
        st.session_state.rag_mode = PipelineMode.BASELINE

    return {
        "messages": st.session_state.messages,
        "rag_mode": st.session_state.rag_mode,
    }


def get_session_state() -> ChatSessionState:
    """Get current typed session state."""
    return {
        "messages": st.session_state.messages,
        "rag_mode": st.session_state.rag_mode,
    }
