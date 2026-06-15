"""Chat page - Talk with ChatBot."""

# Standard library imports
from datetime import datetime

# Third-party imports
import streamlit as st

# Local imports
from chatbot.config.pipeline_config import PipelineMode
from ui.utils.chat import (
    init,
    receive_answer_from_advanced_RAG,
    receive_answer_from_baseline_RAG,
)


init()

# Chat display area
with st.container():
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            # Select avatar based on RAG mode stored with message
            rag_mode = message.get("rag_mode", PipelineMode.BASELINE)
            st.chat_message(
                name="assistant",
                avatar=(
                    st.session_state.baseline_avatar
                    if rag_mode == PipelineMode.BASELINE
                    else st.session_state.advanced_avatar
                ),
            ).write(message["content"])

# Input area
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([0.85, 0.15])

    with col1:
        user_input = st.text_input(
            "Your message:",
            placeholder="Type your message here...",
            label_visibility="collapsed",
        )

    with col2:
        send_button = st.form_submit_button(
            label="Send",
            use_container_width=True,
            type="primary",
        )

# Process input
if send_button and user_input:
    # Add user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat(),
        }
    )

    # Get bot response
    with st.spinner(
        text="Thinking...",
        show_time=True,
    ):
        match st.session_state.rag_mode:
            case PipelineMode.BASELINE:
                bot_response = receive_answer_from_baseline_RAG(user_input)
            case PipelineMode.ADVANCED:
                bot_response = receive_answer_from_advanced_RAG(user_input)
            case _:
                bot_response = f"Error: Invalid RAG mode selected. Available modes: {list(PipelineMode)}"

    # Add bot message with RAG mode info
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": bot_response,
            "timestamp": datetime.now().isoformat(),
            "rag_mode": st.session_state.rag_mode,
        }
    )

    # Rerun to update chat
    st.rerun()
