"""Chat page - Talk with ChatBot."""

# Standard library imports
from datetime import datetime

# Third-party imports
import requests
import streamlit as st

# Local imports
from chatbot.globals import DEFAULT_API_HOST, api_settings, ui_settings
from chatbot.models.chat import Chat


API_URL = f"http://{DEFAULT_API_HOST}:{api_settings.PORT}"
TIMEOUT = 300


# Configure page
st.set_page_config(
    page_title=ui_settings.UI_CHAT_PAGE_TITLE,
    page_icon=ui_settings.UI_CHAT_PAGE_ICON,
    layout="centered",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []


def receive_answer_from_RAG(user_input: str) -> str:
    """Send message to ChatBot API and get response."""
    try:
        response = requests.get(
            f"{API_URL}/chat/query_baseline",
            params={
                "user_input": user_input,
            },
            timeout=TIMEOUT,
        )

        response.raise_for_status()

        data: Chat = Chat(**response.json())

        return data.llm_response
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to API server. Make sure it's running."
    except requests.exceptions.Timeout:
        return f"Error: API request timeout ({TIMEOUT} seconds)"
    except Exception as e:
        return f"Error: {str(e)}"


# Title
st.title("Find answers in your documents")
st.markdown("---")

# Chat display area
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])

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
        bot_response = receive_answer_from_RAG(user_input)

    # Add bot message
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": bot_response,
            "timestamp": datetime.now().isoformat(),
        }
    )

    # Rerun to update chat
    st.rerun()

# Sidebar
st.sidebar.title("Options")
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()
