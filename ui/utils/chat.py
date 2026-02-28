# Third-party imports
import requests
import streamlit as st

# Local imports
from chatbot.config.pipeline_config import PipelineMode
from chatbot.globals import DEFAULT_API_HOST, api_settings, ui_settings
from chatbot.models.chat import Chat
from ui.utils.avatar_utils import get_avatar_path


API_URL = f"http://{DEFAULT_API_HOST}:{api_settings.PORT}"
TIMEOUT = 300

# Cache avatars to avoid recalculation
_AVATAR_CACHE = {}


def receive_answer_from_baseline_RAG(user_input: str) -> str:
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


def receive_answer_from_advanced_RAG(user_input: str) -> str:
    """Send message to ChatBot API and get response from advanced RAG."""
    try:
        response = requests.get(
            f"{API_URL}/chat/query_advanced",
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


def get_bot_avatar(rag_mode: PipelineMode) -> str:
    """Get avatar for the bot based on RAG mode (cached)."""
    if rag_mode in _AVATAR_CACHE:
        return _AVATAR_CACHE[rag_mode]

    if rag_mode == PipelineMode.BASELINE:
        avatar = ui_settings.UI_BASELINE_BOT_AVATAR
    else:
        avatar = ui_settings.UI_ADVANCED_BOT_AVATAR

    result = get_avatar_path(avatar)
    _AVATAR_CACHE[rag_mode] = result
    return result


@st.cache_resource
def _get_sidebar_index(rag_mode: PipelineMode) -> int:
    """Get selectbox index for current RAG mode."""
    return 0 if rag_mode == PipelineMode.BASELINE else 1


def init() -> None:
    """Initialize session state."""
    # Configure page FIRST (must be before any other st commands)
    st.set_page_config(
        page_title=ui_settings.UI_CHAT_PAGE_TITLE,
        page_icon=ui_settings.UI_CHAT_PAGE_ICON,
        layout="centered",
    )

    # Initialize session state with defaults (only once)
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.rag_mode = PipelineMode.BASELINE
        st.session_state.baseline_avatar = get_bot_avatar(
            PipelineMode.BASELINE
        )
        st.session_state.advanced_avatar = get_bot_avatar(
            PipelineMode.ADVANCED
        )

    # Title
    st.title("Find answers in your documents")
    st.markdown("---")

    # Sidebar
    st.sidebar.title("Options")

    # Store selected value in a temporary variable to prevent automatic rerun
    selected_mode = st.sidebar.selectbox(
        label="RAG Mode:",
        options=PipelineMode,
        index=_get_sidebar_index(st.session_state.rag_mode),
    )

    # Only update session_state if the selection actually changed
    if selected_mode != st.session_state.rag_mode:
        st.session_state.rag_mode = selected_mode
        st.rerun()

    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
