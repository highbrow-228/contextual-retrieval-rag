"""Main Streamlit app - Home page."""

# Third-party imports
import streamlit as st

# Local imports
from chatbot.globals import ui_settings


# Configure page
st.set_page_config(
    page_title=ui_settings.UI_HOME_PAGE_TITLE,
    page_icon=ui_settings.UI_HOME_PAGE_ICON,
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title(ui_settings.UI_HOME_PAGE_TITLE)
st.markdown("---")

st.write(
    """
Welcome to **ChatBot RAG** 
AI-powered document assistant.

Upload your documents, ask questions, and get intelligent answers powered by your own knowledge base.

**Get started:**
- Go to **Documents** to upload files
- Head to **Chat** to start asking questions

That's it! 🚀
"""
)

st.sidebar.info(
    "**Tip:** Start by uploading documents in the Documents page, then ask questions in the Chat page!"
)
