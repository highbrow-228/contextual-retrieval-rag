"""API configuration settings for the chatbot UI."""

# Standard library imports
from functools import lru_cache

# Third-party imports
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings


load_dotenv()


class UIConfig(BaseSettings):
    """Configuration for API processing"""

    UI_HOST: str = Field(default="127.0.0.1", description="API host address")
    UI_PORT: int = Field(default=8500, description="API port number")
    UI_RELOAD: bool = Field(
        default=True, description="Enable auto-reload for the API server"
    )
    UI_LOGGER_LEVEL: str = Field(
        default="info", description="Logging level for the UI server"
    )

    # Home page settings
    UI_HOME_PAGE_TITLE: str = Field(
        default="ChatBot UI", description="Title of the Streamlit UI page"
    )
    UI_HOME_PAGE_ICON: str = Field(
        default="🤖", description="Icon for the Streamlit UI page"
    )

    # Chat page settings
    UI_CHAT_PAGE_TITLE: str = Field(
        default="Chat", description="Title of the Chat page"
    )
    UI_CHAT_PAGE_ICON: str = Field(
        default="💬", description="Icon for the Chat page"
    )

    # Documents page settings
    UI_DOCUMENT_PAGE_TITLE: str = Field(
        default="Documents",
        description="Title of the Documents management page",
    )
    UI_DOCUMENT_PAGE_ICON: str = Field(
        default="📋", description="Icon for the Documents management page"
    )

    # Bot avatars for different RAG modes
    UI_BASELINE_BOT_AVATAR: str = Field(
        default="assets/baseline_avatar.png",
        description="Avatar for baseline RAG bot (emoji or path to image)",
    )
    UI_ADVANCED_BOT_AVATAR: str = Field(
        default="assets/advanced_avatar.png",
        description="Avatar for advanced RAG bot (emoji or path to image)",
    )


@lru_cache()
def get_ui_settings() -> UIConfig:
    """Get cached settings instance."""
    return UIConfig()
