"""LangSmith tracing utilities for integrated observability."""

# Standard library imports
import os
from typing import Optional

# Third-party imports
from langsmith import Client


def setup_langsmith() -> Optional[Client]:
    """Initialize and configure LangSmith client based on environment variables.

    Returns:
        LangSmith Client instance if API key is provided, None otherwise.

    Environment variables:
        - LANGSMITH_API_KEY: Your LangSmith API key
        - LANGSMITH_PROJECT: Project name for traces (default: ChatBot-RAG)
        - LANGSMITH_ENDPOINT: LangSmith endpoint (optional)
    """
    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()

    if not api_key:
        return None

    # Set environment variables for LangSmith
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = os.getenv(
        "LANGSMITH_PROJECT", "ChatBot-RAG"
    )

    if endpoint := os.getenv("LANGSMITH_ENDPOINT"):
        os.environ["LANGSMITH_ENDPOINT"] = endpoint

    return Client()
