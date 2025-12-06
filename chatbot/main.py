"""Main entry point for the ChatBot RAG application."""

# Third-party imports
import uvicorn

# Local imports
from chatbot.globals import api_settings


def main() -> None:
    uvicorn.run(
        "chatbot.api.app:app",
        host=api_settings.HOST,
        port=api_settings.PORT,
        reload=api_settings.RELOAD,
    )


if __name__ == "__main__":
    main()
