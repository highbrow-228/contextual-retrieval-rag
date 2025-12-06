# Standard library imports
import os

# Third-party imports
from fastapi import FastAPI
from loguru import logger

# Local imports
from chatbot.api.routers import chat, database
from chatbot.globals import langsmith_settings
from chatbot.pipeline import DocumentProcessingPipeline


# Initialize LangSmith tracing if enabled
if (
    langsmith_settings.LANGSMITH_TRACING
    and langsmith_settings.LANGSMITH_API_KEY
):
    os.environ["LANGSMITH_API_KEY"] = langsmith_settings.LANGSMITH_API_KEY
    os.environ["LANGSMITH_PROJECT"] = langsmith_settings.LANGSMITH_PROJECT
    os.environ["LANGSMITH_ENDPOINT"] = langsmith_settings.LANGSMITH_ENDPOINT
    os.environ["LANGSMITH_TRACING"] = langsmith_settings.LANGSMITH_TRACING

    logger.opt(colors=True).info(
        f"<green>LangSmith tracing enabled</green> for project: <yellow>{langsmith_settings.LANGSMITH_PROJECT}</yellow>"
    )
else:
    logger.opt(colors=True).warning(
        "<yellow>LangSmith tracing disabled</yellow> - set LANGSMITH_TRACING=true and LANGSMITH_API_KEY in .env"
    )


pipeline = DocumentProcessingPipeline()

# Create FastAPI application instance with lifespan
app = FastAPI(
    title="ChatBot API",
    description="An API for interacting with a chatbot powered by Retrieval-Augmented Generation (RAG) and document processing.",
    version="0.1.0",
)

app.include_router(database.router, prefix="/database", tags=["Database"])

app.include_router(chat.router, prefix="/chat", tags=["Chat"])
