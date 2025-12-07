# Standard library imports
import os
import sys

# Third-party imports
from fastapi import FastAPI
from loguru import logger

# Local imports
from chatbot.api.routers import chat, database
from chatbot.globals import langsmith_settings
from chatbot.pipeline import DocumentProcessingPipeline


logger.remove()
logger.add(sys.stdout, level="DEBUG")
logger.add(
    "logs/app.log",
    rotation="10 MB",  # Rotate log file when it reaches 10 MB
    retention="14 days",  # Keep rotated logs for 14 days
    compression="zip",  # Compress rotated log files
    level="INFO",  # Only log INFO level and above to file
    enqueue=True,  # Thread-safe logging queue
    backtrace=True,  # Include full stack trace on exceptions
    diagnose=False,  # Disable verbose diagnostics (only enable for debugging)
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level}</level> | "
        "<cyan>{name}</cyan>:<yellow>{function}</yellow>:<red>{line}</red> - "
        "<level>{message}</level>"
    ),
)

# Initialize LangSmith tracing if enabled
if (
    langsmith_settings.LANGSMITH_TRACING.value == "true"
    and langsmith_settings.LANGSMITH_API_KEY
):
    os.environ["LANGSMITH_API_KEY"] = langsmith_settings.LANGSMITH_API_KEY
    os.environ["LANGSMITH_PROJECT"] = langsmith_settings.LANGSMITH_PROJECT
    os.environ["LANGSMITH_ENDPOINT"] = langsmith_settings.LANGSMITH_ENDPOINT
    os.environ["LANGSMITH_TRACING"] = (
        langsmith_settings.LANGSMITH_TRACING.value
    )

    logger.opt(colors=True).info(
        f"<green>LangSmith tracing enabled</green> for project: <yellow>{langsmith_settings.LANGSMITH_PROJECT}</yellow>"
    )
else:
    logger.opt(colors=True).warning(
        "<yellow>LangSmith tracing disabled</yellow> - set LANGSMITH_TRACING=true and LANGSMITH_API_KEY in .env\n"
        f"LangSmith API key is <yellow>{langsmith_settings.LANGSMITH_API_KEY}</yellow> and\n"
        f"LANGSMITH_TRACING is <yellow>{langsmith_settings.LANGSMITH_TRACING.value}</yellow>"
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
