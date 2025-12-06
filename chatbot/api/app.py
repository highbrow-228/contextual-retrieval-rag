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
    rotation="10 MB",          # або "1 day"
    retention="14 days",       # скільки зберігати
    compression="zip",         # архівація
    level="DEBUG",
    enqueue=True,              # thread / process safe
    backtrace=True,            # stacktrace при exception
    diagnose=False,            # True тільки для debug
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level:<8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
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
    os.environ["LANGSMITH_TRACING"] = langsmith_settings.LANGSMITH_TRACING.value

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
