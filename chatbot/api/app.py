# Third-party imports
from fastapi import FastAPI

# Local imports
from chatbot.api.routers import chat, database
from chatbot.pipeline import DocumentProcessingPipeline


pipeline = DocumentProcessingPipeline()

# Create FastAPI application instance with lifespan
app = FastAPI(
    title="ChatBot API",
    description="An API for interacting with a chatbot powered by Retrieval-Augmented Generation (RAG) and document processing.",
    version="0.1.0",
)

app.include_router(database.router, prefix="/database", tags=["Database"])

app.include_router(chat.router, prefix="/chat", tags=["Chat"])
