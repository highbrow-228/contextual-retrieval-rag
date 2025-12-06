# Third-party imports
from fastapi import APIRouter, Query

# Local imports
from chatbot.models.chat import Chat
from chatbot.utils.log_utils import log_endpoint_execution


router = APIRouter()


@router.get("/query", response_model=Chat)
@log_endpoint_execution
async def chat(
    user_input: str = Query(
        ..., min_length=1, max_length=1000, description="User query text"
    )
) -> Chat:
    """
    Process a user query and return a chat-style response from RAG pipeline.

    Parameters
    ----------
    - **user_input** (str): The user's query/input text to process.

    Returns
    -------
    - **Chat**: Model containing the assistant response and related metadata.

    Raises
    ------
    - **HTTPException**: If the pipeline fails to generate a response.
    """
    # Local imports
    from chatbot.api.app import pipeline

    return pipeline.process(user_input)
