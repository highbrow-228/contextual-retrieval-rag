# Third-party imports
from fastapi import APIRouter, Query

# Local imports
from chatbot.config.pipeline_config import PipelineMode
from chatbot.models.chat import Chat
from chatbot.utils.log_utils import log_endpoint_execution


router = APIRouter()
user_input: str = Query(
    ..., min_length=1, max_length=1000, description="User query text"
)


@router.get("/query_baseline", response_model=Chat)
@log_endpoint_execution
async def chat_baseline(user_input) -> Chat:
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

    return pipeline.process(
        mode=PipelineMode.BASELINE,
        user_input=user_input,
    )


@router.get("/query_advanced", response_model=Chat)
@log_endpoint_execution
async def chat_advanced(user_input) -> Chat:
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

    return pipeline.process(
        mode=PipelineMode.ADVANCED,
        user_input=user_input,
    )
