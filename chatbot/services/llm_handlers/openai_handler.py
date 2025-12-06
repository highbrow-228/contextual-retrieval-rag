"""OpenAI LLM handler implementation.

This file provides a thin wrapper around the OpenAI client used in the
project. It builds a simple RAG-style prompt from retrieved chunks and the
user query, calls the OpenAI Chat API and returns the assistant text.
"""

from __future__ import annotations

# Standard library imports
import asyncio
import time
from typing import Any

# Third-party imports
from langsmith import traceable
from loguru import logger
from openai import (
    APIConnectionError,
    APIError,
    AsyncOpenAI,
    OpenAI,
    RateLimitError,
)
from openai.types.responses.response import Response

# Local imports
from chatbot.base.base_llm import BaseLLMHandler
from chatbot.config.rag_prompts import RagPrompts
from chatbot.utils.llm_utils import get_rag_model_name


class OpenAIHandler(BaseLLMHandler):
    """Handler that invokes OpenAI Chat Completions via internally managed OpenAI clients.
    The handler is intentionally small - it initializes its own `openai.OpenAI` and
    `openai.AsyncOpenAI` client instances using the provided API key and model parameters.
    """

    def __init__(
        self,
        api_key: str,
        rag_model_name: str,
        rag_temperature: float,
        rag_max_tokens: int | None,
        rag_max_retries: int,
        base_delay: int,
    ) -> None:
        self.client: OpenAI = OpenAI(api_key=api_key)

        self.async_client: AsyncOpenAI = AsyncOpenAI(api_key=api_key)

        self.rag_model_name = rag_model_name
        self.rag_temperature = rag_temperature
        self.rag_max_tokens = rag_max_tokens
        self.rag_max_retries = rag_max_retries
        self.base_delay = base_delay

    @traceable(
        run_type="llm",
        name="RAG (sync)",
        metadata={
            "type": "sequential",
            "ls_provider": "OpenAIHandler",
            "model": get_rag_model_name(),
        },
    )
    def invoke_rag(
        self, user_input: str, text_chunks: str, **kwargs: Any
    ) -> str:
        messages = self.build_messages(
            user_input=user_input, text_chunks=text_chunks
        )

        return self._invoke_rag(messages)

    @traceable(
        run_type="llm",
        name="RAG (async)",
        metadata={
            "type": "async",
            "ls_provider": "OpenAIHandler",
            "model": get_rag_model_name(),
        },
    )
    async def ainvoke_rag(
        self, user_input: str, text_chunks: str, **kwargs: Any
    ) -> str:
        """Async wrapper that calls OpenAI async API with retry logic.

        Retries up to max_retries times with exponential backoff only for network/rate limit errors.
        Other errors (invalid params, etc.) fail immediately.
        """
        messages: list[dict[str, str]] = self.build_messages(
            user_input=user_input, text_chunks=text_chunks
        )

        max_retries = self.rag_max_retries
        base_delay = self.base_delay

        for attempt in range(max_retries):
            try:
                response: Response = await self.async_client.responses.create(
                    model=self.rag_model_name,
                    input=messages,
                    temperature=self.rag_temperature,
                    max_output_tokens=self.rag_max_tokens,
                )

                logger.opt(colors=True).info(
                    f"LLM({self.rag_model_name}) response: <green>{response.output_text}</green>"
                )
                return response.output_text

            except (APIConnectionError, RateLimitError) as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)  # 20s, 40s, 80s
                    logger.warning(
                        f"OpenAI API call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.exception(
                        f"OpenAI API call failed after {max_retries} attempts."
                    )
                    raise
            except APIError as e:
                # For other API errors, fail immediately
                logger.exception(f"OpenAI API error (not retryable): {e}")
                raise

    @traceable(
        run_type="prompt",
        name="Build RAG messages",
        metadata={
            "ls_provider": "OpenAIHandler",
            "model": get_rag_model_name(),
        },
    )
    def build_messages(
        self, user_input: str, text_chunks: str
    ) -> list[dict[str, str]]:
        """Compose chat messages for a RAG-style completion.

        Args:
            user_input: The user's question.
            text_chunks: Concatenated context from retrieved chunks.

        Returns:
            A list of messages suitable for the OpenAI chat completion endpoint.
        """
        return [
            {"role": "system", "content": RagPrompts.SYSTEM_RAG_PROMPT},
            {
                "role": "user",
                "content": RagPrompts.HUMAN_RAG_PROMPT.format(
                    user_input=user_input,
                    text_chunks=text_chunks,
                ),
            },
        ]

    @traceable(
        run_type="llm",
        name="llm",
        metadata={
            "type": "sequential",
            "ls_provider": "OpenAIHandler",
            "model": get_rag_model_name(),
        },
    )
    def _invoke_rag(self, messages: list[dict[str, str]]) -> str:
        """Synchronously invoke the OpenAI chat completion endpoint with retry logic.

        Returns the assistant text or a short error message.
        Retries up to 3 times with exponential backoff (20s, 40s, 80s) only for network/rate limit errors.
        Other errors (invalid params, etc.) fail immediately.
        """
        max_retries = self.rag_max_retries
        base_delay = self.base_delay

        for attempt in range(max_retries):
            try:
                response: Response = self.client.responses.create(
                    model=self.rag_model_name,
                    input=messages,
                    temperature=self.rag_temperature,
                    max_output_tokens=self.rag_max_tokens,
                )

                logger.opt(colors=True).info(
                    f"LLM({self.rag_model_name}) response: <green>{response.output_text}</green>"
                )
                return response.output_text

            except (APIConnectionError, RateLimitError) as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)  # 20s, 40s, 80s
                    logger.warning(
                        f"OpenAI API call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.exception(
                        f"OpenAI API call failed after {max_retries} attempts."
                    )
                    raise
            except APIError as e:
                # For other API errors, fail immediately
                logger.exception(f"OpenAI API error (not retryable): {e}")
                raise
