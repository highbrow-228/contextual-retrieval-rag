"""Utility functions and helpers for the ChatBot application."""

# Standard library imports
import asyncio
import functools
import time
from typing import Any, Callable

# Third-party imports
from loguru import logger


def log_initialization(component_name: str = None) -> Callable:
    """
    Decorator for logging initialization of components.
    Logs when initialization starts, completes, and how long it took.

    Args:
        component_name: Name of the component being initialized (e.g., "OpenAI client")
                       If not provided, uses the function name.

    Example:
        @log_initialization("OpenAI client")
        def setup_openai(self):
            self.client = OpenAI(api_key=key)
    """

    def decorator(func: Callable) -> Callable:
        name = component_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger.debug(f"Initializing {name}...")
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.opt(colors=True).debug(
                    f"<green>{name} initialized successfully in {execution_time:.3f}s</green>"
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.opt(colors=True).error(
                    f"<red>Failed to initialize {name} after {execution_time:.3f}s: {str(e)}</red>"
                )
                raise

        return wrapper

    return decorator


def log_endpoint_execution(func: Callable) -> Callable:
    """
    Decorator for logging endpoint execution times.
    """

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        logger.opt(colors=True).info(
            f"Endpoint <red>{func.__name__}</red> started"
        )
        start_time = time.time()

        result = await func(*args, **kwargs)

        execution_time = time.time() - start_time
        logger.opt(colors=True).info(
            f"Endpoint <red>{func.__name__}</red> completed in <green>{execution_time:.2f}</green> seconds"
        )

        return result

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        logger.opt(colors=True).info(f"Endpoint '{func.__name__}' started")
        start_time = time.time()

        result = func(*args, **kwargs)

        execution_time = time.time() - start_time
        logger.opt(colors=True).info(
            f"Endpoint <red>{func.__name__}</red> completed in <green>{execution_time:.2f}</green> seconds"
        )

        return result

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
