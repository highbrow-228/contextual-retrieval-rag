def get_embedding_model_name() -> str:
    """Get the model name used by this handler."""
    # Local imports
    from chatbot.globals import openai_settings

    return openai_settings.MODEL_NAME_EMBEDDING
