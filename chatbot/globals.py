"""Global configuration instances for the chatbot application."""

# Local imports
from chatbot.config.api_config import DEFAULT_HOST, APIConfig, get_api_settings
from chatbot.config.chunk_config import ChunkSettings, get_chunk_settings
from chatbot.config.db_config import DatabaseSettings, get_db_settings
from chatbot.config.langsmith_config import (
    LangSmithSettings,
    get_langsmith_settings,
)
from chatbot.config.ocr_config import OCRConfig, get_ocr_settings
from chatbot.config.openai_config import OpenAISetting, get_openai_settings
from chatbot.config.ui_config import UIConfig, get_ui_settings


# Global default API host
DEFAULT_API_HOST = DEFAULT_HOST

# Global API settings instance
api_settings: APIConfig = get_api_settings()

# Global database settings instance
db_settings: DatabaseSettings = get_db_settings()

# Global OpenAI settings instance
openai_settings: OpenAISetting = get_openai_settings()

# Global Chunk settings instance
chunk_settings: ChunkSettings = get_chunk_settings()

# Global OCR settings instance
ocr_settings: OCRConfig = get_ocr_settings()

# Global UI settings instance
ui_settings: UIConfig = get_ui_settings()

# Global LangSmith settings instance
langsmith_settings: LangSmithSettings = get_langsmith_settings()

# Global default API URL
DEFAULT_API_URL_FOR_UI = f"http://{DEFAULT_API_HOST}:{api_settings.PORT}"

# Request timeouts
TIMEOUT_PER_REQUEST = 30  # seconds for regular requests
TIMEOUT_UPLOAD_REQUEST = 300000  # seconds for upload requests
