"""Streamlit UI for ChatBot RAG system."""

# Standard library imports
import subprocess
import sys

# Local imports
from chatbot.globals import ui_settings


__version__ = "0.1.0"
__author__ = "ChatBot RAG Team"


def main():
    """Entry point for poetry script - runs Streamlit app."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "ui/home.py",
            "--server.port",
            f"{ui_settings.UI_PORT}",
            "--server.address",
            f"{ui_settings.UI_HOST}",
            "--server.runOnSave",
            f"{ui_settings.UI_RELOAD}",
            f"--logger.level={ui_settings.UI_LOGGER_LEVEL}",
        ],
        cwd=".",
    )
