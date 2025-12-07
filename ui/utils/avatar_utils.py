"""Utility functions for handling bot avatars."""

# Standard library imports
from pathlib import Path


def get_avatar_path(avatar: str) -> str:
    """Convert relative avatar path to absolute path.

    Args:
        avatar: Avatar emoji, URL, or relative file path

    Returns:
        Avatar emoji, URL, or absolute file path
    """
    # If it's an emoji or URL, return as is
    if len(avatar) <= 2 or avatar.startswith("http"):
        return avatar

    # If it's a file path, convert to absolute
    avatar_path = Path(avatar)

    # If already absolute, return as string
    if avatar_path.is_absolute():
        return str(avatar_path)

    # Make relative to ui folder
    ui_folder = Path(__file__).parent.parent
    absolute_path = ui_folder / avatar

    return str(absolute_path)
