"""Configuration management for Ansel Photo Album."""

import json
import os
import threading
from pathlib import Path

# Thread lock for photo index file access
_index_lock = threading.Lock()

# Base directories
ANSEL_DIR = Path.home() / ".ansel"
CACHE_DIR = ANSEL_DIR / "cache"
THUMBNAILS_DIR = CACHE_DIR / "thumbnails"
CONFIG_FILE = ANSEL_DIR / "config.json"
TOKEN_FILE = ANSEL_DIR / "dropbox_token.json"
INDEX_FILE = ANSEL_DIR / "photo_index.json"

# Ensure directories exist
ANSEL_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
THUMBNAILS_DIR.mkdir(exist_ok=True)

# Thumbnail settings
THUMBNAIL_SIZE = (300, 300)
THUMBNAIL_QUALITY = 80

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic"}

# Flask settings
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5001
OAUTH_REDIRECT_URI = f"http://localhost:{FLASK_PORT}/oauth/callback"


def load_config():
    """Load configuration from config file."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}


def save_config(config):
    """Save configuration to config file."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_dropbox_credentials():
    """Get Dropbox app credentials from config or environment."""
    config = load_config()
    app_key = os.environ.get("DROPBOX_APP_KEY") or config.get("dropbox_app_key")
    app_secret = os.environ.get("DROPBOX_APP_SECRET") or config.get("dropbox_app_secret")
    return app_key, app_secret


def set_dropbox_credentials(app_key, app_secret):
    """Save Dropbox app credentials to config."""
    config = load_config()
    config["dropbox_app_key"] = app_key
    config["dropbox_app_secret"] = app_secret
    save_config(config)


def load_tokens():
    """Load OAuth tokens from token file."""
    if TOKEN_FILE.exists():
        with open(TOKEN_FILE) as f:
            return json.load(f)
    return None


def save_tokens(tokens):
    """Save OAuth tokens to token file."""
    with open(TOKEN_FILE, "w") as f:
        json.dump(tokens, f, indent=2)


# Default themes for photo classification
DEFAULT_THEMES = [
    "Friends", "Us", "Girma", "McDonald", "Both", "Brooke",
    "Liam", "Christmas", "Easter", "Halloween", "Scouts", "Willow"
]


def load_photo_index():
    """Load photo index from file (thread-safe)."""
    with _index_lock:
        if INDEX_FILE.exists():
            try:
                with open(INDEX_FILE) as f:
                    index = json.load(f)
                    # Ensure themes section exists
                    if "themes" not in index:
                        index["themes"] = {"default": DEFAULT_THEMES, "custom": []}
                    return index
            except json.JSONDecodeError:
                # File is corrupted, try backup
                backup_file = INDEX_FILE.with_suffix(".json.bak")
                if backup_file.exists():
                    with open(backup_file) as f:
                        index = json.load(f)
                        if "themes" not in index:
                            index["themes"] = {"default": DEFAULT_THEMES, "custom": []}
                        return index
                # No backup, return empty
                return {"photos": {}, "years": {}, "themes": {"default": DEFAULT_THEMES, "custom": []}}
        return {"photos": {}, "years": {}, "themes": {"default": DEFAULT_THEMES, "custom": []}}


def save_photo_index(index):
    """Save photo index to file (thread-safe with atomic write)."""
    with _index_lock:
        # Write to temp file first, then rename (atomic on POSIX)
        temp_file = INDEX_FILE.with_suffix(".json.tmp")
        backup_file = INDEX_FILE.with_suffix(".json.bak")

        with open(temp_file, "w") as f:
            json.dump(index, f, indent=2)

        # Create backup of current file
        if INDEX_FILE.exists():
            try:
                INDEX_FILE.rename(backup_file)
            except OSError:
                pass

        # Rename temp to actual (atomic)
        temp_file.rename(INDEX_FILE)


def get_all_themes():
    """Get all available themes (default + custom)."""
    index = load_photo_index()
    themes = index.get("themes", {"default": DEFAULT_THEMES, "custom": []})
    return themes["default"] + themes.get("custom", [])


def add_custom_theme(theme_name):
    """Add a custom theme."""
    index = load_photo_index()
    if "themes" not in index:
        index["themes"] = {"default": DEFAULT_THEMES, "custom": []}

    # Don't add if already exists
    all_themes = index["themes"]["default"] + index["themes"].get("custom", [])
    if theme_name not in all_themes:
        index["themes"]["custom"].append(theme_name)
        save_photo_index(index)
        return True
    return False


def remove_custom_theme(theme_name):
    """Remove a custom theme."""
    index = load_photo_index()
    if "themes" not in index:
        return False

    # Can only remove custom themes, not default
    if theme_name in index["themes"].get("custom", []):
        index["themes"]["custom"].remove(theme_name)
        save_photo_index(index)
        return True
    return False
