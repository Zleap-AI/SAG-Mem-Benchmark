"""Load credentials and endpoints from the repository root .env only."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
REPRO_ROOT = Path(__file__).resolve().parent
_LOADED = False


def prepare_writable_tmp() -> Path:
    """jieba writes marshal cache via tempfile.gettempdir(); /tmp may be unwritable."""
    cache = REPRO_ROOT / ".cache" / "jieba"
    cache.mkdir(parents=True, exist_ok=True)
    for key in ("TMPDIR", "TEMP", "TMP"):
        os.environ.setdefault(key, str(cache))
    return cache


prepare_writable_tmp()


def load_repo_env() -> Path:
    """Apply repository .env to os.environ, overriding prior process env."""
    global _LOADED
    path = REPO_ROOT / ".env"
    if path.is_file():
        load_dotenv(path, override=True)
    _LOADED = True
    return path