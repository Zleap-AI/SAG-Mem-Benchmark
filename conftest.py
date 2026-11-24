import json
import os
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Never load local credentials or rely on a developer's model configuration.
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path(sys.prefix).resolve() / ".cache/tiktoken"))
for key, value in {
    "OPENAI_API_KEY": "offline-test",
    "OPENAI_BASE_URL": "https://example.test/v1",
    "EMBEDDING_API_KEY": "offline-test",
    "EMBEDDING_BASE_URL": "https://example.test/v1",
    "EMBEDDING_MODEL_NAME": "BAAI/bge-large-en-v1.5",
    "EXTRACT_LLM_API_KEY": "offline-test",
    "EXTRACT_LLM_BASE_URL": "https://example.test/v1",
    "OPENAI_EXTRA_BODY": json.dumps({"reasoning": {"enabled": False}, "provider": {
        "only": ["DeepInfra"], "allow_fallbacks": False, "quantizations": ["fp8"]}}),
    "SEARCH_LLM_API_KEY": "offline-test",
    "SEARCH_LLM_BASE_URL": "https://example.test/v1",
    "SEARCH_LLM_MODEL": "qwen/qwen3.6-35b-a3b",
    "JUDGE_LLM_API_KEY": "offline-test",
    "JUDGE_LLM_BASE_URL": "https://example.test/v1",
    "JUDGE_LLM_MODEL": "qwen/qwen3.6-35b-a3b",
    "JEV_API_KEY": "offline-test",
    "JEV_BASE_URL": "https://example.test/decisions",
}.items():
    os.environ[key] = value


@pytest.fixture(scope="session", autouse=True)
def tokenizer_assets():
    # Download the checksum-verified tokenizer once, before tests block external connections.
    import tiktoken
    tiktoken.get_encoding("cl100k_base")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    connect = socket.socket.connect

    def local_only(sock, address):
        if isinstance(address, tuple) and address[0] not in {"127.0.0.1", "::1", "localhost"}:
            raise AssertionError("External network calls are forbidden in tests")
        return connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", local_only)
