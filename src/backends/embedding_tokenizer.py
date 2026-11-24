"""Offline, model-matched embedding token counts and lossless text boundaries."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from tokenizers import Tokenizer

from ..harness.benchmark import file_digest

BGE_MODEL_IDS = {"bge-large-en-v1.5", "baai/bge-large-en-v1.5", "text-embedding-bge-large-en-v1.5"}
TOKENIZER_PATH = Path(__file__).with_name("tokenizer_assets") / "bge-large-en-v1.5.json"
TOKENIZER_SHA256 = "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"


@lru_cache(maxsize=1)
def _bge_tokenizer() -> Tokenizer:
    if file_digest(TOKENIZER_PATH) != TOKENIZER_SHA256:
        raise ValueError("Bundled BGE tokenizer checksum mismatch")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


class EmbeddingTokenizer:
    """Count BGE input including CLS/SEP; never normalize or discard source text."""

    limit = 512

    def __init__(self, model: str):
        if model.strip().lower() not in BGE_MODEL_IDS:
            raise ValueError(f"No verified local embedding tokenizer for {model!r}")
        self.tokenizer = _bge_tokenizer()

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=True).ids)

    def fitting_prefix(self, text: str, *, prefix: str = "", suffix: str = "") -> str:
        """Find a nonempty original prefix that fits when wrapped for the provider."""
        if self.count(prefix + text + suffix) <= self.limit:
            return text
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        # Recount complete candidates: cutting a WordPiece continuation can change its count.
        for index in range(min(self.limit, len(encoded.offsets)), 0, -1):
            end = encoded.offsets[index - 1][1]
            candidate = text[:end]
            if candidate and self.count(prefix + candidate + suffix) <= self.limit:
                return candidate
        raise ValueError("Embedding message header leaves no room for source text within 512 tokens")

    @staticmethod
    def dependencies() -> dict:
        return {"model": "BAAI/bge-large-en-v1.5", "revision": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
                "sha256": file_digest(TOKENIZER_PATH), "limit": 512, "includes_special_tokens": True,
                "implementation": file_digest(Path(__file__))}
