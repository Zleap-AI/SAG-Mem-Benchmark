"""Shared, lossless raw-dialogue chunks for the retrieval ablations."""

from __future__ import annotations

from ..harness.benchmark import History, Session
from .embedding_tokenizer import EmbeddingTokenizer

CHUNK_TOKENS = 512
CHUNK_TOKENIZER = "BAAI/bge-large-en-v1.5"
CHUNK_POLICY = "session_atomic_messages_v2"


def chunk_history(history: History) -> list[dict]:
    """Pack complete messages within sessions, following SAG's atomic-unit design.

    512 tokens is a packing target, including headers and BGE's CLS/SEP tokens.
    A single oversized message stays intact in its own chunk, even above 512.
    Chunks have no overlap. Source labels are metadata only, never Reader text.
    """
    tokenizer = EmbeddingTokenizer(CHUNK_TOKENIZER)
    chunks: list[dict] = []
    for session_position, session in enumerate(history.sessions):
        chunks.extend(_session_chunks(session, session_position, tokenizer))
    return chunks


def _session_chunks(session: Session, position: int, tokenizer: EmbeddingTokenizer) -> list[dict]:
    chunks: list[dict] = []
    parts: list[str] = []
    sources: list[dict] = []
    spans: list[dict] = []

    def flush() -> None:
        if not parts:
            return
        content = "\n".join(parts)
        chunks.append({
            "chunk_id": f"s{position}:c{len(chunks)}",
            "session_id": session.id,
            "content": content,
            "token_count": tokenizer.count(content),
            "sources": sources.copy(),
            "message_spans": spans.copy(),
        })
        parts.clear()
        sources.clear()
        spans.clear()

    for message_position, message in enumerate(session.messages):
        header = f"{message.speaker} (date: {session.date_raw or session.timestamp or 'unknown'}): "
        rendered = header + message.content
        if parts and tokenizer.count("\n".join([*parts, rendered])) > CHUNK_TOKENS:
            flush()
        parts.append(rendered)
        sources.append({"session_id": session.id, "message_id": message.id})
        spans.append({"session_position": position, "message_position": message_position,
                      "message_id": message.id, "start": 0, "end": len(message.content)})
        # Oversized units stand alone; never carry a following message into their chunk.
        if tokenizer.count(rendered) > CHUNK_TOKENS:
            flush()
    flush()
    return chunks
