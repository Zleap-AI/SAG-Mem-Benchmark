"""Pack at most three complete messages, targeting 512 BGE tokens."""

from pathlib import Path

from ..harness.benchmark import History, Session, file_digest
from .embedding_tokenizer import EmbeddingTokenizer
from .raw_chunks import CHUNK_TOKENIZER
from .text_spans import bounded_spans

MAX_MESSAGES = 3
TOKEN_TARGET = 512


def dependencies(*, split_oversized: bool = False) -> dict:
    return {"policy": "max_three_messages_hard512_v3" if split_oversized else "max_three_messages_soft512_v2",
            "unit": "at_most_three_messages", "tokens": TOKEN_TARGET,
            "budget": "hard" if split_oversized else "soft", "messages_per_chunk": MAX_MESSAGES,
            "overlap_tokens": 0,
            "oversized_message": "lossless_source_spans" if split_oversized else "standalone_full_text",
            "tokenizer": EmbeddingTokenizer.dependencies(),
            "implementation_sha256": file_digest(Path(__file__)),
            "span_implementation": file_digest(Path(__file__).with_name("text_spans.py"))}


def chunk_history(history: History, *, token_target: int = TOKEN_TARGET, dated: bool = True,
                  split_oversized: bool = False) -> list[dict]:
    """Pack within sessions, optionally splitting oversized messages losslessly."""
    tokenizer = EmbeddingTokenizer(CHUNK_TOKENIZER)
    return [chunk for position, session in enumerate(history.sessions)
            for chunk in _session_chunks(session, position, tokenizer, token_target, dated, split_oversized)]


def _session_chunks(session: Session, position: int, tokenizer: EmbeddingTokenizer,
                    token_target: int = TOKEN_TARGET, dated: bool = True,
                    split_oversized: bool = False) -> list[dict]:
    chunks, parts, sources, spans = [], [], [], []

    def flush() -> None:
        if not parts:
            return
        content = "\n".join(parts)
        chunks.append({
            "chunk_id": f"s{position}:t{len(chunks)}",
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
        if not dated:
            header = f"{message.speaker}: "
        rendered = header + message.content
        source = {"session_id": session.id, "message_id": message.id}
        span = {"session_position": position, "message_position": message_position,
                "message_id": message.id, "start": 0, "end": len(message.content)}
        if split_oversized and (len(rendered) > 4096 or tokenizer.count(rendered) > token_target):
            flush()
            for start, end in bounded_spans(message.content, tokenizer.count, token_target,
                                            prefix=header, max_chars=4096):
                parts.append(header + message.content[start:end])
                sources.append(source)
                spans.append({**span, "start": start, "end": end})
                flush()
            continue
        if parts and (len(parts) == MAX_MESSAGES
                      or tokenizer.count("\n".join([*parts, rendered])) > token_target):
            flush()
        parts.append(rendered)
        sources.append(source)
        spans.append(span)
        if tokenizer.count(rendered) > token_target:
            flush()
    flush()
    return chunks
