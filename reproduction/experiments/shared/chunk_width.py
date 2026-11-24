"""Optional message caps shared by SAG-Mem and SAG-Mem-Lite."""
from src.backends.embedding_tokenizer import EmbeddingTokenizer

TOKEN_BUDGET = 512


def split_options(options: dict) -> tuple[int | None, dict]:
    """Separate the message cap from the underlying retrieval options."""
    given = dict(options.get("backend_options") or {})
    if "messages_per_chunk" not in given:
        return None, options
    raw = given.pop("messages_per_chunk")
    if type(raw) is not int or not 1 <= raw <= 12:
        raise ValueError("messages_per_chunk must be an integer from 1 to 12")
    return raw, {**options, "backend_options": given}


def _emit(chunks, parts, sources, spans, session_position, session_id, counter, tokenizer):
    if not parts:
        return
    content = "\n".join(parts)
    chunks.append({"chunk_id": f"s{session_position}:t{counter[0]}",
                   "session_id": session_id, "content": content,
                   "token_count": tokenizer.count(content),
                   "sources": sources.copy(), "message_spans": spans.copy()})
    counter[0] += 1
    parts.clear()
    sources.clear()
    spans.clear()


def chunk_history(history, messages_per_chunk: int):
    if type(messages_per_chunk) is not int or not 1 <= messages_per_chunk <= 12:
        raise ValueError("messages_per_chunk must be an integer from 1 to 12")
    tokenizer = EmbeddingTokenizer("BAAI/bge-large-en-v1.5")
    chunks = []
    for session_position, session in enumerate(history.sessions):
        parts, sources, spans = [], [], []
        counter = [0]
        session_id = session.id

        for message_position, message in enumerate(session.messages):
            rendered = (
                f"{message.speaker} (date: {session.date_raw or session.timestamp or 'unknown'}): "
                f"{message.content}"
            )
            if parts and (len(parts) == messages_per_chunk
                          or tokenizer.count("\n".join([*parts, rendered])) > TOKEN_BUDGET):
                _emit(chunks, parts, sources, spans, session_position, session_id, counter, tokenizer)
            parts.append(rendered)
            sources.append({"session_id": session_id, "message_id": message.id})
            spans.append({"session_position": session_position, "message_position": message_position,
                          "message_id": message.id, "start": 0, "end": len(message.content)})
            if tokenizer.count(rendered) > TOKEN_BUDGET:
                _emit(chunks, parts, sources, spans, session_position, session_id, counter, tokenizer)
        _emit(chunks, parts, sources, spans, session_position, session_id, counter, tokenizer)
    return chunks
