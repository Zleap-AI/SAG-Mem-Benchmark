"""Pack messages until 512 BGE tokens. A single longer message stays whole."""
from src.backends.embedding_tokenizer import EmbeddingTokenizer

POLICY = "soft512_atomic_v1"
TOKEN_BUDGET = 512


def chunk_history(history):
    tokenizer = EmbeddingTokenizer("BAAI/bge-large-en-v1.5")
    chunks = []
    for session_position, session in enumerate(history.sessions):
        parts, sources, spans = [], [], []
        session_chunk_count = 0

        def flush():
            nonlocal session_chunk_count
            if not parts:
                return
            content = "\n".join(parts)
            chunks.append({"chunk_id": f"s{session_position}:t{session_chunk_count}",
                           "session_id": session.id, "content": content,
                           "token_count": tokenizer.count(content),
                           "sources": sources.copy(), "message_spans": spans.copy()})
            session_chunk_count += 1
            parts.clear()
            sources.clear()
            spans.clear()

        for message_position, message in enumerate(session.messages):
            rendered = f"{message.speaker} (date: {session.date_raw or session.timestamp or 'unknown'}): {message.content}"
            if parts and tokenizer.count("\n".join([*parts, rendered])) > TOKEN_BUDGET:
                flush()
            parts.append(rendered)
            sources.append({"session_id": session.id, "message_id": message.id})
            spans.append({"session_position": session_position, "message_position": message_position,
                          "message_id": message.id, "start": 0, "end": len(message.content)})
            if tokenizer.count(rendered) > TOKEN_BUDGET:
                flush()
        flush()
    return chunks
