"""Paper experiment configuration, source audit and request accounting."""
from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import os
import time

from src.backends.embedding_tokenizer import EmbeddingTokenizer
from src.harness.benchmark import file_digest, fingerprint
from src.harness.reader_grader import ReaderGrader
from src.llm_client import ConfiguredAsyncOpenAI

from . import chunking
from .profiles import EMBEDDING_PROFILES, PROFILES, api_settings, embedding_profile

__all__ = [
    "EMBEDDING_PROFILES",
    "PROFILES",
    "QUESTION",
    "STAGE",
    "PaperReader",
    "RequestLedger",
    "api_settings",
    "audit_chunks",
    "chunking",
    "embedding_profile",
    "file_digest",
    "fingerprint",
    "sha256",
]

QUESTION = contextvars.ContextVar("question_id", default=None)
STAGE = contextvars.ContextVar("stage", default="prepare")


def sha256(value: str | bytes) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def audit_chunks(history, chunks, messages_per_chunk=None):
    """Check §3.1 on real input: whole messages, one session per chunk, soft 512-token budget.

    The paper setting has no message-count cap (``messages_per_chunk=None``). A positive
    value checks the optional cap used by ``--messages-per-chunk`` sensitivity runs.
    """
    seen, expected = [], []
    oversized = 0
    tokenizer = EmbeddingTokenizer("BAAI/bge-large-en-v1.5")
    for session in history.sessions:
        expected.extend((session.id, message.id) for message in session.messages)
    for chunk in chunks:
        spans = chunk["message_spans"]
        if not spans:
            raise ValueError("Chunk must contain at least one intact message")
        if messages_per_chunk is not None and len(spans) > messages_per_chunk:
            raise ValueError(f"Chunk exceeds the configured cap of {messages_per_chunk} messages")
        rendered = []
        for span, source in zip(spans, chunk["sources"], strict=True):
            session = history.sessions[span["session_position"]]
            message = session.messages[span["message_position"]]
            if (chunk["session_id"] != session.id or source != {
                "session_id": session.id, "message_id": message.id
            } or span["start"] != 0 or span["end"] != len(message.content)):
                raise ValueError("Incomplete source span or crossed session boundary")
            rendered.append(f"{message.speaker} (date: {session.date_raw or session.timestamp or 'unknown'}): "
                            + message.content)
            seen.append((session.id, message.id))
        if "\n".join(rendered) != chunk["content"]:
            raise ValueError("Chunk text differs from complete source messages")
        count = tokenizer.count(chunk["content"])
        if count != chunk["token_count"] or (count > 512 and len(spans) != 1):
            raise ValueError("Soft token limit violated")
        oversized += count > 512
    if seen != expected:
        raise ValueError("Source messages are missing, duplicated or reordered")
    return {"messages": len(expected), "chunks": len(chunks), "oversized_chunks": oversized,
            "max_bge_tokens": max((c["token_count"] for c in chunks), default=0),
            "history_sha256": history.digest, "chunks_sha256": fingerprint(chunks),
            "source_messages_complete": True}


class RequestLedger:
    """Audit SDK responses without modifying Reader/Judge prompts or parsing."""
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("a", encoding="utf-8")
        self.records = []

    def wrap(self, client, method, label):
        container, name = method
        original = getattr(container, name)

        async def invoke(*args, **kwargs):
            started = time.perf_counter()
            body = kwargs.get("messages", kwargs.get("input"))
            record = {"question_id": QUESTION.get(), "stage": STAGE.get(), "method": label,
                      "model": kwargs["model"], "request_content_sha256": fingerprint(body),
                      "request_characters": len(json.dumps(body, ensure_ascii=False)),
                      "timestamp": time.time()}
            try:
                response = await original(*args, **kwargs)
                value = response.model_dump(mode="json")
                usage = value.get("usage")
                record.update(response_id=value.get("id"), response_model=value.get("model"),
                              provider=value.get("provider"), usage=usage,
                              response_sha256=fingerprint(value), status="succeeded")
                if usage is None:
                    raise ValueError("Provider did not return token usage")
                record["cost_usd"] = usage.get("cost")
                if (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0):
                    raise ValueError("Provider charged reasoning tokens despite disabled reasoning")
                choices = value.get("choices", [])
                if any(choice.get("finish_reason") == "length" for choice in choices):
                    raise ValueError("Model response hit output limit")
                if any(choice.get("message", {}).get("reasoning") for choice in choices):
                    raise ValueError("Provider returned reasoning despite disabled reasoning")
                return response
            except asyncio.CancelledError:
                record.update(status="interrupted", error_type="CancelledError")
                raise
            except Exception as error:
                record.update(status="failed", error_type=type(error).__name__, error=str(error))
                raise
            finally:
                record["elapsed_seconds"] = time.perf_counter() - started
                self.records.append(record)
                self.stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                self.stream.flush()
                os.fsync(self.stream.fileno())

        setattr(container, name, invoke)

    def close(self):
        self.stream.close()


class PaperReader(ReaderGrader):
    """Use the paper prompts and grading; endpoints come from repository .env."""

    def __init__(self, profile, ledger):
        from src.config import settings

        reader_model = settings.search_llm.model or profile["model"]
        judge_model = settings.judge_llm.model or profile["model"]
        super().__init__(reader_model, judge_model)
        for role, endpoint in (("reader", settings.search_llm), ("judge", settings.judge_llm)):
            client = ConfiguredAsyncOpenAI(
                api_key=endpoint.api_key,
                base_url=endpoint.base_url,
                request_extra_body=endpoint.extra_body,
                timeout=240,
                max_retries=0,
            )
            self.__dict__[f"_{role}_client"] = client
            ledger.wrap(client, (client.chat.completions, "create"), role + ".create")
            ledger.wrap(client, (client.chat.completions, "parse"), role + ".parse")

    async def respond(self, *args, **kwargs):
        import re

        if re.sub(r"\[session\s+[^\]\r\n]+\][ \t]*", "", args[0]) != args[0]:
            raise ValueError("Reader would remove text from a source message")
        token = STAGE.set("reader")
        try:
            return await super().respond(*args, **kwargs)
        finally:
            STAGE.reset(token)

    async def grade(self, *args, **kwargs):
        token = STAGE.set("judge")
        try:
            return await super().grade(*args, **kwargs)
        finally:
            STAGE.reset(token)
