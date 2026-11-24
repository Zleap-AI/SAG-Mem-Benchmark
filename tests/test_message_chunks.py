from collections import defaultdict
from itertools import pairwise

import numpy as np
import pytest

from src.backends import message_chunks
from src.backends.embedding_tokenizer import EmbeddingTokenizer
from src.backends.raw_chunks import chunk_history as original_chunks
from src.backends.sag2_chunk_time import annotate_chunk
from src.harness.benchmark import BuildSpec, History, Message, Session, fingerprint


def test_message_cap_and_session_boundaries():
    history = History("test", (
        Session("a", tuple(Message(str(i), "Alice", "hello") for i in range(8)), "2025-05-12"),
        Session("b", (Message("next", "Bob", "world"),), "2025-05-13"),
    ))
    chunks = message_chunks.chunk_history(history)
    assert [len(c["message_spans"]) for c in chunks] == [3, 3, 2, 1]
    assert [c["session_id"] for c in chunks] == ["a", "a", "a", "b"]
    assert [s["message_id"] for c in chunks for s in c["sources"]] == [str(i) for i in range(8)] + ["next"]
    assert len(original_chunks(History("test", history.sessions[:1]))) == 1


def test_token_cap_precedes_message_cap_and_counts_headers():
    tokenizer = EmbeddingTokenizer("BAAI/bge-large-en-v1.5")
    header = "Alice (date: 2025-05-12): "
    body = "hello " * (512 - tokenizer.count(header))
    assert tokenizer.count(header + body) == 512
    history = History("test", (Session("a", (Message("full", "Alice", body),
                                               Message("next", "Alice", "more")), "2025-05-12"),))
    chunks = message_chunks.chunk_history(history)
    assert len(chunks) == 2
    assert chunks[0]["token_count"] == 512
    assert all(len(c["message_spans"]) == 1 and c["token_count"] <= 512 for c in chunks)


@pytest.mark.parametrize("text", ["hello world!\n" * 600, "你好🙂 e\u0301\tMixed CASE\n" * 300])
def test_oversized_messages_preserve_all_characters_and_offsets(text):
    history = History("test", (Session("a", (Message("first", "Bob", "before"),
                                               Message("long", "Alice", text),
                                               Message("last", "Bob", "yesterday")), "2025-05-12"),))
    chunks = message_chunks.chunk_history(history)
    assert len(chunks) == 3
    restored = defaultdict(str)
    expected_offsets = defaultdict(int)
    tokenizer = EmbeddingTokenizer("BAAI/bge-large-en-v1.5")
    for chunk in chunks:
        assert chunk["token_count"] == tokenizer.count(chunk["content"])
        assert len(chunk["message_spans"]) == 1
        span = chunk["message_spans"][0]
        name = "Alice" if span["message_id"] == "long" else "Bob"
        body = chunk["content"].removeprefix(f"{name} (date: 2025-05-12): ")
        assert span["start"] == expected_offsets[span["message_id"]]
        assert len(body) == span["end"] - span["start"]
        expected_offsets[span["message_id"]] = span["end"]
        restored[span["message_id"]] += body
        annotate_chunk({**chunk, "kind": "chunk"})
    assert chunks[1]["token_count"] > 512
    assert chunks[1]["message_spans"][0]["start"] == 0
    assert chunks[1]["message_spans"][0]["end"] == len(text)
    assert restored == {"first": "before", "long": text, "last": "yesterday"}


def test_hard_split_keeps_every_oversized_source_span_under_512_tokens():
    text = "long trajectory step; " * 1500
    history = History("test", (Session("a", (Message("long", "Trajectory", text),), "2025-05-12"),))
    chunks = message_chunks.chunk_history(history, split_oversized=True)
    tokenizer = EmbeddingTokenizer("BAAI/bge-large-en-v1.5")
    spans = [chunk["message_spans"][0] for chunk in chunks]

    assert len(chunks) > 1
    assert all(chunk["token_count"] <= 512 for chunk in chunks)
    assert all(tokenizer.count(chunk["content"]) <= 512 for chunk in chunks)
    assert spans[0]["start"] == 0 and spans[-1]["end"] == len(text)
    assert all(left["end"] == right["start"] for left, right in pairwise(spans))
    assert "".join(text[span["start"]:span["end"]] for span in spans) == text
    for chunk in chunks:
        annotate_chunk({**chunk, "kind": "chunk"})


def test_empty_messages_and_oversized_headers_are_kept():
    history = History("test", (Session("a", (Message("empty", "Alice", ""),), "2025-05-12"),))
    chunks = message_chunks.chunk_history(history)
    assert chunks[0]["sources"] == [{"session_id": "a", "message_id": "empty"}]
    assert chunks[0]["message_spans"][0]["start"] == chunks[0]["message_spans"][0]["end"] == 0
    oversized = History("test", (Session("a", (Message("m", "name " * 600, ""),), "2025-05-12"),))
    assert message_chunks.chunk_history(oversized)[0]["token_count"] > 512


@pytest.mark.parametrize("method", ["sag-mem", "sag-mem-lite"])
def test_paper_chunking_policy_is_identical_across_datasets(method):
    from reproduction.run import configure, parse_cli
    from src.harness.benchmark import resolve

    policies = []
    for dataset in ("locomo", "longmemeval"):
        options = configure(parse_cli(["--method", method, "--dataset", dataset]))
        policies.append(resolve("backend", options["backend"]).dependencies(options)["chunking"])
    assert policies[0] == policies[1]
    assert policies[0]["budget"] == "soft"
    assert policies[0]["messages_per_chunk"] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["sag-mem", "sag-mem-lite"])
async def test_paper_methods_apply_default_cap_and_matching_build_identity(tmp_path, method):
    from reproduction.run import configure, parse_cli
    from src.harness.benchmark import resolve

    options = configure(parse_cli(["--method", method, "--dataset", "locomo",
                                   "--cache-root", str(tmp_path / "cache")]))
    options["retained_sources_root"] = str(tmp_path / "retained")
    backend_type = resolve("backend", options["backend"])
    backend = backend_type(options)
    history = History("test", (Session("a", tuple(Message(str(i), "Alice", f"hello {i}")
                          for i in range(8)), "2025-05-12"),))
    deps = backend_type.dependencies(options)
    spec = BuildSpec(fingerprint({"history": history.digest, "deps": deps}), method, str(tmp_path), deps)

    async def embed(texts):
        vectors = np.zeros((len(texts), 4096), dtype=np.float32)
        vectors[:, 0] = 1
        return vectors, {"requests": 0, "texts": len(texts), "prompt_tokens": 0, "cost_usd": 0}

    backend.embed = embed
    try:
        await backend.prepare(history, spec)
        assert [len(c["message_spans"]) for c in backend.mapping.values()] == [3, 3, 2]
        assert [source["message_id"] for chunk in backend.mapping.values()
                for source in chunk["sources"]] == [str(i) for i in range(8)]
        assert deps["chunking"]["budget"] == "soft"
        assert deps["chunking"]["messages_per_chunk"] == 3
    finally:
        await backend.aclose()
