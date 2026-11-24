import json

import numpy as np
import pytest

from reproduction.experiments.lite.backend import BatchedLite, LiteBackend
from reproduction.experiments.main.paper_backend import PaperSAGMem
from reproduction.run import configure, parse_cli, preflight
from src.harness.benchmark import BuildSpec, History, Message, Session, fingerprint


@pytest.mark.parametrize("method", ["sag-mem", "sag-mem-lite"])
@pytest.mark.parametrize("dataset", ["locomo", "longmemeval"])
def test_cli_message_cap_changes_preflight_chunks(tmp_path, method, dataset):
    messages = [{"dia_id": f"m{i}", "speaker": "Alice", "text": f"Message {i} about Paris."}
                for i in range(6)]
    if dataset == "locomo":
        sample = {"sample_id": "cap", "conversation": {
            "session_1_date_time": "2026-11-05", "session_1": messages},
            "qa": [{"question": "Which city?", "answer": "Paris", "category": 4}]}
    else:
        sample = {"question_id": "cap", "question_type": "single-session-user",
                  "question": "Which city?", "answer": "Paris", "question_date": "2026-11-06",
                  "answer_session_ids": ["s1"], "haystack_session_ids": ["s1"],
                  "haystack_dates": ["2026-11-05"], "haystack_sessions": [[
                      {"role": "user", "content": message["text"]} for message in messages]]}
    data = tmp_path / "data.json"
    data.write_text(json.dumps([sample]))
    for cap, expected_chunks in [(None, 2), (0, 1), (3, 2), (1, 6)]:
        args = parse_cli(["--method", method, "--dataset", dataset, "--data-path", str(data),
                          *(["--messages-per-chunk", str(cap)] if cap is not None else []), "--preflight"])
        assert args.messages_per_chunk == (3 if cap is None else cap)
        audit = preflight(configure(args))
        assert audit["chunks"] == expected_chunks, (method, dataset, cap, audit)
        assert audit["messages"] == 6
        assert audit["model_calls"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_type", [PaperSAGMem, LiteBackend, BatchedLite])
@pytest.mark.parametrize("cap,expected_chunks", [(None, 1), (3, 2), (1, 6)])
async def test_index_and_prefetch_use_configured_message_cap(tmp_path, monkeypatch, backend_type,
                                                          cap, expected_chunks):
    history = History("cap", (Session("s", tuple(
        Message(f"m{i}", "user", f"Message {i} about Paris.") for i in range(6)
    ), "2026-11-05"),))
    options = {"backend_options": {"candidate_k": 100, "max_hops": 2},
               "cache_root": str(tmp_path / "cache"),
               "retained_sources_root": str(tmp_path / "retained")}
    if cap is not None:
        options["backend_options"]["messages_per_chunk"] = cap
    embedded = []

    async def embed(self, texts):
        embedded.extend(texts)
        return np.array([[1.0] + [0.0] * 4095] * len(texts), dtype=np.float32), {
            "requests": 0, "texts": len(texts), "prompt_tokens": 0, "cost_usd": 0}

    monkeypatch.setattr(LiteBackend, "embed", embed)
    dependencies = backend_type.dependencies(options)
    assert dependencies["chunking"]["messages_per_chunk"] == cap
    policy = "soft512_atomic_v1" if cap is None else f"{cap}_messages_soft512_atomic_v1"
    assert dependencies["chunking"]["policy"] == policy
    spec = BuildSpec(fingerprint(dependencies), "cap", str(tmp_path / "indexes"), dependencies)
    backend = backend_type(options)
    try:
        await backend.prepare(history, spec)
        sources = json.loads((spec.directory / "sources.json").read_text())
        assert len(sources) == expected_chunks
        assert sum(len(row["sources"]) for row in sources.values()) == 6
        if cap is not None:
            assert all(len(row["sources"]) <= cap for row in sources.values())
        assert backend.build_stats["messages_per_chunk"] == cap
        assert backend.build_stats["chunk_policy"] == policy
        if issubclass(backend_type, LiteBackend):
            assert set(embedded) == {row["content"] for row in sources.values()}
        manifest = json.loads((spec.directory / "build.json").read_text())
        assert manifest["dependencies"]["chunking"] == dependencies["chunking"]
    finally:
        await backend.aclose()


def test_message_caps_have_distinct_build_identities():
    options = {"backend_options": {}}
    uncapped = LiteBackend.dependencies(options)
    capped = LiteBackend.dependencies({"backend_options": {"messages_per_chunk": 3}})
    assert fingerprint(uncapped) != fingerprint(capped)
