import json

import pytest

from src.backends import sag2_adapter
from src.backends.sag2_chunk_time import SAG2ChunkTimeBackend
from src.harness import benchmark as b


@pytest.mark.parametrize(("supplied", "candidate_k", "max_hops"), [
    ({}, 50, 1),
    ({"candidate_k": 100}, 100, 1),
    ({"max_hops": 0}, 50, 0),
    ({"candidate_k": 100, "max_hops": 2}, 100, 2),
])
def test_time_defaults_and_explicit_overrides(supplied, candidate_k, max_hops, monkeypatch):
    options = {"backend_options": supplied}
    before = json.dumps(options, sort_keys=True)
    monkeypatch.setattr(sag2_adapter, "_source", lambda config: {"package": "test"})
    monkeypatch.setattr(sag2_adapter, "_service", lambda config: ({"model": "test"}, "private"))
    backend = SAG2ChunkTimeBackend(options)
    runtime = SAG2ChunkTimeBackend.runtime_dependencies(options)["retrieval"]
    assert backend.config["candidate_k"] == runtime["candidate_k"] == candidate_k
    assert backend.config["max_hops"] == runtime["max_hops"] == max_hops
    assert json.dumps(options, sort_keys=True) == before
    original = sag2_adapter.SAG2Backend({})
    assert original.config["candidate_k"] == 100 and original.config["max_hops"] == 2


def test_validation_uses_time_candidate_default_before_loading_services():
    with pytest.raises(ValueError, match="candidate_k must be at least top_k"):
        SAG2ChunkTimeBackend.validate({"top_k": 51, "explicit": []}, [])


def test_registry_and_runtime_identity(monkeypatch):
    options = {"backend_options": {"candidate_k": 50, "max_hops": 1}, "top_k": 20}
    monkeypatch.setattr(sag2_adapter, "_source", lambda config: {"package": "test"})
    monkeypatch.setattr(sag2_adapter, "_service", lambda config: ({"model": "test"}, "private"))
    assert b.resolve("backend", "sag2-chunk-time") is SAG2ChunkTimeBackend
    plain = sag2_adapter.SAG2Backend.runtime_dependencies(options)
    timed = SAG2ChunkTimeBackend.runtime_dependencies(options)
    assert timed["retrieval"] == plain["retrieval"]
    assert timed["jev"] == plain["jev"]
    assert timed["reader_context"]["day_display"] == "YYYY/MM/DD"
    assert "private" not in json.dumps(timed)


@pytest.mark.asyncio
async def test_real_retrieval_keeps_index_and_jev_input_unchanged(tmp_path):
    pytest.importorskip("zleap.sag2")
    options = {"backend_options": {"candidate_k": 50, "max_hops": 1}, "top_k": 2}
    history = b.History("books", (
        b.Session("start", (b.Message("m1", "user", "I started the book today."),),
                  "2023/01/10 (Tue) 10:34"),
        b.Session("finish", (b.Message("m2", "user", "I finished the book yesterday."),),
                  "2023/01/31 (Tue) 23:59"),
    ))
    dependencies = SAG2ChunkTimeBackend.dependencies(options)
    assert dependencies == sag2_adapter.SAG2Backend.dependencies(options)
    assert SAG2ChunkTimeBackend.runtime_dependencies(options) != sag2_adapter.SAG2Backend.runtime_dependencies(options)
    spec = b.BuildSpec("shared-index", "test", str(tmp_path), dependencies)
    calls = []

    class Scorer:
        async def score(self, query, candidates, evidence):
            calls.append((query, candidates, evidence))
            return ([{**item, "jev_score": 0.9} for item in candidates],
                    {"requests": 1, "judgments": len(candidates), "input_tokens": 100,
                     "output_tokens": 10, "batches": []})

    results = []
    for backend_type in (SAG2ChunkTimeBackend, SAG2ChunkTimeBackend):
        backend = backend_type(options)
        await backend.prepare(history, spec)
        backend.client = Scorer()
        try:
            retrieve = sag2_adapter.SAG2Backend.retrieve if not results else SAG2ChunkTimeBackend.retrieve
            results.append(await retrieve(backend, b.Query("When did I read the book?"), options))
            if len(results) == 2:
                assert backend.build_stats["build_reused"] is True
        finally:
            await backend.aclose()
    plain, timed = results
    assert calls[0] == calls[1]
    assert all("[Message date:" not in item["content"] for item in calls[0][1])
    assert plain.items == timed.items
    assert plain.stats["sag2_trace"] == timed.stats["sag2_trace"]
    assert plain.stats["jev_usage"] == timed.stats["jev_usage"]
    assert "user: [Message date: 2023/01/10 10:34]\nI started the book today（2023/01/10）." in timed.context
    assert "yesterday（2023/01/30）" in timed.context
    assert "today（" not in plain.context
    assert len(timed.stats["time_explanations"]) == 2


@pytest.mark.asyncio
async def test_empty_retrieval_and_source_metadata_are_preserved(monkeypatch):
    original = b.RetrievalResult("", [], {"jev_usage": {"requests": 0}}, "message", "chunk")

    async def retrieve(self, query, options):
        return original

    monkeypatch.setattr(sag2_adapter.SAG2Backend, "retrieve", retrieve)
    backend = SAG2ChunkTimeBackend({"backend_options": {}})
    result = await backend.retrieve(b.Query("unknown"), {})
    assert result.context == ""
    assert result.items == []
    assert result.source_granularity == "message" and result.metric_family == "chunk"
    assert result.stats == {"jev_usage": {"requests": 0}, "time_explanations": []}
    assert original.stats == {"jev_usage": {"requests": 0}}
