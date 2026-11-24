import math
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("zleap.sag2")

from src.backends import sag2_embedding as se
from src.harness.benchmark import BuildSpec, History, Message, Query, Session, fingerprint


def vector(score):
    value = np.zeros(se.DIMENSIONS, dtype=np.float32)
    value[:2] = [score, math.sqrt(max(0, 1 - score * score))]
    return value


def options(**overrides):
    return {"top_k": 10, "backend_options": {}, "explicit": [], **overrides}


@pytest.mark.asyncio
async def test_final_order_keeps_high_similarity_near_duplicates():
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE cosine_embeddings (document_id TEXT PRIMARY KEY, vector BLOB)")
    db.executemany("INSERT INTO cosine_embeddings VALUES (?, ?)",
                   [(str(i), vector(score).astype("<f4").tobytes())
                    for i, score in enumerate([0.99, 0.98, 0.80])])

    class Index:
        def match_query(self, query):
            return query

        def retrieve(self, expression, excluded, limit):
            return [{"id": str(i), "document_id": str(i), "content": text}
                    for i, text in enumerate([
                        "alpha shared shared", "alpha shared shared update", "omega unrelated"])]

    result = await se.search_cosine(Index(), db, "alpha", vector(1), top_k=3,
                                    candidate_k=100, seed_k=10, max_hops=0, expansion_term_limit=20)
    assert [c["id"] for c in result["candidates"]] == ["0", "1", "2"]
    assert result["selected_ids"] == ["0", "1", "2"]
    db.close()


@pytest.mark.asyncio
async def test_every_third_round_candidate_is_scored_and_seeds_are_not_reused():
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE cosine_embeddings (document_id TEXT PRIMARY KEY, vector BLOB)")
    db.executemany("INSERT INTO cosine_embeddings VALUES (?, ?)",
                   [(str(i), vector(0.1 + i / 400).astype("<f4").tobytes()) for i in range(300)])

    class Index:
        def __init__(self):
            self.seeds = []
            self.queries = []

        def match_query(self, query):
            return query

        def retrieve(self, expression, excluded, limit):
            return [{"id": f"chunk-{i}", "document_id": i, "content": f"project evidence {i}"}
                    for i in range(100)]

        def expansion_terms(self, query, seeds, limit):
            self.queries.append(query)
            self.seeds.append([c["id"] for c in seeds])
            return [{"term": "project", "score": 1}]

        def retrieve_expansion(self, expression, terms, excluded, limit):
            start = len(self.seeds) * 100
            return ([{"id": f"chunk-{i}", "document_id": i, "content": f"project evidence {i}"}
                     for i in range(start, start + 100)], [])

    index = Index()
    result = await se.search_cosine(index, db, "original question", vector(1), top_k=10,
                                    candidate_k=100, seed_k=10, max_hops=2, expansion_term_limit=20)
    assert [r["new_candidate_count"] for r in result["trace"]] == [100, 100, 100]
    assert len(result["trace"][2]["ranked"]) == 125  # 5 protected + 20 seeds + 100 new
    assert not set(index.seeds[0]) & set(index.seeds[1])
    assert index.queries == ["original question", "original question"]
    assert result["selected_ids"] == [f"chunk-{i}" for i in range(299, 289, -1)]
    assert len(result["candidates"]) == 10
    db.close()


@pytest.mark.asyncio
async def test_sqlite_embeddings_reused_and_only_one_query_embedding_for_all_hops(tmp_path):
    history = History("test", tuple(Session(str(i), (Message(str(i), "Alice", text),), "2025-05-12")
                                   for i, text in enumerate([
                                       "The project alpha uses Cedar library.",
                                       "The project beta is a weather service.",
                                       "Cedar library was installed yesterday.",
                                   ])))
    opts = options(top_k=2)
    backend = se.SAG2Embedding(opts)
    build = BuildSpec(fingerprint({"history": history.digest, "deps": backend.dependencies(opts)}),
                      "test", str(tmp_path), backend.dependencies(opts))
    calls = []

    async def embed(texts):
        calls.append(texts)
        values = [vector(1 if t.startswith("Instruct:") else 0.9 if "alpha" in t else 0.5) for t in texts]
        return np.stack(values), {"requests": 1, "texts": len(texts), "prompt_tokens": len(texts) * 7}

    backend.embed = embed
    await backend.prepare(history, build)
    assert backend.build_stats["embedding_build_new_texts"] == 3
    assert backend.vector_db.execute("SELECT COUNT(*) FROM cosine_embeddings").fetchone()[0] == 3
    calls.clear()
    result = await backend.retrieve(Query("Which project uses Cedar?"), opts)
    assert len(calls) == 1
    assert calls[0] == [f"Instruct: {se.QUERY_INSTRUCTION}\nQuery:Which project uses Cedar?"]
    assert result.items[0]["score"] == pytest.approx(0.9)
    assert result.stats["jev_usage"]["requests"] == 0
    assert result.stats["extraction_build_requests"] == 0
    assert result.stats["selected_ids"] == [item["sag2_chunk_id"] for item in result.items]
    assert [item["score"] for item in result.items] == sorted(
        [item["score"] for item in result.items], reverse=True)
    assert all("date: 2025-05-12" in item["content"] for item in result.items)
    await backend.aclose()

    reopened = se.SAG2Embedding(opts)
    reopened.embed = embed
    calls.clear()
    await reopened.prepare(history, build)
    assert calls == []
    assert reopened.build_stats["build_reused"] is True
    assert reopened.build_stats["embedding_build_new_texts"] == 0
    await reopened.aclose()


@pytest.mark.asyncio
async def test_empty_history_needs_no_embedding(tmp_path):
    backend = se.SAG2Embedding(options())
    history = History("empty", ())
    build = BuildSpec("empty", "test", str(tmp_path), backend.dependencies(options()))
    await backend.prepare(history, build)
    result = await backend.retrieve(Query("A question"), options())
    assert result.items == []
    assert backend.client is None
    await backend.aclose()


@pytest.mark.parametrize("bad", [np.zeros((1, se.DIMENSIONS)), [[1, 2]],
                                np.full((1, se.DIMENSIONS), float("nan"))])
def test_invalid_provider_vectors_are_rejected(bad):
    with pytest.raises(ValueError, match="Embedding API"):
        se.normalized_vectors(bad, 1)


@pytest.mark.asyncio
async def test_bge_profile_uses_native_dimensions_without_dimensions_parameter(tmp_path):
    from reproduction.experiments.lite.backend import LiteBackend

    opts = options(embedding_profile="bge", cache_root=str(tmp_path))
    backend = LiteBackend(opts)
    calls = []

    class FakeEmbeddings:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                data=[SimpleNamespace(index=0, embedding=np.ones(1024, dtype=np.float32).tolist())],
                usage=SimpleNamespace(prompt_tokens=3, model_dump=dict),
            )

    class FakeClient:
        embeddings = FakeEmbeddings()

        async def close(self):
            return None

    backend.client = FakeClient()
    vectors, usage = await backend.embed(["test input"])

    assert vectors.shape == (1, 1024)
    assert "dimensions" not in calls[0]
    assert calls[0]["model"] == backend.profile["model"]
    assert backend.dependencies(opts)["embedding"]["dimensions"] == 1024
    assert backend.profile["send_dimensions"] is False
    assert usage["prompt_tokens"] == 3
    await backend.aclose()


def test_embedding_model_uses_explicit_model_option():
    opts = options(backend_options={"embedding_model": "Qwen/Qwen3-Embedding-8B"})
    assert se.config_for(opts)["embedding_model"] == "Qwen/Qwen3-Embedding-8B"


def test_retrieval_changes_do_not_rebuild_embeddings():
    a = options()
    b = options(top_k=20, backend_options={"max_hops": 1})
    assert se.SAG2Embedding.dependencies(a) == se.SAG2Embedding.dependencies(b)
    assert se.SAG2Embedding.runtime_dependencies(a) != se.SAG2Embedding.runtime_dependencies(b)
