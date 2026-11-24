"""SAG2-Chunk ablation: persistent embeddings replace JEV judgments."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI

from ..config import settings
from ..harness.benchmark import (
    BuildSpec,
    History,
    Query,
    RetrievalResult,
    atomic_json,
    file_digest,
    fingerprint,
    register,
)
from .adapters import manifest_matches, publish, reject_options
from .embedding_tokenizer import EmbeddingTokenizer
from .raw_chunks import CHUNK_POLICY, CHUNK_TOKENIZER, CHUNK_TOKENS, chunk_history
from .sag2_adapter import _chunk_counts, _corpus, _source
from .sag2_package import CorpusIndex, normalize_chunk

MODEL = "qwen/qwen3-embedding-8b"
DIMENSIONS = 4096
QUERY_INSTRUCTION = "Given a question, retrieve relevant conversation passages that answer the question"
DEFAULTS = {"candidate_k": 100, "seed_k": 10, "max_hops": 2, "expansion_term_limit": 20,
            "embedding_batch_size": 32}


def config_for(options: dict) -> dict:
    given = options.get("backend_options", {})
    if unknown := set(given) - {*DEFAULTS, "embedding_model", "embedding_base_url"}:
        raise ValueError(f"Unknown sag2_embedding options: {sorted(unknown)}")
    config = {**DEFAULTS, "embedding_model": MODEL,
              "embedding_base_url": os.getenv("SAG2_EMBEDDING_BASE_URL") or settings.openai_base_url,
              **given}
    for name in DEFAULTS:
        minimum = 0 if name == "max_hops" else 1
        if type(config[name]) is not int or config[name] < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if config["max_hops"] > 2:
        raise ValueError("At most two expansion rounds are supported")
    if options.get("top_k", 10) > config["candidate_k"]:
        raise ValueError("top_k must not exceed candidate_k")
    return config


def normalized_vectors(values, count: int) -> np.ndarray:
    vectors = np.asarray(values, dtype=np.float32)
    if vectors.shape != (count, DIMENSIONS) or not np.isfinite(vectors).all():
        raise ValueError(f"Embedding API must return {count} finite {DIMENSIONS}-dimensional vectors")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding API returned a zero vector")
    return vectors / norms


async def search_cosine(index, vector_db, query: str, query_vector: np.ndarray, *, top_k: int,
                        candidate_k: int, seed_k: int, max_hops: int,
                        expansion_term_limit: int) -> dict:
    """Keep SAG2 lexical expansion and history retention; score every new candidate."""
    started = time.perf_counter()
    retained, carried, protected = {}, {}, {}
    seen, expanded, frontier, trace = set(), set(), [], []

    def ranked():
        return sorted(retained.values(), key=lambda c: (-c["cosine_score"], -c["fusion_score"], c["id"]))

    expression = await asyncio.to_thread(index.match_query, query)
    for hop in range(max_hops + 1):
        if not expression or (hop and not frontier):
            break
        # SAG2's expansion weighting expects a nonnegative relevance score.
        # This monotone mapping preserves cosine order, including negative values.
        seeds = [{**c, "jev_score": (c["cosine_score"] + 1) / 2} for c in frontier]
        terms = await asyncio.to_thread(index.expansion_terms, query, seeds, expansion_term_limit) if hop else []
        if hop and not terms:
            break
        hits, queries = (
            await asyncio.to_thread(index.retrieve_expansion, expression, terms, seen, candidate_k)
            if hop else (await asyncio.to_thread(index.retrieve, expression, seen, candidate_k), [])
        )
        incoming = [{**hit, "hop": hop, "fusion_score": 1 / (61 + i)} for i, hit in enumerate(hits)]
        if not incoming:
            break
        if hop == 0:
            protected = {c["id"]: c for c in incoming[:5]}
        carried.update({c["id"]: c for c in frontier})
        pool = list({c["id"]: c for c in [*protected.values(), *carried.values(), *incoming]}.values())
        ids = [str(c["document_id"]) for c in pool]
        placeholders = ",".join("?" for _ in ids)
        vectors = {key: np.frombuffer(blob, dtype="<f4") for key, blob in vector_db.execute(
            f"SELECT document_id, vector FROM cosine_embeddings WHERE document_id IN ({placeholders})", ids)}
        scores = np.clip(np.stack([vectors[key] for key in ids]) @ query_vector, -1, 1)
        retained = {c["id"]: {**c, "cosine_score": float(score)}
                    for c, score in zip(pool, scores, strict=True)}
        seen.update(retained)
        frontier = [c for c in ranked() if c["id"] not in expanded][:seed_k]
        trace.append({"hop": hop, "stage": "expansion" if hop else "bm25",
                      "new_candidate_ids": [c["id"] for c in incoming], "new_candidate_count": len(incoming),
                      "retained_candidate_count": len(pool) - len(incoming),
                      "terms": terms, "expansion_queries": queries,
                      "ranked": [{"id": c["id"], "score": c["cosine_score"]} for c in ranked()],
                      "next_seed_ids": [c["id"] for c in frontier]})
        expanded.update(c["id"] for c in frontier)

    selected = ranked()[:top_k]
    return {"candidates": selected, "trace": trace, "selected_ids": [c["id"] for c in selected],
            "elapsed_ms": round((time.perf_counter() - started) * 1000)}


class SAG2Embedding:
    def __init__(self, options: dict):
        self.options, self.config = options, config_for(options)
        self.index = self.vector_db = self.client = None

    @classmethod
    def validate(cls, options: dict, examples: list) -> None:
        reject_options(options, {"strategy", "return_type", "evidence_mode", "extraction_profile", "graph_limit"})
        if options.get("time_filter", "none") != "none":
            raise ValueError("sag2_embedding does not implement query-time filtering")
        config = config_for(options)
        if not config["embedding_base_url"] or not (os.getenv("SAG2_EMBEDDING_API_KEY") or settings.openai_api_key):
            raise ValueError("Configure SAG2_EMBEDDING_BASE_URL/API_KEY or OPENAI_BASE_URL/API_KEY")
        _source({})
        EmbeddingTokenizer(CHUNK_TOKENIZER)

    @classmethod
    def dependencies(cls, options: dict) -> dict:
        config = config_for(options)
        return {"family": "sag2_embedding", "version": 1, "source": _source({}),
                "chunking": {"tokens": CHUNK_TOKENS, "policy": CHUNK_POLICY,
                             "tokenizer": EmbeddingTokenizer.dependencies()},
                "embedding": {"model": config["embedding_model"], "base_url": config["embedding_base_url"],
                              "dimensions": DIMENSIONS, "normalization": "l2_float32", "document_prefix": ""},
                "storage": "sqlite_bm25_and_embeddings", "duplicate_policy": "merge_sources",
                "implementation": {name: file_digest(Path(__file__).with_name(name))
                                   for name in ("sag2_embedding.py", "raw_chunks.py", "sag2_package.py")}}

    @classmethod
    def runtime_dependencies(cls, options: dict) -> dict:
        config = config_for(options)
        return {"algorithm": "SAG2_CHUNK_BM25_COSINE", "retrieval": {k: config[k] for k in DEFAULTS},
                "query_instruction": QUERY_INSTRUCTION, "final_order": "cosine_descending", "dpp": False,
                "history_retention": "first_five_bm25_and_all_previous_seeds",
                "third_round_new_candidate_cap": config["candidate_k"]}

    async def embed(self, texts: list[str]) -> tuple[np.ndarray, dict]:
        if self.client is None:
            self.client = AsyncOpenAI(base_url=self.config["embedding_base_url"],
                                      api_key=os.getenv("SAG2_EMBEDDING_API_KEY") or settings.openai_api_key,
                                      timeout=120, max_retries=0)
        response = await self.client.embeddings.create(
            model=self.config["embedding_model"], input=texts, dimensions=DIMENSIONS, encoding_format="float")
        records = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in records] != list(range(len(texts))):
            raise ValueError("Embedding API returned missing or duplicate input indices")
        return normalized_vectors([item.embedding for item in records], len(texts)), {
            "requests": 1, "texts": len(texts), "prompt_tokens": response.usage.prompt_tokens}

    async def prepare(self, history: History, build_spec: BuildSpec) -> None:
        if fingerprint(build_spec.dependencies) != fingerprint(self.dependencies(self.options)):
            raise ValueError("Embedding build dependencies changed")
        build_spec.directory.mkdir(parents=True, exist_ok=True)
        path, mapping_path = build_spec.directory / "sag2.sqlite", build_spec.directory / "sources.json"
        corpus, self.mapping = _corpus(chunk_history(history), normalize_chunk)
        reused = manifest_matches(build_spec)
        if reused:
            manifest = json.loads((build_spec.directory / "build.json").read_text())
            if file_digest(path) != manifest["index_sha256"] or file_digest(mapping_path) != manifest["sources_sha256"]:
                raise ValueError("Verified embedding index or sources are corrupt")
        self.index = CorpusIndex(path, corpus, history.digest)
        self.vector_db = sqlite3.connect(path)
        self.vector_db.execute("CREATE TABLE IF NOT EXISTS cosine_embeddings "
                               "(document_id TEXT PRIMARY KEY, vector BLOB NOT NULL)")
        self.vector_db.execute("CREATE TABLE IF NOT EXISTS cosine_embedding_usage "
                               "(texts INTEGER NOT NULL, prompt_tokens INTEGER NOT NULL)")
        completed = {row[0] for row in self.vector_db.execute("SELECT document_id FROM cosine_embeddings")}
        missing = [document for document in corpus if document["idx"] not in completed]
        if reused and missing:
            raise ValueError("Verified embedding index is incomplete")
        batch_size = self.config["embedding_batch_size"]
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            vectors, usage = await self.embed([document["text"] for document in batch])
            with self.vector_db:
                self.vector_db.executemany("INSERT INTO cosine_embeddings VALUES (?, ?)",
                                          [(document["idx"], vector.astype("<f4").tobytes())
                                           for document, vector in zip(batch, vectors, strict=True)])
                self.vector_db.execute("INSERT INTO cosine_embedding_usage VALUES (?, ?)",
                                       (usage["texts"], usage["prompt_tokens"]))
        if not reused:
            self.vector_db.commit()
            atomic_json(mapping_path, self.mapping)
            publish(build_spec, history_digest=history.digest, index_sha256=file_digest(path),
                    sources_sha256=file_digest(mapping_path), **_chunk_counts(self.mapping))
        requests, texts, tokens = self.vector_db.execute(
            "SELECT COUNT(*), COALESCE(SUM(texts), 0), COALESCE(SUM(prompt_tokens), 0) "
            "FROM cosine_embedding_usage").fetchone()
        self.build_stats = {"build_reused": reused, **_chunk_counts(self.mapping),
                            "embedding_model": self.config["embedding_model"], "embedding_dimensions": DIMENSIONS,
                            "embedding_build_requests": requests, "embedding_build_texts": texts,
                            "embedding_build_prompt_tokens": tokens,
                            "embedding_build_new_texts": len(missing), "extraction_build_requests": 0}

    async def retrieve(self, query: Query, options: dict) -> RetrievalResult:
        if not self.mapping or not query.text.strip():
            return RetrievalResult("", [], dict(self.build_stats), "message", "chunk")
        started = time.perf_counter()
        # One embedding of the original question is reused for every hop.
        vector, usage = await self.embed([f"Instruct: {QUERY_INSTRUCTION}\nQuery:{query.text}"])
        result = await search_cosine(self.index.query_view(), self.vector_db, query.text, vector[0],
                                     top_k=options["top_k"], **{k: self.config[k] for k in DEFAULTS
                                                              if k != "embedding_batch_size"})
        items = [{**self.mapping[str(hit["document_id"])], "score": hit["cosine_score"], "kind": "chunk",
                  "sag2_chunk_id": hit["id"], "hop": hit["hop"]} for hit in result["candidates"]]
        stats = {**self.build_stats, "algorithm": "SAG2_CHUNK_BM25_COSINE", "sag2_trace": result["trace"],
                 "embedding_query_usage": usage,
                 "retrieval_elapsed_ms": round((time.perf_counter() - started) * 1000),
                 "cosine_search_elapsed_ms": result["elapsed_ms"],
                 "selected_ids": result["selected_ids"],
                 "jev_usage": {"requests": 0, "judgments": 0, "input_tokens": 0, "output_tokens": 0}}
        return RetrievalResult("\n\n".join(item["content"] for item in items), items, stats, "message", "chunk")

    async def aclose(self) -> None:
        if self.index is not None:
            self.index.close()
        if self.vector_db is not None:
            self.vector_db.close()
        if self.client is not None:
            await self.client.close()


register("backend", "sag2_embedding", SAG2Embedding)
