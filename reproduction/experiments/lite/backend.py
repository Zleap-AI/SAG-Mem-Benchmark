"""SAG-Mem-Lite (§3.5): the shared chunks, the SAG2 lexical expansion, and cosine scores.

The encoder is an explicit profile (``experiments/shared/profiles.py``). A chunk longer than the
encoder window is never shortened silently: the default rule is ``error``; ``prefix`` sends the
longest original prefix that fits the window while the stored and Reader-visible text stays whole.
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI
from src.backends.adapters import manifest_matches, publish
from src.backends.embedding_tokenizer import TOKENIZER_PATH as BGE_TOKENIZER
from src.backends.embedding_tokenizer import EmbeddingTokenizer
from src.backends.sag2_adapter import _chunk_counts, _corpus
from src.backends.sag2_chunk_time import annotate_chunk
from src.backends.sag2_embedding import SAG2Embedding
from src.backends.sag2_package import CorpusIndex
from src.harness.benchmark import atomic_json
from tokenizers import Tokenizer

from ..shared import chunk_width
from ..shared.profiles import api_settings, embedding_profile
from ..shared.runtime import audit_chunks, chunking, file_digest, fingerprint, sha256

ROOT = Path(__file__).resolve().parent
TOKENIZERS = {"qwen3": ROOT / "assets/tokenizer.json", "bge": BGE_TOKENIZER}


def tokenizer_path(options: dict) -> Path:
    return Path(options.get("embedding_tokenizer") or TOKENIZERS[embedding_profile(options)["tokenizer"]])


class LiteBackend(SAG2Embedding):
    @classmethod
    def validate(cls, options, examples):
        _, retrieval_options = chunk_width.split_options(options)
        super().validate(retrieval_options, examples)

    @classmethod
    def dependencies(cls, options):
        messages_per_chunk, retrieval_options = chunk_width.split_options(options)
        profile = embedding_profile(options)
        result = super().dependencies(retrieval_options)
        result["family"] = "sag_mem_lite"
        result["chunking"].update(policy=chunking.POLICY, budget="soft", messages_per_chunk=None,
                                  oversized_message="keep_complete_and_alone", cross_session=False)
        chunker = chunking
        if messages_per_chunk is not None:
            result["chunking"].update(policy=f"{messages_per_chunk}_messages_soft512_atomic_v1",
                                      messages_per_chunk=messages_per_chunk)
            chunker = chunk_width
        result["embedding"].update(profile=profile["name"], model=profile["model"],
                                    provider=profile["extra_body"].get("provider"),
                                    dimensions=profile["dimensions"],
                                    max_input_tokens=profile["max_input_tokens"],
                                    overflow=profile["overflow"],
                                    truncation=profile["overflow"] != "error", full_source_text=True)
        result["implementation"].update(shared_chunker=file_digest(Path(chunker.__file__)),
                                        lite_backend=file_digest(Path(__file__)),
                                        common=file_digest(ROOT.parent / "shared/runtime.py"),
                                        profiles=file_digest(ROOT.parent / "shared/profiles.py"),
                                        embedding_tokenizer=file_digest(tokenizer_path(options)))
        return result

    @classmethod
    def runtime_dependencies(cls, options):
        from src.harness import relative_time
        _, retrieval_options = chunk_width.split_options(options)
        return {**super().runtime_dependencies(retrieval_options), "name": "SAG-Mem/Lite", "jev_calls": 0,
                "context_truncation": False, "full_source_text": True,
                "reader_time_rules_sha256": file_digest(Path(relative_time.__file__)),
                "reader_format": "source_date_relative_time_inline_v8"}

    def __init__(self, options, ledger=None):
        messages_per_chunk, retrieval_options = chunk_width.split_options(options)
        super().__init__(retrieval_options)
        self.options = options
        self.messages_per_chunk = messages_per_chunk
        self.chunk_policy = (chunking.POLICY if messages_per_chunk is None else
                             f"{messages_per_chunk}_messages_soft512_atomic_v1")
        self.ledger = ledger
        self.profile = embedding_profile(options)
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path(options)))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        cache_root = Path(options["cache_root"])
        cache_root.mkdir(parents=True, exist_ok=True)
        self.cache = sqlite3.connect(cache_root / "embedding_cache.sqlite")
        self.cache.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, text_hash TEXT NOT NULL, "
                           "vector BLOB NOT NULL, vector_hash TEXT NOT NULL)")

    def _chunks(self, history):
        if self.messages_per_chunk is None:
            return chunking.chunk_history(history)
        return chunk_width.chunk_history(history, self.messages_per_chunk)

    def encoder_input(self, text: str) -> tuple[str, int]:
        """Text sent to the encoder and its token count; cache and index stay keyed by the full text."""
        limit = self.profile["max_input_tokens"]
        count = len(self.tokenizer.encode(text).ids)
        if count <= limit:
            return text, count
        if self.profile["overflow"] == "error":
            raise ValueError(f"Complete embedding input exceeds {limit} tokens; choose an encoder with a "
                             "larger window or select the explicit 'prefix' overflow rule")
        if self.profile["tokenizer"] != "bge":
            raise ValueError("The prefix overflow rule is implemented for the BGE tokenizer only")
        shortened = EmbeddingTokenizer("BAAI/bge-large-en-v1.5").fitting_prefix(text)
        return shortened, len(self.tokenizer.encode(shortened).ids)

    async def embed(self, texts):
        profile = self.profile
        chunker = chunking if self.messages_per_chunk is None else chunk_width
        policy = {"model": profile["model"], "dimensions": profile["dimensions"], "body": profile["extra_body"],
                  "chunking": self.chunk_policy,
                  "chunker_sha256": file_digest(Path(chunker.__file__))}
        if profile["overflow"] != "error":
            policy["overflow"] = f"{profile['overflow']}_{profile['max_input_tokens']}"
        keys = [fingerprint({**policy, "text_sha256": sha256(text)}) for text in texts]
        vectors, missing = {}, {}
        for key, text in zip(keys, texts, strict=True):
            cached = self.cache.execute("SELECT text_hash,vector,vector_hash FROM vectors WHERE key=?",
                                        (key,)).fetchone()
            if cached:
                if cached[0] != sha256(text) or cached[2] != sha256(cached[1]):
                    raise ValueError("Embedding cache content hash mismatch")
                vectors[key] = np.frombuffer(cached[1], dtype="<f4")
            else:
                missing[key] = text
        usage = {"requests": 0, "texts": 0, "prompt_tokens": 0, "cost_usd": 0,
                 "cache_hits": len(texts) - len(missing), "shortened_inputs": 0}
        if missing:
            prepared = [self.encoder_input(text) for text in missing.values()]
            inputs, counts = [sent for sent, _ in prepared], [count for _, count in prepared]
            shortened = sum(sent != text for sent, text in zip(inputs, missing.values(), strict=True))
            if self.client is None:
                credentials = api_settings(require_jev=False)
                self.client = AsyncOpenAI(base_url=self.config["embedding_base_url"],
                    api_key=os.getenv("SAG2_EMBEDDING_API_KEY") or credentials["api_key"],
                    timeout=240, max_retries=0)
                if self.ledger:
                    self.ledger.wrap(self.client, (self.client.embeddings, "create"), "embedding")
            request = {"model": profile["model"], "input": inputs, "encoding_format": "float"}
            if profile["send_dimensions"]:
                request["dimensions"] = profile["dimensions"]
            if profile["extra_body"]:
                request["extra_body"] = profile["extra_body"]
            response = await self.client.embeddings.create(**request)
            ordered = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in ordered] != list(range(len(missing))):
                raise ValueError("Embedding response has missing or duplicate input indices")
            if profile["name"] == "qwen3" and response.usage.prompt_tokens < sum(counts):
                raise ValueError("Embedding usage is shorter than complete local token count")
            matrix = np.asarray([item.embedding for item in ordered], dtype=np.float32)
            if matrix.shape != (len(missing), profile["dimensions"]) or not np.isfinite(matrix).all():
                raise ValueError(f"Embedding API must return {len(missing)} finite "
                                 f"{profile['dimensions']}-dimensional vectors")
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            if np.any(norms == 0):
                raise ValueError("Embedding API returned a zero vector")
            matrix /= norms
            with self.cache:
                for (key, text), vector in zip(missing.items(), matrix, strict=True):
                    blob = vector.astype("<f4").tobytes()
                    self.cache.execute("INSERT INTO vectors VALUES (?,?,?,?)",
                                       (key, sha256(text), blob, sha256(blob)))
                    vectors[key] = vector
            usage.update(requests=1, texts=len(missing), prompt_tokens=response.usage.prompt_tokens,
                         local_input_tokens=sum(counts), shortened_inputs=shortened,
                         cost_usd=response.usage.model_dump().get("cost"))
        return np.stack([vectors[key] for key in keys]), usage

    async def prepare(self, history, build_spec):
        started = time.perf_counter()
        if fingerprint(build_spec.dependencies) != fingerprint(self.dependencies(self.options)):
            raise ValueError("Build dependency hash changed")
        chunks = self._chunks(history)
        audit = audit_chunks(history, chunks, self.messages_per_chunk)
        corpus, self.mapping = _corpus(chunks, normalize=lambda text: text)
        directory = build_spec.directory
        directory.mkdir(parents=True, exist_ok=True)
        path, sources = directory / "sag2.sqlite", directory / "sources.json"
        reused = manifest_matches(build_spec)
        if reused:
            if not path.exists() and (directory / "index_archive.json").exists():
                from .archive import restore_index
                restore_index(directory)
            manifest = json.loads((directory / "build.json").read_text())
            if (manifest["index_sha256"] != file_digest(path)
                    or manifest["sources_sha256"] != file_digest(sources)
                    or json.loads(sources.read_text()) != self.mapping):
                raise ValueError("Saved index or source content hash mismatch")
        else:
            pending = directory / "pending.json"
            identity = {"build_id": build_spec.id, "dependencies": build_spec.dependencies, **audit}
            if pending.exists() and json.loads(pending.read_text()) != identity:
                raise ValueError("Partial index was built with different source content or rules")
            atomic_json(pending, identity)
            atomic_json(sources, self.mapping)
        self.index = CorpusIndex(path, corpus, history.digest)
        self.vector_db = sqlite3.connect(path)
        self.vector_db.execute("CREATE TABLE IF NOT EXISTS cosine_embeddings "
                               "(document_id TEXT PRIMARY KEY, vector BLOB NOT NULL, "
                               "vector_sha256 TEXT NOT NULL, text_sha256 TEXT NOT NULL)")
        completed = {}
        for identifier, blob, vector_hash, text_hash in self.vector_db.execute("SELECT * FROM cosine_embeddings"):
            if sha256(blob) != vector_hash or sha256(self.mapping[identifier]["content"]) != text_hash:
                raise ValueError("Persisted embedding is corrupt or no longer matches source")
            completed[identifier] = True
        missing = [document for document in corpus if document["idx"] not in completed]
        if reused and missing:
            raise ValueError("Verified index is missing vectors")
        self.build_stats = {"build_reused": reused, **audit, **_chunk_counts(self.mapping),
                            "messages_per_chunk": self.messages_per_chunk, "chunk_policy": self.chunk_policy,
                            "embedding_build_new_texts": len(missing), "embedding_build_requests": 0,
                            "embedding_build_prompt_tokens": 0, "embedding_build_cost_usd": 0,
                            "embedding_build_shortened_inputs": 0,
                            "embedding_profile": self.profile["name"],
                            "embedding_model": self.profile["model"],
                            "embedding_dimensions": self.profile["dimensions"],
                            "extraction_build_requests": 0, "chunk_budget": "soft"}
        for offset in range(0, len(missing), 64):
            batch = missing[offset:offset + 64]
            matrix, usage = await self.embed([document["text"] for document in batch])
            with self.vector_db:
                for document, vector in zip(batch, matrix, strict=True):
                    blob = vector.astype("<f4").tobytes()
                    self.vector_db.execute("INSERT INTO cosine_embeddings VALUES (?,?,?,?)",
                        (document["idx"], blob, sha256(blob), sha256(document["text"])))
            for target, origin in [("embedding_build_requests", "requests"),
                                   ("embedding_build_prompt_tokens", "prompt_tokens"),
                                   ("embedding_build_shortened_inputs", "shortened_inputs")]:
                self.build_stats[target] += usage.get(origin, 0)
            if usage["cost_usd"] is None:
                self.build_stats["embedding_build_cost_usd"] = None
            elif self.build_stats["embedding_build_cost_usd"] is not None:
                self.build_stats["embedding_build_cost_usd"] += usage["cost_usd"]
            print(json.dumps({"stage": "embedding_build", "corpus_id": history.corpus_id,
                              "completed": min(offset + 64, len(missing)), "total": len(missing)}), flush=True)
        self.vector_db.commit()
        self.build_stats["build_elapsed_seconds"] = time.perf_counter() - started
        if not reused:
            publish(build_spec, history_digest=history.digest, index_sha256=file_digest(path),
                    sources_sha256=file_digest(sources), stats=self.build_stats)

    async def retrieve(self, query, options):
        result = await super().retrieve(query, options)
        rendered, explanations = [], []
        for item in result.items:
            text, matches = annotate_chunk(item)
            rendered.append(text)
            explanations.extend(matches)
        return replace(result, context="\n\n".join(rendered),
                       stats={**result.stats, "time_explanations": explanations})

    async def aclose(self):
        await super().aclose()
        self.cache.close()


class BatchedLite(LiteBackend):
    """The paper's bounded embedding prefetch; retrieval and chunking stay unchanged."""

    slots = asyncio.Semaphore(8)

    @classmethod
    def dependencies(cls, options):
        result = super().dependencies(options)
        result["implementation"]["embedding_prefetch"] = file_digest(Path(__file__))
        return result

    async def prepare(self, history, build_spec):
        if manifest_matches(build_spec):
            await super().prepare(history, build_spec)
            return
        corpus, _ = _corpus(self._chunks(history), normalize=lambda text: text)

        async def prefetch(batch):
            async with self.slots:
                _, usage = await self.embed([item["text"] for item in batch])
                return usage

        async with asyncio.TaskGroup() as tasks:
            batches = [tasks.create_task(prefetch(corpus[offset:offset + 64]))
                       for offset in range(0, len(corpus), 64)]
        usages = [batch.result() for batch in batches]
        await super().prepare(history, build_spec)
        self.build_stats["prefetch"] = {
            "global_embedding_batch_concurrency": 8,
            "requests": sum(usage["requests"] for usage in usages),
            "texts": sum(usage["texts"] for usage in usages),
            "prompt_tokens": sum(usage["prompt_tokens"] for usage in usages),
            "cost_usd": (sum(usage["cost_usd"] for usage in usages)
                         if all(usage["cost_usd"] is not None for usage in usages) else None),
        }
