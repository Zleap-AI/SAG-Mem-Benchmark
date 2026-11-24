"""Offline checks for selection and query-cache isolation."""
import sqlite3
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from reproduction.experiments.latency.run import fresh_query_cache, select
from reproduction.experiments.lite.backend import LiteBackend


class LatencyContracts(unittest.TestCase):
    def test_locomo_selection_is_order_independent_and_balanced_by_sample(self):
        examples = [SimpleNamespace(id=f"sample-{sample}:{n}", history_digest=f"h{sample}",
                                    target=SimpleNamespace(category=str(n % 5)))
                    for sample in range(10) for n in range(12)]
        selected = select(examples, "locomo")
        self.assertEqual(len(selected), 100)
        self.assertEqual(Counter(e.history_digest for e in selected), {f"h{n}": 10 for n in range(10)})
        self.assertEqual([e.id for e in selected], [e.id for e in select(list(reversed(examples)), "locomo")])

    def test_query_cache_reset_does_not_modify_persisted_document_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.sqlite"
            connection = sqlite3.connect(path)
            connection.execute("CREATE TABLE vectors (key TEXT PRIMARY KEY)")
            connection.execute("INSERT INTO vectors VALUES ('document')")
            connection.commit()
            backend = SimpleNamespace(cache=connection)
            fresh_query_cache(backend)
            self.assertEqual(backend.cache.execute("SELECT COUNT(*) FROM vectors").fetchone()[0], 0)
            backend.cache.close()
            with sqlite3.connect(path) as saved:
                self.assertEqual(saved.execute("SELECT key FROM vectors").fetchone()[0], "document")


class LocalBgeEmbeddingContracts(unittest.IsolatedAsyncioTestCase):
    async def test_local_bge_accepts_1024_dimensions_without_requesting_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            options = {"backend_options": {"embedding_base_url": "http://127.0.0.1/v1"},
                       "embedding_profile": "bge", "cache_root": directory, "top_k": 20}
            backend = LiteBackend(options)
            requests = []

            async def create(**kwargs):
                requests.append(kwargs)
                usage = SimpleNamespace(prompt_tokens=10, model_dump=dict)
                return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0] * 1024)], usage=usage)

            async def close():
                pass

            backend.client = SimpleNamespace(embeddings=SimpleNamespace(create=create), close=close)
            try:
                vectors, usage = await backend.embed(["short source text"])
                self.assertEqual(vectors.shape, (1, 1024))
                self.assertEqual(usage["requests"], 1)
                self.assertNotIn("dimensions", requests[0])
                self.assertNotIn("extra_body", requests[0])
            finally:
                await backend.aclose()


if __name__ == "__main__":
    unittest.main()
