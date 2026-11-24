"""The paper backend builds complete-message indexes without any service call."""
import json

import pytest
from reproduction.experiments.main.paper_backend import PaperSAGMem
from src.backends import sag2_adapter
from src.harness.benchmark import BuildSpec, History, Message, Session, fingerprint


@pytest.mark.asyncio
async def test_main_keeps_oversized_source_and_retains_build_metadata(tmp_path):
    history = History("test", (Session("s", (
        Message("a", "user", "First detail."),
        Message("b", "assistant", "Acknowledged."),
        Message("c", "user", "Another detail."),
        Message("d", "user", "complete " * 700),
    ), "2023/05/21"),))
    options = {"backend_options": {"candidate_k": 50, "max_hops": 1},
               "retained_sources_root": str(tmp_path / "retained")}
    dependencies = PaperSAGMem.dependencies(options)
    spec = BuildSpec(fingerprint({"history": history.digest, "dependencies": dependencies}),
                     "test", str(tmp_path / "indexes"), dependencies)
    original = sag2_adapter.chunk_history
    backend = PaperSAGMem(options)
    try:
        await backend.prepare(history, spec)
        assert sag2_adapter.chunk_history is original
        assert len(backend.mapping) == 2
        assert any("complete " * 700 in row["content"] for row in backend.mapping.values())
        retained = tmp_path / "retained" / spec.id
        assert json.loads((retained / "sources.json").read_text()) == backend.mapping
        assert (retained / "build.json").read_bytes() == (spec.directory / "build.json").read_bytes()
    finally:
        await backend.aclose()
