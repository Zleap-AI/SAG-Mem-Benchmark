import pytest

from reproduction.experiments.lite.backend import BatchedLite, LiteBackend
from reproduction.experiments.main.paper_backend import PaperSAGMem
from reproduction.run import configure, parse_cli
from src.harness.benchmark import resolve


@pytest.mark.parametrize("method", ["sag-mem", "sag-mem-lite"])
@pytest.mark.parametrize("dataset", ["locomo", "longmemeval"])
def test_paper_entrypoint_registers_current_backend(method, dataset):
    options = configure(parse_cli(["--method", method, "--dataset", dataset]))
    expected = PaperSAGMem if method == "sag-mem" else (LiteBackend if dataset == "locomo" else BatchedLite)
    assert resolve("backend", options["backend"]) is expected
    assert options["namespace"] == method
    assert options["top_k"] == 20


@pytest.mark.parametrize("method", ["sag-mem", "sag-mem-lite"])
def test_paper_entrypoint_preserves_explicit_retrieval_options(method):
    options = configure(parse_cli(["--method", method, "--dataset", "locomo",
        "--candidate-k", "30", "--max-hops", "0", "--top-k", "7", "--concurrency", "3"]))
    assert options["backend_options"]["candidate_k"] == 30
    assert options["backend_options"]["max_hops"] == 0
    assert options["top_k"] == 7
    assert options["concurrency"] == 3
