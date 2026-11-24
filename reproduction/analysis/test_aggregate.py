from copy import deepcopy
from pathlib import Path

import pytest
from reproduction.analysis.aggregate import bootstrap, fingerprint, verify_manifest


def manifest():
    service = {"model": "qwen/qwen3.6-35b-a3b", "extra_body": {
        "reasoning": {"enabled": False}, "provider": {
            "only": ["DeepInfra"], "allow_fallbacks": False, "quantizations": ["fp8"]}}}
    identity = {"config": {"reader_model": service["model"], "judge_model": service["model"]},
                "reader_service": deepcopy(service), "judge_service": deepcopy(service)}
    return {"identity": identity, "run_id": fingerprint(identity)}


def test_model_drift_is_rejected_even_with_valid_manifest_hash():
    item = manifest()
    item["identity"]["reader_service"]["model"] = "gpt-4o"
    item["identity"]["config"]["reader_model"] = "gpt-4o"
    item["run_id"] = fingerprint(item["identity"])
    with pytest.raises(ValueError, match="Wrong reader model"):
        verify_manifest(item, Path("test-only.json"), "Full Context", "qwen")


def test_manifest_content_cannot_change_under_an_old_identity():
    item = manifest()
    item["identity"]["config"]["reader_model"] = "gpt-4o"
    with pytest.raises(ValueError, match="identity digest mismatch"):
        verify_manifest(item, Path("test-only.json"), "Full Context", "qwen")


def test_bootstrap_resamples_histories_not_individual_questions():
    rows = {str(i): {"history": "first" if i < 100 else "second", "grade": i < 100}
            for i in range(200)}
    assert bootstrap(rows) == [0.0, 100.0]
    assert bootstrap(rows, rows) == [0.0, 0.0]


def test_verified_replay_rejects_changed_source(tmp_path, monkeypatch):
    import json

    from reproduction.analysis import aggregate

    monkeypatch.setattr(aggregate, "ROOT", tmp_path)
    source = tmp_path / "source.jsonl"
    source.write_text("original evidence\n")
    source_manifest = source.with_suffix(".jsonl.manifest.json")
    source_manifest.write_text(json.dumps({"run_id": "original-run"}))
    item = manifest()
    item["identity"]["config"].update(backend_options={"candidate_k": 50, "max_hops": 1}, top_k=20)
    item["identity"]["build_dependencies"] = {
        "chunking": {"policy": "three_messages_soft512_atomic_v1", "messages_per_chunk": 3}}
    item["identity"]["verified_replay"] = {
        "source_relative_path": "source.jsonl", "source_run_id": "original-run",
        "source_sha256": aggregate.digest(source),
        "source_manifest_sha256": aggregate.digest(source_manifest)}
    item["run_id"] = fingerprint(item["identity"])
    verify_manifest(item, tmp_path / "replay.manifest.json", "SAG-Mem", "qwen")
    source.write_text("changed evidence\n")
    with pytest.raises(ValueError, match="replay source content changed"):
        verify_manifest(item, tmp_path / "replay.manifest.json", "SAG-Mem", "qwen")


def test_replay_retains_logical_jev_cost(tmp_path, monkeypatch):
    import json

    from reproduction.analysis import aggregate

    monkeypatch.setattr(aggregate, "ROOT", tmp_path)
    monkeypatch.setattr(aggregate, "independent_questions", lambda dataset: {
        "q1": ("question", "1", "answer", False, "")})
    item = manifest()
    item["identity"]["config"].update(backend_options={"candidate_k": 50, "max_hops": 1}, top_k=20)
    item["identity"]["build_dependencies"] = {
        "chunking": {"policy": "three_messages_soft512_atomic_v1", "messages_per_chunk": 3}}
    item["run_id"] = fingerprint(item["identity"])
    path = tmp_path / "run.jsonl"
    path.with_suffix(".jsonl.manifest.json").write_text(json.dumps(item))
    row = {"run_id": item["run_id"], "question_id": "q1", "query": {"text": "question", "date_raw": ""},
           "question_type": "1", "gold_answer": "answer", "is_abstention": False, "grade": True,
           "status": "succeeded", "corpus_id": "history", "build_id": "build", "context_truncated": False,
           "backend": "sag-mem", "backend_stats": {
               "jev_usage": {"input_tokens": 0, "output_tokens": 0, "requests": 0, "judgments": 0},
               "source_jev_usage": {"input_tokens": 109, "output_tokens": 3, "requests": 2, "judgments": 10}}}
    path.write_text(json.dumps(row) + "\n")
    rows, _, _ = aggregate.read_run([path], "locomo", "SAG-Mem", "qwen")
    assert rows["q1"]["jev_input_tokens"] == 109
    assert rows["q1"]["jev_requests"] == 2


def test_lite_requires_current_retrieval_settings(tmp_path):
    item = manifest()
    item["identity"]["config"].update(backend_options={"candidate_k": 100, "max_hops": 2}, top_k=20)
    item["identity"]["build_dependencies"] = {
        "chunking": {"policy": "three_messages_soft512_atomic_v1", "messages_per_chunk": 3}}
    item["run_id"] = fingerprint(item["identity"])
    verify_manifest(item, tmp_path / "lite.manifest.json", "SAG-Mem-Lite", "qwen")
    item["identity"]["config"]["backend_options"] = {"candidate_k": 50, "max_hops": 1}
    item["run_id"] = fingerprint(item["identity"])
    with pytest.raises(ValueError, match="Primary retrieval settings differ"):
        verify_manifest(item, tmp_path / "lite.manifest.json", "SAG-Mem-Lite", "qwen")


@pytest.mark.parametrize("method,candidates,hops", [("SAG-Mem", 50, 1), ("SAG-Mem-Lite", 100, 2)])
@pytest.mark.parametrize("policy,cap", [("soft512_atomic_v1", None), ("3_messages_soft512_atomic_v1", 3)])
def test_current_chunk_policy_is_accepted(tmp_path, method, candidates, hops, policy, cap):
    item = manifest()
    item["identity"]["config"].update(backend_options={"candidate_k": candidates, "max_hops": hops}, top_k=20)
    item["identity"]["build_dependencies"] = {
        "chunking": {"policy": policy, "messages_per_chunk": cap}}
    item["run_id"] = fingerprint(item["identity"])
    verify_manifest(item, tmp_path / "run.manifest.json", method, "qwen")
    item["identity"]["build_dependencies"]["chunking"]["messages_per_chunk"] = 2
    item["run_id"] = fingerprint(item["identity"])
    with pytest.raises(ValueError, match="Chunk policy differs"):
        verify_manifest(item, tmp_path / "changed.manifest.json", method, "qwen")
