from types import SimpleNamespace

import pytest

from src.harness.reader_grader import (
    format_sag_time_evidence,
    memory_retrieval_metrics,
    retrieval_metrics,
)

RECORD = {
    "answer_session_ids": ["session-1", "session-2"],
    "gold_answer": "window seat",
}
ITEMS = [
    {"session_id": "session-1", "content": "The user prefers a window seat."},
    {"session_id": "session-1", "content": "The user likes looking outside."},
    {"session_id": "session-3", "content": "The user travels frequently."},
]


def test_chunk_metrics_use_distinct_gold_session_coverage() -> None:
    """Repeated hits from one of two gold sessions cover only half the evidence."""
    metrics = retrieval_metrics(RECORD, ITEMS)

    assert metrics["chunk_session_recall"] == 0.5
    assert metrics["answer_chunk_recall"] == pytest.approx(1 / 3)
    assert metrics["retrieved_chunk_count"] == 3
    assert metrics["top5_session_recall"] == 0.5


def test_recall_at_k_uses_all_sources_and_counts_each_gold_id_once():
    from src.harness.benchmark import Target, RetrievalResult
    from src.harness.reader_grader import evidence_metrics, aggregate_metrics

    target = Target("answer", "1", session_ids=("s1", "s2"),
                    message_ids=("m1", "m2", "m3"), evidence_status="valid")
    first = {"content": "answer", "sources": [{"session_id": "s1", "message_id": "m1"},
                                                {"session_id": "s2", "message_id": "m2"}]}
    tail = {"content": "answer", "sources": [{"session_id": "s2", "message_id": "m3"}]}
    result = RetrievalResult("answer", [first] * 5 + [tail], source_granularity="message", metric_family="chunk")
    metrics = evidence_metrics(target, result)
    assert metrics["top5_session_recall"] == 1
    assert metrics["top5_message_recall"] == pytest.approx(2 / 3)
    assert metrics["top10_message_recall"] == metrics["top20_message_recall"] == 1
    assert metrics["chunk_session_recall"] == metrics["session_coverage"] == 1
    invalid = evidence_metrics(Target("answer", "1", session_ids=("s1",),
                                      evidence_status="invalid"), result)
    assert invalid["top5_session_recall"] is None
    summary = aggregate_metrics([{"grade": True, **metrics}])
    assert summary["top5_message_recall"] == pytest.approx(2 / 3)
    assert summary["top5_message_recall_valid_count"] == 1
    missing = evidence_metrics(Target("answer", "1"), result)
    assert missing["top5_session_recall"] is None
    assert missing["top5_message_recall"] is None


def test_saved_metrics_summary_recomputes_old_recall_values():
    from dataclasses import asdict
    from src.harness.benchmark import Target
    from src.harness.reader_grader import aggregate_metrics

    target = Target("answer", "1", session_ids=("s1",), evidence_status="valid")
    saved = {"grade": True, "target": asdict(target), "source_granularity": "message",
             "evidence": [{"session_id": "s1"}] + [{"session_id": "other"}] * 4,
             "top5_session_recall": 0.2, "chunk_session_recall": 0.2}
    result = aggregate_metrics([saved])
    assert result["top5_session_recall"] == result["chunk_session_recall"] == 1
    assert saved["top5_session_recall"] == 0.2


def test_longmemeval_turn_recall_uses_dataset_answer_flags(tmp_path):
    import json
    from src.data.loader import LongMemEval
    from scripts.run_benchmark import parse_args

    item = {"question_id": "q", "question_type": "single-session-user", "question": "Where?",
            "answer": "Paris", "question_date": "2026-11-06", "answer_session_ids": ["s"],
            "haystack_session_ids": ["s"], "haystack_dates": ["2026-11-05"],
            "haystack_sessions": [[{"role": "user", "content": "Paris", "has_answer": True},
                                    {"role": "assistant", "content": "OK", "has_answer": False}]]}
    path = tmp_path / "data.json"
    path.write_text(json.dumps([item]))
    example = LongMemEval().scan(parse_args(["--data-path", str(path)]))[0]
    assert example.target.message_ids == ("s0:s#t0",)


def test_memory_metrics_use_gold_session_coverage_denominator() -> None:
    """Event definition: share of gold sessions covered, deduped by session."""
    metrics = memory_retrieval_metrics(RECORD, ITEMS)

    assert metrics["memory_session_recall"] == 0.5
    assert metrics["answer_memory_hit"] == 1.0
    assert metrics["retrieved_memory_count"] == 3
    assert metrics["gold_session_memory_count"] == 2
    assert metrics["answer_memory_count"] == 1
    assert metrics["top5_memory_session_recall"] == 0.5


def test_the_two_metric_families_share_no_names() -> None:
    """Guards the whole point of the split: no name means two definitions."""
    assert not (set(retrieval_metrics(RECORD, ITEMS)) & set(memory_retrieval_metrics(RECORD, ITEMS)))


def test_sag_time_evidence_does_not_expose_session_ids() -> None:
    hit = SimpleNamespace(
        session_id="answer_secret_session",
        start_time=None,
        end_time=None,
        temporal_status="not_future",
        content="The relevant fact is here.",
    )
    evidence = format_sag_time_evidence([hit])
    assert "answer_secret_session" not in evidence
    assert "The relevant fact is here." in evidence


import asyncio
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

from scripts.run_benchmark import parse_args
from src.backends.adapters import FullContext
from src.backends.sag2_adapter import SAG2Backend
from src.config import settings
from src.data.loader import LoCoMo, LongMemEval, parse_time
from src.harness import benchmark as b
from src.harness.reader_grader import (
    LoCoMoProtocol,
    LongMemEvalProtocol,
    aggregate_metrics,
    evidence_metrics,
)


def options(tmp_path, *args):
    return parse_args(["--backend", "full_context", "--output", str(tmp_path / "out.jsonl"),
                       "--index-root", str(tmp_path / "indexes"), *args])


def lme_fixture(tmp_path, **changes):
    row = {"question_id": "q1", "question_type": "temporal-reasoning", "question": "What color?",
           "answer": "blue", "question_date": "2023/01/01 (Sun) 10:00", "answer_session_ids": ["s1"],
           "haystack_session_ids": ["s1"], "haystack_dates": ["2023/01/01 (Sun) 19:00"],
           "haystack_sessions": [[{"role": "user", "content": "My color is blue.", "has_answer": True},
                                   {"role": "assistant", "content": "Noted."}]]}
    row.update(changes)
    path = tmp_path / "data.json"
    path.write_text(json.dumps([row, {**row, "question_id": "q2", "question": "Recall my color?"}]))
    return path


@pytest.mark.asyncio
async def test_repeated_lme_sessions_preserve_dates_and_gold_mapping(tmp_path):
    from src.backends.adapters import legacy_history

    turns = [{"role": "user", "content": "My color is blue."}]
    path = lme_fixture(tmp_path, haystack_sessions=[turns, turns, turns],
                       haystack_session_ids=["s1", "s1", "s1#occurrence-1"],
                       haystack_dates=["2023/01/01 10:00", "2023/01/02 10:00", "2023/01/03 10:00"],
                       answer_session_ids=["s1"])
    original = path.read_bytes()
    opts = options(tmp_path, "--data-path", str(path))
    examples, deps, _ = b.prepare_run(opts)
    example = examples[0]
    history = LongMemEval().load_history(example.history_ref)
    assert [s.id for s in history.sessions] == ["s1", "s1", "s1#occurrence-1"]
    assert len({m.id for s in history.sessions for m in s.messages}) == 3
    assert example.target.session_ids == ("s1",)
    assert history.digest == example.history_digest
    assert legacy_history(history) == legacy_history(LongMemEval().load_history(example.history_ref))
    assert len(history.prefix(2).sessions) == 2
    adapter = FullContext(opts)
    await adapter.prepare(history, b.make_build(example, opts, deps))
    result = await adapter.retrieve(example.query, opts)
    await adapter.aclose()
    assert [s.date_raw for s in history.sessions] == ["2023/01/01 10:00", "2023/01/02 10:00", "2023/01/03 10:00"]
    assert len(result.items) == 3
    assert result.items[0]["sources"][0]["session_id"] == "s1"
    assert evidence_metrics(example.target, result)["session_coverage"] == 1.0
    assert path.read_bytes() == original


def test_lme_still_rejects_empty_session_ids(tmp_path):
    path = lme_fixture(tmp_path, haystack_session_ids=[""])
    with pytest.raises(ValueError, match="Empty session IDs"):
        LongMemEval().scan(options(tmp_path, "--data-path", str(path)))


def locomo_fixture(tmp_path):
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps([{"sample_id": "c", "conversation": {
        "session_1_date_time": "9:00 am on 8 May, 2023",
        "session_1": [{"speaker": "Alice", "dia_id": "D1:1", "text": "hello", "blip_caption": "a cat"}],
        "session_2_date_time": "bad", "session_2": [{"speaker": "Bob", "dia_id": "D2:1", "text": "blue"}]},
        "qa": [{"question": "Unanswerable?", "category": 5, "adversarial_answer": "trap", "evidence": ["D1:1"]},
               {"question": "Color?", "category": 4, "answer": "blue", "evidence": ["D9:1"]},
               {"question": "Who?", "category": 1, "answer": "Bob", "evidence": []}]}]))
    return path


def test_cli_defaults_and_validation(tmp_path):
    default = parse_args([])
    assert Path(default["output"]).name == "run.jsonl"
    assert Path(default["output"]).parent.parent == settings.results_dir.resolve()
    assert "_longmemeval_full_context_" in Path(default["output"]).parent.name
    assert (default["dataset"], default["backend"], default["split"], default["top_k"]) == ("longmemeval", "full_context", "s", 20)
    assert (default["concurrency"], default["worker_timeout"], default["worker_retries"], default["model_timeout"]) == (2, 18000, 1, 180)
    assert not any(default[k] for k in ("resume", "mlflow", "limit", "context_tokens"))
    assert default["extraction_profile"] == "relaxed.en"
    assert parse_args(["--extraction-profile", ""])["extraction_profile"] == ""
    for args in (["--questions", "q", "--limit", "1"],
                 ["--stratified", "1", "--start", "1"],
                 ["--worker-retries", "-1"], ["--namespace", "../outside"], ["--context-tokens", "0"],
                 ["--regrade-from", "x", "--retrieval-only"], ["--resume", "--rerun-incorrect"]):
        with pytest.raises(SystemExit):
            options(tmp_path, *args)


def test_question_type_filter_selects_exact_longmemeval_category(tmp_path):
    path = lme_fixture(tmp_path)
    rows = json.loads(path.read_text())
    rows[0]["question_type"] = "multi-session"
    rows[1]["question_type"] = "temporal-reasoning"
    path.write_text(json.dumps(rows))

    opts = options(tmp_path, "--data-path", str(path), "--question-type", "multi-session", "--limit", "1")
    examples, _, manifest = b.prepare_run(opts)

    assert [example.id for example in examples] == ["q1"]
    assert manifest["identity"]["config"]["question_types"] == ["multi-session"]


def test_reuse_build_source_preserves_governance_build_order(tmp_path):
    first = b.Example(
        "q1", b.Query("first"), b.Target("a", "multi-session"), {}, "history-1", "corpus-1"
    )
    second = b.Example(
        "q2", b.Query("second"), b.Target("b", "multi-session"), {}, "history-2", "corpus-2"
    )
    build_id = "b" * 64
    source = tmp_path / "prior.jsonl"
    source.write_text(
        "\n".join(
            (
                json.dumps({"question_id": "q2", "question_type": "multi-session", "corpus_id": "corpus-2", "build_id": build_id}),
                json.dumps({"question_id": "q1", "question_type": "multi-session", "corpus_id": "corpus-1", "build_id": "a" * 64}),
            )
        )
        + "\n"
    )
    root = tmp_path / "indexes"
    manifest = root / "sag" / build_id / "build.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"status": "succeeded", "build_id": build_id, "dependencies": {"family": "sag"}}))
    options = {
        "reuse_builds_from": str(source),
        "question_types": ["multi-session"],
        "questions": [],
        "stratified": 0,
        "start": 0,
        "limit": 1,
        "namespace": "sag",
        "index_root": str(root),
    }

    selected = b._select_examples_from_reuse_source([first, second], options)
    options["_reused_builds"] = b._load_reused_builds(options, selected)
    spec = b.make_build(selected[0], options, {"current": "dependencies"})

    assert [example.id for example in selected] == ["q2"]
    assert spec.id == build_id
    assert spec.dependencies == {"family": "sag"}


def test_prepare_run_preserves_protocol_filter_and_reused_build_cohort(tmp_path, monkeypatch):
    path = lme_fixture(tmp_path)
    rows = json.loads(path.read_text())
    rows.append({**rows[0], "question_id": "q3", "question_type": "single-session-user"})
    path.write_text(json.dumps(rows))
    checked = []

    class TemporalProtocol(LongMemEvalProtocol):
        @staticmethod
        def accepts_target(target):
            checked.append(target.category)
            return target.category == "temporal-reasoning"

    monkeypatch.setitem(b.REGISTRIES["protocol"], "merge_temporal", TemporalProtocol)
    source = tmp_path / "prior.jsonl"
    source.write_text("".join(json.dumps({
        "question_id": qid, "question_type": "temporal-reasoning", "build_id": build_id,
    }) + "\n" for qid, build_id in (("q2", "b" * 64), ("q1", "a" * 64))))
    opts = options(tmp_path, "--data-path", str(path), "--protocol", "merge_temporal",
                   "--reuse-builds-from", str(source), "--question-type", "temporal-reasoning", "--limit", "1")
    build_id = "b" * 64
    build = Path(opts["index_root"]) / opts["namespace"] / build_id / "build.json"
    build.parent.mkdir(parents=True)
    build.write_text(json.dumps({"status": "succeeded", "build_id": build_id,
                                 "dependencies": {"family": "sag"}}))

    examples, dependencies, manifest = b.prepare_run(opts)

    assert checked == ["temporal-reasoning", "temporal-reasoning", "single-session-user"]
    assert [example.id for example in examples] == ["q2"]
    assert [question["id"] for question in manifest["identity"]["questions"]] == ["q2"]
    assert b.make_build(examples[0], opts, dependencies).id == build_id


def test_lme_mapping_strip_labels_reuse_and_prefix(tmp_path):
    opts = options(tmp_path, "--data-path", str(lme_fixture(tmp_path)))
    adapter = LongMemEval()
    examples = adapter.scan(opts)
    assert examples[0].history_digest == examples[1].history_digest
    history = adapter.load_history(examples[0].history_ref)
    payload = json.dumps(asdict(history))
    assert not any(label in payload for label in ("has_answer", "question", '"answer"', "evidence"))
    assert history.prefix(1).digest != history.digest
    assert len(history.prefix(1).sessions[0].messages) == 1
    assert len(history.prefix(0).sessions) == 0
    with pytest.raises(ValueError):
        history.prefix(99)


@pytest.mark.parametrize("changes", [{"haystack_dates": []}, {"haystack_session_ids": ["s1", "s1"]}])
def test_history_validation(tmp_path, changes):
    with pytest.raises(ValueError):
        LongMemEval().scan(options(tmp_path, "--data-path", str(lme_fixture(tmp_path, **changes))))


def test_duplicate_question_ids_fail(tmp_path):
    path = lme_fixture(tmp_path)
    rows = json.loads(path.read_text())
    rows[1]["question_id"] = rows[0]["question_id"]
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="duplicate question"):
        LongMemEval().scan(options(tmp_path, "--data-path", str(path)))


def test_locomo_mapping_evidence_dates_and_caption(tmp_path):
    opts = options(tmp_path, "--dataset", "locomo", "--data-path", str(locomo_fixture(tmp_path)))
    adapter = LoCoMo()
    examples = adapter.scan(opts)
    assert len(examples) == 3
    assert examples[0].target.answer == "No information available"
    assert examples[0].target.abstention
    assert [e.target.evidence_status for e in examples] == ["valid", "invalid", "missing"]
    assert examples[0].query.timestamp is None
    history = adapter.load_history(examples[0].history_ref)
    assert history.sessions[0].messages[0].speaker == "Alice"
    assert "cat" not in history.sessions[0].messages[0].content
    assert not history.sessions[1].time_known
    assert history.sessions[1].date_raw == "bad"
    opts["text_mode"] = "text_caption"
    example = adapter.scan(opts)[0]
    assert "cat" in adapter.load_history(example.history_ref).sessions[0].messages[0].content
    assert example.history_digest != examples[0].history_digest


def test_build_identity_independent_reader_and_topk(tmp_path):
    opts = options(tmp_path, "--data-path", str(lme_fixture(tmp_path)))
    example = LongMemEval().scan(opts)[0]
    deps = SAG2Backend.dependencies(opts)
    a = b.make_build(example, opts, deps)
    other = {**opts, "top_k": 1, "reader_model": "different", "judge_model": "other"}
    assert a.id == b.make_build(example, other, SAG2Backend.dependencies(other)).id
    assert a.id != b.make_build(replace(example, history_digest="changed"), opts, deps).id
    assert a.id != b.make_build(example, opts, {**deps, "sdk": "new"}).id


@pytest.mark.parametrize("role", ["reader", "judge"])
def test_model_endpoint_change_rejects_resume_but_preserves_build(tmp_path, monkeypatch, role):
    from src import config

    settings = replace(config.settings, openai_base_url="https://shared.test/v1", openai_api_key="shared-key",
                       reader_base_url=None, judge_base_url=None, reader_api_key="", judge_api_key="")
    monkeypatch.setattr(config, "settings", settings)
    opts = options(tmp_path, "--backend", "sag2", "--data-path", str(lme_fixture(tmp_path)))
    examples, deps, original = b.prepare_run(opts)
    build = b.make_build(examples[0], opts, deps)
    store = b.ResultStore(Path(opts["output"]), original, False)
    store.close()

    changed = replace(settings, **{f"{role}_base_url": f"https://{role}.test/v1"})
    monkeypatch.setattr(config, "settings", changed)
    examples, deps, updated = b.prepare_run(opts)

    assert updated["run_id"] != original["run_id"]
    assert b.make_build(examples[0], opts, deps).id == build.id
    assert original["identity"][f"{role}_service"]["base_url"] == "https://shared.test/v1"
    assert updated["identity"][f"{role}_service"]["base_url"] == f"https://{role}.test/v1"
    with pytest.raises(ValueError, match="identity mismatch"):
        b.ResultStore(Path(opts["output"]), updated, True)

    monkeypatch.setattr(config, "settings", replace(changed, **{f"{role}_api_key": "rotated-key"}))
    _, _, rotated = b.prepare_run(opts)
    assert rotated["run_id"] == updated["run_id"]
    assert "shared-key" not in json.dumps(rotated)
    assert "rotated-key" not in json.dumps(rotated)


def test_date_text_separate_from_filter(tmp_path):
    opts = options(tmp_path, "--data-path", str(lme_fixture(tmp_path, question_date="bad")))
    example = LongMemEval().scan(opts)[0]
    protocol = LongMemEvalProtocol()
    assert protocol.question(example.query, True).startswith("(date: bad)")
    assert protocol.question(example.query, False) == example.query.text
    assert parse_time("bad") is None


def test_locomo_rules_are_separate():
    p = LoCoMoProtocol()
    assert p.score("cats", b.Target("cat", "4"))["locomo_rule_score"] == 1
    assert p.score("red", b.Target("red, blue", "1"))["locomo_rule_score"] == .5
    assert p.score("blue", b.Target("blue; explanation", "3"))["locomo_rule_score"] == 1
    assert p.score("No information available", b.Target("ignored", "5", True))["locomo_rule_score"] == 1
    assert p.score("trap", b.Target("ignored", "5", True))["locomo_rule_score"] == 0
    assert "off-by-one tolerance" not in p.judge_prompt("q", "a", b.Target("a", "2"))
    assert "same date or time period" in p.judge_prompt("q", "a", b.Target("a", "2"))


def test_evidence_metrics_null_and_multisource():
    target = b.Target("blue", "1", session_ids=("s1", "s2"), message_ids=("m1", "m2"), evidence_status="valid")
    result = b.RetrievalResult("blue", [{"content": "blue", "sources": [{"session_id": "s1", "message_id": "m1"},
                                                                                    {"session_id": "s2", "message_id": "m2"}]}],
                               source_granularity="message")
    assert evidence_metrics(target, result)["session_coverage"] == 1
    assert evidence_metrics(target, result)["message_coverage"] == 1
    assert evidence_metrics(replace(target, evidence_status="invalid"), result)["session_coverage"] is None
    result.source_granularity = "unavailable"
    assert evidence_metrics(target, result)["session_coverage"] is None
    result.source_granularity = "session"
    result.items = []
    assert evidence_metrics(target, result)["session_coverage"] == 0
    assert evidence_metrics(target, result)["message_coverage"] is None


def test_result_repair_and_lock(tmp_path):
    path = tmp_path / "records.jsonl"
    manifest = {"run_id": "a"}
    store = b.ResultStore(path, manifest)
    store.append({"question_id": "q", "run_id": "a", "status": "succeeded"})
    with pytest.raises(BlockingIOError):
        b.ResultStore(path, manifest, True)
    store.close()
    with path.open("ab") as stream:
        stream.write(b'{"question_id":')
    store = b.ResultStore(path, manifest, True)
    assert set(store.records) == {"q"}
    store.close()
    with pytest.raises(ValueError, match="identity mismatch"):
        b.ResultStore(path, {"run_id": "b"}, True)
    path.write_bytes(path.read_bytes() + b'{bad}\n')
    with pytest.raises(ValueError, match="Corrupt"):
        b.ResultStore(path, manifest, True)


@pytest.mark.asyncio
async def test_full_context_never_topk_crops(tmp_path):
    opts = options(tmp_path, "--data-path", str(lme_fixture(tmp_path)), "--top-k", "1")
    example = LongMemEval().scan(opts)[0]
    backend = FullContext(opts)
    history = LongMemEval().load_history(example.history_ref)
    await backend.prepare(history, b.make_build(example, opts, backend.dependencies(opts)))
    result = await backend.retrieve(example.query, opts)
    assert len(result.items) == 2
    assert "Noted" in result.context


@pytest.mark.asyncio
async def test_group_once_resume_and_retrieval_contract(tmp_path, monkeypatch):
    opts = options(tmp_path, "--data-path", str(lme_fixture(tmp_path)), "--retrieval-only")
    starts, queries = [], []

    class FakeWorker:
        process = None

        def __init__(self, options):
            pass

        async def start(self, example, spec):
            starts.append(spec.id)

        async def retrieve(self, query):
            queries.append(asdict(query))
            return b.RetrievalResult("blue", source_granularity="session")

        async def close(self, **kwargs):
            pass

    monkeypatch.setattr(b, "BackendProcess", FakeWorker)
    summary = await b.run(opts)
    assert len(starts) == 1 and len(queries) == 2 and summary["errors"] == 0
    assert set(queries[0]) == {"text", "date_raw", "timestamp"}
    await b.run({**opts, "resume": True})
    assert len(queries) == 2
    with pytest.raises(ValueError, match="identity mismatch"):
        await b.run({**opts, "resume": True, "reader_model": "different"})


@pytest.mark.asyncio
async def test_judge_failure_preserves_rule_score_and_denominator(tmp_path):
    class Reader:
        async def respond(self, *args, **kwargs):
            return SimpleNamespace(hypothesis="blue", duration=0, prompt_tokens=3, completion_tokens=1)

        async def grade(self, *args, **kwargs):
            raise RuntimeError("provider unavailable")

    record = {"context": "blue"}
    await b.score_record(record, b.Target("blue", "4"), b.Query("color?"), LoCoMoProtocol(), Reader(), options(tmp_path))
    assert record["locomo_rule_score"] == 1 and record["grade"] is None
    assert record["failure_stage"] == "judge"
    metrics = aggregate_metrics([record, {"grade": False}])
    assert metrics["total"] == 2 and metrics["valid_grades"] == 1 and metrics["errors"] == 1
    assert metrics["locomo_rule_score_valid_count"] == 1 and metrics["accuracy"] == 0
    assert record["bleu1"] == 1 and record["metrics"]["bleu1"] == 1
    assert metrics["bleu1"] == 1 and metrics["bleu1_valid_count"] == 1


@pytest.mark.asyncio
async def test_real_worker_plugin_and_no_backend_sdks(tmp_path, monkeypatch):
    import os

    (tmp_path / "worker_fixture.py").write_text(
        "from src.backends.adapters import FullContext\n"
        "from src.harness.benchmark import register\n"
        "class Keyword(FullContext): pass\n"
        "register('backend', 'keyword', Keyword)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    path = lme_fixture(tmp_path)
    opts = options(tmp_path, "--data-path", str(path), "--plugin", "worker_fixture",
                   "--backend", "keyword", "--retrieval-only")
    summary = await b.run(opts)
    assert summary["errors"] == 0
    rows = b.read_records(Path(opts["output"]))
    assert len(rows) == 2 and rows[0]["build_id"] == rows[1]["build_id"]
    process = await asyncio.create_subprocess_exec(sys.executable, "-c",
        "from src.harness.benchmark import resolve; resolve('backend','full_context'); import sys; "
        "assert 'graphiti_core' not in sys.modules; assert 'zleap.sag' not in sys.modules",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    _stdout, stderr = await process.communicate()
    assert process.returncode == 0, stderr.decode()


@pytest.mark.asyncio
async def test_worker_timeout_only_kills_own_process(tmp_path):
    opts = options(tmp_path)
    opts["worker_timeout"] = .05
    worker = b.BackendProcess(opts)
    worker.process = await asyncio.create_subprocess_exec(sys.executable, "-c", "import time; time.sleep(20)",
                                                         stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
    other = await asyncio.create_subprocess_exec(sys.executable, "-c", "import time; time.sleep(20)")
    try:
        with pytest.raises(b.WorkerFailure, match="timeout"):
            await worker.call({"method": "prepare"})
        assert worker.process.returncode is not None
        assert other.returncode is None
    finally:
        other.kill()
        await other.wait()


@pytest.mark.asyncio
async def test_regrade_saved_answers_never_opens_backend(tmp_path, monkeypatch):
    from src.harness import reader_grader

    path = lme_fixture(tmp_path)
    opts = options(tmp_path, "--data-path", str(path))
    prepared = b.prepare_run(opts)
    example = prepared[0][0]
    source = Path(opts["output"])
    store = b.ResultStore(source, prepared[2])
    store.append({"question_id": example.id, "run_id": prepared[2]["run_id"], "hypothesis": "blue",
                  "query": asdict(example.query), "target": asdict(example.target), "question_type": "temporal-reasoning"})
    store.close()
    original = source.read_bytes()

    class Judge:
        def __init__(self, *args):
            pass

        async def grade(self, *args, **kwargs):
            return SimpleNamespace(is_correct=True, prompt_tokens=1, completion_tokens=1)

        async def aclose(self):
            pass

    monkeypatch.setattr(reader_grader, "ReaderGrader", Judge)
    def no_backend(*args, **kwargs):
        raise AssertionError("No backend allowed during regrade")
    monkeypatch.setattr(b, "BackendProcess", no_backend)
    new = parse_args(["--regrade-from", str(source), "--output", str(tmp_path / "regrade.jsonl")])
    summary = await b.regrade(new)
    assert summary["accuracy"] == 1 and source.read_bytes() == original
    result = b.read_records(Path(new["output"]))[0]
    assert result["question"].startswith("(date:") and result["bleu1"] == 1




@pytest.mark.asyncio
async def test_prefix_cli_excludes_future_before_worker_prepares(tmp_path):
    opts = options(tmp_path, "--data-path", str(lme_fixture(tmp_path)), "--history-prefix", "1", "--retrieval-only")
    await b.run(opts)
    row = b.read_records(Path(opts["output"]))[0]
    assert "blue" in row["context"] and "Noted" not in row["context"]
    assert len(row["evidence"]) == 1


@pytest.mark.asyncio
async def test_retry_and_rerun_incorrect_only(tmp_path, monkeypatch):
    opts = options(tmp_path, "--data-path", str(lme_fixture(tmp_path)), "--retrieval-only")
    requested = []

    class Worker:
        process = None

        def __init__(self, options):
            pass

        async def start(self, *args):
            pass

        async def retrieve(self, query):
            requested.append(query.text)
            if len(requested) == 1:
                raise b.WorkerFailure("retrieve", "transient")
            return b.RetrievalResult("blue")

        async def close(self, **kwargs):
            pass

    monkeypatch.setattr(b, "BackendProcess", Worker)
    result = await b.run(opts)
    assert result["errors"] == 0 and len(requested) == 3
    path = Path(opts["output"])
    rows = b.read_records(path)
    rows[0]["grade"] = False
    rows[1].update(status="failed", error="judge failed", grade=None)
    path.write_text(path.read_text() + "".join(json.dumps(r) + "\n" for r in rows))
    await b.run({**opts, "rerun_incorrect": True})
    assert len(requested) == 4 and requested[-1] == rows[0]["question"]


@pytest.mark.asyncio
async def test_group_load_waits_for_execution_slot(tmp_path, monkeypatch):
    opts = options(tmp_path, "--data-path", str(lme_fixture(tmp_path)), "--retrieval-only", "--concurrency", "1")
    examples, deps, manifest = b.prepare_run(opts)
    examples[1] = replace(examples[1], history_digest="different-prefix", prefix=1)
    events = []

    class Worker:
        process = None

        def __init__(self, options):
            pass

        async def start(self, example, spec):
            events.append("load-" + example.id)

        async def retrieve(self, query):
            await asyncio.sleep(.01)
            events.append("retrieve")
            return b.RetrievalResult("blue")

        async def close(self, **kwargs):
            events.append("close")

    monkeypatch.setattr(b, "BackendProcess", Worker)
    await b.run(opts, (examples, deps, manifest))
    assert events == ["load-q1", "retrieve", "close", "load-q2", "retrieve", "close"]


def test_context_budget_keeps_raw_and_uses_independent_estimate():
    raw = "blue " * 100
    context, stats = b.budget_context(raw, 5)
    assert len(context) < len(raw) and stats["context_truncated"]
    assert stats["reader_context_estimated_tokens"] == 5
    context, stats = b.budget_context(raw, None)
    assert context == raw and not stats["context_truncated"]


@pytest.mark.parametrize("failure, message", [
    ("missing_input", "Merge input is missing or not a file"),
    ("input_directory", "Merge input is missing or not a file"),
    ("missing_manifest", "Missing merge manifest"),
    ("invalid_manifest_json", "Invalid merge manifest JSON"),
    ("invalid_manifest_encoding", "Invalid merge manifest JSON"),
    ("corrupt_result", "Corrupt result at"),
])
def test_merge_input_errors_identify_source_without_writing_output(tmp_path, failure, message):
    data = lme_fixture(tmp_path)
    opts = options(tmp_path, "--data-path", str(data))
    manifest = b.prepare_run(opts)[2]
    source = Path(opts["output"])
    manifest_path = source.with_suffix(source.suffix + ".manifest.json")
    source.write_text(json.dumps({"question_id": "q1", "run_id": manifest["run_id"]}) + "\n")
    b.atomic_json(manifest_path, manifest)
    if failure in {"missing_input", "input_directory"}:
        source.unlink()
        if failure == "input_directory":
            source.mkdir()
    elif failure == "missing_manifest":
        manifest_path.unlink()
    elif failure == "invalid_manifest_json":
        manifest_path.write_text("{")
    elif failure == "invalid_manifest_encoding":
        manifest_path.write_bytes(b"\xff")
    else:
        source.write_text(source.read_text() + "{")
    originals = {p: p.read_bytes() for p in (source, manifest_path) if p.is_file()}
    output = tmp_path / "merge-output" / "merged.jsonl"
    merge = parse_args(["--merge-from", str(source), "--output", str(output)])

    with pytest.raises(ValueError, match=message) as error:
        b.merge_results(merge)

    assert str(manifest_path if "manifest" in failure else source) in str(error.value)
    assert not output.parent.exists()
    assert all(p.read_bytes() == content for p, content in originals.items())


@pytest.mark.asyncio
async def test_merge_compatible_shards_and_reject_mixed_configs(tmp_path):
    path = lme_fixture(tmp_path)
    opts1 = options(tmp_path, "--data-path", str(path), "--retrieval-only", "--limit", "1")
    opts2 = {**opts1, "start": 1, "output": str(tmp_path / "second.jsonl")}
    await b.run(opts1)
    await b.run(opts2)
    merge = parse_args(["--merge-from", opts1["output"], opts2["output"], "--output", str(tmp_path / "merged.jsonl")])
    summary = b.merge_results(merge)
    assert summary["total"] == 2
    rows = b.read_records(Path(merge["output"]))
    assert all(row["source_run_id"] for row in rows)
    manifest_path = Path(opts2["output"] + ".manifest.json")
    data = json.loads(manifest_path.read_text())
    data["identity"]["config"]["reader_model"] = "different"
    manifest_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="different experiment"):
        b.merge_results({**merge, "output": str(tmp_path / "bad.jsonl")})


def test_custom_protocol_metric_is_aggregated():
    summary = aggregate_metrics([{"metrics": {"custom_score": .5}}, {"metrics": {"custom_score": 1}}, {}])
    assert summary["custom_score"] == .75 and summary["custom_score_valid_count"] == 2


def test_protocol_mismatch_rejected_before_services(tmp_path):
    opts = options(tmp_path, "--dataset", "locomo", "--data-path", str(locomo_fixture(tmp_path)),
                   "--protocol", "longmemeval")
    with pytest.raises(ValueError, match="Unsupported LongMemEval"):
        b.prepare_run(opts)


@pytest.mark.asyncio
async def test_invalid_judge_response_is_error_not_false(monkeypatch):
    from unittest.mock import AsyncMock

    from src.harness.reader_grader import ReaderGrader

    grader = object.__new__(ReaderGrader)
    grader.judge_model = "test"
    completions = SimpleNamespace(parse=AsyncMock(side_effect=RuntimeError("not supported")),
                                  create=AsyncMock(return_value=SimpleNamespace(
                                      choices=[SimpleNamespace(message=SimpleNamespace(content="maybe yes maybe no"))])))
    grader._judge_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    with pytest.raises(ValueError, match="Invalid judge"):
        await grader.grade("q", "a", "answer", "single-session-user")


@pytest.mark.asyncio
async def test_build_lock_wait_and_release(tmp_path):
    spec = b.BuildSpec("id", "namespace", str(tmp_path), {})
    first = await b.acquire_build_lock(spec)
    contender = asyncio.create_task(b.acquire_build_lock(spec))
    await asyncio.sleep(.02)
    assert not contender.done()
    first.close()
    second = await asyncio.wait_for(contender, 1)
    second.close()
