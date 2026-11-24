import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_benchmark import parse_args
from src.harness import benchmark as b
from src.harness import reader_grader
from src.harness.locomo_metrics import CATEGORIES, LoCoMoReporter


def manifest(categories):
    return {"run_id": "test", "identity": {"config": {"protocol": "locomo"}, "questions": [
        {"id": f"q{i}", "target": {"category": str(category)}} for i, category in enumerate(categories)]}}


def row(i, category, grade, **extra):
    return {"question_id": f"q{i}", "question_type": str(category), "grade": grade,
            "status": "succeeded", **extra}


@pytest.mark.parametrize("category", ["1", "2", "3", "4"])
def test_answerable_bleu_uses_full_reference_and_shared_formula(category):
    protocol = reader_grader.LoCoMoProtocol()
    assert protocol.score("blue", b.Target("blue", category))["bleu1"] == 1
    assert protocol.score("blue extra", b.Target("blue", category))["bleu1"] == .5
    assert protocol.score("red", b.Target("blue", category))["bleu1"] == 0
    assert protocol.score("", b.Target("blue", category))["bleu1"] == 0
    # Unlike the category-specific F1, BLEU uses the entire reference without splitting it.
    assert protocol.score("blue", b.Target("blue; explanation", category))["bleu1"] == pytest.approx(
        0.36787944117144233)


@pytest.mark.parametrize("abstention", [True, False])
def test_adversarial_bleu_is_missing_and_excluded_from_denominator(abstention):
    protocol = reader_grader.LoCoMoProtocol()
    scores = protocol.score("No information available", b.Target("No information available", "5", abstention))
    assert "bleu1" not in scores
    rows = [{"metrics": protocol.score("blue", b.Target("blue", "4"))},
            {"metrics": protocol.score("red", b.Target("blue", "2"))},
            {"metrics": scores}, {}]  # Old/unscored rows must not become zero scores.
    summary = reader_grader.aggregate_metrics(rows)
    assert summary["bleu1"] == .5
    assert summary["bleu1_valid_count"] == 2
    assert reader_grader.aggregate_metrics(rows[2:])["bleu1"] is None


def test_weighted_total_macro_zero_coverage_and_independent_rule_denominator():
    m = manifest([1, 1, 2, 3, 4, 5])
    rows = [row(0, 1, True, locomo_rule_score=.25), row(1, 1, False, locomo_rule_score=.75),
            row(2, 2, True), row(3, 3, True), row(4, 4, True)]
    report = LoCoMoReporter(m, rows)
    partial = report.snapshot()
    assert partial["overall"]["judge_accuracy"] == .8
    assert partial["overall"]["rule_score"] == .5
    assert partial["overall"]["rule_valid_count"] == 2
    assert partial["judge_macro_accuracy"] is None
    assert partial["by_category"]["adversarial"]["judge_accuracy"] is None
    assert "judge_accuracy_adversarial" not in report.metrics(partial, "live")
    assert "adversarial=N/A(0/0)" in report.terminal(partial)
    final = report.update(row(5, 5, False))
    assert final["overall"]["judge_accuracy"] == pytest.approx(4 / 6)
    assert final["judge_macro_accuracy"] == .7
    assert final["evaluation_complete"]
    assert final["judge_protocol_version"] == "legacy_unverified"
    # Returned snapshots remain immutable as the next question completes.
    assert partial["overall"]["completed"] == 5


def test_resume_replaces_latest_and_excludes_unselected_results():
    m = manifest([4, 5, 2])
    report = LoCoMoReporter(m, [row(0, 4, True), row(1, 5, None, error="timeout", status="failed"),
                               row(99, 1, True)])
    assert report.snapshot()["step"] == 0
    first = report.update(row(1, 5, False))
    assert first["overall"]["completed"] == 2
    assert first["overall"]["errors"] == 0
    assert first["overall"]["judge_accuracy"] == .5
    assert not first["evaluation_complete"]  # q2 remains missing, despite successful saved rows.
    second = report.update(row(0, 4, None, status="failed", error="judge failed", locomo_rule_score=1))
    assert second["overall"]["graded"] == 1
    assert second["overall"]["rule_score"] == 1
    assert second["overall"]["judge_accuracy"] == 0
    report.update(row(0, 4, None, status="failed", error="judge failed", locomo_rule_score=1))
    assert report.snapshot()["overall"] == second["overall"]


def test_retrieval_only_invalid_grades_rules_and_category_mismatch():
    report = LoCoMoReporter(manifest([4, 5]), [row(0, 4, "yes", locomo_rule_score=float("nan")),
                                            row(1, 5, True, retrieval_only=True)])
    snapshot = report.snapshot()
    assert snapshot["overall"]["graded"] == 0
    assert snapshot["overall"]["rule_valid_count"] == 0
    assert not snapshot["evaluation_complete"]
    assert snapshot["overall"]["judge_accuracy"] is None
    assert len(snapshot["by_category"]) == 5
    assert report.metrics(snapshot, "live") == report.metrics(snapshot, "final") == {}
    with pytest.raises(ValueError, match="category mismatch"):
        report.update(row(0, 1, True))


def test_legacy_regrade_categories_recovered_without_claiming_new_judge_version():
    old = {"identity": {"config": {"protocol": "locomo"}}, "question_order": ["q0", "q1"]}
    rows = [row(0, 4, True), row(1, 5, False)]
    reporter = LoCoMoReporter(old, rows)
    assert reporter.snapshot()["overall"]["planned"] == 2
    assert reporter.snapshot()["judge_protocol_version"] == "legacy_unverified"
    assert reporter.empty().snapshot()["overall"]["completed"] == 0
    with pytest.raises(ValueError, match="category metadata"):
        LoCoMoReporter(old, rows[:1])


def test_old_judge_fixture_remains_historical_and_reader_is_unchanged():
    fixture = json.loads((Path(__file__).parent / "fixtures" / "locomo_judge_v1.json").read_text())
    protocol = reader_grader.LoCoMoProtocol()
    assert reader_grader.GRADER_SYSTEM_PROMPT == fixture["system"]
    assert protocol.reader_system == fixture["reader_system"]
    assert protocol.reporting_metadata()["judge_protocol_version"] == "locomo-mem0-abstention-v1"
    old_manifest = {**manifest([1, 2, 3, 4, 5]), "judge_reporting": fixture["metadata"]}
    report = LoCoMoReporter(old_manifest).snapshot()
    assert report["judge_protocol_version"] == "locomo-semantic-v1"
    assert report["overall"]["planned"] == 5 and "adversarial" in report["by_category"]


def test_all_1986_questions_have_one_step_each():
    categories = [1] * 282 + [2] * 321 + [3] * 96 + [4] * 841 + [5] * 446
    report = LoCoMoReporter(manifest(categories))
    for i, cat in enumerate(categories):
        snapshot = report.update(row(i, cat, True))
        assert snapshot["step"] == i + 1
    assert snapshot["overall"]["planned"] == snapshot["overall"]["graded"] == 1986
    assert snapshot["evaluation_complete"]


@pytest.mark.parametrize("refusal_grade", [True, False, None])
def test_refusal_result_does_not_change_mem0_denominator_or_macro(refusal_grade):
    m = {**manifest([1, 1, 2, 3, 4, 5]), "judge_reporting": reader_grader.LoCoMoProtocol.reporting_metadata()}
    rows = [row(0, 1, True), row(1, 1, False), row(2, 2, True), row(3, 3, True), row(4, 4, True)]
    reporter = LoCoMoReporter(m, rows)
    before = reporter.snapshot()["mem0"]
    assert before["planned"] == before["graded"] == 5
    assert before["judge_accuracy"] == .8
    assert before["judge_macro_accuracy"] == .875
    refusal = row(5, 5, refusal_grade, status="failed" if refusal_grade is None else "succeeded",
                  error="timeout" if refusal_grade is None else None)
    after = reporter.update(refusal)
    assert after["mem0"] == before
    assert after["overall"]["planned"] == 6
    assert after["overall"]["graded"] == (5 if refusal_grade is None else 6)
    assert after["evaluation_complete"] is (refusal_grade is not None)
    for phase in ("live", "final"):
        metrics = reporter.metrics(after, phase)
        assert metrics["mem0_judge_accuracy"] == .8
        assert "mem0_planned" not in metrics
    assert reporter.metrics(after, "final")["mem0_judge_macro_accuracy"] == .875
    assert reporter.metrics(after, "curve") == {}
    assert reporter.tags(after)["locomo.mem0_planned"] == 5


def test_legacy_four_category_manifest_keeps_its_original_scope():
    m = {**manifest([1, 2, 3, 4]), "judge_reporting": {
        "judge_protocol_version": "locomo-mem0-judge-v1", "categories": ["1", "2", "3", "4"]}}
    reporter = LoCoMoReporter(m, [row(i, i + 1, True) for i in range(4)])
    report = reporter.snapshot()
    assert report["judge_protocol_version"] == "locomo-mem0-judge-v1"
    assert "adversarial" not in report["by_category"]
    assert report["evaluation_complete"] and report["overall"]["planned"] == 4


def test_unified_metric_inventory_has_11_names_including_7_running_scores():
    m = {**manifest([1, 2, 3, 4, 5]), "judge_reporting": reader_grader.LoCoMoProtocol.reporting_metadata()}
    rows = [row(i, i + 1, True, locomo_rule_score=.5, bleu1=.25) for i in range(5)]
    reporter = LoCoMoReporter(m, rows)
    snapshot = reporter.snapshot()
    live = reporter.metrics(snapshot, "live")
    final = reporter.metrics(snapshot, "final")
    assert len(live) == 7
    assert len(final) == 11
    assert set(live) == {"judge_accuracy", "mem0_judge_accuracy"} | {
        f"judge_accuracy_{category}" for category in CATEGORIES.values()}
    assert set(final) == set(live) | {
        "judge_macro_accuracy", "mem0_judge_macro_accuracy", "rule_score", "bleu1"}
    assert len(set(live) | set(final)) == 11
    assert not any(key.startswith("locomo_") for key in set(live) | set(final))
    assert all(final[key] == value for key, value in live.items())
    assert reporter.metrics(snapshot, "curve") == {}
    assert snapshot["reporting_version"] == "locomo-reporting-v5"
    assert reporter.tags(snapshot)["locomo.graded"] == 5
    assert reporter.tags(snapshot)["locomo.bleu1_valid_count"] == 4


def test_compact_bleu_keeps_valid_answerable_scores_on_judge_failure():
    rows = [row(0, 1, None, status="failed", error="judge timeout", metrics={"bleu1": .5}),
            row(1, 2, True, bleu1=.25), row(2, 3, True, bleu1=float("nan")),
            row(3, 4, True, bleu1=True), row(4, 5, True, bleu1=1)]
    reporter = LoCoMoReporter(manifest([1, 2, 3, 4, 5]), rows)
    snapshot = reporter.snapshot()
    assert reporter.metrics(snapshot, "final")["bleu1"] == .375
    assert reporter.tags(snapshot)["locomo.bleu1_valid_count"] == 2
    assert reporter.tags(snapshot)["locomo.errors"] == 1
    assert not snapshot["evaluation_complete"]
    with pytest.raises(ValueError, match="reporting phase"):
        reporter.metrics(snapshot, "unknown")


@pytest.mark.asyncio
async def test_run_resume_regrade_replay_and_merge_share_reporting(tmp_path, monkeypatch, capsys):
    data = tmp_path / "locomo.json"
    data.write_text(json.dumps([{"sample_id": "c", "conversation": {
        "session_1": [{"dia_id": "D1:1", "speaker": "A", "text": "blue"}],
        "session_1_date_time": "2023/01/01 (Sun) 10:00"}, "qa": [
        {"category": cat, "question": "color?", "answer": "blue", "evidence": ["D1:1"]}
        for cat in range(1, 6)]}]))
    calls = []

    class Worker:
        def __init__(self, *args):
            self.process = SimpleNamespace(pid=1)

        async def start(self, *args):
            pass

        async def retrieve(self, query):
            return b.RetrievalResult("blue", [])

        async def close(self):
            pass

    class Reader:
        def __init__(self, *args):
            pass

        async def respond(self, *args, **kwargs):
            return SimpleNamespace(hypothesis="blue", duration=0, prompt_tokens=1, completion_tokens=1)

        async def grade(self, question, gold, answer, category, **kwargs):
            calls.append(category)
            return SimpleNamespace(is_correct=category != "5", prompt_tokens=1, completion_tokens=1)

        async def aclose(self):
            pass

    class Tracking:
        def __init__(self):
            self.points = []

        def bind(self, manifest):
            pass

        async def record(self, record, *, extra_metrics=None):
            self.points.append(extra_metrics)

    monkeypatch.setattr(b, "BackendProcess", Worker)
    monkeypatch.setattr(reader_grader, "ReaderGrader", Reader)
    opts = parse_args(["--dataset", "locomo", "--backend", "full_context", "--data-path", str(data),
                       "--output", str(tmp_path / "out.jsonl"), "--text-mode", "text_caption"])
    tracking = Tracking()
    result = await b.run(opts, tracking=tracking)
    assert result["bleu1"] == 1 and result["bleu1_valid_count"] == 4
    report = result["locomo_evaluation"]
    assert report["overall"]["judge_accuracy"] == .8
    assert report["mem0"]["judge_accuracy"] == 1
    assert report["evaluation_complete"]
    assert report["judge_protocol_version"] == "locomo-mem0-abstention-v1"
    assert len(tracking.points) == 5
    assert tracking.points[-1]["judge_accuracy"] == .8
    assert tracking.points[-1]["mem0_judge_accuracy"] == 1
    assert tracking.points[-1]["judge_accuracy_adversarial"] == 0
    for cat in CATEGORIES.values():
        assert report["by_category"][cat]["graded"] == 1
        assert f"locomo_graded_{cat}" not in tracking.points[-1]
    assert len(tracking.points[-1]) == 7
    terminal = capsys.readouterr().out
    assert "adversarial=" in terminal and "mem0_1_4=" in terminal
    resumed = await b.run({**opts, "resume": True}, tracking=tracking)
    assert len(calls) == 5 and len(tracking.points) == 5
    assert resumed["locomo_evaluation"] == report
    assert resumed["bleu1"] == 1 and resumed["bleu1_valid_count"] == 4
    for mode in ("regrade", "replay"):
        replay_opts = parse_args([f"--{mode}-from", opts["output"],
                                  "--output", str(tmp_path / f"{mode}.jsonl")])
        refreshed = await b.regrade(replay_opts)
        assert refreshed["locomo_evaluation"] == report
        assert refreshed["bleu1"] == 1 and refreshed["bleu1_valid_count"] == 4

    # A missing row in a shard must still count in the merged expected denominator.
    out = Path(opts["output"])
    rows = b.read_records(out)
    for saved in rows:
        if saved["question_type"] == "5":
            assert "bleu1" not in saved and "bleu1" not in saved["metrics"]
        else:
            assert saved["bleu1"] == saved["metrics"]["bleu1"] == 1
    out.write_text("".join(json.dumps(r) + "\n" for r in rows[:-1]))
    merged = b.merge_results({"merge_from": [str(out)], "output": str(tmp_path / "merged.jsonl")})
    assert merged["locomo_evaluation"]["overall"]["planned"] == 5
    assert merged["locomo_evaluation"]["overall"]["missing"] == 1
    assert not merged["locomo_evaluation"]["evaluation_complete"]
    assert merged["locomo_evaluation"]["mem0"]["graded"] == 4
    assert merged["bleu1"] == 1 and merged["bleu1_valid_count"] == 4
