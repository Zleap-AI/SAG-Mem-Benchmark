"""Legacy result selections and plugin loading through real postprocessing entry points."""

import json
import sys
from pathlib import Path

import pytest

from scripts import run_benchmark as cli
from src.harness import benchmark as b


def save_source(tmp_path, *, missing=False, known_categories=False, config=None):
    path = tmp_path / "source.jsonl"
    manifest = {"run_id": "legacy", "identity": {"config": config or {"protocol": "locomo"}},
                "question_order": ["q0", "q1"]}
    if known_categories:
        manifest["identity"]["questions"] = [
            {"id": f"q{i}", "target": {"category": category}}
            for i, category in enumerate(("4", "5"))]
    rows = [{"run_id": "legacy", "question_id": f"q{i}", "question_type": category,
             "status": "succeeded", "grade": i == 0, "bleu1": .5 if i == 0 else None}
            for i, category in enumerate(("4", "5"))]
    if missing:
        rows.pop()
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    path.with_suffix(".jsonl.manifest.json").write_text(json.dumps(manifest))
    return path


def test_merge_legacy_regrade_recovers_categories_and_preserves_source(tmp_path):
    source = save_source(tmp_path)
    original = source.read_bytes()
    output = tmp_path / "merged.jsonl"
    summary = b.merge_results({"merge_from": [str(source)], "output": str(output)})
    report = summary["locomo_evaluation"]
    assert report["overall"]["planned"] == report["overall"]["graded"] == 2
    assert report["overall"]["judge_accuracy"] == .5
    assert report["evaluation_complete"] and report["judge_protocol_version"] == "legacy_unverified"
    manifest = json.loads(output.with_suffix(".jsonl.manifest.json").read_text())
    assert manifest["reporting_questions"] == [{"id": "q0", "category": "4"}, {"id": "q1", "category": "5"}]
    assert source.read_bytes() == original


def test_merge_missing_legacy_category_fails_before_output_creation(tmp_path):
    source = save_source(tmp_path, missing=True)
    output = tmp_path / "merged.jsonl"
    with pytest.raises(ValueError, match="category metadata"):
        b.merge_results({"merge_from": [str(source)], "output": str(output)})
    assert not output.exists()
    assert not output.with_suffix(".jsonl.manifest.json").exists()


def test_merge_missing_legacy_row_keeps_known_expected_denominator(tmp_path):
    source = save_source(tmp_path, missing=True, known_categories=True)
    summary = b.merge_results({"merge_from": [str(source)], "output": str(tmp_path / "merged.jsonl")})
    report = summary["locomo_evaluation"]
    assert report["overall"]["planned"] == 2 and report["overall"]["missing"] == 1
    assert not report["evaluation_complete"]


@pytest.mark.parametrize("mode", ["metrics-only", "merge-from"])
@pytest.mark.parametrize("plugin_source", ["manifest", "cli"])
def test_postprocessing_loads_custom_protocol_plugins(tmp_path, monkeypatch, mode, plugin_source):
    module_name = "reporting_compat_test_plugin"
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    # Restore the registry and imported module after each case.
    monkeypatch.setitem(sys.modules, module_name, None)
    monkeypatch.delitem(sys.modules, module_name)
    monkeypatch.setattr(b, "REGISTRIES", {**b.REGISTRIES, "protocol": dict(b.REGISTRIES["protocol"])})
    (tmp_path / f"{module_name}.py").write_text(
        "from src.harness.benchmark import register\n"
        "from src.harness.reader_grader import LoCoMoProtocol\n"
        "register('protocol', 'reporting_custom', LoCoMoProtocol)\n")
    plugins = [module_name] if plugin_source == "manifest" else []
    source = save_source(tmp_path, config={"protocol": "reporting_custom", "plugins": plugins})
    output = source if mode == "metrics-only" else tmp_path / "merged.jsonl"
    args = (["--metrics-only"] if mode == "metrics-only" else ["--merge-from", str(source)])
    args += ["--output", str(output)]
    if plugin_source == "cli":
        args += ["--plugin", module_name]
    options = cli.parse_args(args)
    monkeypatch.setattr(cli, "parse_args", lambda: options)
    assert cli.main() == 0
    summary = json.loads(Path(output).with_suffix(".summary.json").read_text())
    assert summary["locomo_evaluation"]["overall"]["planned"] == 2
    assert summary["locomo_evaluation"]["evaluation_complete"]
