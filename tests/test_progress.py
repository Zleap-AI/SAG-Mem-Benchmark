from scripts.run_benchmark import parse_args
from src.harness.progress import TerminalProgress


def test_completed_retry_failure_and_finish_are_distinct(tmp_path, capsys):
    options = parse_args(["--output", str(tmp_path / "out.jsonl")])
    progress = TerminalProgress(5, 2, 2, options)
    progress.result({"question_id": "q4", "grade": False})
    progress.retry("q5", 2, RuntimeError("provider unavailable"))
    progress.result({"question_id": "q5", "grade": None, "error": "disk full", "failure_stage": "prepare"})
    progress.finish({"total": 5, "valid_grades": 4, "errors": 1, "accuracy": .75})
    output = capsys.readouterr().out
    assert "[4/5] question_id=q4 MISS" in output
    assert "retry question_id=q5 attempt=2/2" in output
    assert "[5/5] question_id=q5 ERR" in output
    assert "stage=prepare disk full" in output
    assert "DONE records=5 graded=4 accuracy=0.75 errors=1" in output
