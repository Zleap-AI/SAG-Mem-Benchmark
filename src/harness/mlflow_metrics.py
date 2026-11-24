"""MLflow metric contract from dd7f20fda43a72263d05af013b012ac48efea526.

Source: scripts/run_experiment.py. Keep names, denominators, missing-value
behavior, and the live/completion-order versus final/dataset-order distinction.
The generic local summary is deliberately independent of this remote contract.
"""

from collections.abc import Iterator

# 远端沿用历史 MLflow 指标名称与口径；本地 summary 的扩展指标由 reader_grader 单独汇总。
TRACKED_RECORD_METRICS = (
    "bleu1",
    "chunk_session_recall",
    "answer_chunk_recall",
    "top5_session_recall",
    "top10_session_recall",
    "top20_session_recall",
    "memory_session_recall",
    "answer_memory_hit",
    "top5_memory_session_recall",
    "top10_memory_session_recall",
    "top20_memory_session_recall",
)


def aggregate_metrics(records: list[dict]) -> dict:
    # 保留旧版缺失值规则：无有效评分不生成 accuracy，数值指标不把缺失项补成零。
    graded = [r for r in records if r.get("grade") is not None]
    metrics: dict[str, float] = {}
    if graded:
        metrics["accuracy"] = sum(1 for r in graded if r["grade"]) / len(graded)
    by_type: dict[str, list[bool]] = {}
    for r in graded:
        by_type.setdefault(r["question_type"], []).append(bool(r["grade"]))
    for qtype, flags in by_type.items():
        metrics[f"accuracy_{qtype}"] = sum(flags) / len(flags)
    abs_flags = [bool(r["grade"]) for r in graded if r.get("is_abstention")]
    if abs_flags:
        metrics["accuracy_abstention"] = sum(abs_flags) / len(abs_flags)
    for name in TRACKED_RECORD_METRICS:
        values = [r[name] for r in records if isinstance(r.get(name), (int, float))]
        if values:
            metrics[name] = sum(values) / len(values)
    metrics["n_records"] = float(len(records))
    metrics["n_graded"] = float(len(graded))
    metrics["n_errors"] = float(sum(1 for r in records if r.get("error")))
    return metrics


def step_metrics(records: list[dict]) -> Iterator[tuple[str, float, int]]:
    """Legacy final curves, emitted in dataset order with one-based steps."""
    # 最终曲线按数据集顺序重建；运行中的 live 曲线按本次完成顺序，二者的 step 不混用。
    correct = evaluated = 0
    type_stats: dict[str, list[int]] = {}
    for step, record in enumerate(records, start=1):
        grade = record.get("grade")
        if grade is not None:
            evaluated += 1
            if grade is True:
                correct += 1
        qtype = record.get("question_type", "unknown")
        type_stats.setdefault(qtype, [0, 0])
        if grade is not None:
            type_stats[qtype][1] += 1
            if grade:
                type_stats[qtype][0] += 1
            yield "grade", 1.0 if grade else 0.0, step
        yield "llm_judge", correct / evaluated if evaluated else 0.0, step
        for name in TRACKED_RECORD_METRICS:
            value = record.get(name)
            if isinstance(value, (int, float)):
                yield name, float(value), step
        tc, tn = type_stats[qtype]
        if tn:
            yield f"llm_judge_{qtype}", tc / tn, step


def ordered_records(records: dict[str, dict], manifest: dict | None) -> list[dict]:
    """Select and order results using the persisted dataset selection, including resume."""
    # 合并与回放优先用显式 question_order，普通运行回退到 manifest 中的选题顺序。
    manifest = manifest or {}
    order = manifest.get("question_order")
    if order is None:
        questions = manifest.get("identity", {}).get("questions")
        if questions is not None:
            order = [q["id"] for q in questions]
    return [records[qid] for qid in order if qid in records] if order is not None else list(records.values())
