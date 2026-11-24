"""LoCoMo reporting only: these metrics do not change Reader or Judge behavior."""

import math

from .reporting import selected_questions

CATEGORIES = {"1": "multi_hop", "2": "temporal", "3": "open_domain",
              "4": "single_hop", "5": "adversarial"}
REPORTING_VERSION = "locomo-reporting-v5"


class LoCoMoReporter:
    summary_key = "locomo_evaluation"

    def __init__(self, manifest: dict, records=()) -> None:
        records = list(records)
        self.manifest = manifest
        self.categories = {key: name for key, name in CATEGORIES.items()
                           if key in manifest.get("judge_reporting", {}).get("categories", CATEGORIES)}
        self.selection = {q["id"]: str(q["category"]) for q in selected_questions(manifest, records)}
        if not self.selection:
            raise ValueError("LoCoMo reporting requires a nonempty manifest question selection")
        if any(category not in self.categories for category in self.selection.values()):
            raise ValueError("LoCoMo reporting requires category metadata for every selected question")
        self.records: dict[str, dict] = {}
        self.step = 0
        for row in records:
            self._save(row)

    def empty(self):
        """Create a zero-result reporter with the same validated selection for final curves."""
        manifest = {**self.manifest, "reporting_questions": [
            {"id": qid, "category": category} for qid, category in self.selection.items()]}
        return type(self)(manifest)

    def _save(self, record: dict) -> None:
        qid = record["question_id"]
        if qid not in self.selection:
            return
        if str(record.get("question_type")) != self.selection[qid]:
            raise ValueError(f"LoCoMo result category mismatch: {qid}")
        # Copy only metric inputs: later record mutation must not change previous snapshots.
        rule = record.get("metrics", {}).get("locomo_rule_score", record.get("locomo_rule_score"))
        bleu = record.get("metrics", {}).get("bleu1", record.get("bleu1"))
        self.records[qid] = {"grade": record.get("grade"), "error": bool(record.get("error")),
                             "status": record.get("status"), "rule": rule,
                             "bleu1": bleu,
                             "retrieval_only": bool(record.get("retrieval_only"))}

    def update(self, record: dict) -> dict:
        self._save(record)
        self.step += 1
        return self.snapshot()

    def _stats(self, ids: list[str]) -> dict:
        rows = [self.records[qid] for qid in ids if qid in self.records]
        grades = [r["grade"] for r in rows if type(r["grade"]) is bool and not r["retrieval_only"]]
        rules = [r["rule"] for r in rows if type(r["rule"]) in (float, int)
                 and math.isfinite(r["rule"]) and not r["retrieval_only"]]
        return {"planned": len(ids), "completed": len(rows), "missing": len(ids) - len(rows),
                "graded": len(grades), "correct": sum(grades), "errors": sum(r["error"] for r in rows),
                "judge_accuracy": sum(grades) / len(grades) if grades else None,
                "judge_coverage": len(grades) / len(ids) if ids else None,
                "rule_score": sum(rules) / len(rules) if rules else None, "rule_valid_count": len(rules)}

    def snapshot(self) -> dict:
        overall = self._stats(list(self.selection))
        categories = {name: self._stats([qid for qid, cat in self.selection.items() if cat == category])
                      for category, name in self.categories.items()}
        scores = [stats["judge_accuracy"] for stats in categories.values()]
        complete = bool(self.selection) and overall["graded"] == overall["planned"] and all(
            r["status"] == "succeeded" and not r["error"] for r in self.records.values())
        provenance = self.manifest.get("judge_reporting", {})
        result = {"reporting_version": REPORTING_VERSION,
                "judge_protocol_version": provenance.get("judge_protocol_version", "legacy_unverified"),
                "evaluation_complete": complete, "status": "complete" if complete else "partial",
                "step": self.step, "overall": overall, "by_category": categories,
                "judge_macro_accuracy": sum(scores) / len(scores) if all(s is not None for s in scores)
                else None}
        mem0_categories = provenance.get("mem0_categories")
        if mem0_categories:
            mem0 = self._stats([qid for qid, cat in self.selection.items() if cat in mem0_categories])
            mem0_scores = [categories[CATEGORIES[cat]]["judge_accuracy"] for cat in mem0_categories]
            mem0["judge_macro_accuracy"] = (sum(mem0_scores) / len(mem0_scores)
                                            if all(s is not None for s in mem0_scores) else None)
            result["mem0"] = mem0
        bleu = [r["bleu1"] for qid, r in self.records.items()
                if self.selection[qid] != "5" and not r["retrieval_only"]
                and type(r["bleu1"]) in (float, int) and math.isfinite(r["bleu1"])]
        result["bleu1"] = sum(bleu) / len(bleu) if bleu else None
        result["bleu1_valid_count"] = len(bleu)
        return result

    @staticmethod
    def metrics(snapshot: dict, phase: str) -> dict:
        if phase not in {"live", "curve", "final"}:
            raise ValueError(f"Unknown reporting phase: {phase}")
        # The live curve already records actual completion order. Do not reconstruct another copy.
        if phase == "curve":
            return {}
        # Running snapshots and the final aggregate share the same statistic and metric name.
        values = {"judge_accuracy": snapshot["overall"]["judge_accuracy"]}
        for category, stats in snapshot["by_category"].items():
            values["judge_accuracy_" + category] = stats["judge_accuracy"]
        mem0 = snapshot.get("mem0", {})
        if mem0:
            values["mem0_judge_accuracy"] = mem0["judge_accuracy"]
        if phase == "final":
            values.update({"judge_macro_accuracy": snapshot["judge_macro_accuracy"],
                           "rule_score": snapshot["overall"]["rule_score"],
                           "bleu1": snapshot.get("bleu1")})
            if mem0:
                values["mem0_judge_macro_accuracy"] = mem0["judge_macro_accuracy"]
        return {key: value for key, value in values.items() if value is not None}

    @staticmethod
    def tags(snapshot: dict) -> dict:
        tags = {f"locomo.{key}": snapshot[key] for key in
                ("reporting_version", "judge_protocol_version", "evaluation_complete", "status")}
        counts = ("planned", "completed", "missing", "graded", "correct", "errors", "judge_coverage",
                  "rule_valid_count")
        tags.update({f"locomo.{key}": snapshot["overall"][key] for key in counts})
        tags["locomo.bleu1_valid_count"] = snapshot["bleu1_valid_count"]
        for category, stats in snapshot["by_category"].items():
            tags.update({f"locomo.{key}_{category}": stats[key] for key in ("planned", "graded", "errors")})
        tags.update({f"locomo.mem0_{key}": stats for key, stats in snapshot.get("mem0", {}).items()
                     if key in counts})
        return tags

    @staticmethod
    def terminal(snapshot: dict) -> str:
        def score(stats):
            accuracy = stats["judge_accuracy"]
            text = "N/A" if accuracy is None else f"{accuracy:.1%}"
            return f"{text}({stats['correct']}/{stats['graded']})"

        total = snapshot["overall"]
        categories = " ".join(f"{cat}={score(stats)}" for cat, stats in snapshot["by_category"].items())
        mem0 = f" mem0_1_4={score(snapshot['mem0'])}" if "mem0" in snapshot else ""
        return (f"LoCoMo step={snapshot['step']} completed={total['completed']}/{total['planned']} "
                f"graded={total['graded']}/{total['planned']} errors={total['errors']} "
                f"judge={score(total)}{mem0} {categories} status={snapshot['status']}")
