"""Optional protocol reporting, shared by terminal, saved summaries and tracking."""


def selected_questions(manifest: dict, records=()) -> list[dict]:
    """Keep manifest IDs, recovering legacy categories without dropping missing questions."""
    if "reporting_questions" in manifest:
        return manifest["reporting_questions"]
    questions = [{"id": q["id"], "category": str(q.get("target", {}).get("category", ""))}
                 for q in manifest.get("identity", {}).get("questions", [])]
    if "question_order" in manifest:
        categories = {r["question_id"]: str(r.get("question_type", "")) for r in records}
        categories.update({q["id"]: q["category"] for q in questions if q["category"]})
        return [{"id": qid, "category": categories.get(qid, "")} for qid in manifest["question_order"]]
    return questions


def _reporter_factory(manifest: dict | None):
    from .benchmark import resolve

    if not manifest:
        return None
    protocol = manifest.get("identity", {}).get("config", {}).get("protocol")
    if not protocol:
        return None
    return getattr(resolve("protocol", protocol), "reporter", None)


def has_reporter(manifest: dict | None) -> bool:
    """Protocols with a reporter own their MLflow metrics instead of adding legacy duplicates."""
    return _reporter_factory(manifest) is not None


def make_reporter(manifest: dict | None, records=()):
    factory = _reporter_factory(manifest)
    return factory(manifest, records) if factory else None
