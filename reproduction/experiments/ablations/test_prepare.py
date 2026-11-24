"""Three offline contract tests. No model clients or credentials are imported."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from reproduction.experiments.ablations.prepare import (
    CONDITIONS,
    _corpus,
    comparison_profile,
    file_hash,
    fingerprint,
    h0_items,
    input_identity,
    normalize_passage,
    render,
    verify_mapping,
)


def evidence(identifier, body, date="2026-11-05"):
    return {"chunk_id": f"s0:t{identifier}", "sag2_chunk_id": identifier, "kind": "chunk",
            "session_id": "s", "content": f"user (date: {date}): {body}",
            "token_count": 10, "message_spans": [{"start": 0, "end": len(body),
                "message_id": str(identifier), "session_position": 0, "message_position": identifier}],
            "sources": [{"session_id": "s", "message_id": str(identifier)}],
            "source_chunk_ids": [f"s0:t{identifier}"]}


class OfflineContracts(unittest.TestCase):
    def test_native_replay_inherits_only_retrieval_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.jsonl"
            source.write_text('{"question_id":"test"}\n')
            original = {"build_dependencies": {"chunking": "frozen"}, "questions": [{"id": "test"}],
                        "data_checksums": {"data": "digest"}, "backend_runtime": {"jev": "frozen"},
                        "dataset_code": "dataset", "backend_code": "backend",
                        "reader_service": {"model": "rejected-model"}}
            source.with_suffix(".jsonl.manifest.json").write_text(json.dumps({
                "identity": original, "run_id": fingerprint(original)}))
            final = {"evaluation_mode": "reader_judge_replay", "source_sha256": file_hash(source),
                     "source_run": fingerprint(original), "reader_service": {"model": "qwen"}}
            manifest = {"identity": final, "source_path": str(source)}
            result, lineage = input_identity(Path(folder) / "replayed.manifest.json", manifest)
            self.assertEqual(result["reader_service"]["model"], "qwen")
            self.assertEqual(result["backend_runtime"], original["backend_runtime"])
            self.assertEqual(len(lineage), 2)
            source.write_text('changed\n')
            with self.assertRaises(ValueError):
                input_identity(Path(folder) / "replayed.manifest.json", manifest)

    def test_transport_change_preserves_semantic_checks(self):
        original = {"reader_service": {"base_url": "https://openrouter.ai/api/v1", "model": "qwen",
                    "extra_body": {"reasoning": {"enabled": False}, "provider": {
                        "order": ["DeepInfra"], "allow_fallbacks": False, "quantizations": ["fp8"]}}},
                    "retrieval": {"jev": {"endpoint": "https://openrouter.ai/api/alpha/decisions",
                                           "model": "jev-fixed"}, "retrieval": {"max_hops": 1}}}
        migrated = copy.deepcopy(original)
        service = migrated["reader_service"]
        service["base_url"] = "https://proxy.example.test/v1"
        provider = service["extra_body"]["provider"]
        provider["only"] = provider.pop("order")
        migrated["retrieval"]["jev"]["endpoint"] = "https://proxy.example.test/decisions"
        snapshot = copy.deepcopy(original)
        self.assertEqual(comparison_profile(original), comparison_profile(migrated))
        self.assertEqual(original, snapshot)
        migrated["retrieval"]["retrieval"]["max_hops"] = 2
        self.assertNotEqual(comparison_profile(original), comparison_profile(migrated))
        migrated["retrieval"]["retrieval"]["max_hops"] = 1
        migrated["reader_service"]["model"] = "different-model"
        self.assertNotEqual(comparison_profile(original), comparison_profile(migrated))

    def test_time_conditions_preserve_body_and_original_parentheses(self):
        for body, date in [("Yesterday I moved. Original（2020/01/01）", "2026-11-05"),
                           ("Last month I moved.", "1:56 pm on 8 May, 2023"),
                           ('He said "yesterday". In two days.', "2026/11/05 13:30")]:
            items = [evidence(1, body, date)]
            original = copy.deepcopy(items)
            contexts = render(items)
            self.assertEqual(tuple(contexts), CONDITIONS)
            self.assertEqual(contexts["raw"], items[0]["content"])
            self.assertTrue(contexts["normalized"].endswith(body))
            self.assertEqual(items, original)
            self.assertIn("[Message date:", contexts["normalized"])

    def test_h0_restores_candidates_then_applies_dpp_without_mutation(self):
        mapping = {}
        for identifier, body in [(1, "apple apple"), (2, "apple apple"), (3, "banana banana")]:
            item = evidence(identifier, body)
            # Distinct source dates avoid duplicate index documents in this fixture.
            item["indexed_content"] = body
            mapping[str(identifier)] = item
        row = {"backend_stats": {"sag2_trace": [
            {"stage": "bm25", "hop": 0, "candidate_ids": [1, 2, 3], "candidate_document_ids": [1, 2, 3]},
            {"stage": "jev", "hop": 0, "scores": [{"id": 1, "score": .95},
                {"id": 2, "score": .9}, {"id": 3, "score": .7}]},
            {"stage": "jev", "hop": 1, "scores": [{"id": 2, "score": 1}]},
        ], "jev_usage": {"batches": [{"hop": 0, "candidate_count": 3, "input_tokens": 12, "output_tokens": 3},
                                       {"hop": 1, "candidate_count": 3, "input_tokens": 99, "output_tokens": 8}]}}}
        snapshot = copy.deepcopy((row, mapping))
        selected, usage = h0_items(row, mapping, 50, 3)
        self.assertEqual([i["sag2_chunk_id"] for i in selected], [1, 3, 2])
        self.assertEqual(usage["input_tokens"], 12)
        self.assertEqual(usage["output_tokens"], 3)
        self.assertEqual((row, mapping), snapshot)
        for item in selected:
            self.assertEqual(item["content"], mapping[str(item["sag2_chunk_id"])]["content"])

    def test_source_mapping_rejects_changed_record(self):
        item = evidence(0, "Unchanged original.")
        item.pop("sag2_chunk_id")
        _, mapping = _corpus([item], normalize_passage)
        verify_mapping(mapping, [item])
        mapping["1"]["content"] += " changed"
        with self.assertRaises(ValueError):
            verify_mapping(mapping, [item])


if __name__ == "__main__":
    unittest.main()
