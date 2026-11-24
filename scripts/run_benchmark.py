#!/usr/bin/env python3
"""Unified offline text QA: run, preflight, replay saved evidence, regrade, or refresh metrics."""

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings
from src.harness.benchmark import (
    atomic_json,
    latest_records,
    load_plugins,
    merge_results,
    prepare_run,
    read_records,
    regrade,
    run,
    worker_main,
)
from src.harness.tracking import ExperimentTracking


def parse_args(argv=None) -> dict:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="longmemeval")
    p.add_argument("--split", choices=("s", "oracle"), default="s")
    p.add_argument("--data-path")
    p.add_argument("--backend", default="full_context")
    p.add_argument("--protocol", help="Defaults to dataset; locomo uses Mem0 Judge on 1-4 and refusal Judge on 5")
    p.add_argument("--plugin", dest="plugins", action="append", default=[])
    p.add_argument("--backend-options", type=json.loads, default={})
    p.add_argument("--text-mode", choices=("text", "text_caption"), default="text")
    p.add_argument("--history-prefix", type=int)
    p.add_argument("--question-date-text", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--time-filter", choices=("none", "query_time"), default="none")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--graph-limit", type=int, default=20)
    p.add_argument("--strategy", default="pruned_expand_llm", choices=(
        "pruned_expand_llm", "pruned_expand_rff"))
    p.add_argument("--return-type", choices=("chunk", "event"), default="chunk")
    p.add_argument(
        "--evidence-mode",
        choices=("chunk", "event", "chunk_dedup", "event_chunk"),
        default="chunk",
    )
    p.add_argument("--extraction-profile", default="relaxed.en",
                   help='SAG extraction profile (default: relaxed.en); pass "" for SDK built-in prompts')
    p.add_argument("--reader-model", default=settings.response_model)
    p.add_argument("--judge-model", default=settings.grader_model)
    p.add_argument("--context-tokens", type=int)
    p.add_argument("--model-timeout", type=float, default=180)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--worker-timeout", type=float, default=18000)
    p.add_argument("--worker-close-timeout", type=float, default=300,
                   help="Seconds allowed for worker shutdown and closed-index maintenance")
    p.add_argument("--worker-retries", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--questions", nargs="+", default=[])
    p.add_argument(
        "--question-type",
        dest="question_types",
        action="append",
        default=[],
        help="Only run exact question types; repeatable",
    )
    p.add_argument("--stratified", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--namespace", default="benchmark")
    p.add_argument("--index-root", default=str(Path(__file__).resolve().parents[1] / "reproduction" / "indexes"))
    p.add_argument(
        "--reuse-builds-from",
        help="Prior result JSONL whose build IDs and verified build manifests should be reused",
    )
    p.add_argument("--output")
    p.add_argument("--retrieval-only", action="store_true",
                   help="Skip Reader/Judge; backend ingestion/retrieval/governance may still call models")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--metrics-only", action="store_true")
    mode.add_argument("--regrade-from")
    mode.add_argument("--replay-from", help="Regenerate Reader answers and Judge from saved context; no retrieval")
    mode.add_argument("--merge-from", nargs="+")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--rerun-incorrect", action="store_true")
    p.add_argument("--mlflow", action="store_true")
    p.add_argument("--run-name")
    args = vars(p.parse_args(argv))
    # 区分用户显式参数和 CLI 默认值，重评/回放才能保留源实验配置。
    explicit = {token.split("=")[0].removeprefix("--").replace("-", "_") for token in argv if token.startswith("--")}
    if "no_question_date_text" in explicit:
        explicit.add("question_date_text")
    if "question_type" in explicit:
        explicit.add("question_types")
    args["explicit"] = sorted(explicit)
    args["protocol"] = args["protocol"] or args["dataset"]
    for key in ("concurrency", "worker_timeout", "worker_close_timeout", "model_timeout", "top_k", "graph_limit"):
        if args[key] <= 0:
            p.error(f"--{key.replace('_', '-')} must be positive")
    for key in ("limit", "start", "worker_retries", "stratified"):
        if args[key] < 0:
            p.error(f"--{key.replace('_', '-')} cannot be negative")
    if args["history_prefix"] is not None and args["history_prefix"] < 0:
        p.error("--history-prefix cannot be negative")
    if args["context_tokens"] is not None and args["context_tokens"] <= 0:
        p.error("--context-tokens must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args["namespace"]):
        p.error("namespace must contain only letters, numbers, underscores and hyphens")
    if not isinstance(args["backend_options"], dict):
        p.error("--backend-options must be a JSON object")
    if args["questions"] and (args["question_types"] or args["stratified"] or args["start"] or args["limit"]):
        p.error("--questions cannot combine with --question-type/--stratified/--start/--limit")
    if args["stratified"] and (args["start"] or args["limit"]):
        p.error("--stratified cannot combine with --start/--limit")
    if args["rerun_incorrect"] and (args["resume"] or args["regrade_from"] or args["replay_from"] or args["merge_from"] or args["metrics_only"] or args["preflight"]):
        p.error("--rerun-incorrect cannot combine with another execution mode or --resume")
    if (args["regrade_from"] or args["replay_from"]) and args["retrieval_only"]:
        p.error("Saved-result evaluation cannot combine with --retrieval-only")
    if args["metrics_only"] and args["resume"]:
        p.error("--metrics-only cannot combine with --resume")
    if (args["resume"] or args["rerun_incorrect"] or args["metrics_only"] or args["regrade_from"] or args["replay_from"] or args["merge_from"]) and not args["output"]:
        p.error("This mode requires an explicit --output")
    if (args["metrics_only"] or args["rerun_incorrect"]) and not Path(args["output"]).exists():
        p.error("This mode requires an existing --output")
    if args["merge_from"] and args["resume"]:
        p.error("--merge-from requires a fresh output")
    if args["regrade_from"] or args["replay_from"] or args["metrics_only"] or args["merge_from"]:
        ignored = explicit & {"dataset", "data_path", "backend", "top_k", "strategy", "return_type", "time_filter",
                              "history_prefix", "extraction_profile", "questions", "question_types", "start",
                              "limit", "stratified", "reuse_builds_from"}
        if ignored:
            p.error(f"Options not supported in postprocessing mode: {sorted(ignored)}")
    if not args["output"]:
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S%f")
        run_dir = settings.results_dir / f"{stamp[:8]}_{args['dataset']}_{args['backend']}_{stamp[9:]}"
        args["output"] = str(run_dir / "run.jsonl")
    args["output"] = str(Path(args["output"]).resolve())
    args["index_root"] = str(Path(args["index_root"]).resolve())
    return args


def main() -> int:
    if sys.argv[1:] == ["--internal-worker"]:
        asyncio.run(worker_main())
        return 0
    options = parse_args()
    try:
        # 将数据准备和后处理纳入 tracking 生命周期，记录启动后的失败与中断。
        with ExperimentTracking(options) as tracking:
            if options["metrics_only"]:
                print(f"Refreshing metrics from {options['output']}", flush=True)
                from src.harness.reader_grader import aggregate_metrics
                output = Path(options["output"])
                manifest_path = output.with_suffix(output.suffix + ".manifest.json")
                manifest = None
                if manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text())
                    load_plugins(list(dict.fromkeys([
                        *manifest.get("identity", {}).get("config", {}).get("plugins", []),
                        *options.get("plugins", []),
                    ])))
                    tracking.bind(manifest)
                rows = latest_records(read_records(output))
                summary = aggregate_metrics(list(rows.values()), manifest)
                atomic_json(output.with_suffix(".summary.json"), summary)
            elif options["merge_from"]:
                print(f"Merging {len(options['merge_from'])} files into {options['output']}", flush=True)
                summary = merge_results(options, tracking=tracking)
            elif options["regrade_from"] or options["replay_from"]:
                action = "Replaying saved context" if options["replay_from"] else "Regrading saved answers"
                print(f"{action} from {options['replay_from'] or options['regrade_from']}", flush=True)
                summary = asyncio.run(regrade(options, tracking=tracking))
            else:
                print(f"Validating dataset={options['dataset']} split={options['split']} "
                      f"backend={options['backend']} start={options['start']} limit={options['limit']} "
                      f"data={options['data_path'] or 'default'}", flush=True)
                prepared = prepare_run(options)
                if options["preflight"]:
                    print(json.dumps({"questions": len(prepared[0]),
                                      "history_groups": len({e.history_digest for e in prepared[0]}),
                                      "unknown_history_time_questions": sum(e.unknown_time for e in prepared[0]),
                                      "run_id": prepared[2]["run_id"], "output": options["output"]}, indent=2))
                    return 0
                tracking.bind(prepared[2])
                summary = asyncio.run(run(options, prepared, tracking=tracking))
            tracking.complete(summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 1 if summary["errors"] else 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, ExceptionGroup) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
