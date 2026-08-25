import argparse
import asyncio
from pathlib import Path

from promptwatch.config import PromptConfig
from promptwatch.dataset import GoldenDataset
from promptwatch.diff import DEFAULT_CRITICAL, DEFAULT_WARN, diff_runs, format_diff
from promptwatch.gemini import DEFAULT_MODEL
from promptwatch.judge import JudgeConfig
from promptwatch.results import DEFAULT_DB, connect, latest_run, load_run, save_run
from promptwatch.runner import (
    DEFAULT_CONCURRENCY,
    DEFAULT_REQUESTS_PER_MINUTE,
    run_dataset,
)

DEFAULT_DATASET = Path("datasets/golden_v1.json")
DEFAULT_JUDGE = Path("prompts/judge_v1.yaml")

EXIT_CODES = {"pass": 0, "warn": 1, "critical": 2, "no_data": 2}


def _summarise(run) -> str:
    return "\n".join(
        [
            f"run           {run.run_id}",
            f"prompt        {run.prompt_version}   model {run.model}"
            f"   judge {run.judge_version}",
            f"cases         {len(run.cases)}  "
            f"(scored {len(run.scored)}, "
            f"out of contract {run.count('out_of_contract')}, "
            f"off contract output {run.count('off_contract_output')}, "
            f"errors {run.count('error')})",
            f"accuracy      {run.category_accuracy:.2%}",
            f"summary mean  {run.mean_summary_score:.2f}",
            f"latency       mean {run.mean_latency_ms:.0f}ms  "
            f"max {run.max_latency_ms}ms",
            f"tokens        {run.total_tokens}",
        ]
    )


def _run(args: argparse.Namespace) -> int:
    prompt_config = PromptConfig.load(args.prompt)
    dataset = GoldenDataset.load(args.dataset)
    judge_config = None if args.skip_judge else JudgeConfig.load(args.judge)

    connection = connect(args.db)
    previous = latest_run(connection, prompt_config.version)

    run = asyncio.run(
        run_dataset(
            prompt_config,
            dataset,
            judge_config,
            model=args.model,
            concurrency=args.concurrency,
            requests_per_minute=args.rpm,
            limit=args.limit,
        )
    )
    save_run(connection, run)
    print(_summarise(run))

    if previous is None:
        print("\nno earlier run for this prompt version, nothing to diff against")
        return 0

    diff = diff_runs(previous, run, warn=args.warn, critical=args.critical)
    print("\n" + format_diff(diff))
    return EXIT_CODES[diff.verdict]


def _diff(args: argparse.Namespace) -> int:
    connection = connect(args.db)
    diff = diff_runs(
        load_run(connection, args.base),
        load_run(connection, args.head),
        warn=args.warn,
        critical=args.critical,
    )
    print(format_diff(diff))
    return EXIT_CODES[diff.verdict]


def _runs(args: argparse.Namespace) -> int:
    connection = connect(args.db)
    rows = connection.execute(
        "SELECT run_id, prompt_version, model, judge_version, started_at "
        "FROM runs ORDER BY started_at DESC, run_id DESC"
    ).fetchall()
    if not rows:
        print("no runs recorded")
        return 0
    for row in rows:
        print(
            f"  {row['run_id']:28} prompt {row['prompt_version']:4} "
            f"model {row['model']:24} judge {row['judge_version']}"
        )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a prompt version.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--warn", type=float, default=DEFAULT_WARN)
    parser.add_argument("--critical", type=float, default=DEFAULT_CRITICAL)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="evaluate a prompt against the dataset")
    run.add_argument("prompt", type=Path)
    run.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    run.add_argument("--judge", type=Path, default=DEFAULT_JUDGE)
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--limit", type=int)
    run.add_argument("--skip-judge", action="store_true")
    run.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    run.add_argument("--rpm", type=int, default=DEFAULT_REQUESTS_PER_MINUTE)
    run.set_defaults(handler=_run)

    diff = sub.add_parser("diff", help="compare two recorded runs")
    diff.add_argument("base")
    diff.add_argument("head")
    diff.set_defaults(handler=_diff)

    runs = sub.add_parser("runs", help="list recorded runs")
    runs.set_defaults(handler=_runs)
    return parser


if __name__ == "__main__":
    args = _parser().parse_args()
    raise SystemExit(args.handler(args))
