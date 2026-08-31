# CLI reference

Every command PromptWatch exposes. The [README](../README.md) covers why the tool works the way it does; this file covers how to drive it.


## Smoke test a backend

One request, one email, no database write. The fastest way to prove a backend is reachable and on contract.

```powershell
python -m promptwatch.classifier ollama
python -m promptwatch.classifier gemini
python -m promptwatch.classifier groq
```

Prints the backend, a validated `ClassificationResult`, and the token counts.

## Evaluate a prompt

```powershell
promptwatch run prompts/v3.yaml
```

Runs every case, scores it, stores the run, and diffs against the most recent earlier run *of the same prompt on the same backend*. `python -m promptwatch run ...` is identical if you would rather not rely on the console script.

| Flag | Default | What it does |
|---|---|---|
| `--provider` | `ollama` | Backend under test: `ollama`, `gemini`, `groq` |
| `--model` | the provider's own | Override the model, e.g. `--model qwen3:8b` |
| `--judge-provider` | `groq` | Backend that grades summaries. Deliberately independent of `--provider` |
| `--judge-model` | the judge backend's own | Override the grader's model |
| `--judge` | `prompts/judge_v1.yaml` | Judge prompt version |
| `--skip-judge` | off | Classification only. Halves the calls and removes all Groq traffic |
| `--dataset` | `datasets/golden_v1.json` | Golden corpus to run against |
| `--limit` | all 94 | First N cases only. For smoke runs, not sampling |
| `--quiet` | off | Suppress the per-case progress lines |
| `--concurrency` | the provider's own | In-flight requests. Ollama is 1, hosted is 5 |
| `--rpm` | the provider's own | Requests per minute for the classifier; `0` disables pacing |
| `--db` | `runs.db` | Database file. Use a throwaway for experiments |
| `--warn` | `0.05` | Accuracy drop that triggers WARN |
| `--critical` | `0.08` | Accuracy drop that triggers CRITICAL |

`--db` and the thresholds are global flags and go *before* the subcommand:

```powershell
promptwatch --db smoke.db run prompts/v3.yaml --limit 5 --skip-judge
promptwatch --warn 0.01 --critical 0.05 run prompts/v3.yaml
```

## Full dataset across all three backends

Run these **sequentially**. All three send judge traffic to the same Groq account, and three in-process rate limiters cannot see each other — running them in parallel produces nothing but 429s. One line guarantees the ordering:

```powershell
promptwatch run prompts/v3.yaml --provider ollama; promptwatch run prompts/v3.yaml --provider groq; promptwatch run prompts/v3.yaml --provider gemini
```

Accuracy needs no judge, so measure that first if that is the question — it is faster and puts no load on Groq at all:

```powershell
promptwatch run prompts/v3.yaml --provider ollama --skip-judge; promptwatch run prompts/v3.yaml --provider gemini --skip-judge; promptwatch run prompts/v3.yaml --provider groq --skip-judge
```

Rough cost and duration for one full 94-case run:

| Backend | Classifier calls | Judge calls | Wall clock | Notes |
|---|---|---|---|---|
| `ollama` | 94, free, unpaced | 94 on Groq | judge-bound, ~16 min | ~4 min with `--skip-judge` |
| `gemini` | 94 at 12/min | 94 on Groq | ~16 min | ~8 min with `--skip-judge` |
| `groq` | 94 at 6/min | 94, same limiter | ~30 min | classifier and judge share one quota |

## Inspect and compare runs

```powershell
promptwatch runs                                  # every recorded run, newest first
promptwatch diff <base_run_id> <head_run_id>      # compare any two, no API calls
promptwatch --db smoke.db runs                    # against a throwaway database
```

`diff` works across backends and will tell you the comparison is confounded rather than silently pretending it is clean.

## Exit codes

`run` and `diff` share them, so either can gate CI directly.

| Code | Meaning |
|---|---|
| `0` | PASS |
| `1` | WARN — accuracy dropped more than `--warn` |
| `2` | CRITICAL — dropped more than `--critical`, or too few cases scored to judge |

## Development

```powershell
ruff check .              # lint
ruff check --fix .        # lint and autofix
mypy                      # strict over src/ and tools/
pytest                    # 111 tests, no API calls, fails under 80% coverage
pytest -q --no-cov        # fast, no coverage gate
pytest tests/test_runner.py -q
pytest -k determinism -q
```

All three must pass before a commit. There is no CI yet; Phase 5 adds it.

## Building the dataset

```powershell
python -m tools.fetch_emails "subject:(interview)" --limit 20 --out raw_emails/batch.txt
```

Pulls from Gmail over IMAP into a labelling worksheet. Needs `GMAIL_ADDRESS` and a Gmail app password, not your account password. Output lands in `raw_emails/`, which is gitignored and must never be committed — it is unredacted mail. Redaction helpers live in `tools/redact.py` and are applied while hand-assembling cases.

## Aborted runs

A case is marked `error` only after five retries with up to ninety seconds of
backoff. Anything that survives that is a quota or configuration problem, not a
blip, so the run is abandoned, nothing is written to the database, and the exit
code is `2`. Partial runs are never recorded.

Pass `--quiet` to suppress the per-case progress lines.
