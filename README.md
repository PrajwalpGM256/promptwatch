# PromptWatch

Regression testing for LLM prompts.

Teams treat prompts as code but rarely test them like code. You tweak the wording, it might look better on the few examples,then you ship it, and you have no idea what else changed. PromptWatch runs a prompt version against a hand labelled dataset, scores the output, and diffs it against a previous run so you can see exactly what changed before it ships.

The feature under test is an email classifier for a job search inbox: given an email, return one of six categories plus a one sentence summary.

## What it produces

```
run           20260827T073919-v2
prompt        v2   judge v1
backend       gemini   model gemini-3.5-flash-lite
judged by     groq   model openai/gpt-oss-20b
cases         94  (scored 94, out of contract 0, off contract output 0, errors 0)
accuracy      90.43%
summary mean  3.63
latency       mean 885ms  max 5907ms
tokens        77930

20260827T073919-v2  vs  20260825T075108-v2

NOT A CLEAN PROMPT COMPARISON: judge backend gemini -> groq; judge model gemini-3.5-flash-lite -> openai/gpt-oss-20b

category accuracy   0.91 -> 0.90   (-1.1%)   PASS
summary mean        4.32 -> 3.63   (-0.69)   reported only
cases scored        base 100%   head 100%

regressions (2)
  gc-082  job_alert -> misc
  gc-086  misc -> newsletter
improvements (1)
  gc-053  interview_invite -> application_ack

out of contract     base: 0   head: 0
errors              base: 0   head: 0
```

Two things in that output are the whole product.

The per-case flip list, because an aggregate hides compensating errors: two cases break, one gets fixed, the headline barely moves, and you never find out which.

And the confounder banner. That `summary mean` fell 0.69, which reads as a quality collapse until you see the grader itself changed between the two runs. Without that line you would spend a day debugging a prompt that never got worse.

## Install

```powershell
python -m venv promptwatch-env
promptwatch-env\Scripts\activate
pip install -e ".[dev]"
```

`src/` layout, so the editable install is required — the package is not importable from a bare checkout. Run everything from the repo root: `prompts/` and `datasets/` resolve against the working directory, not the package.

The default backend is a local model through [Ollama](https://ollama.com), so a full run costs nothing and has no quota:

```powershell
winget install Ollama.Ollama    # or download from ollama.com
ollama pull gemma3:4b
ollama list                     # confirm it is registered
ollama ps                       # confirm it loads on GPU, not CPU
```

Ollama installs a background service on port 11434. The Python side talks to it over HTTP and never shells out, so the `ollama` binary does not need to be on `PATH`.

Hosted backends need keys in `.env`. Neither is required to run locally, but the judge defaults to Groq, so `GROQ_API_KEY` is needed for any run that scores summaries.

```
GEMINI_API_KEY=your-key              # --provider gemini
GROQ_API_KEY=your-key                # --provider groq, and the default judge
OLLAMA_HOST=http://localhost:11434   # optional, this is the default
GMAIL_ADDRESS=you@gmail.com          # only for tools/fetch_emails.py
GMAIL_APP_PASSWORD=your-app-password
```

## Commands

### Smoke test a backend

One request, one email, no database write. The fastest way to prove a backend is reachable and on contract.

```powershell
python -m promptwatch.classifier ollama
python -m promptwatch.classifier gemini
python -m promptwatch.classifier groq
```

Prints the backend, a validated `ClassificationResult`, and the token counts.

### Evaluate a prompt

```powershell
promptwatch run prompts/v2.yaml
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
| `--concurrency` | the provider's own | In-flight requests. Ollama is 1, hosted is 5 |
| `--rpm` | the provider's own | Requests per minute for the classifier; `0` disables pacing |
| `--db` | `runs.db` | Database file. Use a throwaway for experiments |
| `--warn` | `0.03` | Accuracy drop that triggers WARN |
| `--critical` | `0.08` | Accuracy drop that triggers CRITICAL |

`--db` and the thresholds are global flags and go *before* the subcommand:

```powershell
promptwatch --db smoke.db run prompts/v2.yaml --limit 5 --skip-judge
promptwatch --warn 0.01 --critical 0.05 run prompts/v2.yaml
```

### Full dataset across all three backends

Run these **sequentially**. All three send judge traffic to the same Groq account, and three in-process rate limiters cannot see each other — running them in parallel produces nothing but 429s. One line guarantees the ordering:

```powershell
promptwatch run prompts/v2.yaml --provider ollama; promptwatch run prompts/v2.yaml --provider groq; promptwatch run prompts/v2.yaml --provider gemini
```

Accuracy needs no judge, so measure that first if that is the question — it is faster and puts no load on Groq at all:

```powershell
promptwatch run prompts/v2.yaml --provider ollama --skip-judge; promptwatch run prompts/v2.yaml --provider gemini --skip-judge; promptwatch run prompts/v2.yaml --provider groq --skip-judge
```

Rough cost and duration for one full 94-case run:

| Backend | Classifier calls | Judge calls | Wall clock | Notes |
|---|---|---|---|---|
| `ollama` | 94, free, unpaced | 94 on Groq | judge-bound, ~16 min | ~4 min with `--skip-judge` |
| `gemini` | 94 at 12/min | 94 on Groq | ~16 min | ~8 min with `--skip-judge` |
| `groq` | 94 at 6/min | 94, same limiter | ~30 min | classifier and judge share one quota |

### Inspect and compare runs

```powershell
promptwatch runs                                  # every recorded run, newest first
promptwatch diff <base_run_id> <head_run_id>      # compare any two, no API calls
promptwatch --db smoke.db runs                    # against a throwaway database
```

`diff` works across backends and will tell you the comparison is confounded rather than silently pretending it is clean.

### Exit codes

`run` and `diff` share them, so either can gate CI directly.

| Code | Meaning |
|---|---|
| `0` | PASS |
| `1` | WARN — accuracy dropped more than `--warn` |
| `2` | CRITICAL — dropped more than `--critical`, or too few cases scored to judge |

### Development

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

### Building the dataset

```powershell
python -m tools.fetch_emails "subject:(interview)" --limit 20 --out raw_emails/batch.txt
```

Pulls from Gmail over IMAP into a labelling worksheet. Needs `GMAIL_ADDRESS` and a Gmail app password, not your account password. Output lands in `raw_emails/`, which is gitignored and must never be committed — it is unredacted mail. Redaction helpers live in `tools/redact.py` and are applied while hand-assembling cases.

## How it works

```
cli.py                            CLI
  runner.py                       runs all cases concurrently, paced
    classifier.py                 the feature under test
    judge.py                      scores summaries 1-5
    results.py                    SQLite storage
  diff.py                         compares two runs, produces a verdict
config.py  dataset.py             typed, validated inputs
models.py                         the category contract
provider.py                       one transport interface
  ollama.py  gemini.py  groq.py   three backends behind it
```

A run is `(prompt version, provider, model, dataset version, judge version, judge provider, judge model, started_at)` plus one scored result per case, stored in SQLite with the constraints enforced by the database rather than only in Python. Every element of that tuple is something that can silently change a score, which is why all of it is recorded and any change to it shows up as a confounder.

## Design decisions worth reading

**Prompt versions are hermetic.** Each prompt YAML declares its own category list, and the response schema is built from that list rather than from the `Category` type in code. This exists because of a real bug: adding a sixth category to the type silently changed what `v1.yaml` returned, despite that file never being edited. For a tool that answers "did this prompt change break anything", a code change appearing as a prompt regression is fatal. Now each YAML is a complete record of what that version could emit.

**Out-of-contract cases are separated from failures.** v1 cannot produce `interview_invite`, so the 8 cases labelled that way are marked out of contract and excluded from the accuracy denominator instead of counted wrong. Without this, the first v1-versus-v2 comparison reports a fake 8-case regression.

**Transport errors are quarantined from contract violations.** A model returning a category the prompt forbids is real signal and counts as wrong. A network failure is noise: it is retried, then excluded from the denominator entirely. A run with five dropped connections must not read as five wrong answers.

**Classification runs at temperature 0.** This is the least glamorous decision here and the one that matters most. The judge was pinned at temperature 0 from the start but the classifier was not, so it ran at each provider's default of around 1.0. Two runs of the *same prompt on the same model against the same dataset* then disagreed with each other, and the diff dutifully reported those disagreements as regressions. Three cases — `gc-053`, `gc-082`, `gc-086` — were caught flipping back and forth across runs where nothing had changed. A regression harness that manufactures regressions cannot be used to detect them, and every number produced before this fix is not comparable to anything after it. Temperature 0 is greedy decoding rather than a guarantee of determinism, but it removes the sampling variance that was drowning the signal.

**Confounders are detected.** If two runs differ in provider, model, dataset version, judge version, judge backend or judge model, the diff prints `NOT A CLEAN PROMPT COMPARISON` and names what changed. This was added after a diff proudly reported a 20% improvement that was entirely due to a model swap, and it earned its keep again the first time the judge backend moved.

**Pacing and concurrency belong to the backend, not the runner.** A local model on one GPU serialises anyway, so five requests in flight bought nothing and made Ollama return 500 for 15 of 94 cases; its concurrency is 1. Groq's binding constraint is tokens per minute rather than requests, so its request pacing is set to fit the token budget rather than the request ceiling. And when the classifier and judge share a backend they share one rate limiter, because one account is one quota however many call sites draw on it. `TransientError` also carries the server's `Retry-After` header when it sends one — guessing an exponential backoff against a token bucket that refills on the server's schedule just burns retry attempts.

**The judge is pinned to one backend, whatever is under test.** Summary scores are only comparable if the grader is constant, so `--judge-provider` defaults to Groq and does not follow `--provider`. Letting it inherit meant gpt-oss-20b wrote summaries and then graded them, and a Gemini run and a Groq run reported 4.60 and 3.60 with no way to tell whether that was summary quality or grader strictness. Both numbers were measuring two things at once.

**The backend is part of a run's identity, not a setting.** A run records its provider and model, and "the previous run to compare against" is scoped to both. The alternative — keying baselines on prompt version alone — means a local run silently diffs against a hosted one and reports the backend swap as a prompt regression. That is the exact failure the tool exists to catch, so it cannot be allowed in the tool itself. It also turns a constraint into a feature: the same golden suite runs against three backends, so "is the cheap local model good enough" becomes a question the harness answers rather than one you guess at.

**Dataset validation is two-tier.** Schema checks are always-fatal validators. The class-balance check is a separate method you call deliberately, because a dataset is legitimately unbalanced while you are labelling case 12 of 94, and a load-time balance invariant would make the file unloadable for most of its life.

**Aggregates are computed, never stored.** Accuracy, mean score and token totals are derived on read, so out-of-contract cases are structurally absent from the calculation rather than filtered out by remembering to filter.

## The dataset

`datasets/golden_v1.json` holds 94 emails from a real inbox, redacted and hand-verified. Each case carries a stable append-only id, the correct category, two to four key facts a good summary must convey, a difficulty, optional tags, and a note on why it is worth keeping.

Real mail is far messier than anything synthetic. The corpus includes an email whose subject reads "Thank you for applying" over a body that is a rejection, a job alert with a broken template merge splicing a job title into an unrelated sentence, an advert whose subject contains "Interview", and a case whose entire body is a name and a phone number.

Redaction replaces people, addresses, phone numbers, requisition ids and tracking links, while deliberately preserving typos, casing and template artefacts, because those are the signal under test.

This is a balanced regression suite, not an accuracy estimate. The class mix is deliberate, so the headline score is not inbox accuracy.

## Results

**Baselines are being re-established.** Every run recorded before the temperature fix used non-greedy sampling and is not comparable to anything produced since, so the run history was cleared rather than left to imply a continuity that does not exist. The figures below are those historical runs, kept because the observations they produced are still true, but they should not be read as current.

Historical, 2026-08-25, `--provider gemini`, `gemini-3.5-flash-lite`, all 94 cases, non-greedy sampling:

| | v1 | v2 |
|---|---|---|
| accuracy | 87.21% | 91.49% |
| cases scored | 86 | 94 |
| out of contract | 8 | 0 |
| off-contract output | 1 | 0 |
| errors | 0 | 0 |
| summary mean | 4.32 | 4.32 |

A later Gemini run scored 90.43% on v2 — a one-case difference, which is exactly the sampling noise that motivated the temperature fix.

The single off-contract output in v1 is worth noting: the model returned a summary longer than 200 characters. The response schema constrained the category but said nothing about summary length, which is a live demonstration that structured output is a strong constraint rather than a guarantee, and why the result is validated again after parsing.

Of v2's eight failures, five are the same disagreement: promotional and product emails labelled `misc` that the model calls `newsletter`. That is one unstable category boundary rather than eight independent errors, and it is the concrete next thing to fix in the prompt.

## Status

Phases 1 to 3 are complete and validated end to end against all three backends on the full 94-case dataset. Phase 4 (HTML diff report, Slack alerts, rolling-average drift detection) and Phase 5 (GitHub Action on PRs touching `prompts/`, Docker) are designed but not built.

Known limitations, honestly:

- **Latency is only trustworthy when nothing is throttling.** Retries happen inside the measured window, so a throttled call is indistinguishable from a slow one. Under quota pressure the same prompt measured 941ms mean in one run and 10,048ms in another. A clean unthrottled Gemini run measures 885ms mean with a 5.9s max, which is believable — but read the `errors` count before trusting any latency figure.
- **The judge uses a coarse scale.** Scores cluster at 5 and 3 with almost nothing at 4, so small movements in the mean should not be read as meaningful. Summary score is reported but deliberately never gates the verdict.
- **The `misc` category is heterogeneous**, covering non-job mail, product announcements, staffing marketing and relationship mail, so a `misc` regression does not say which kind broke.
- **The local backend has not been benchmarked.** `gemma3:4b` runs and stays on contract, but has no clean full-dataset accuracy figure yet. Runs are not comparable across backends and the diff will say so.
- **Groq's real limit is tokens, not requests.** The free tier allows far more requests per day than this suite needs, but a judge call is around 900 tokens and the token-per-minute ceiling is what actually throttles. Request pacing is set conservatively to fit the lowest tier; raise `--rpm` if your account allows more.
- **gpt-oss occasionally returns an empty generation.** Roughly one judge call in 94 fails with `400 json_validate_failed` and an empty `failed_generation`, which is the model spending its output budget on reasoning tokens. It is counted as a judge failure and excluded rather than silently scored.
- **Rate limiting is per-process.** The limiter cannot see other running instances, so two concurrent evaluations against the same account will throttle each other. Run them sequentially.
- **CI cannot use the local backend.** GitHub runners have no GPU, so Phase 5 will run against Gemini or Groq. On Groq, schema-constrained output is only available on the gpt-oss models.

## Layout

```
pyproject.toml           deps and all tool config
src/promptwatch/
  cli.py                 CLI: run, diff, runs
  classifier.py          the feature under test
  models.py              Category, EmailInput, ClassificationResult
  config.py              PromptConfig, per-version category declaration
  dataset.py             GoldenCase, GoldenDataset, check_balance
  provider.py            Turn, Completion, Provider protocol, get_provider
  ollama.py              local backend, the default
  gemini.py              Gemini backend
  groq.py                Groq backend
  judge.py               summary scoring
  runner.py              concurrent execution, pacing, failure handling
  results.py             result models and SQLite storage
  diff.py                run comparison and verdict
prompts/                 v1.yaml, v2.yaml, judge_v1.yaml
datasets/golden_v1.json  94 labelled cases
tools/                   Gmail fetcher and redaction helpers
tests/                   111 tests, no API calls
```

Everything reached over the network sits behind the `Provider` protocol, and `FakeProvider` in `tests/conftest.py` implements it in memory. That is why the runner, classifier and judge are all testable offline: the suite makes no API calls and needs no keys.
