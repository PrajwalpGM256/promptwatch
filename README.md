# PromptWatch

Regression testing for LLM prompts.

Teams treat prompts as code but rarely test them like code. You tweak the wording, it looks better on the two examples you tried, you ship it, and you have no idea what else moved. PromptWatch runs a prompt version against a hand-labelled dataset, scores the output, and diffs it against a previous run so you can see exactly what changed before it ships.

The feature under test is an email classifier for a job-search inbox: given an email, return one of six categories plus a one-sentence summary.

## What it produces

```
20260825T075108-v2  vs  20260825T082041-v1

category accuracy   0.87 -> 0.91   (+4.3%)   PASS
summary mean        4.32 -> 4.32   (+0.00)   reported only
cases scored        base 100%   head 100%

regressions (2)
  gc-006  misc -> newsletter
  gc-053  application_ack -> interview_invite
improvements (5)
  gc-082  misc -> job_alert
  gc-083  misc -> application_ack
  gc-084  None -> application_ack
  gc-086  newsletter -> misc
  gc-087  newsletter -> misc

out of contract     base: 8   head: 0
errors              base: 0   head: 0
```

The per-case flip list is the point. An aggregate score hides compensating errors: two cases break, two get fixed, the headline does not move, and you never find out.

## Quickstart

```bash
python -m venv promptwatch-env
promptwatch-env\Scripts\activate
pip install -e ".[dev]"
```

The default backend is a local model through [Ollama](https://ollama.com), so a full run costs nothing and has no quota:

```bash
ollama pull gemma3:4b
```

Hosted backends need a key in `.env`. Neither is required to run locally:

```
GEMINI_API_KEY=your-key              # --provider gemini
GROQ_API_KEY=your-key                # --provider groq
OLLAMA_HOST=http://localhost:11434   # optional, this is the default
GMAIL_ADDRESS=you@gmail.com          # only needed for tools/fetch_emails.py
GMAIL_APP_PASSWORD=your-app-password
```

```bash
promptwatch run prompts/v2.yaml                 # evaluate, store, auto-diff
promptwatch run prompts/v2.yaml --limit 5 --skip-judge   # quick smoke run
promptwatch run prompts/v2.yaml --provider gemini        # hosted backend
promptwatch runs                                # list recorded runs
promptwatch diff <base_run_id> <head_run_id>
```

`python -m promptwatch` works identically if you'd rather not rely on the console script. Run from the repo root either way — `prompts/` and `datasets/` resolve against the working directory.

```bash
ruff check .    # lint
mypy            # strict over src/ and tools/
pytest          # 100 tests, no API calls, fails under 80% coverage
```

`run` exits 0 on pass, 1 on warn, 2 on critical or when there is too little data to judge, so it can gate CI directly.

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

A run is `(prompt version, provider, model, dataset version, judge version, timestamp)` plus one scored result per case, stored in SQLite with the constraints enforced by the database rather than only in Python.

## Design decisions worth reading

**Prompt versions are hermetic.** Each prompt YAML declares its own category list, and the response schema is built from that list rather than from the `Category` type in code. This exists because of a real bug: adding a sixth category to the type silently changed what `v1.yaml` returned, despite that file never being edited. For a tool that answers "did this prompt change break anything", a code change appearing as a prompt regression is fatal. Now each YAML is a complete record of what that version could emit.

**Out-of-contract cases are separated from failures.** v1 cannot produce `interview_invite`, so the 8 cases labelled that way are marked out of contract and excluded from the accuracy denominator instead of counted wrong. Without this, the first v1-versus-v2 comparison reports a fake 8-case regression.

**Transport errors are quarantined from contract violations.** A model returning a category the prompt forbids is real signal and counts as wrong. A network failure is noise: it is retried, then excluded from the denominator entirely. A run with five dropped connections must not read as five wrong answers.

**Confounders are detected.** If two runs differ in provider, model, dataset version or judge version, the diff prints `NOT A CLEAN PROMPT COMPARISON` and names what changed. This was added after a diff proudly reported a 20% improvement that was entirely due to a model swap.

**The backend is part of a run's identity, not a setting.** A run records its provider and model, and "the previous run to compare against" is scoped to both. The alternative — keying baselines on prompt version alone — means a local run silently diffs against a hosted one and reports the backend swap as a prompt regression. That is the exact failure the tool exists to catch, so it cannot be allowed in the tool itself. It also turns a constraint into a feature: the same golden suite runs against three backends, so "is the cheap local model good enough" becomes a question the harness answers rather than one you guess at.

**Dataset validation is two-tier.** Schema checks are always-fatal validators. The class-balance check is a separate method you call deliberately, because a dataset is legitimately unbalanced while you are labelling case 12 of 94, and a load-time balance invariant would make the file unloadable for most of its life.

**Aggregates are computed, never stored.** Accuracy, mean score and token totals are derived on read, so out-of-contract cases are structurally absent from the calculation rather than filtered out by remembering to filter.

## The dataset

`datasets/golden_v1.json` holds 94 emails from a real inbox, redacted and hand-verified. Each case carries a stable append-only id, the correct category, two to four key facts a good summary must convey, a difficulty, optional tags, and a note on why it is worth keeping.

Real mail is far messier than anything synthetic. The corpus includes an email whose subject reads "Thank you for applying" over a body that is a rejection, a job alert with a broken template merge splicing a job title into an unrelated sentence, an advert whose subject contains "Interview", and a case whose entire body is a name and a phone number.

Redaction replaces people, addresses, phone numbers, requisition ids and tracking links, while deliberately preserving typos, casing and template artefacts, because those are the signal under test.

This is a balanced regression suite, not an accuracy estimate. The class mix is deliberate, so the headline score is not inbox accuracy.

## Results

Full runs on 2026-08-25, `--provider gemini`, `gemini-3.5-flash-lite`, all 94 cases:

| | v1 | v2 |
|---|---|---|
| accuracy | 87.21% | 91.49% |
| cases scored | 86 | 94 |
| out of contract | 8 | 0 |
| off-contract output | 1 | 0 |
| errors | 0 | 0 |
| summary mean | 4.32 | 4.32 |

The single off-contract output in v1 is worth noting: the model returned a summary longer than 200 characters. The response schema constrained the category but said nothing about summary length, which is a live demonstration that structured output is a strong constraint rather than a guarantee, and why the result is validated again after parsing.

Of v2's eight failures, five are the same disagreement: promotional and product emails labelled `misc` that the model calls `newsletter`. That is one unstable category boundary rather than eight independent errors, and it is the concrete next thing to fix in the prompt.

## Status

Phases 1 to 3 are complete and validated. Phase 4 (HTML diff report, Slack alerts, rolling-average drift detection) and Phase 5 (GitHub Action on PRs touching `prompts/`, Docker) are designed but not built.

Known limitations, honestly:

- **Latency is not yet a usable metric.** The Gemini SDK retries internally on rate limiting and that backoff falls inside the measured window, so a throttled call is indistinguishable from a slow one. The same prompt measured 941ms mean in one run and 10,048ms in another purely due to quota pressure.
- **The judge uses a coarse scale.** Scores cluster at 5 and 3 with almost nothing at 4, so small movements in the mean should not be read as meaningful. Summary score is reported but deliberately never gates the verdict.
- **The `misc` category is heterogeneous**, covering non-job mail, product announcements, staffing marketing and relationship mail, so a `misc` regression does not say which kind broke.
- **The published numbers are Gemini's.** A full judged run is 188 requests, which is why iteration moved to a local model; the local backend has its own accuracy and has not been benchmarked against the suite yet. Runs are not comparable across backends and the diff will say so.
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
tests/                   100 tests, no API calls
```
