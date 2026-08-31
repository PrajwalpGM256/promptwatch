# PromptWatch

**Regression testing for LLM prompts.** Run a prompt version against a hand labelled dataset, score it, and diff it against the previous run, so you find out what a wording change broke *before* you ship it.

Teams treat prompts as code but rarely test them like code. when the wording is changed, it might look better on few examples, then we ship, and we have no idea what else moved. This is the missing CI step.

The feature under test is an email classifier for a job search inbox. Given an email, return one of six categories plus a one sentence summary.

## Does it work?

Yes, and here is the evidence. The tool identified five cases as unstable, and they pointed at one weak category boundary in the prompt. Rewriting that boundary as `v3` produced:

| backend | model | v2 | v3 | change |
|---|---|---|---|---|
| gemini | `gemini-3.5-flash-lite` | 90.43% | **97.87%** | +7 cases |
| groq | `openai/gpt-oss-20b` | 79.79% | **90.43%** | +10 cases |
| ollama | `gemma3:4b` (local, free) | 78.72% | **86.17%** | +7 cases |

The same gain appears on three unrelated models. Replication across backends rules out sampling luck in any single model, which a result from one backend cannot do at this magnitude.

The gain also clears the project's measured noise floor. Four runs of an identical configuration, meaning the same prompt, model, dataset and temperature, disagreed with each other by 2 to 4 cases, because hosted inference is not reproducible even at temperature 0. 89 of 94 cases were stable across all four runs. Any movement smaller than roughly 5 cases is therefore indistinguishable from noise, and the warn threshold is set at 5% to sit above it.

An alerting threshold that falls inside a harness's own measurement noise produces false alarms and trains its users to ignore the tool.

## Quickstart

```powershell
python -m venv promptwatch-env
promptwatch-env\Scripts\activate
pip install -e ".[dev]"
```

The default backend is a local model via [Ollama](https://ollama.com), so a full run costs nothing and needs no API key:

```powershell
ollama pull gemma3:4b
promptwatch run prompts/v3.yaml --skip-judge
```

Hosted backends read `GEMINI_API_KEY` or `GROQ_API_KEY` from `.env`. Full flag reference in [docs/CLI.md](docs/CLI.md).

## What a run looks like

```
[ 52/94] gc-052  ok    misc                                   662ms
[ 53/94] gc-053  MISS  interview_invite != application_ack     770ms

run           20260828T203946-v3
backend       gemini   model gemini-3.5-flash-lite
judged by     groq     model openai/gpt-oss-20b
cases         94  (scored 94, out of contract 0, errors 0)
accuracy      97.87%

20260828T203946-v3  vs  20260827T203721-v2

NOT A CLEAN PROMPT COMPARISON: judge backend gemini -> groq

category accuracy   0.90 -> 0.98   (+7.4%)   PASS
regressions (0)
improvements (7)
  gc-053  interview_invite -> application_ack
  gc-086  newsletter -> misc
```

Two elements of that output carry the value.

The first is the per case flip list. An aggregate score can hide compensating errors. Two cases might break and two might get fixed, so the headline barely moves and the change goes unnoticed.

The second is the confounder banner, which states when a comparison is not clean rather than presenting a confounded result as a valid one.

`run` exits 0 on pass, 1 on warn, 2 on critical, so it gates CI directly.

## How it works

```
cli.py                      run, diff, runs. Turns a verdict into an exit code
  runner.py                 concurrent execution, pacing, failure handling
    classifier.py           the feature under test
    judge.py                scores summaries 1-5 against hand-labelled facts
    results.py              typed results, SQLite storage, schema migrations
  diff.py                   compares two runs, produces the verdict
provider.py                 one transport interface
  ollama.py gemini.py groq.py    three backends behind it
config.py  dataset.py       versioned prompts and the golden corpus
models.py                   the category contract
```

A run is `(prompt version, provider, model, dataset version, judge version, judge provider, judge model, started_at)` plus one result per case. Every element of that tuple can silently move a score, so all of it is recorded and any change to it is reported as a confounder.

The dataset is `datasets/golden_v1.json`: 94 emails from a real inbox, redacted and hand-verified, each with a stable append-only id, the correct category, the key facts a good summary must convey, and a note on why it earns its place. Real mail is messier than anything synthetic. Examples include a subject reading "Thank you for applying" over a rejection body, a broken template merge splicing a job title into an unrelated sentence, an advert with "Interview" in the subject. Redaction preserves typos, casing and template artefacts, because those are the signal. The class mix is deliberate and does not track inbox frequencies, so the headline score measures performance against a balanced suite.

## Why it is built this way

**Prompt versions are hermetic.** Each YAML declares its own category list, and the response schema is built from that list rather than from the `Category` type in code. This came from a real bug: adding a sixth category to the type silently changed what `v1.yaml` returned, despite that file never being edited. For a tool that answers "did this change break anything", a code change surfacing as a prompt regression is fatal.

**Classification runs at temperature 0.** The least glamorous decision here and the one that mattered most. The judge was pinned at 0 from the start; the classifier was not, so it ran at the provider default near 1.0. Two runs of the same prompt on the same model then disagreed with each other, and the diff reported those disagreements as regressions. A harness that manufactures regressions cannot be used to detect them. Greedy decoding removes the sampling variance. It does not make hosted inference deterministic, which is why the noise floor above is measured rather than assumed.

**A run is identified by its backend as well as its prompt version.** Each run records its provider and model, and the search for "the previous run" filters on both. Scoping baselines to the prompt version alone would let a local run diff against a hosted one and report the backend swap as a prompt regression, which is the exact failure this tool exists to catch. The same scoping turns a constraint into a feature: one suite runs against three backends, so the question of whether the free local model is good enough becomes something the harness answers directly.

**The judge runs on a fixed backend.** `--judge-provider` defaults to Groq and deliberately does not follow `--provider`. Letting it inherit meant gpt-oss-20b wrote summaries and then graded them, and a Gemini run and a Groq run reported 4.60 and 3.60 with no way to tell summary quality from grader strictness. Groq specifically, because it is the only backend that behaves identically on a laptop and on a GPU-less CI runner.

**Failures are separated into three kinds.** A model returning a category the prompt forbids counts as wrong, because that is real signal about the prompt. A case whose expected category postdates the prompt version is marked out of contract and excluded from the denominator, which keeps v1 from being punished for the eight `interview_invite` cases it cannot emit. A case still failing after five retries and ninety seconds of backoff indicates quota exhaustion, so the run is abandoned and nothing is written. A half finished run in the history is worse than no run at all.

**Aggregates are computed on read.** Accuracy, mean score and token totals are derived from the stored cases each time they are accessed. Excluded cases are therefore structurally absent from every calculation, so no code path has to remember to filter them.

## Limits

- **Latency is only trustworthy when nothing throttled.** Retries fall inside the measured window, so check the `errors` count before believing a latency figure.
- **The judge scale is coarse.** Scores cluster at 5 and 3, so small movements in the mean are not meaningful. Summary score is reported but never gates the verdict.
- **`misc` is heterogeneous**, covering non-job mail, product updates, staffing marketing and relationship mail, so a `misc` regression does not say which kind broke.
- **Scores do not transfer between backends.** The diff will say so when you compare across one.
- **Groq's ceiling is tokens per minute.** The daily request allowance is generous, but a judge call runs to roughly 900 tokens, so the token rate is what throttles. gpt-oss also returns an empty generation occasionally, when reasoning consumes its output budget.
- **Rate limiting is per-process.** Two concurrent runs against one account throttle each other. Run them sequentially.

## Status

Phases 1 to 3 are complete and validated end to end against all three backends on the full dataset. Phase 4 (HTML diff report, Slack alerts, rolling-average drift detection) and Phase 5 (GitHub Action on PRs touching `prompts/`, Docker) are designed but not built. CI will use a hosted backend, since GitHub runners have no GPU.

`ruff check .`, `mypy` and `pytest` all pass: 115 tests, no API calls, 80% coverage floor.
