# CLAUDE.md

Project memory for PromptWatch. Read this before making changes.

## What this is

PromptWatch is a CI/CD-style regression testing system for LLM prompts. It runs a prompt version against a hand-labeled golden dataset, scores the outputs across multiple dimensions, diffs against the previous run, and alerts on regressions before they ship. The feature under test is a job-search inbox email classifier.

The point of the project: most teams ship prompt/model changes blind. This proves the "what happens after deployment" mindset.

## Current status

**Phases 1 to 3 complete**, validated end to end against all three backends on the full 94-case dataset. No alerting or CI exists yet. Do not scaffold ahead. Next: Phase 4.

`v3.yaml` is the current prompt. It scores 97.87% on Gemini, 90.43% on Groq and 86.17% on Ollama, up from 90.43 / 79.79 / 78.72 for v2. Use v3 in examples and new runs unless a comparison specifically needs an older version.

Providers were made pluggable because a full judged run is 188 requests against Gemini's 500/day cap, roughly 2.6 runs, which is not enough to iterate on a prompt. Ollama is the default so iteration is unmetered; Gemini and Groq stay for hosted runs and CI.

Two behaviours worth knowing before changing the runner. Classification runs at temperature 0, because at the provider default two identical runs disagreed with each other and the diff reported that as a regression. And a case reaching `error` status has already survived five retries and up to ninety seconds of backoff, so the whole run is abandoned and nothing is written rather than recording a partial run.

## Non-negotiables

- **Providers are pluggable, and never OpenAI or Anthropic.** Three backends implement the `Provider` protocol in `promptwatch/provider.py`: `ollama` (local, the default, no quota), `gemini` (`google-genai`, `GEMINI_API_KEY`), and `groq` (`GROQ_API_KEY`, raw HTTP). Groq's endpoint is OpenAI-shaped but the `openai` SDK is not a dependency and must not become one; hosted providers are reached with `httpx` directly. Never introduce `OPENAI_API_KEY` or an Anthropic SDK. Only `promptwatch/gemini.py` may import `google.genai` — nothing else in the package knows which backend it is talking to.
- **The judge is pinned, never inherited.** `--judge-provider` defaults to `groq` and is deliberately independent of `--provider`. A grader that changes with the thing it grades makes summary scores incomparable, and lets a model mark its own homework. Groq specifically because it is the only backend that behaves identically on a laptop and on a GPU-less CI runner — pinning to Ollama makes CI unable to reproduce a score, pinning to Gemini puts every "free" local run back under the 500/day cap.
- **The noise floor is measured, and thresholds sit above it.** Four runs of an identical configuration disagreed by 2 to 4 cases out of 94: hosted inference is not reproducible even at temperature 0, and 89 of 94 cases were stable across all four. `DEFAULT_WARN` is therefore 0.05, not the 0.03 it started at. Do not lower it without re-measuring the floor first. An alerting threshold inside the measurement noise fires on runs where nothing changed, which is the same failure the tool exists to catch.
- **Classification runs at temperature 0.** `classify_email` passes it explicitly and every backend forwards it. The judge already ran at 0; the classifier did not, and that alone made runs disagree with themselves. Greedy decoding removes sampling variance and does not make hosted inference deterministic, which is why the floor above still exists.
- **A run is identified by prompt version *and* backend.** `RunResult` records `provider`, `model`, `judge_provider` and `judge_model`; `latest_run()` filters on the classifier pair, and `diff.py` flags a change to either pair as a confounder. Comparing a local run against a hosted one measures the swap, not the prompt.
- **Types are Pydantic.** Data models in `promptwatch/models.py`; config models in `promptwatch/config.py`. Any structured data crossing a function boundary is a typed model, not a loose dict.
- **Prompts are versioned YAML** in `prompts/` (e.g. `v1.yaml`). A prompt change is a new version file, not an in-place edit. These files are the "code" CI runs against.
- **Golden dataset is hand-labeled, never LLM-generated.** Human-verified ground truth is the whole point. Each case: stable ID, input, expected output, difficulty tag, notes on why it matters.
- **Redaction:** recruiters and other third parties get consistent fake names; Prajwal keeps his first name only (surname dropped, never replaced). Requisition IDs zeroed, email addresses to `example.com`, tracking links to `example.com`. Typos, casing, and template artifacts are preserved — they are the signal.
- **One file at a time.** Build and verify a single file before moving on.

## Coding style

- **Keep it minimal; think like a senior dev.** Smallest change that does the job. No speculative abstraction, no defensive scaffolding for problems that don't exist yet, no restating the obvious in comments, docstrings, or commit messages. This applies to prose too — commit messages and explanations should be short and carry only what isn't already evident from the diff.
- Python 3.11+, full type hints on every function signature. Hints are load-bearing, not decoration — `mypy --strict` must pass on `src/` with no `type: ignore` unless a stdlib stub is genuinely wrong (there is one, in `tools/fetch_emails.py`).
- Prefer stdlib and deps already in `pyproject.toml`. Ask before adding a new dependency.
- **Name the concept, not the shape.** A recurring structural type gets an alias in the module that owns it — `JsonSchema` in `provider.py`, not `dict[str, Any]` repeated across six files. Bare `dict` and `list` in a signature are a lint failure under strict mode anyway.
- **Be modular.** Every function does one thing and its seams are obvious — build, call, validate are separate steps, not one long body. Modular means clean boundaries, not file count: extract a function when a block has a name, split a file when it has two reasons to change. Don't scatter a small module across files for its own sake.
- **Never write comments unless asked.** No inline `#` comments, no module docstrings, no class docstrings. The code and its names carry the meaning; if a line needs explaining, that's a signal to rename or restructure it, not to annotate it. When a *why* is genuinely non-obvious (a workaround, a subtle constraint), it belongs in the commit message or CLAUDE.md, not the source.
- Small, single-purpose functions. Docstring on public functions only — what it does, what it returns, what it raises.
- Config from env via `python-dotenv`. Never hardcode keys or model names; make the model a parameter with a sensible default.
- Fail loud and early: validate model output against the `Category` contract and raise a clear error on anything off-contract. No silent fallbacks.
- **Depend on a protocol, not a vendor.** Anything reached over the network sits behind a `Protocol` in this package, with the vendor SDK confined to one module. `runner.py` must never import `google.genai`. The test for this: adding a fourth backend is a new file, not a refactor.
- **Every seam gets a fake.** A protocol that can't be faked in `conftest.py` is the wrong shape. `FakeProvider` is why `runner.py` is testable offline; new seams follow it.
- Match the style of existing files rather than introducing new patterns.

## Tooling gates

All three must pass before a commit. There is no CI yet, so they are run by hand.

```bash
ruff check .
mypy
pytest
```

- `mypy` reads its config from `pyproject.toml`: strict over `src/` and `tools/`, tests excluded. Tests are excluded deliberately — several deliberately construct invalid input to assert Pydantic rejects it, so type-checking them reports false positives that can only be silenced by weakening the test.
- `pytest` fails under 80% coverage, configured in `addopts`. Current is ~83%. The uncovered remainder is the three provider transports, which is correct: those are live HTTP and SDK calls. Raise the floor when it drifts up; do not lower it to make a commit pass.
- Coverage is a floor, not a target. Do not write tests to move the number.

## Project layout

`src/` layout. Everything importable lives in `src/promptwatch/`; data, prompts and dev scripts sit at the repo root. Install with `pip install -e ".[dev]"` — the package is not importable otherwise, which is the point: tests cannot accidentally pass against loose files in the working directory.

```
pyproject.toml    — deps, ruff, mypy, pytest and coverage config, all of it
src/promptwatch/
  __main__.py     — python -m promptwatch
  cli.py          — CLI: run, diff, runs
  classifier.py   — classify_email(): Provider + PromptConfig + EmailInput -> ClassificationResult
  models.py       — EmailInput, ClassificationResult, Category literal type
  config.py       — FewShotExample, PromptConfig (with load() for YAML)
  dataset.py      — GoldenCase, GoldenDataset, Tag, Difficulty, check_balance()
  provider.py     — Turn, Completion, Provider protocol, TransientError, get_provider()
  gemini.py       — Gemini backend (the only google-genai importer)
  ollama.py       — local backend over /api/chat, default
  groq.py         — Groq backend, strict json_schema (gpt-oss models only)
  judge.py        — JudgeConfig, score_summary()
  runner.py       — concurrent execution, pacing, failure handling
  results.py      — CaseResult, RunResult, SQLite storage
  diff.py         — run-vs-run diffing and the verdict
tests/
  conftest.py     — FakeProvider, fixtures, run/case builders
prompts/v1.yaml   — versioned system prompt + few-shot examples (v3.yaml is current)
datasets/golden_v1.json — the labelled corpus
tools/
  fetch_emails.py — Gmail IMAP pull into a labelling worksheet
  redact.py       — shared clean/redact/cap helpers for assembling cases
raw_emails/       — unredacted worksheets (gitignored, never commit)
.env              — GEMINI_API_KEY, GROQ_API_KEY, OLLAMA_HOST, GMAIL_ADDRESS, GMAIL_APP_PASSWORD (gitignored)
```

Imports are always package-qualified: `from promptwatch.models import ...`, never bare `from models import ...`. Run commands from the repo root — `prompts/` and `datasets/` are resolved relative to the working directory, not the package.

Where a new module goes: inside `src/promptwatch/` if anything imports it, `tools/` if it is a script a human runs by hand. `tools/` is not part of the shipped package and is held to a looser mypy bar.

## Categories (the contract)

`application_ack`, `interview_invite`, `rejection`, `job_alert`, `newsletter`, `misc`. Changing this set is a breaking change.

`interview_invite` was added in v2 after a live run showed interview invitations — the highest-value email in the inbox — landing in `misc` under the five-category set.

**Prompt versions are hermetic.** Each YAML declares its own `categories` list, and `classifier.py` builds both the response schema and the validation check from `prompt_config.categories` — never from the `Category` type. Widening `Category` in code therefore cannot change what an older version emits; v1 still returns `misc` for interview invites, as it did the day it was written. Keep it that way: a new category means a new prompt version, never an edit to an existing one.

`Category` in `models.py` remains the union of every value valid across all versions — it types `ClassificationResult` and constrains what a YAML may declare, but it is never handed to the API.

## Golden dataset conventions

`datasets/golden_v1.json` — 94 cases, all sourced from Prajwal's real inbox. Filename carries the version; a breaking change (new category set, mass relabel) becomes `golden_v2.json`, never an in-place rewrite. Adding a case does not bump the version.

- **IDs are append-only.** `gc-NNN`, never renumber, never reuse, never delete — leave the gap. Renumbering silently re-maps every historical run-vs-run diff to the wrong case.
- **`categories` is declared per dataset**, same hermeticity rule as `PromptConfig`. A case labelled `interview_invite` is not a valid test against v1, whose schema cannot emit it.
- **Two-tier validation.** Schema validity (unique IDs, declared categories, no unredacted addresses) is a `model_validator` and always fatal. Fitness for eval (`check_balance`: floor 6 per category, ceiling 30%) is a separate method — balance is meaningless mid-labelling, so it must not block loading.
- **Bodies capped at 2000 chars** with a `[...truncated]` marker. Newsletters ran to 60kb; uncapped they would dominate per-run token cost.
- **This is a balanced regression suite, not an accuracy estimate.** The class mix is deliberate, so the headline score is not inbox accuracy.

Known weaknesses to revisit in Phase 3: `misc` is heterogeneous (non-job mail, product announcements, staffing-agency marketing, relationship mail with no ask), so a `misc` regression will not say which kind broke. `must_mention`'s minimum of 2 is awkward for near-empty mail (gc-033, gc-094). No tag expresses "misleading subject" despite gc-017, gc-034, and gc-053 all exhibiting it.

## Roadmap (don't build ahead)

1. ~~**Feature under test**~~ — classifier fn + Pydantic contract + versioned prompts. Done.
2. ~~**Golden dataset**~~ — 94 hand-labeled cases, balanced, versioned JSON. Done.
3. ~~**Eval engine**~~ — test runner (async batching), multi-dim scoring (exact category match, LLM-as-judge summary 1–5, latency, tokens), run-vs-run diffing, warn >5% / critical >8% thresholds (configurable, and set above the measured noise floor). Done.
4. **Alerting + reporting** — HTML diff report (scorecard, side-by-side regressions, trend chart), Slack webhook alerts, rolling-average slow-drift detection.
5. **CI/CD** — GitHub Action on PRs touching `prompts/`; runs eval, comments status, blocks merge on critical regressions. Dockerize. README as onboarding docs, not a tutorial. **CI must use a hosted provider** — GitHub runners have no GPU, so `--provider ollama` is local-only. The Docker image must not bundle model weights; point it at a host Ollama via `OLLAMA_HOST`.
6. **Portfolio polish** — Loom walkthrough, short writeup of the problem/approach/one proud design decision.

## Tooling / infra choices

- Storage: SQLite + JSON files (git-friendly, zero infra).
- Scheduling: GitHub Actions. Alerting: Slack webhooks. Report: HTML (or Streamlit). Container: Docker.

## Workflow notes

- **Never run git commands.** Prajwal runs every git command manually. This is enforced by a `deny` rule in `.claude/settings.local.json` (`Bash(git *)`, `PowerShell(git *)`), including read-only ones like `status` and `diff`. When a commit is warranted, write out the commit message and the exact commands, and let them run it.
- **Prajwal runs and tests the app.** Don't execute the application or its scripts to verify behaviour — write the code, then hand over the exact commands to run and say what output to expect. Applies to eval runs, smoke tests, and anything that calls the Gemini API.
- **No external commit gate.** `no-mistakes` was dropped — it broke on a CLI incompatibility and Phase 5 builds project-specific CI (GitHub Actions) instead. Run `ruff check .`, `mypy` and `pytest` before committing; use `/code-review` on meaningful diffs.
- Keep diffs focused and reviewable. Don't bundle unrelated changes.
