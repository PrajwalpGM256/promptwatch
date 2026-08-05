# CLAUDE.md

Project memory for PromptWatch. Read this before making changes.

## What this is

PromptWatch is a CI/CD-style regression testing system for LLM prompts. It runs a prompt version against a hand-labeled golden dataset, scores the outputs across multiple dimensions, diffs against the previous run, and alerts on regressions before they ship. The feature under test is a job-search inbox email classifier.

The point of the project: most teams ship prompt/model changes blind. This proves the "what happens after deployment" mindset.

## Current status

**Phase 1 (Define the feature under test).** The classifier and its contract are written; `classify_email()` has been verified up to the API call but not yet against a live Gemini key. Nothing downstream (dataset, eval engine, alerting, CI) exists yet. Do not scaffold ahead into later phases.

## Non-negotiables

- **LLM provider is Gemini, not OpenAI.** The spec says OpenAI; we deliberately use Gemini instead. Always use the `google-genai` SDK and the `GEMINI_API_KEY` env var. Never introduce `openai`, `OPENAI_API_KEY`, or Anthropic SDKs.
- **Types are Pydantic.** Data models in `promptwatch/models.py`; config models in `promptwatch/config.py`. Any structured data crossing a function boundary is a typed model, not a loose dict.
- **Prompts are versioned YAML** in `prompts/` (e.g. `v1.yaml`). A prompt change is a new version file, not an in-place edit. These files are the "code" CI runs against.
- **Golden dataset is hand-labeled, never LLM-generated.** Human-verified ground truth is the whole point. Each case: stable ID, input, expected output, difficulty tag, notes on why it matters.
- **One file at a time.** Build and verify a single file before moving on.

## Coding style

- **Keep it minimal; think like a senior dev.** Smallest change that does the job. No speculative abstraction, no defensive scaffolding for problems that don't exist yet, no restating the obvious in comments, docstrings, or commit messages. This applies to prose too — commit messages and explanations should be short and carry only what isn't already evident from the diff.
- Python 3.11+, full type hints on every function signature.
- Prefer stdlib and deps already in `requirements.txt`. Ask before adding a new dependency.
- Small, single-purpose functions. Short docstring on every public function: what it does, what it returns, what it raises.
- Config from env via `python-dotenv`. Never hardcode keys or model names; make the model a parameter with a sensible default.
- Fail loud and early: validate model output against the `Category` contract and raise a clear error on anything off-contract. No silent fallbacks.
- Match the style of existing files rather than introducing new patterns.

## Project layout

Package code lives under `promptwatch/`; the entry point, prompts, and env sit at the repo root.

```
promptwatch/
  models.py     — EmailInput, ClassificationResult, Category literal type
  config.py     — FewShotExample, PromptConfig (with load() for YAML)
classifier.py   — classify_email(): PromptConfig + EmailInput -> ClassificationResult via Gemini
prompts/v1.yaml — versioned system prompt + few-shot examples
.env            — GEMINI_API_KEY (gitignored)
```

Imports are always package-qualified: `from promptwatch.models import ...`, never bare `from models import ...`. Run from the repo root so `promptwatch` resolves. There is no `promptwatch/__init__.py` — it works as an implicit namespace package, which is fine for now but will need a real `__init__.py` if this ever becomes pip-installable.

## Categories (the contract)

`application_ack`, `rejection`, `job_alert`, `newsletter`, `misc`. Changing this set is a breaking change.

## Roadmap (don't build ahead)

1. **Feature under test** — classifier fn + Pydantic contract + versioned prompts. ← current
2. **Golden dataset** — 50–100 hand-labeled cases incl. edge cases (ambiguous, short, typos, mixed-language, sarcastic); versioned JSON.
3. **Eval engine** — test runner (async batching), multi-dim scoring (exact category match, LLM-as-judge summary 1–5, latency, tokens), run-vs-run diffing, warn >3% / critical >8% thresholds (configurable).
4. **Alerting + reporting** — HTML diff report (scorecard, side-by-side regressions, trend chart), Slack webhook alerts, rolling-average slow-drift detection.
5. **CI/CD** — GitHub Action on PRs touching `prompts/`; runs eval, comments status, blocks merge on critical regressions. Dockerize. README as onboarding docs, not a tutorial.
6. **Portfolio polish** — Loom walkthrough, short writeup of the problem/approach/one proud design decision.

## Tooling / infra choices

- Storage: SQLite + JSON files (git-friendly, zero infra).
- Scheduling: GitHub Actions. Alerting: Slack webhooks. Report: HTML (or Streamlit). Container: Docker.

## Workflow notes

- **Never run git commands.** Prajwal runs every git command manually. This is enforced by a `deny` rule in `.claude/settings.local.json` (`Bash(git *)`, `PowerShell(git *)`), including read-only ones like `status` and `diff`. When a commit is warranted, write out the commit message and the exact commands, and let them run it.
- Commits are gated through `no-mistakes` before reaching the GitHub remote.
- Keep diffs focused and reviewable. Don't bundle unrelated changes.
