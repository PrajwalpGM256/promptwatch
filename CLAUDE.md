# CLAUDE.md

Project memory for PromptWatch. Read this before making changes.

## What this is

PromptWatch is a CI/CD-style regression testing system for LLM prompts. It runs a prompt version against a hand-labeled golden dataset, scores the outputs across multiple dimensions, diffs against the previous run, and alerts on regressions before they ship. The feature under test is a job-search inbox email classifier.

The point of the project: most teams ship prompt/model changes blind. This proves the "what happens after deployment" mindset.

## Current status

**Phases 1 and 2 complete.** `classify_email()` is verified end-to-end against live Gemini, and `datasets/golden_v1.json` holds 94 labelled cases that pass `check_balance()`. No eval engine, alerting, or CI exists yet. Do not scaffold ahead. Next: Phase 3, the eval engine.

## Non-negotiables

- **LLM provider is Gemini, not OpenAI.** The spec says OpenAI; we deliberately use Gemini instead. Always use the `google-genai` SDK and the `GEMINI_API_KEY` env var. Never introduce `openai`, `OPENAI_API_KEY`, or Anthropic SDKs.
- **Types are Pydantic.** Data models in `promptwatch/models.py`; config models in `promptwatch/config.py`. Any structured data crossing a function boundary is a typed model, not a loose dict.
- **Prompts are versioned YAML** in `prompts/` (e.g. `v1.yaml`). A prompt change is a new version file, not an in-place edit. These files are the "code" CI runs against.
- **Golden dataset is hand-labeled, never LLM-generated.** Human-verified ground truth is the whole point. Each case: stable ID, input, expected output, difficulty tag, notes on why it matters.
- **Redaction:** recruiters and other third parties get consistent fake names; Prajwal keeps his first name only (surname dropped, never replaced). Requisition IDs zeroed, email addresses to `example.com`, tracking links to `example.com`. Typos, casing, and template artifacts are preserved — they are the signal.
- **One file at a time.** Build and verify a single file before moving on.

## Coding style

- **Keep it minimal; think like a senior dev.** Smallest change that does the job. No speculative abstraction, no defensive scaffolding for problems that don't exist yet, no restating the obvious in comments, docstrings, or commit messages. This applies to prose too — commit messages and explanations should be short and carry only what isn't already evident from the diff.
- Python 3.11+, full type hints on every function signature.
- Prefer stdlib and deps already in `requirements.txt`. Ask before adding a new dependency.
- **Be modular.** Every function does one thing and its seams are obvious — build, call, validate are separate steps, not one long body. Modular means clean boundaries, not file count: extract a function when a block has a name, split a file when it has two reasons to change. Don't scatter a small module across files for its own sake.
- **Never write comments unless asked.** No inline `#` comments, no module docstrings, no class docstrings. The code and its names carry the meaning; if a line needs explaining, that's a signal to rename or restructure it, not to annotate it. When a *why* is genuinely non-obvious (a workaround, a subtle constraint), it belongs in the commit message or CLAUDE.md, not the source.
- Small, single-purpose functions. Docstring on public functions only — what it does, what it returns, what it raises.
- Config from env via `python-dotenv`. Never hardcode keys or model names; make the model a parameter with a sensible default.
- Fail loud and early: validate model output against the `Category` contract and raise a clear error on anything off-contract. No silent fallbacks.
- Match the style of existing files rather than introducing new patterns.

## Project layout

Package code lives under `promptwatch/`; the entry point, prompts, and env sit at the repo root.

```
promptwatch/
  models.py       — EmailInput, ClassificationResult, Category literal type
  config.py       — FewShotExample, PromptConfig (with load() for YAML)
  dataset.py      — GoldenCase, GoldenDataset, Tag, Difficulty, check_balance()
classifier.py     — classify_email(): PromptConfig + EmailInput -> ClassificationResult
prompts/v1.yaml   — versioned system prompt + few-shot examples (v2.yaml is current)
datasets/golden_v1.json — the labelled corpus
tools/
  fetch_emails.py — Gmail IMAP pull into a labelling worksheet
  redact.py       — shared clean/redact/cap helpers for assembling cases
raw_emails/       — unredacted worksheets (gitignored, never commit)
.env              — GEMINI_API_KEY, GMAIL_ADDRESS, GMAIL_APP_PASSWORD (gitignored)
```

Imports are always package-qualified: `from promptwatch.models import ...`, never bare `from models import ...`. Run from the repo root so `promptwatch` resolves. There is no `promptwatch/__init__.py` — it works as an implicit namespace package, which is fine for now but will need a real `__init__.py` if this ever becomes pip-installable.

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
3. **Eval engine** — test runner (async batching), multi-dim scoring (exact category match, LLM-as-judge summary 1–5, latency, tokens), run-vs-run diffing, warn >3% / critical >8% thresholds (configurable). ← current
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
