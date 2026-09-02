from html import escape
from pathlib import Path

from promptwatch.chart import headline, render_trend, subtitle, tiles
from promptwatch.dataset import GoldenCase, GoldenDataset
from promptwatch.diff import MIN_SCORED_RATIO, CaseFlip, RunDiff, format_score
from promptwatch.drift import DriftReport
from promptwatch.results import CaseResult, RunResult

_VERDICT_BADGE = {
    "pass": "badge-success",
    "warn": "badge-warning",
    "critical": "badge-error",
    "no_data": "badge-neutral",
    "insufficient_history": "badge-neutral",
}

_HEAD = """<!DOCTYPE html>
<html lang="en" data-theme="business">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/daisyui@5.5.19/daisyui.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/daisyui@5.5.19/themes.css">
<script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4.2.4/dist/index.global.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  :where(.grid, .flex) > * {{ min-width: 0; }}
  :where(p, h1, h2, h3, li, td, th, .badge) {{ overflow-wrap: anywhere; }}
  .stat-value {{ font-variant-numeric: proportional-nums; }}
  .pw-mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  .pw-panel {{ background: var(--color-base-100); border-radius: .75rem;
               padding: 1.25rem; }}

  .pw-chart {{ width: 100%; height: auto; display: block;
               color: var(--color-base-content); opacity: .95; }}
  .pw-grid {{ stroke: currentColor; stroke-width: 1; opacity: .22; }}
  .pw-axis {{ stroke: currentColor; stroke-width: 1; opacity: .45; }}
  .pw-line {{ fill: none; stroke: var(--color-primary); stroke-width: 2;
              stroke-linejoin: round; stroke-linecap: round; }}
  .pw-dot {{ fill: var(--color-primary); stroke: var(--color-base-100);
             stroke-width: 2; }}
  .pw-dot-recent {{ r: 5.5; }}
  .pw-hit {{ fill: transparent; }}
  .pw-window {{ fill: currentColor; opacity: .10; }}
  .pw-tone-warn {{ fill: var(--color-warning); opacity: .20; }}
  .pw-tone-critical {{ fill: var(--color-error); opacity: .20; }}
  .pw-tolerance {{ fill: var(--color-success); opacity: .16; }}
  .pw-baseline {{ stroke: currentColor; stroke-width: 1.5; opacity: .7; }}
  .pw-threshold {{ stroke: var(--color-warning); stroke-width: 1;
                   stroke-dasharray: 4 3; opacity: 1; }}
  .pw-threshold-critical {{ stroke: var(--color-error); }}
  .pw-key, .pw-tick {{ fill: currentColor; font-size: 10.5px; opacity: .9; }}
  .pw-axis-title {{ fill: currentColor; font-size: 11px; opacity: .8; }}
</style>
</head>
<body class="bg-base-200 text-base-content">
<div class="max-w-5xl mx-auto px-5 py-10 space-y-8">
"""

_FOOT = """</div>
</body>
</html>
"""


def _esc(value: object) -> str:
    return escape(str(value), quote=True)


def _badge(verdict: str) -> str:
    css = _VERDICT_BADGE.get(verdict, "badge-neutral")
    return f'<span class="badge {css}">{_esc(verdict.upper())}</span>'


def _judged_by(run: RunResult) -> str:
    if run.judge_provider == "none":
        return "not judged"
    return (
        f"judged by {_esc(run.judge_provider)}/{_esc(run.judge_model)}"
    )


def _header(diff: RunDiff, base: RunResult, head: RunResult) -> str:
    return (
        '<header class="space-y-2">'
        '<div class="flex flex-wrap items-center gap-2">'
        f"{_badge(diff.verdict)}"
        f'<span class="badge badge-soft">prompt {_esc(head.prompt_version)}</span>'
        f'<span class="badge badge-soft">{_esc(head.provider)}/'
        f"{_esc(head.model)}</span>"
        "</div>"
        '<h1 class="text-2xl font-bold">'
        f"category accuracy {base.category_accuracy:.2%} &rarr; "
        f"{head.category_accuracy:.2%}"
        f" <span class='text-base-content/60 text-lg'>"
        f"({diff.accuracy_delta:+.1%})</span></h1>"
        f'<p class="text-sm text-base-content/70 pw-mono">'
        f"{_esc(head.run_id)} vs {_esc(base.run_id)}</p>"
        f'<p class="text-sm text-base-content/60">{_judged_by(head)}'
        f" &nbsp;·&nbsp; {_esc(head.started_at)}</p>"
        "</header>"
    )


def _alerts(diff: RunDiff) -> str:
    parts = []
    if diff.confounders:
        listed = "; ".join(_esc(item) for item in diff.confounders)
        parts.append(
            '<div class="alert alert-warning alert-soft"><span>'
            f"<strong>NOT A CLEAN PROMPT COMPARISON:</strong> {listed}"
            "</span></div>"
        )
    if diff.verdict == "no_data":
        parts.append(
            '<div class="alert alert-error alert-soft"><span>'
            "<strong>VERDICT WITHHELD:</strong> too few cases scored to compare "
            f"(minimum {MIN_SCORED_RATIO:.0%})</span></div>"
        )
    return "".join(parts)


def _stat(label: str, value: str, note: str = "") -> str:
    desc = f'<div class="stat-desc">{note}</div>' if note else ""
    return (
        f'<div class="stat"><div class="stat-title">{_esc(label)}</div>'
        f'<div class="stat-value text-2xl">{value}</div>{desc}</div>'
    )


def _scorecard(diff: RunDiff) -> str:
    delta = diff.summary_delta
    summary_note = "reported only" if delta is None else f"{delta:+.2f}"
    return (
        '<div class="stats stats-vertical sm:stats-horizontal bg-base-100 w-full">'
        + _stat(
            "category accuracy",
            f"{diff.head_accuracy:.2%}",
            f"{diff.accuracy_delta:+.1%} vs base",
        )
        + _stat(
            "summary mean",
            _esc(format_score(diff.head_summary_score)),
            summary_note,
        )
        + _stat(
            "regressions",
            str(len(diff.regressions)),
            f"{len(diff.improvements)} improvements",
        )
        + _stat(
            "cases scored",
            f"{diff.head_scored_ratio:.0%}",
            f"base {diff.base_scored_ratio:.0%}",
        )
        + _stat(
            "errors",
            str(diff.head_errors),
            f"{diff.head_out_of_contract} out of contract",
        )
        + "</div>"
    )


def _summary_cell(label: str, case: CaseResult | None, tone: str) -> str:
    if case is None:
        return (
            f'<div><div class="text-xs uppercase tracking-wide '
            f'text-base-content/50">{_esc(label)}</div>'
            '<p class="text-sm italic text-base-content/50">not in this run</p></div>'
        )
    text = case.summary or "no summary recorded"
    return (
        f'<div><div class="text-xs uppercase tracking-wide '
        f'text-base-content/50">{_esc(label)}</div>'
        f'<span class="badge {tone} badge-sm my-1">'
        f"{_esc(case.actual_category)}</span>"
        f'<p class="text-sm text-base-content/80">{_esc(text)}</p></div>'
    )


def _case_context(case: GoldenCase | None) -> str:
    if case is None:
        return ""
    facts = "".join(
        f'<li>{_esc(fact)}</li>' for fact in case.must_mention
    )
    return (
        f'<p class="text-sm font-medium">{_esc(case.subject)}</p>'
        f'<div class="text-xs text-base-content/60">'
        f'<span class="badge badge-ghost badge-xs">{_esc(case.difficulty)}</span>'
        f" must mention<ul class=\"list-disc list-inside\">{facts}</ul></div>"
    )


def _flip_card(
    flip: CaseFlip,
    base_cases: dict[str, CaseResult],
    head_cases: dict[str, CaseResult],
    golden: dict[str, GoldenCase],
    regressed: bool,
) -> str:
    before_tone = "badge-success" if regressed else "badge-error"
    after_tone = "badge-error" if regressed else "badge-success"
    return (
        '<div class="card card-border bg-base-100"><div class="card-body gap-3">'
        '<div class="flex flex-wrap items-center gap-2">'
        f'<span class="pw-mono font-semibold">{_esc(flip.case_id)}</span>'
        f'<span class="badge badge-soft badge-sm">expected '
        f"{_esc(flip.expected_category)}</span></div>"
        + _case_context(golden.get(flip.case_id))
        + '<div class="grid gap-4 sm:grid-cols-2">'
        + _summary_cell("before", base_cases.get(flip.case_id), before_tone)
        + _summary_cell("after", head_cases.get(flip.case_id), after_tone)
        + "</div></div></div>"
    )


def _flips(
    diff: RunDiff,
    base: RunResult,
    head: RunResult,
    dataset: GoldenDataset | None,
) -> str:
    base_cases, head_cases = base.by_id(), head.by_id()
    golden = dataset.by_id() if dataset else {}

    def cards(flips: list[CaseFlip], regressed: bool) -> str:
        return "".join(
            _flip_card(flip, base_cases, head_cases, golden, regressed)
            for flip in flips
        )

    parts = [
        '<section class="space-y-4">'
        f'<h2 class="text-xl font-semibold">Regressions '
        f"({len(diff.regressions)})</h2>"
    ]
    if diff.regressions:
        parts.append(f'<div class="space-y-3">{cards(diff.regressions, True)}</div>')
    else:
        parts.append(
            '<p class="text-sm text-base-content/60">No case got worse.</p>'
        )

    if diff.improvements:
        parts.append(
            '<details class="collapse collapse-arrow bg-base-100">'
            '<summary class="collapse-title font-medium">'
            f"Improvements ({len(diff.improvements)})</summary>"
            f'<div class="collapse-content space-y-3">'
            f"{cards(diff.improvements, False)}</div></details>"
        )
    parts.append("</section>")
    return "".join(parts)


def _point_table(drift: DriftReport) -> str:
    rows = "".join(
        f"<tr><td>{n + 1}</td>"
        f'<td class="pw-mono text-xs">{_esc(point.run_id)}</td>'
        f"<td>{point.accuracy:.2%}</td>"
        f"<td>{point.scored_ratio:.0%}</td>"
        f"<td>{point.case_count}</td></tr>"
        for n, point in enumerate(drift.points)
    )
    return (
        '<div class="overflow-x-auto"><table class="table table-sm">'
        "<thead><tr><th>run</th><th>id</th><th>accuracy</th>"
        "<th>scored</th><th>cases</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


def _drift_section(drift: DriftReport | None) -> str:
    if drift is None:
        return ""

    figures = "".join(
        _stat(label, _esc(value)) for label, value in tiles(drift)
    )
    chart = render_trend(drift)
    body = (
        f'<div class="pw-panel">{chart}</div>'
        if chart
        else '<p class="text-sm text-base-content/60">'
        "Fewer than four runs, so there is nothing worth plotting yet.</p>"
    )
    return (
        '<section class="space-y-4">'
        '<div class="flex flex-wrap items-center gap-2">'
        f'<h2 class="text-xl font-semibold">{_esc(headline(drift))}</h2>'
        f"{_badge(drift.verdict)}</div>"
        f'<p class="text-sm text-base-content/60">{_esc(subtitle(drift))}</p>'
        '<div class="stats stats-vertical sm:stats-horizontal bg-base-100 w-full">'
        f"{figures}</div>"
        f"{body}"
        f"{_point_table(drift)}"
        "</section>"
    )


def _raw(diff: RunDiff) -> str:
    return (
        '<details class="collapse collapse-arrow bg-base-100">'
        '<summary class="collapse-title font-medium">Raw diff (JSON)</summary>'
        '<div class="collapse-content">'
        f'<pre class="pw-mono text-xs overflow-x-auto">'
        f"{_esc(diff.model_dump_json(indent=2))}</pre></div></details>"
    )


def render_report(
    diff: RunDiff,
    base: RunResult,
    head: RunResult,
    drift: DriftReport | None = None,
    dataset: GoldenDataset | None = None,
) -> str:
    """Build a standalone HTML diff report.

    Both runs are needed, not just the diff: `CaseFlip` carries categories only,
    and the side-by-side view reads the generated summary text off the runs.
    Passing `dataset` adds the email subject and key facts to each flipped case.

    Returns:
        A complete HTML document.

    Raises:
        ValueError: if `drift` describes a different series than `head`, which
            would put an unrelated trend beside the diff.
    """
    if drift is not None and (
        drift.prompt_version,
        drift.provider,
        drift.model,
    ) != (head.prompt_version, head.provider, head.model):
        raise ValueError(
            f"drift is for {drift.prompt_version} on {drift.provider}/"
            f"{drift.model}, but the run is {head.prompt_version} on "
            f"{head.provider}/{head.model}"
        )

    title = f"PromptWatch {head.run_id} vs {base.run_id}"
    return (
        _HEAD.format(title=_esc(title))
        + _header(diff, base, head)
        + _alerts(diff)
        + _scorecard(diff)
        + _flips(diff, base, head, dataset)
        + _drift_section(drift)
        + _raw(diff)
        + _FOOT
    )


def write_report(path: Path, html: str) -> None:
    """Write a report, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
