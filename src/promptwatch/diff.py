from typing import Literal

from pydantic import BaseModel

from promptwatch.models import Category
from promptwatch.results import RunResult

Verdict = Literal["pass", "warn", "critical", "no_data"]

DEFAULT_WARN = 0.03
DEFAULT_CRITICAL = 0.08
MIN_SCORED_RATIO = 0.8


class CaseFlip(BaseModel):
    case_id: str
    expected_category: Category
    before: Category | None
    after: Category | None


class RunDiff(BaseModel):
    base_run_id: str
    head_run_id: str
    base_accuracy: float
    head_accuracy: float
    base_summary_score: float
    head_summary_score: float
    base_scored_ratio: float
    head_scored_ratio: float
    regressions: list[CaseFlip]
    improvements: list[CaseFlip]
    base_out_of_contract: int
    head_out_of_contract: int
    base_errors: int
    head_errors: int
    only_in_base: list[str]
    only_in_head: list[str]
    confounders: list[str]
    verdict: Verdict

    @property
    def accuracy_delta(self) -> float:
        return self.head_accuracy - self.base_accuracy

    @property
    def summary_delta(self) -> float:
        return self.head_summary_score - self.base_summary_score


def _confounders(base: RunResult, head: RunResult) -> list[str]:
    changed = []
    for label, before, after in (
        ("provider", base.provider, head.provider),
        ("model", base.model, head.model),
        ("dataset", base.dataset_version, head.dataset_version),
        ("judge", base.judge_version, head.judge_version),
        ("judge backend", base.judge_provider, head.judge_provider),
        ("judge model", base.judge_model, head.judge_model),
    ):
        if before != after:
            changed.append(f"{label} {before} -> {after}")
    return changed


def _verdict(delta: float, warn: float, critical: float) -> Verdict:
    drop = round(-delta, 6)
    if drop > critical:
        return "critical"
    if drop > warn:
        return "warn"
    return "pass"


def diff_runs(
    base: RunResult,
    head: RunResult,
    warn: float = DEFAULT_WARN,
    critical: float = DEFAULT_CRITICAL,
) -> RunDiff:
    """Compare two runs and decide whether the change is acceptable.

    Only category accuracy drives the verdict. The summary score is reported
    but never gates, because a holistic judge score drifts between runs and
    gating on it would raise false alarms.

    Returns:
        A RunDiff naming every case that flipped in either direction.
    """
    base_cases = base.by_id()
    head_cases = head.by_id()
    shared = sorted(set(base_cases) & set(head_cases))

    regressions: list[CaseFlip] = []
    improvements: list[CaseFlip] = []
    for case_id in shared:
        before, after = base_cases[case_id], head_cases[case_id]
        if before.category_match is None or after.category_match is None:
            continue
        if before.category_match == after.category_match:
            continue
        flip = CaseFlip(
            case_id=case_id,
            expected_category=after.expected_category,
            before=before.actual_category,
            after=after.actual_category,
        )
        if before.category_match:
            regressions.append(flip)
        else:
            improvements.append(flip)

    delta = head.category_accuracy - base.category_accuracy
    if min(base.scored_ratio, head.scored_ratio) < MIN_SCORED_RATIO:
        verdict: Verdict = "no_data"
    else:
        verdict = _verdict(delta, warn, critical)

    return RunDiff(
        base_run_id=base.run_id,
        head_run_id=head.run_id,
        base_accuracy=base.category_accuracy,
        head_accuracy=head.category_accuracy,
        base_summary_score=base.mean_summary_score,
        head_summary_score=head.mean_summary_score,
        base_scored_ratio=base.scored_ratio,
        head_scored_ratio=head.scored_ratio,
        regressions=regressions,
        improvements=improvements,
        base_out_of_contract=base.count("out_of_contract"),
        head_out_of_contract=head.count("out_of_contract"),
        base_errors=base.count("error"),
        head_errors=head.count("error"),
        only_in_base=sorted(set(base_cases) - set(head_cases)),
        only_in_head=sorted(set(head_cases) - set(base_cases)),
        confounders=_confounders(base, head),
        verdict=verdict,
    )


def format_diff(diff: RunDiff) -> str:
    """Render a RunDiff as the terminal report."""
    lines = [f"{diff.head_run_id}  vs  {diff.base_run_id}", ""]
    if diff.confounders:
        lines.append(
            "NOT A CLEAN PROMPT COMPARISON: " + "; ".join(diff.confounders)
        )
        lines.append("")
    lines += [
        f"category accuracy   {diff.base_accuracy:.2f} -> {diff.head_accuracy:.2f}"
        f"   ({diff.accuracy_delta:+.1%})   {diff.verdict.upper()}",
        f"summary mean        {diff.base_summary_score:.2f} -> "
        f"{diff.head_summary_score:.2f}   ({diff.summary_delta:+.2f})   reported only",
        f"cases scored        base {diff.base_scored_ratio:.0%}"
        f"   head {diff.head_scored_ratio:.0%}",
        "",
    ]
    if diff.verdict == "no_data":
        lines.insert(
            2,
            f"VERDICT WITHHELD: too few cases scored to compare "
            f"(minimum {MIN_SCORED_RATIO:.0%})\n",
        )

    lines.append(f"regressions ({len(diff.regressions)})")
    for flip in diff.regressions:
        lines.append(f"  {flip.case_id}  {flip.before} -> {flip.after}")

    lines.append(f"improvements ({len(diff.improvements)})")
    for flip in diff.improvements:
        lines.append(f"  {flip.case_id}  {flip.before} -> {flip.after}")

    lines += [
        "",
        f"out of contract     base: {diff.base_out_of_contract}"
        f"   head: {diff.head_out_of_contract}",
        f"errors              base: {diff.base_errors}   head: {diff.head_errors}",
    ]
    if diff.only_in_base or diff.only_in_head:
        lines.append(
            f"only in one run     base: {len(diff.only_in_base)}"
            f"   head: {len(diff.only_in_head)}"
        )
    return "\n".join(lines)
