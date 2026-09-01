from statistics import mean
from typing import Literal

from pydantic import BaseModel

from promptwatch.diff import MIN_SCORED_RATIO
from promptwatch.results import RunResult

DriftVerdict = Literal["pass", "warn", "critical", "insufficient_history"]

DRIFT_WINDOW = 3
DRIFT_WARN = 0.03
DRIFT_CRITICAL = 0.05


class DriftPoint(BaseModel):
    run_id: str
    started_at: str
    accuracy: float
    scored_ratio: float
    case_count: int


class DriftReport(BaseModel):
    """A comparison of the last `window` runs against the `window` before them.

    `points` covers the whole series, not just the compared windows, so a chart
    can show the context the verdict was drawn from.
    """

    prompt_version: str
    provider: str
    model: str
    window: int
    points: list[DriftPoint]
    prior_run_ids: list[str]
    recent_run_ids: list[str]
    prior_mean: float | None
    recent_mean: float | None
    verdict: DriftVerdict
    reason: str | None

    @property
    def delta(self) -> float | None:
        if self.prior_mean is None or self.recent_mean is None:
            return None
        return self.recent_mean - self.prior_mean


def _points(history: list[RunResult]) -> list[DriftPoint]:
    return [
        DriftPoint(
            run_id=run.run_id,
            started_at=run.started_at,
            accuracy=run.category_accuracy,
            scored_ratio=run.scored_ratio,
            case_count=len(run.cases),
        )
        for run in history
    ]


def _blocker(window: list[DriftPoint]) -> str | None:
    """Why this window cannot be compared, or None if it can."""
    for point in window:
        if point.scored_ratio < MIN_SCORED_RATIO:
            return (
                f"{point.run_id} scored {point.scored_ratio:.0%} of cases, "
                f"drift needs {MIN_SCORED_RATIO:.0%}"
            )

    first = window[0]
    for point in window[1:]:
        if point.case_count != first.case_count:
            return (
                f"dataset size changed {first.case_count} -> {point.case_count} "
                f"at {point.run_id}"
            )
    return None


def _drift_verdict(delta: float, warn: float, critical: float) -> DriftVerdict:
    drop = round(-delta, 6)
    if drop > critical:
        return "critical"
    if drop > warn:
        return "warn"
    return "pass"


def detect_drift(
    history: list[RunResult],
    window: int = DRIFT_WINDOW,
    warn: float = DRIFT_WARN,
    critical: float = DRIFT_CRITICAL,
) -> DriftReport:
    """Compare the mean accuracy of the last `window` runs against the previous.

    Args:
        history: runs of one prompt on one backend, oldest first.

    Returns:
        A DriftReport whose verdict is insufficient_history, with a reason, when
        the series is too short or the compared runs are not comparable.

    Raises:
        ValueError: if `history` is empty.
    """
    if not history:
        raise ValueError("cannot detect drift without any runs")

    points = _points(history)
    latest = history[-1]
    needed = window * 2

    prior_ids: list[str] = []
    recent_ids: list[str] = []
    prior_mean: float | None = None
    recent_mean: float | None = None
    verdict: DriftVerdict = "insufficient_history"

    if len(points) < needed:
        reason: str | None = (
            f"{len(points)} of {needed} runs of {latest.prompt_version} on "
            f"{latest.provider}/{latest.model}"
        )
    else:
        compared = points[-needed:]
        prior, recent = compared[:window], compared[window:]
        prior_ids = [point.run_id for point in prior]
        recent_ids = [point.run_id for point in recent]
        reason = _blocker(compared)
        if reason is None:
            prior_mean = mean(point.accuracy for point in prior)
            recent_mean = mean(point.accuracy for point in recent)
            verdict = _drift_verdict(recent_mean - prior_mean, warn, critical)

    return DriftReport(
        prompt_version=latest.prompt_version,
        provider=latest.provider,
        model=latest.model,
        window=window,
        points=points,
        prior_run_ids=prior_ids,
        recent_run_ids=recent_ids,
        prior_mean=prior_mean,
        recent_mean=recent_mean,
        verdict=verdict,
        reason=reason,
    )


def format_drift(report: DriftReport) -> str:
    """Render a DriftReport as the terminal section."""
    prior, recent, delta = report.prior_mean, report.recent_mean, report.delta
    if prior is None or recent is None or delta is None:
        return f"drift               {report.reason}, not enough history"

    return "\n".join(
        [
            f"drift (window {report.window})    "
            f"prior {prior:.2%} -> recent {recent:.2%}"
            f"   ({delta:+.1%})   {report.verdict.upper()}",
            f"  prior             {', '.join(report.prior_run_ids)}",
            f"  recent            {', '.join(report.recent_run_ids)}",
        ]
    )
