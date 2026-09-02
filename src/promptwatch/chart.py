from html import escape

from pydantic import BaseModel

from promptwatch.drift import DriftPoint, DriftReport

WIDTH = 720
HEIGHT = 220
PAD_LEFT = 52
PAD_RIGHT = 74
PAD_TOP = 14
PAD_BOTTOM = 42

MIN_POINTS = 4
MIN_SPAN = 0.04
GRID_LINES = 4
MAX_TICKS = 8

_VERDICT_TONE = {
    "pass": "pw-tone-ok",
    "warn": "pw-tone-warn",
    "critical": "pw-tone-critical",
    "insufficient_history": "pw-tone-muted",
}


class Bounds(BaseModel):
    low: float
    high: float

    @property
    def span(self) -> float:
        return self.high - self.low


def headline(report: DriftReport) -> str:
    """The finding, as a sentence, for use as the section title."""
    delta = report.delta
    if delta is None:
        return f"Not enough history to judge drift: {report.reason}"

    runs = f"the last {report.window} runs"
    if abs(delta) <= report.warn:
        return f"Accuracy held steady across {runs}"
    direction = "fell" if delta < 0 else "rose"
    return f"Accuracy {direction} {abs(delta) * 100:.1f} points over {runs}"


def subtitle(report: DriftReport) -> str:
    """Which series the finding is about."""
    return (
        f"{report.prompt_version} on {report.provider}/{report.model}"
        f" · {len(report.points)} runs"
    )


def tiles(report: DriftReport) -> list[tuple[str, str]]:
    """Label and value pairs for the figures beside the chart."""
    latest = report.points[-1]
    rows = [("latest accuracy", f"{latest.accuracy:.2%}")]
    if report.prior_mean is None or report.recent_mean is None:
        rows.append(("runs recorded", f"{len(report.points)} of {report.window * 2}"))
        return rows

    delta = report.recent_mean - report.prior_mean
    rows = [
        (f"prior {report.window} runs", f"{report.prior_mean:.2%}"),
        (f"recent {report.window} runs", f"{report.recent_mean:.2%}"),
        ("change", f"{delta * 100:+.1f} pp"),
        *rows,
    ]
    return rows


def _references(report: DriftReport) -> list[float]:
    """Values the y axis must contain so the threshold rules stay visible."""
    prior = report.prior_mean
    if prior is None:
        return []
    return [prior, prior - report.warn, prior - report.critical]


def _bounds(points: list[DriftPoint], extra: list[float]) -> Bounds:
    values = [point.accuracy for point in points] + extra
    low, high = min(values), max(values)
    if high - low < MIN_SPAN:
        middle = (high + low) / 2
        low, high = middle - MIN_SPAN / 2, middle + MIN_SPAN / 2
    margin = (high - low) * 0.15
    return Bounds(low=low - margin, high=high + margin)


def _x(index: int, count: int) -> float:
    if count < 2:
        return PAD_LEFT + (WIDTH - PAD_LEFT - PAD_RIGHT) / 2
    step = (WIDTH - PAD_LEFT - PAD_RIGHT) / (count - 1)
    return PAD_LEFT + index * step


def _y(value: float, bounds: Bounds) -> float:
    plot = HEIGHT - PAD_TOP - PAD_BOTTOM
    return PAD_TOP + plot * (1 - (value - bounds.low) / bounds.span)


def _grid(bounds: Bounds) -> str:
    parts = []
    for step in range(GRID_LINES + 1):
        value = bounds.low + bounds.span * step / GRID_LINES
        y = _y(value, bounds)
        parts.append(
            f'<line class="pw-grid" x1="{PAD_LEFT}" y1="{y:.1f}" '
            f'x2="{WIDTH - PAD_RIGHT}" y2="{y:.1f}" />'
        )
        parts.append(
            f'<text class="pw-tick" x="{PAD_LEFT - 8}" y="{y + 4:.1f}" '
            f'text-anchor="end">{value:.1%}</text>'
        )
    return "".join(parts)


def _thresholds(report: DriftReport, bounds: Bounds) -> str:
    """The prior mean, the tolerance band beneath it, and the critical rule.

    Fixed references are what make severity readable: the y axis rescales to
    each series, so a shallow drop and a steep one look alike without them.
    """
    prior = report.prior_mean
    if prior is None:
        return ""

    right = WIDTH - PAD_RIGHT
    top = _y(prior, bounds)
    warn_y = _y(prior - report.warn, bounds)
    critical_y = _y(prior - report.critical, bounds)
    return (
        f'<rect class="pw-tolerance" x="{PAD_LEFT}" y="{top:.1f}" '
        f'width="{right - PAD_LEFT}" height="{max(warn_y - top, 0):.1f}" />'
        f'<line class="pw-baseline" x1="{PAD_LEFT}" y1="{top:.1f}" '
        f'x2="{right}" y2="{top:.1f}" />'
        f'<line class="pw-threshold" x1="{PAD_LEFT}" y1="{warn_y:.1f}" '
        f'x2="{right}" y2="{warn_y:.1f}" />'
        f'<line class="pw-threshold pw-threshold-critical" x1="{PAD_LEFT}" '
        f'y1="{critical_y:.1f}" x2="{right}" y2="{critical_y:.1f}" />'
        f'<text class="pw-key" x="{right + 6}" y="{top + 4:.1f}">prior</text>'
        f'<text class="pw-key" x="{right + 6}" y="{warn_y + 4:.1f}">warn</text>'
        f'<text class="pw-key" x="{right + 6}" y="{critical_y + 4:.1f}">crit</text>'
    )


def _recent_span(report: DriftReport, bounds: Bounds) -> str:
    if report.recent_mean is None or not report.recent_run_ids:
        return ""
    index = {point.run_id: n for n, point in enumerate(report.points)}
    positions = [index[run_id] for run_id in report.recent_run_ids if run_id in index]
    if not positions:
        return ""

    count = len(report.points)
    start = min(positions)
    left = (_x(start - 1, count) + _x(start, count)) / 2 if start else PAD_LEFT
    tone = _VERDICT_TONE[report.verdict]
    return (
        f'<rect class="pw-window {tone}" x="{left:.1f}" y="{PAD_TOP}" '
        f'width="{WIDTH - PAD_RIGHT - left:.1f}" '
        f'height="{HEIGHT - PAD_TOP - PAD_BOTTOM}" />'
    )


def _series(points: list[DriftPoint], bounds: Bounds) -> str:
    coordinates = " ".join(
        f"{_x(n, len(points)):.1f},{_y(point.accuracy, bounds):.1f}"
        for n, point in enumerate(points)
    )
    return f'<polyline class="pw-line" points="{coordinates}" />'


def _markers(report: DriftReport, bounds: Bounds) -> str:
    points = report.points
    recent = set(report.recent_run_ids)
    parts = []
    for n, point in enumerate(points):
        x, y = _x(n, len(points)), _y(point.accuracy, bounds)
        css = "pw-dot pw-dot-recent" if point.run_id in recent else "pw-dot"
        label = escape(f"run {n + 1} · {point.run_id} · {point.accuracy:.2%}")
        parts.append(
            f"<g><title>{label}</title>"
            f'<circle class="pw-hit" cx="{x:.1f}" cy="{y:.1f}" r="14" />'
            f'<circle class="{css}" cx="{x:.1f}" cy="{y:.1f}" r="4.5" />'
            f"</g>"
        )
    return "".join(parts)


def _axis(points: list[DriftPoint]) -> str:
    baseline = HEIGHT - PAD_BOTTOM
    count = len(points)
    every = max(1, -(-count // MAX_TICKS))
    ticks = [
        f'<text class="pw-tick" x="{_x(n, count):.1f}" y="{baseline + 16}" '
        f'text-anchor="middle">{n + 1}</text>'
        for n in range(count)
        if n % every == 0 or n == count - 1
    ]
    return (
        f'<line class="pw-axis" x1="{PAD_LEFT}" y1="{baseline}" '
        f'x2="{WIDTH - PAD_RIGHT}" y2="{baseline}" />'
        + "".join(ticks)
        + f'<text class="pw-axis-title" x="{(PAD_LEFT + WIDTH - PAD_RIGHT) / 2:.0f}" '
        f'y="{baseline + 34}" text-anchor="middle">run, oldest to newest</text>'
    )


def render_trend(report: DriftReport) -> str | None:
    """Inline SVG of category accuracy per run, with the recent window marked.

    The prior-window mean is drawn as a rule across the full width so the drop
    is a measurable distance. Every label sits outside the plot area.

    Returns:
        The SVG markup, or None when the series has too few points to plot
        honestly.
    """
    points = report.points
    if len(points) < MIN_POINTS:
        return None

    bounds = _bounds(points, _references(report))
    return (
        f'<svg class="pw-chart" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
        f'aria-label="Category accuracy across {len(points)} runs">'
        f"{_recent_span(report, bounds)}"
        f"{_grid(bounds)}"
        f"{_axis(points)}"
        f"{_thresholds(report, bounds)}"
        f"{_series(points, bounds)}"
        f"{_markers(report, bounds)}"
        "</svg>"
    )
