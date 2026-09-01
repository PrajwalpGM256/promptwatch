import pytest
from conftest import make_case_result, make_run, make_series

from promptwatch.drift import (
    DRIFT_WINDOW,
    _blocker,
    _drift_verdict,
    _points,
    detect_drift,
    format_drift,
)

FLAT = [0.90] * 6


def test_points_carry_the_series_shape():
    points = _points(make_series([0.90, 0.80]))
    assert [p.run_id for p in points] == ["r01", "r02"]
    assert points[0].accuracy == pytest.approx(0.90, abs=0.01)
    assert points[0].case_count == 94
    assert points[0].scored_ratio == 1.0


def test_empty_history_raises():
    with pytest.raises(ValueError, match="without any runs"):
        detect_drift([])


def test_short_series_refuses_and_names_the_shortfall():
    report = detect_drift(make_series([0.90] * 5))
    assert report.verdict == "insufficient_history"
    assert report.reason is not None
    assert "5 of 6 runs" in report.reason
    assert report.delta is None


def test_flat_series_passes():
    report = detect_drift(make_series(FLAT))
    assert report.verdict == "pass"
    assert report.delta == pytest.approx(0.0)
    assert report.prior_run_ids == ["r01", "r02", "r03"]
    assert report.recent_run_ids == ["r04", "r05", "r06"]


@pytest.mark.parametrize(
    "recent,expected",
    [(0.90, "pass"), (0.88, "pass"), (0.86, "warn"), (0.83, "critical")],
)
def test_decline_crosses_the_thresholds(recent, expected):
    report = detect_drift(make_series([0.90] * 3 + [recent] * 3))
    assert report.verdict == expected


def test_improvement_never_alarms():
    report = detect_drift(make_series([0.70] * 3 + [0.95] * 3))
    assert report.verdict == "pass"
    assert report.delta is not None and report.delta > 0


def test_points_cover_the_whole_series_not_just_the_window():
    report = detect_drift(make_series([0.90] * 10))
    assert len(report.points) == 10
    assert len(report.prior_run_ids) == DRIFT_WINDOW


def test_low_scored_ratio_refuses_and_names_the_run():
    runs = make_series(FLAT)
    runs[-1] = make_run(
        "bad",
        [make_case_result(f"gc-{n:03}", "misc", "misc") for n in range(50)]
        + [
            make_case_result(f"gc-{n:03}", "misc", None, status="error")
            for n in range(50, 94)
        ],
        started_at="2026-09-07T00:00:00Z",
    )

    report = detect_drift(runs)

    assert report.verdict == "insufficient_history"
    assert report.reason is not None and "bad" in report.reason


def test_case_count_change_refuses_and_names_the_run():
    runs = make_series([0.90] * 5) + make_series([0.90], total=95)
    runs[-1] = make_run(
        "grown",
        runs[-1].cases,
        started_at="2026-09-07T00:00:00Z",
    )

    report = detect_drift(runs)

    assert report.verdict == "insufficient_history"
    assert report.reason is not None
    assert "dataset size changed" in report.reason
    assert "grown" in report.reason


def test_blocker_passes_a_clean_window():
    assert _blocker(_points(make_series(FLAT))) is None


@pytest.mark.parametrize(
    "delta,expected",
    [(0.0, "pass"), (0.10, "pass"), (-0.03, "pass"), (-0.04, "warn"),
     (-0.05, "warn"), (-0.06, "critical")],
)
def test_drift_verdict_boundaries(delta, expected):
    assert _drift_verdict(delta, 0.03, 0.05) == expected


def test_alarm_clears_once_the_decline_leaves_the_window():
    verdicts = []
    for extra in range(DRIFT_WINDOW + 2):
        series = [0.95] * 3 + [0.85] * 3 + [0.85] * extra
        verdicts.append(detect_drift(make_series(series)).verdict)

    assert verdicts[0] != "pass"
    assert verdicts[-1] == "pass"
    assert sum(1 for v in verdicts if v != "pass") <= DRIFT_WINDOW


def test_format_refusal_is_one_line():
    text = format_drift(detect_drift(make_series([0.90] * 3)))
    assert "\n" not in text
    assert "not enough history" in text


def test_format_names_both_window_means():
    text = format_drift(detect_drift(make_series([0.95] * 3 + [0.85] * 3)))
    assert "prior" in text and "recent" in text
    assert "r01" in text and "r06" in text
