import pytest
from conftest import make_series

from promptwatch.chart import (
    HEIGHT,
    MIN_POINTS,
    MIN_SPAN,
    PAD_BOTTOM,
    PAD_LEFT,
    PAD_RIGHT,
    PAD_TOP,
    WIDTH,
    _bounds,
    _references,
    _x,
    _y,
    headline,
    render_trend,
    subtitle,
    tiles,
)
from promptwatch.drift import detect_drift


def report_of(accuracies):
    return detect_drift(make_series(accuracies))


def points_of(accuracies):
    return report_of(accuracies).points


def test_headline_states_a_decline_in_points():
    text = headline(report_of([0.95] * 3 + [0.84] * 3))
    assert text.startswith("Accuracy fell")
    assert "points over the last 3 runs" in text


def test_headline_states_stability_rather_than_a_zero():
    text = headline(report_of([0.90] * 6))
    assert text == "Accuracy held steady across the last 3 runs"


def test_headline_states_an_improvement_as_a_rise():
    assert headline(report_of([0.70] * 3 + [0.95] * 3)).startswith("Accuracy rose")


def test_headline_explains_insufficient_history():
    text = headline(report_of([0.90] * 3))
    assert text.startswith("Not enough history")
    assert "3 of 6 runs" in text


def test_subtitle_names_the_series_and_run_count():
    text = subtitle(report_of([0.90] * 6))
    assert "v2 on gemini/gemini-3.5-flash-lite" in text
    assert "6 runs" in text


def test_tiles_carry_both_means_and_the_change():
    labels = dict(tiles(report_of([0.95] * 3 + [0.84] * 3)))
    assert "prior 3 runs" in labels
    assert "recent 3 runs" in labels
    assert labels["change"].startswith("-10")
    assert labels["change"].endswith("pp")


def test_tiles_report_the_shortfall_when_history_is_short():
    labels = dict(tiles(report_of([0.90] * 4)))
    assert labels["runs recorded"] == "4 of 6"
    assert "change" not in labels


def test_bounds_are_not_zero_based():
    assert _bounds(points_of([0.90, 0.95, 0.92, 0.94]), []).low > 0.5


def test_flat_series_gets_a_minimum_span():
    assert _bounds(points_of([0.90] * 4), []).span >= MIN_SPAN


def test_bounds_include_the_window_means():
    points = points_of([0.90] * 4)
    assert _bounds(points, [0.5]).low < 0.5


def test_higher_accuracy_maps_to_a_smaller_y():
    bounds = _bounds(points_of([0.80, 0.95, 0.85, 0.90]), [])
    assert _y(0.95, bounds) < _y(0.80, bounds)


def test_points_stay_inside_the_plot_area():
    points = points_of([0.80, 0.95, 0.85, 0.90])
    bounds = _bounds(points, [])
    for n, point in enumerate(points):
        assert PAD_LEFT <= _x(n, len(points)) <= WIDTH
        assert PAD_TOP <= _y(point.accuracy, bounds) <= HEIGHT - PAD_BOTTOM


def test_a_single_point_does_not_divide_by_zero():
    assert PAD_LEFT < _x(0, 1) < WIDTH


def test_short_series_is_not_plotted():
    assert render_trend(report_of([0.90] * (MIN_POINTS - 1))) is None


def test_svg_envelope_and_point_count():
    svg = render_trend(report_of([0.90, 0.91, 0.89, 0.92, 0.90]))
    assert svg is not None
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert svg.count("pw-dot") == 5
    assert 'role="img"' in svg


def test_axis_is_labelled_by_run_index_not_run_id():
    svg = render_trend(report_of([0.90, 0.91, 0.89, 0.92]))
    assert svg is not None
    assert "run, oldest to newest" in svg
    assert 'text-anchor="middle">1</text>' in svg
    assert "r01" not in svg.split("<title>")[0]


def test_run_ids_appear_only_in_tooltips():
    svg = render_trend(report_of([0.90, 0.91, 0.89, 0.92]))
    assert svg is not None
    assert svg.count("<title>") == 4
    assert "r01" in svg


def test_run_ids_are_escaped():
    report = report_of([0.90, 0.91, 0.89, 0.92])
    report.points[0].run_id = "<script>x</script>"

    svg = render_trend(report)

    assert svg is not None
    assert "<script>" not in svg
    assert "&lt;script&gt;" in svg


def test_references_and_window_absent_when_history_is_insufficient():
    report = report_of([0.90, 0.91, 0.89, 0.92, 0.90])
    assert report.verdict == "insufficient_history"

    svg = render_trend(report)

    assert svg is not None
    assert "pw-baseline" not in svg
    assert "pw-threshold" not in svg
    assert "pw-window" not in svg


def test_thresholds_are_drawn_once_the_window_is_full():
    svg = render_trend(report_of([0.90] * 6))
    assert svg is not None
    assert svg.count("pw-baseline") == 1
    assert svg.count("pw-threshold") == 3
    assert svg.count("pw-tolerance") == 1
    assert svg.count("pw-window") == 1
    assert svg.count("pw-dot-recent") == 3
    assert ">prior<" in svg and ">warn<" in svg and ">crit<" in svg


def test_threshold_rules_stay_visible_on_a_shallow_drop():
    report = report_of([0.90] * 3 + [0.895] * 3)
    bounds = _bounds(report.points, _references(report))
    assert report.prior_mean is not None
    assert bounds.low < report.prior_mean - report.critical


def test_the_y_axis_span_does_not_shrink_below_the_critical_threshold():
    report = report_of([0.90] * 6)
    bounds = _bounds(report.points, _references(report))
    assert bounds.span > report.critical


def test_the_recent_window_reaches_the_right_edge():
    svg = render_trend(report_of([0.90] * 6))
    assert svg is not None
    rect = svg.split('class="pw-window')[1].split("/>")[0]
    left = float(rect.split('x="')[1].split('"')[0])
    width = float(rect.split('width="')[1].split('"')[0])
    assert left + width == pytest.approx(WIDTH - PAD_RIGHT)


@pytest.mark.parametrize(
    "recent,tone",
    [(0.90, "pw-tone-ok"), (0.86, "pw-tone-warn"), (0.83, "pw-tone-critical")],
)
def test_window_tone_follows_the_verdict(recent, tone):
    svg = render_trend(report_of([0.90] * 3 + [recent] * 3))
    assert svg is not None
    assert tone in svg
