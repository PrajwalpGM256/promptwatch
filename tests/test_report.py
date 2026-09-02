import pytest
from conftest import make_case_result, make_run, make_series

from promptwatch.dataset import GoldenCase, GoldenDataset
from promptwatch.diff import diff_runs
from promptwatch.drift import detect_drift
from promptwatch.report import _esc, render_report, write_report


def pair(base_cases, head_cases, **kwargs):
    base = make_run("base-run", base_cases)
    head = make_run("head-run", head_cases, **kwargs)
    return diff_runs(base, head), base, head


def flipped(before, after, summary_before, summary_after):
    base_case = make_case_result("gc-047", "interview_invite", before)
    base_case.summary = summary_before
    head_case = make_case_result("gc-047", "interview_invite", after)
    head_case.summary = summary_after
    padding_base = [make_case_result(f"c{i}", "misc", "misc") for i in range(9)]
    padding_head = [make_case_result(f"c{i}", "misc", "misc") for i in range(9)]
    return pair([base_case, *padding_base], [head_case, *padding_head])


def test_escaping_makes_markup_inert():
    assert _esc("<b>&</b>") == "&lt;b&gt;&amp;&lt;/b&gt;"


def test_document_envelope_and_theme():
    diff, base, head = pair(
        [make_case_result("a", "misc", "misc")],
        [make_case_result("a", "misc", "misc")],
    )
    html = render_report(diff, base, head)
    assert html.startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")
    assert 'data-theme="business"' in html


def test_header_names_both_runs():
    diff, base, head = pair(
        [make_case_result("a", "misc", "misc")],
        [make_case_result("a", "misc", "misc")],
    )
    html = render_report(diff, base, head)
    assert "head-run" in html and "base-run" in html


@pytest.mark.parametrize(
    "wrong,css",
    [(0, "badge-success"), (6, "badge-warning"), (12, "badge-error")],
)
def test_verdict_badge_matches_the_verdict(wrong, css):
    base_cases = [make_case_result(f"c{i}", "misc", "misc") for i in range(100)]
    head_cases = [
        make_case_result(f"c{i}", "misc", "misc" if i >= wrong else "rejection")
        for i in range(100)
    ]
    diff, base, head = pair(base_cases, head_cases)
    assert css in render_report(diff, base, head)


def test_confounder_alert_appears_only_when_earned():
    clean, base, head = pair(
        [make_case_result("a", "misc", "misc")],
        [make_case_result("a", "misc", "misc")],
    )
    assert "NOT A CLEAN PROMPT COMPARISON" not in render_report(clean, base, head)

    swapped, base2, head2 = pair(
        [make_case_result("a", "misc", "misc")],
        [make_case_result("a", "misc", "misc")],
        provider="ollama",
    )
    html = render_report(swapped, base2, head2)
    assert "NOT A CLEAN PROMPT COMPARISON" in html
    assert "provider" in html


def test_withheld_verdict_alert():
    base_cases = [make_case_result(f"c{i}", "misc", "misc") for i in range(10)]
    head_cases = [
        make_case_result(f"c{i}", "misc", None, status="error") for i in range(10)
    ]
    diff, base, head = pair(base_cases, head_cases)
    assert "VERDICT WITHHELD" in render_report(diff, base, head)


def test_unjudged_scorecard_says_not_judged_with_no_delta():
    unjudged = [
        make_case_result(f"c{i}", "misc", "misc", summary_score=None)
        for i in range(10)
    ]
    judged = [make_case_result(f"c{i}", "misc", "misc") for i in range(10)]
    diff, base, head = pair(judged, unjudged)

    html = render_report(diff, base, head)

    assert "not judged" in html
    assert "reported only" in html


def test_regression_shows_both_summaries():
    diff, base, head = flipped(
        "interview_invite",
        "misc",
        "Recruiter proposes a Thursday phone screen.",
        "Automated notice about an unrelated role.",
    )

    html = render_report(diff, base, head)

    assert "Recruiter proposes a Thursday phone screen." in html
    assert "Automated notice about an unrelated role." in html
    assert "gc-047" in html
    assert "Regressions (1)" in html


def test_summaries_are_escaped():
    diff, base, head = flipped(
        "interview_invite",
        "misc",
        "<script>alert(1)</script>",
        "safe & sound",
    )

    html = render_report(diff, base, head)

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert "safe &amp; sound" in html


def test_improvements_are_present_but_collapsed():
    diff, base, head = flipped(
        "misc",
        "interview_invite",
        "wrong before",
        "right after",
    )

    html = render_report(diff, base, head)

    assert "Improvements (1)" in html
    assert "<details" in html
    assert "No case got worse." in html


def test_drift_section_omitted_without_a_report():
    diff, base, head = pair(
        [make_case_result("a", "misc", "misc")],
        [make_case_result("a", "misc", "misc")],
    )
    assert "runs recorded" not in render_report(diff, base, head)


def test_drift_section_embeds_the_chart_when_plottable():
    diff, base, head = pair(
        [make_case_result("a", "misc", "misc")],
        [make_case_result("a", "misc", "misc")],
    )
    drift = detect_drift(make_series([0.90] * 6))

    html = render_report(diff, base, head, drift)

    assert "<svg" in html
    assert "Accuracy held steady" in html
    assert "<table" in html


def test_drift_section_falls_back_to_the_table_when_too_short():
    diff, base, head = pair(
        [make_case_result("a", "misc", "misc")],
        [make_case_result("a", "misc", "misc")],
    )
    drift = detect_drift(make_series([0.90] * 3))

    html = render_report(diff, base, head, drift)

    assert "<svg" not in html
    assert "nothing worth plotting yet" in html
    assert "<table" in html


def test_raw_diff_is_embedded():
    diff, base, head = pair(
        [make_case_result("a", "misc", "misc")],
        [make_case_result("a", "misc", "misc")],
    )
    html = render_report(diff, base, head)
    assert "Raw diff (JSON)" in html
    assert "head_run_id" in html


def test_write_report_creates_parent_directories(tmp_path):
    target = tmp_path / "nested" / "deeper" / "report.html"

    write_report(target, "<html>hi</html>")

    assert target.read_text(encoding="utf-8") == "<html>hi</html>"


def test_unjudged_header_says_not_judged():
    unjudged = [
        make_case_result(f"c{i}", "misc", "misc", summary_score=None)
        for i in range(10)
    ]
    base = make_run("base-run", unjudged, judge_provider="none", judge_model="none")
    head = make_run("head-run", unjudged, judge_provider="none", judge_model="none")

    html = render_report(diff_runs(base, head), base, head)

    assert "not judged" in html
    assert "none/none" not in html


def test_flip_card_shows_the_email_subject_and_key_facts(golden_case):
    diff, base, head = flipped(
        "interview_invite", "misc", "before text", "after text"
    )
    dataset = GoldenDataset(
        version="v1",
        timestamp="2026-09-01T00:00:00Z",
        categories=["interview_invite", "misc"],
        cases=[GoldenCase(**{**golden_case.model_dump(), "id": "gc-047"})],
    )

    html = render_report(diff, base, head, dataset=dataset)

    assert _esc(dataset.cases[0].subject) in html
    assert _esc(dataset.cases[0].must_mention[0]) in html


def test_drift_from_another_series_is_refused():
    diff, base, head = pair(
        [make_case_result("a", "misc", "misc")],
        [make_case_result("a", "misc", "misc")],
    )
    other = detect_drift(make_series([0.90] * 6, provider="ollama", model="gemma3:4b"))

    with pytest.raises(ValueError, match="but the run is"):
        render_report(diff, base, head, other)
