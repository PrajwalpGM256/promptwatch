import pytest
from conftest import make_case_result, make_run

from promptwatch.diff import MIN_SCORED_RATIO, diff_runs, format_diff


def run_of(run_id: str, correct: int, wrong: int, **kwargs):
    cases = [make_case_result(f"c{i}", "misc", "misc") for i in range(correct)]
    cases += [
        make_case_result(f"w{i}", "misc", "rejection") for i in range(wrong)
    ]
    return make_run(run_id, cases, **kwargs)


def test_flips_are_named_per_case():
    base = make_run(
        "base",
        [
            make_case_result("a", "misc", "misc"),
            make_case_result("b", "job_alert", "misc"),
            make_case_result("c", "rejection", "rejection"),
        ],
    )
    head = make_run(
        "head",
        [
            make_case_result("a", "misc", "newsletter"),
            make_case_result("b", "job_alert", "job_alert"),
            make_case_result("c", "rejection", "rejection"),
        ],
    )
    diff = diff_runs(base, head)
    assert [f.case_id for f in diff.regressions] == ["a"]
    assert [f.case_id for f in diff.improvements] == ["b"]
    assert diff.regressions[0].before == "misc"
    assert diff.regressions[0].after == "newsletter"


def test_cases_present_in_only_one_run_are_reported():
    base = make_run("base", [make_case_result("a", "misc", "misc")])
    head = make_run(
        "head",
        [
            make_case_result("a", "misc", "misc"),
            make_case_result("new", "misc", "misc"),
        ],
    )
    diff = diff_runs(base, head)
    assert diff.only_in_head == ["new"]
    assert diff.only_in_base == []


@pytest.mark.parametrize(
    "wrong_in_head,expected",
    [
        (0, "pass"),
        (4, "pass"),
        (5, "pass"),
        (6, "warn"),
        (8, "warn"),
        (9, "critical"),
        (20, "critical"),
    ],
)
def test_verdict_thresholds(wrong_in_head, expected):
    base = run_of("base", correct=100, wrong=0)
    head = run_of("head", correct=100 - wrong_in_head, wrong=wrong_in_head)
    assert diff_runs(base, head).verdict == expected


def test_verdict_withheld_when_too_few_cases_scored():
    base = run_of("base", correct=10, wrong=0)
    head = make_run(
        "head",
        [make_case_result(f"c{i}", "misc", None, status="error") for i in range(10)],
    )
    assert diff_runs(base, head).verdict == "no_data"
    assert "VERDICT WITHHELD" in format_diff(diff_runs(base, head))


def test_out_of_contract_cases_do_not_withhold_the_verdict():
    base = run_of("base", correct=10, wrong=0)
    head = make_run(
        "head",
        [make_case_result(f"c{i}", "misc", "misc") for i in range(8)]
        + [
            make_case_result(
                f"x{i}", "interview_invite", None, status="out_of_contract"
            )
            for i in range(2)
        ],
    )
    diff = diff_runs(base, head)
    assert diff.head_scored_ratio == 1.0
    assert diff.verdict == "pass"
    assert diff.head_out_of_contract == 2


def test_scored_ratio_floor_is_the_boundary():
    base = run_of("base", correct=10, wrong=0)
    cases = [make_case_result(f"c{i}", "misc", "misc") for i in range(8)]
    cases += [make_case_result(f"e{i}", "misc", None, status="error") for i in range(2)]
    head = make_run("head", cases)
    assert head.scored_ratio == pytest.approx(MIN_SCORED_RATIO)
    assert diff_runs(base, head).verdict != "no_data"


@pytest.mark.parametrize(
    "kwargs,fragment",
    [
        ({"provider": "ollama"}, "provider"),
        ({"model": "gemini-2.5-flash"}, "model"),
        ({"dataset_version": "v2"}, "dataset"),
        ({"judge_version": "none"}, "judge"),
    ],
)
def test_confounders_detected(kwargs, fragment):
    base = run_of("base", correct=10, wrong=0)
    head = run_of("head", correct=10, wrong=0, **kwargs)
    diff = diff_runs(base, head)
    assert any(fragment in c for c in diff.confounders)
    assert "NOT A CLEAN PROMPT COMPARISON" in format_diff(diff)


def test_no_confounders_when_everything_matches():
    diff = diff_runs(run_of("base", 10, 0), run_of("head", 10, 0))
    assert diff.confounders == []
    assert "NOT A CLEAN PROMPT COMPARISON" not in format_diff(diff)


def unjudged(run_id):
    cases = [
        make_case_result(f"c{i}", "misc", "misc", summary_score=None)
        for i in range(10)
    ]
    return make_run(run_id, cases, judge_version="none", judge_provider="none")


def test_delta_is_absent_when_either_run_was_unjudged():
    judged = run_of("judged", correct=10, wrong=0)

    assert diff_runs(judged, unjudged("head")).summary_delta is None
    assert diff_runs(unjudged("base"), judged).summary_delta is None
    assert diff_runs(unjudged("base"), unjudged("head")).summary_delta is None


def test_unjudged_side_renders_as_not_judged():
    text = format_diff(diff_runs(run_of("base", correct=10, wrong=0), unjudged("head")))

    assert "not judged" in text
    assert "0.00" not in text.split("summary mean")[1].split("\n")[0]


def test_judged_pair_still_reports_a_delta():
    text = format_diff(
        diff_runs(
            run_of("base", correct=10, wrong=0),
            run_of("head", correct=10, wrong=0),
        )
    )
    assert "reported only" in text
    assert "not judged" not in text
