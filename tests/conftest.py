import pytest

from promptwatch.dataset import GoldenCase
from promptwatch.results import CaseResult, RunResult

ALL_CATEGORIES = [
    "application_ack",
    "interview_invite",
    "rejection",
    "job_alert",
    "newsletter",
    "misc",
]


@pytest.fixture
def valid_case_fields() -> dict:
    return {
        "id": "gc-001",
        "subject": "Re: Backend Engineer application",
        "body": "We'd like a 30-minute phone screen. Reply to noreply@example.com.",
        "expected_category": "interview_invite",
        "must_mention": ["phone screen", "Backend Engineer"],
        "difficulty": "easy",
        "tags": [],
        "notes": "Straightforward invite.",
    }


@pytest.fixture
def golden_case(valid_case_fields) -> GoldenCase:
    return GoldenCase(**valid_case_fields)


def make_case_result(
    case_id: str,
    expected: str,
    actual: str | None,
    status: str = "scored",
    summary_score: int | None = 4,
) -> CaseResult:
    scored = status == "scored"
    attempted = status != "out_of_contract"
    return CaseResult(
        case_id=case_id,
        status=status,
        expected_category=expected,
        actual_category=actual if scored else None,
        category_match=(actual == expected) if scored else None,
        summary="a summary" if scored else None,
        summary_score=summary_score if scored else None,
        latency_ms=100 if attempted else 0,
        prompt_tokens=10 if scored else 0,
        output_tokens=5 if scored else 0,
    )


def make_run(
    run_id: str,
    cases: list[CaseResult],
    prompt_version: str = "v2",
    model: str = "gemini-3.5-flash-lite",
    dataset_version: str = "v1",
    judge_version: str = "v1",
) -> RunResult:
    return RunResult(
        run_id=run_id,
        prompt_version=prompt_version,
        model=model,
        dataset_version=dataset_version,
        judge_version=judge_version,
        started_at="2026-08-25T00:00:00Z",
        cases=cases,
    )
