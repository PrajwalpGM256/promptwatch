import asyncio
import json

import pytest

from promptwatch.dataset import GoldenCase
from promptwatch.provider import Completion, JsonSchema, Turn
from promptwatch.results import CaseResult, RunResult

ALL_CATEGORIES = [
    "application_ack",
    "interview_invite",
    "rejection",
    "job_alert",
    "newsletter",
    "misc",
]


class FakeProvider:
    """An in-memory Provider that answers from canned values.

    Distinguishes a judge call from a classify call by the schema it is given,
    so it stays correct however the runner interleaves them.
    """

    name = "fake"
    default_model = "fake-1"
    default_requests_per_minute = 0
    default_concurrency = 5

    def __init__(
        self,
        category: str = "misc",
        summary: str = "A short summary.",
        score: int = 5,
        error: Exception | None = None,
        text: str | None = None,
    ) -> None:
        self.category = category
        self.summary = summary
        self.score = score
        self.error = error
        self.text = text
        self.calls: list[str] = []
        self.temperatures: list[float | None] = []

    async def generate_json(
        self,
        system_prompt: str,
        turns: list[Turn],
        schema: JsonSchema,
        model: str,
        temperature: float | None = None,
    ) -> Completion:
        await asyncio.sleep(0)
        judging = "score" in schema["properties"]
        self.calls.append("judge" if judging else "classify")
        self.temperatures.append(temperature)
        if self.error is not None:
            raise self.error
        if self.text is not None:
            payload = self.text
        elif judging:
            payload = json.dumps({"score": self.score})
        else:
            payload = json.dumps(
                {"category": self.category, "summary": self.summary}
            )
        return Completion(text=payload, prompt_tokens=10, output_tokens=5)


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()


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
    provider: str = "gemini",
    model: str = "gemini-3.5-flash-lite",
    dataset_version: str = "v1",
    judge_version: str = "v1",
    judge_provider: str = "groq",
    judge_model: str = "openai/gpt-oss-20b",
    started_at: str = "2026-08-25T00:00:00Z",
) -> RunResult:
    return RunResult(
        run_id=run_id,
        prompt_version=prompt_version,
        provider=provider,
        model=model,
        dataset_version=dataset_version,
        judge_version=judge_version,
        judge_provider=judge_provider,
        judge_model=judge_model,
        started_at=started_at,
        cases=cases,
    )
