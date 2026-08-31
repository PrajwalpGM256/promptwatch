import asyncio
from time import perf_counter

import pytest
from conftest import FakeProvider
from tenacity import Future, RetryCallState

from promptwatch.config import PromptConfig
from promptwatch.dataset import GoldenCase, GoldenDataset
from promptwatch.judge import JudgeConfig
from promptwatch.provider import ProviderError, TransientError, get_provider
from promptwatch.runner import (
    RateLimiter,
    RunAborted,
    _run_case,
    _wait,
    run_dataset,
)


def case_expecting(category: str, fields: dict) -> GoldenCase:
    return GoldenCase(**{**fields, "expected_category": category})


def dataset_of(*cases: GoldenCase) -> GoldenDataset:
    return GoldenDataset(
        version="v1",
        timestamp="2026-08-26T00:00:00Z",
        categories=["application_ack", "interview_invite", "rejection", "misc"],
        cases=list(cases),
    )


def two_cases(fields: dict) -> GoldenDataset:
    first = case_expecting("interview_invite", fields)
    second = GoldenCase(
        **{**fields, "id": "gc-002", "expected_category": "misc"}
    )
    return dataset_of(first, second)


@pytest.mark.asyncio
async def test_out_of_contract_case_never_calls_the_api(valid_case_fields):
    v1 = PromptConfig.load("prompts/v1.yaml")
    case = case_expecting("interview_invite", valid_case_fields)

    result = await _run_case(
        get_provider("ollama"),
        case,
        v1,
        None,
        "no-such-model",
        FakeProvider(),
        "no-such-judge-model",
        asyncio.Semaphore(1),
        RateLimiter(600),
        RateLimiter(600),
    )

    assert result.status == "out_of_contract"
    assert result.category_match is None
    assert result.prompt_tokens == 0
    assert result.latency_ms == 0


@pytest.mark.asyncio
async def test_in_contract_case_is_not_short_circuited(valid_case_fields):
    v2 = PromptConfig.load("prompts/v2.yaml")
    case = case_expecting("interview_invite", valid_case_fields)
    assert case.expected_category in v2.categories


@pytest.mark.asyncio
async def test_rate_limiter_spaces_acquisitions():
    limiter = RateLimiter(per_minute=600)
    started = perf_counter()
    await asyncio.gather(*(limiter.acquire() for _ in range(5)))
    elapsed = perf_counter() - started
    assert elapsed >= 0.4


@pytest.mark.asyncio
async def test_rate_limiter_first_acquisition_is_immediate():
    limiter = RateLimiter(per_minute=60)
    started = perf_counter()
    await limiter.acquire()
    assert perf_counter() - started < 0.1


@pytest.mark.asyncio
async def test_rate_limiter_zero_does_not_pace():
    limiter = RateLimiter(per_minute=0)
    started = perf_counter()
    await asyncio.gather(*(limiter.acquire() for _ in range(50)))
    assert perf_counter() - started < 0.1


@pytest.mark.asyncio
async def test_run_dataset_scores_and_records_the_backend(valid_case_fields):
    v2 = PromptConfig.load("prompts/v2.yaml")
    provider = FakeProvider(category="misc", summary="A note.")

    run = await run_dataset(provider, v2, two_cases(valid_case_fields))

    assert run.provider == "fake"
    assert run.model == "fake-1"
    assert [c.status for c in run.cases] == ["scored", "scored"]
    assert [c.category_match for c in run.cases] == [False, True]
    assert run.category_accuracy == 0.5
    assert run.total_tokens == 30
    assert provider.calls == ["classify", "classify"]


@pytest.mark.asyncio
async def test_run_dataset_judges_only_when_given_a_judge(valid_case_fields):
    v2 = PromptConfig.load("prompts/v2.yaml")
    judge = JudgeConfig.load("prompts/judge_v1.yaml")
    dataset = two_cases(valid_case_fields)

    unjudged = await run_dataset(FakeProvider(), v2, dataset)
    assert [c.summary_score for c in unjudged.cases] == [None, None]
    assert unjudged.judge_version == "none"
    assert unjudged.judge_provider == "none"

    grader = FakeProvider(score=4)
    judged = await run_dataset(
        FakeProvider(), v2, dataset, judge, judge_provider=grader
    )
    assert [c.summary_score for c in judged.cases] == [4, 4]
    assert judged.judge_version == judge.version
    assert judged.mean_summary_score == 4
    assert grader.calls == ["judge", "judge"]


@pytest.mark.asyncio
async def test_judge_runs_on_its_own_backend(valid_case_fields):
    v2 = PromptConfig.load("prompts/v2.yaml")
    judge = JudgeConfig.load("prompts/judge_v1.yaml")
    under_test = FakeProvider(category="misc")
    grader = FakeProvider(score=2)

    run = await run_dataset(
        under_test,
        v2,
        two_cases(valid_case_fields),
        judge,
        judge_provider=grader,
    )

    assert run.provider == "fake"
    assert run.judge_provider == "fake"
    assert run.judge_model == "fake-1"
    assert under_test.calls == ["classify", "classify"]
    assert grader.calls == ["judge", "judge"]


@pytest.mark.asyncio
async def test_undeclared_category_is_off_contract_output(valid_case_fields):
    v2 = PromptConfig.load("prompts/v2.yaml")
    provider = FakeProvider(text='{"category": "spam", "summary": "x"}')

    run = await run_dataset(provider, v2, two_cases(valid_case_fields))

    assert {c.status for c in run.cases} == {"off_contract_output"}
    assert all(c.category_match is False for c in run.cases)
    assert run.category_accuracy == 0.0


@pytest.mark.asyncio
async def test_provider_failure_abandons_the_run(valid_case_fields):
    v2 = PromptConfig.load("prompts/v2.yaml")
    provider = FakeProvider(error=ProviderError("no route to host"))

    with pytest.raises(RunAborted, match="no route to host"):
        await run_dataset(provider, v2, two_cases(valid_case_fields))


@pytest.mark.asyncio
async def test_abort_stops_spending_requests(valid_case_fields):
    v2 = PromptConfig.load("prompts/v2.yaml")
    dataset = dataset_of(
        *(
            GoldenCase(**{**valid_case_fields, "id": f"gc-{n:03}"})
            for n in range(1, 21)
        )
    )
    provider = FakeProvider(error=ProviderError("quota exhausted"))

    with pytest.raises(RunAborted):
        await run_dataset(provider, v2, dataset, concurrency=2)

    assert len(provider.calls) < 20


@pytest.mark.asyncio
async def test_progress_is_reported_per_case(valid_case_fields):
    v2 = PromptConfig.load("prompts/v2.yaml")
    seen: list[tuple[str, int, int]] = []

    await run_dataset(
        FakeProvider(),
        v2,
        two_cases(valid_case_fields),
        on_case=lambda result, done, total: seen.append(
            (result.case_id, done, total)
        ),
    )

    assert [(done, total) for _, done, total in seen] == [(1, 2), (2, 2)]
    assert {case_id for case_id, _, _ in seen} == {"gc-001", "gc-002"}


@pytest.mark.asyncio
async def test_out_of_contract_case_costs_no_provider_call(valid_case_fields):
    v1 = PromptConfig.load("prompts/v1.yaml")
    provider = FakeProvider()

    run = await run_dataset(provider, v1, two_cases(valid_case_fields))

    statuses = {c.case_id: c.status for c in run.cases}
    assert statuses["gc-001"] == "out_of_contract"
    assert statuses["gc-002"] == "scored"
    assert provider.calls == ["classify"]


@pytest.mark.asyncio
async def test_limit_truncates_the_case_list(valid_case_fields):
    v2 = PromptConfig.load("prompts/v2.yaml")
    provider = FakeProvider()

    run = await run_dataset(provider, v2, two_cases(valid_case_fields), limit=1)

    assert len(run.cases) == 1
    assert provider.calls == ["classify"]


@pytest.mark.asyncio
async def test_judge_is_paced_by_its_own_backend(valid_case_fields, monkeypatch):
    intervals = []
    original = RateLimiter.__init__

    def record(self, per_minute):
        intervals.append(per_minute)
        original(self, per_minute)

    monkeypatch.setattr(RateLimiter, "__init__", record)

    judge = JudgeConfig.load("prompts/judge_v1.yaml")
    grader = FakeProvider()
    grader.name = "fake-judge"
    grader.default_requests_per_minute = 600

    await run_dataset(
        FakeProvider(),
        PromptConfig.load("prompts/v2.yaml"),
        two_cases(valid_case_fields),
        judge,
        judge_provider=grader,
        requests_per_minute=0,
    )

    assert intervals == [0, 600]


@pytest.mark.asyncio
async def test_one_backend_gets_one_limiter(valid_case_fields, monkeypatch):
    built = []
    original = RateLimiter.__init__

    def record(self, per_minute):
        built.append(per_minute)
        original(self, per_minute)

    monkeypatch.setattr(RateLimiter, "__init__", record)

    same = FakeProvider()
    await run_dataset(
        same,
        PromptConfig.load("prompts/v2.yaml"),
        two_cases(valid_case_fields),
        JudgeConfig.load("prompts/judge_v1.yaml"),
        judge_provider=same,
        requests_per_minute=0,
    )

    assert built == [0]


def _state(exception):
    state = RetryCallState(None, None, (), {})
    state.outcome = Future.construct(1, exception, True)
    state.attempt_number = 1
    return state


def test_wait_honours_the_provider_hint():
    assert _wait(_state(TransientError("429", retry_after=45))) == 45


def test_wait_falls_back_to_backoff_without_a_hint():
    assert _wait(_state(TransientError("429"))) == 3
