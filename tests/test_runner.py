import asyncio
from time import perf_counter

import pytest

from promptwatch.config import PromptConfig
from promptwatch.dataset import GoldenCase
from promptwatch.runner import RateLimiter, _run_case


def case_expecting(category: str, fields: dict) -> GoldenCase:
    return GoldenCase(**{**fields, "expected_category": category})


@pytest.mark.asyncio
async def test_out_of_contract_case_never_calls_the_api(valid_case_fields):
    v1 = PromptConfig.load("prompts/v1.yaml")
    case = case_expecting("interview_invite", valid_case_fields)

    result = await _run_case(
        case, v1, None, "no-such-model", asyncio.Semaphore(1), RateLimiter(600)
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
