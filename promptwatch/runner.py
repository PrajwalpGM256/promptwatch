import asyncio
from datetime import UTC, datetime
from time import perf_counter

from google.genai import errors
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from classifier import classify_email_async
from promptwatch.config import PromptConfig
from promptwatch.dataset import GoldenCase, GoldenDataset
from promptwatch.gemini import DEFAULT_MODEL
from promptwatch.judge import JudgeConfig, score_summary
from promptwatch.models import EmailInput
from promptwatch.results import CaseResult, RunResult

DEFAULT_CONCURRENCY = 5
DEFAULT_REQUESTS_PER_MINUTE = 12



class RateLimiter:
    """Paces API calls to a fixed number per minute across all workers.

    Gemini's free tier rejects bursts with 429 regardless of how few requests
    are in flight, so capping concurrency is not enough on its own.
    """

    def __init__(self, per_minute: int) -> None:
        self._interval = 60 / per_minute
        self._lock = asyncio.Lock()
        self._next_slot = 0.0

    async def acquire(self) -> None:
        """Block until the next request is allowed to go out."""
        async with self._lock:
            now = perf_counter()
            wait = max(0.0, self._next_slot - now)
            self._next_slot = max(now, self._next_slot) + self._interval
        if wait:
            await asyncio.sleep(wait)


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, errors.ServerError):
        return True
    return isinstance(exc, errors.ClientError) and exc.code == 429


_transient_retry = retry(
    retry=retry_if_exception(_is_transient),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=3, min=3, max=60),
    reraise=True,
)


def _run_id(prompt_version: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{prompt_version}"


async def _classify(
    prompt_config: PromptConfig,
    case: GoldenCase,
    model: str,
    limiter: RateLimiter,
) -> CaseResult:
    email = EmailInput(subject=case.subject, body=case.body)
    elapsed_ms = 0

    async def call():
        nonlocal elapsed_ms
        await limiter.acquire()
        started = perf_counter()
        try:
            return await classify_email_async(prompt_config, email, model)
        finally:
            elapsed_ms = int((perf_counter() - started) * 1000)

    try:
        result, usage = await _transient_retry(call)()
    except ValueError as exc:
        return CaseResult(
            case_id=case.id,
            status="off_contract_output",
            expected_category=case.expected_category,
            category_match=False,
            latency_ms=elapsed_ms,
            error=str(exc),
        )
    except errors.APIError as exc:
        return CaseResult(
            case_id=case.id,
            status="error",
            expected_category=case.expected_category,
            latency_ms=elapsed_ms,
            error=f"{type(exc).__name__}: {exc}",
        )

    return CaseResult(
        case_id=case.id,
        status="scored",
        expected_category=case.expected_category,
        actual_category=result.category,
        category_match=result.category == case.expected_category,
        summary=result.summary,
        latency_ms=elapsed_ms,
        prompt_tokens=usage.prompt_token_count if usage else 0,
        output_tokens=usage.candidates_token_count if usage else 0,
    )


async def _judge(
    result: CaseResult,
    case: GoldenCase,
    judge_config: JudgeConfig,
    model: str,
    limiter: RateLimiter,
) -> None:
    async def call():
        await limiter.acquire()
        return await score_summary(judge_config, case, result.summary, model)

    try:
        result.summary_score = await _transient_retry(call)()
    except (ValueError, errors.APIError) as exc:
        result.error = f"judge failed: {type(exc).__name__}: {exc}"


async def _run_case(
    case: GoldenCase,
    prompt_config: PromptConfig,
    judge_config: JudgeConfig | None,
    model: str,
    semaphore: asyncio.Semaphore,
    limiter: RateLimiter,
) -> CaseResult:
    if case.expected_category not in prompt_config.categories:
        return CaseResult(
            case_id=case.id,
            status="out_of_contract",
            expected_category=case.expected_category,
        )

    async with semaphore:
        result = await _classify(prompt_config, case, model, limiter)
        if judge_config is not None and result.summary:
            await _judge(result, case, judge_config, model, limiter)
    return result


async def run_dataset(
    prompt_config: PromptConfig,
    dataset: GoldenDataset,
    judge_config: JudgeConfig | None = None,
    model: str = DEFAULT_MODEL,
    concurrency: int = DEFAULT_CONCURRENCY,
    requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
    limit: int | None = None,
) -> RunResult:
    """Evaluate every case in `dataset` against one prompt version.

    Cases whose expected category the prompt version cannot emit are marked
    out_of_contract without spending an API call, so an older prompt is never
    penalised for a category that postdates it. Passing `judge_config` as None
    skips summary scoring and halves the API calls.

    Returns:
        A RunResult holding one CaseResult per case evaluated.
    """
    cases = dataset.cases[:limit] if limit else dataset.cases
    semaphore = asyncio.Semaphore(concurrency)
    limiter = RateLimiter(requests_per_minute)

    results = await asyncio.gather(
        *(
            _run_case(case, prompt_config, judge_config, model, semaphore, limiter)
            for case in cases
        )
    )

    return RunResult(
        run_id=_run_id(prompt_config.version),
        prompt_version=prompt_config.version,
        model=model,
        dataset_version=dataset.version,
        judge_version=judge_config.version if judge_config else "none",
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        cases=list(results),
    )
