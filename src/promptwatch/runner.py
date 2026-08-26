import asyncio
from datetime import UTC, datetime
from time import perf_counter

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from promptwatch.classifier import classify_email
from promptwatch.config import PromptConfig
from promptwatch.dataset import GoldenCase, GoldenDataset
from promptwatch.judge import JudgeConfig, score_summary
from promptwatch.models import ClassificationResult, EmailInput
from promptwatch.provider import (
    Completion,
    Provider,
    ProviderError,
    TransientError,
)
from promptwatch.results import CaseResult, RunResult

DEFAULT_CONCURRENCY = 5


class RateLimiter:
    """Paces provider calls to a fixed number per minute across all workers.

    Hosted free tiers reject bursts with 429 regardless of how few requests are
    in flight, so capping concurrency is not enough on its own. A `per_minute`
    of 0 disables pacing, which is what a local provider wants.
    """

    def __init__(self, per_minute: int) -> None:
        self._interval = 60 / per_minute if per_minute else 0.0
        self._lock = asyncio.Lock()
        self._next_slot = 0.0

    async def acquire(self) -> None:
        """Block until the next request is allowed to go out."""
        if not self._interval:
            return
        async with self._lock:
            now = perf_counter()
            wait = max(0.0, self._next_slot - now)
            self._next_slot = max(now, self._next_slot) + self._interval
        if wait:
            await asyncio.sleep(wait)


_transient_retry = retry(
    retry=retry_if_exception_type(TransientError),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=3, min=3, max=60),
    reraise=True,
)


def _run_id(prompt_version: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{prompt_version}"


async def _classify(
    provider: Provider,
    prompt_config: PromptConfig,
    case: GoldenCase,
    model: str,
    limiter: RateLimiter,
) -> CaseResult:
    email = EmailInput(subject=case.subject, body=case.body)
    elapsed_ms = 0

    async def call() -> tuple[ClassificationResult, Completion]:
        nonlocal elapsed_ms
        await limiter.acquire()
        started = perf_counter()
        try:
            return await classify_email(provider, prompt_config, email, model)
        finally:
            elapsed_ms = int((perf_counter() - started) * 1000)

    try:
        result, completion = await _transient_retry(call)()
    except ValueError as exc:
        return CaseResult(
            case_id=case.id,
            status="off_contract_output",
            expected_category=case.expected_category,
            category_match=False,
            latency_ms=elapsed_ms,
            error=str(exc),
        )
    except ProviderError as exc:
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
        prompt_tokens=completion.prompt_tokens,
        output_tokens=completion.output_tokens,
    )


async def _judge(
    provider: Provider,
    result: CaseResult,
    case: GoldenCase,
    judge_config: JudgeConfig,
    model: str,
    limiter: RateLimiter,
) -> None:
    summary = result.summary
    if summary is None:
        return

    async def call() -> int:
        await limiter.acquire()
        return await score_summary(provider, judge_config, case, summary, model)

    try:
        result.summary_score = await _transient_retry(call)()
    except (ValueError, ProviderError) as exc:
        result.error = f"judge failed: {type(exc).__name__}: {exc}"


async def _run_case(
    provider: Provider,
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
        result = await _classify(provider, prompt_config, case, model, limiter)
        if judge_config is not None and result.summary:
            await _judge(provider, result, case, judge_config, model, limiter)
    return result


async def run_dataset(
    provider: Provider,
    prompt_config: PromptConfig,
    dataset: GoldenDataset,
    judge_config: JudgeConfig | None = None,
    model: str | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    requests_per_minute: int | None = None,
    limit: int | None = None,
) -> RunResult:
    """Evaluate every case in `dataset` against one prompt version.

    Cases whose expected category the prompt version cannot emit are marked
    out_of_contract without spending a provider call, so an older prompt is
    never penalised for a category that postdates it. Passing `judge_config` as
    None skips summary scoring and halves the calls. A `requests_per_minute` of
    None takes the provider's own pacing; 0 disables pacing entirely.

    Returns:
        A RunResult holding one CaseResult per case evaluated.
    """
    model = model or provider.default_model
    if requests_per_minute is None:
        requests_per_minute = provider.default_requests_per_minute
    cases = dataset.cases[:limit] if limit else dataset.cases
    semaphore = asyncio.Semaphore(concurrency)
    limiter = RateLimiter(requests_per_minute)

    results = await asyncio.gather(
        *(
            _run_case(
                provider,
                case,
                prompt_config,
                judge_config,
                model,
                semaphore,
                limiter,
            )
            for case in cases
        )
    )

    return RunResult(
        run_id=_run_id(prompt_config.version),
        prompt_version=prompt_config.version,
        provider=provider.name,
        model=model,
        dataset_version=dataset.version,
        judge_version=judge_config.version if judge_config else "none",
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        cases=list(results),
    )
