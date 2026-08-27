import asyncio
from datetime import UTC, datetime
from time import perf_counter

from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from promptwatch.classifier import classify_email
from promptwatch.config import PromptConfig
from promptwatch.dataset import GoldenCase, GoldenDataset
from promptwatch.judge import DEFAULT_JUDGE_PROVIDER, JudgeConfig, score_summary
from promptwatch.models import ClassificationResult, EmailInput
from promptwatch.provider import (
    Completion,
    Provider,
    ProviderError,
    TransientError,
    get_provider,
)
from promptwatch.results import CaseResult, RunResult


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


_backoff = wait_exponential(multiplier=3, min=3, max=60)


def _wait(state: RetryCallState) -> float:
    """Wait as long as the provider asked, or back off exponentially."""
    exception = state.outcome.exception() if state.outcome else None
    if isinstance(exception, TransientError) and exception.retry_after:
        return exception.retry_after
    return _backoff(state)


_transient_retry = retry(
    retry=retry_if_exception_type(TransientError),
    stop=stop_after_attempt(5),
    wait=_wait,
    reraise=True,
)


def _run_id(prompt_version: str, started: datetime) -> str:
    return f"{started.strftime('%Y%m%dT%H%M%S')}-{prompt_version}"


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
    judge_provider: Provider,
    judge_model: str,
    semaphore: asyncio.Semaphore,
    limiter: RateLimiter,
    judge_limiter: RateLimiter,
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
            await _judge(
                judge_provider,
                result,
                case,
                judge_config,
                judge_model,
                judge_limiter,
            )
    return result


async def run_dataset(
    provider: Provider,
    prompt_config: PromptConfig,
    dataset: GoldenDataset,
    judge_config: JudgeConfig | None = None,
    model: str | None = None,
    judge_provider: Provider | None = None,
    judge_model: str | None = None,
    concurrency: int | None = None,
    requests_per_minute: int | None = None,
    limit: int | None = None,
) -> RunResult:
    """Evaluate every case in `dataset` against one prompt version.

    Cases whose expected category the prompt version cannot emit are marked
    out_of_contract without spending a provider call, so an older prompt is
    never penalised for a category that postdates it. Passing `judge_config` as
    None skips summary scoring and halves the calls. A `requests_per_minute` of
    None takes the provider's own pacing; 0 disables pacing entirely. The
    judge is paced from its own backend's limit when it differs from the one
    under test, and shares the same limiter when it does not, because one
    account is one quota however many call sites draw on it.

    The judge runs on `judge_provider`, which is deliberately independent of
    the backend under test: a grader that changes with the thing it grades
    makes summary scores incomparable between runs.

    Returns:
        A RunResult holding one CaseResult per case evaluated.
    """
    model = model or provider.default_model
    concurrency = concurrency or provider.default_concurrency
    judge_provider = judge_provider or get_provider(DEFAULT_JUDGE_PROVIDER)
    judge_model = judge_model or judge_provider.default_model
    if requests_per_minute is None:
        requests_per_minute = provider.default_requests_per_minute
    cases = dataset.cases[:limit] if limit else dataset.cases
    started = datetime.now(UTC)
    semaphore = asyncio.Semaphore(concurrency)
    limiter = RateLimiter(requests_per_minute)
    judge_limiter = (
        limiter
        if judge_provider.name == provider.name
        else RateLimiter(judge_provider.default_requests_per_minute)
    )

    results = await asyncio.gather(
        *(
            _run_case(
                provider,
                case,
                prompt_config,
                judge_config,
                model,
                judge_provider,
                judge_model,
                semaphore,
                limiter,
                judge_limiter,
            )
            for case in cases
        )
    )

    return RunResult(
        run_id=_run_id(prompt_config.version, started),
        prompt_version=prompt_config.version,
        provider=provider.name,
        model=model,
        dataset_version=dataset.version,
        judge_version=judge_config.version if judge_config else "none",
        judge_provider=judge_provider.name if judge_config else "none",
        judge_model=judge_model if judge_config else "none",
        started_at=started.isoformat(timespec="seconds"),
        cases=list(results),
    )
