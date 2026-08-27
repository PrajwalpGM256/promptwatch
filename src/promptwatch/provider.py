import importlib
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel

Role = Literal["user", "assistant"]
JsonSchema = dict[str, Any]

_IMPLEMENTATIONS = {
    "gemini": "promptwatch.gemini",
    "ollama": "promptwatch.ollama",
    "groq": "promptwatch.groq",
}


class Turn(BaseModel):
    role: Role
    text: str


class Completion(BaseModel):
    text: str
    prompt_tokens: int = 0
    output_tokens: int = 0


class ProviderError(Exception):
    """A provider was unreachable or returned an unusable response."""


class TransientError(ProviderError):
    """A rate limit or server-side failure worth retrying.

    `retry_after` carries the server's own Retry-After hint in seconds when it
    sends one. A token-bucket limit refills on the server's schedule, not ours,
    so guessing a backoff wastes attempts.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class Provider(Protocol):
    name: str
    default_model: str
    default_requests_per_minute: int
    default_concurrency: int

    async def generate_json(
        self,
        system_prompt: str,
        turns: list[Turn],
        schema: JsonSchema,
        model: str,
        temperature: float | None = None,
    ) -> Completion:
        """Generate a response constrained to `schema`, as JSON text.

        Raises:
            TransientError: on rate limiting or a server-side failure.
            ProviderError: on any other transport or credential failure.
        """
        ...


def names() -> list[str]:
    """Return the registered provider names."""
    return sorted(_IMPLEMENTATIONS)


def get_provider(name: str) -> Provider:
    """Load the provider registered under `name`.

    Implementations are imported lazily, so a missing SDK or API key for one
    provider cannot break the others.

    Raises:
        ValueError: if `name` is not a registered provider.
    """
    try:
        module = _IMPLEMENTATIONS[name]
    except KeyError:
        raise ValueError(
            f"unknown provider {name!r}; expected one of {names()}"
        ) from None
    return cast(Provider, importlib.import_module(module).provider())
