import os
from functools import lru_cache

import httpx
from dotenv import load_dotenv

from promptwatch.provider import (
    Completion,
    JsonSchema,
    ProviderError,
    TransientError,
    Turn,
)

load_dotenv()

DEFAULT_MODEL = "gemma3:4b"
DEFAULT_HOST = "http://localhost:11434"
REQUEST_TIMEOUT = 300.0


def host() -> str:
    """Return the Ollama base URL, from OLLAMA_HOST or the local default."""
    return os.environ.get("OLLAMA_HOST", DEFAULT_HOST).rstrip("/")


@lru_cache(maxsize=1)
def _cached_client(base_url: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=base_url, timeout=REQUEST_TIMEOUT)


def client() -> httpx.AsyncClient:
    """Return a cached HTTP client bound to the configured Ollama host."""
    return _cached_client(host())


def _messages(system_prompt: str, turns: list[Turn]) -> list[dict[str, str]]:
    return [{"role": "system", "content": system_prompt}] + [
        {"role": turn.role, "content": turn.text} for turn in turns
    ]


class OllamaProvider:
    name = "ollama"
    default_model = DEFAULT_MODEL
    default_requests_per_minute = 0

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
            TransientError: on a 429 or a 5xx from the Ollama server.
            ProviderError: if Ollama is unreachable, the model is not pulled,
                or the response cannot be read.
        """
        payload: JsonSchema = {
            "model": model,
            "messages": _messages(system_prompt, turns),
            "format": schema,
            "stream": False,
        }
        if temperature is not None:
            payload["options"] = {"temperature": temperature}

        try:
            response = await client().post("/api/chat", json=payload)
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise ProviderError(
                f"cannot reach Ollama at {host()}. Is `ollama serve` running?"
            ) from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 429 or status >= 500:
                raise TransientError(f"Ollama returned {status}") from exc
            if status == 404:
                raise ProviderError(
                    f"Ollama has no model {model!r}. Run `ollama pull {model}`."
                ) from exc
            raise ProviderError(
                f"Ollama returned {status}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"{type(exc).__name__}: {exc}") from exc

        data = response.json()
        content = data.get("message", {}).get("content")
        if not content:
            raise ProviderError(f"Ollama returned no message content: {data!r}")

        return Completion(
            text=content,
            prompt_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )


def provider() -> OllamaProvider:
    """Return the Ollama provider."""
    return OllamaProvider()
