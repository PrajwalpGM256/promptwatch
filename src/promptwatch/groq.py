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

DEFAULT_MODEL = "openai/gpt-oss-20b"
BASE_URL = "https://api.groq.com/openai/v1"
REQUEST_TIMEOUT = 120.0


def api_key() -> str:
    """Read GROQ_API_KEY from the environment.

    Raises:
        ProviderError: if it is not set.
    """
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise ProviderError("GROQ_API_KEY is not set. Add it to your .env file.")
    return key


@lru_cache(maxsize=1)
def _cached_client(key: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=BASE_URL,
        timeout=REQUEST_TIMEOUT,
        headers={"Authorization": f"Bearer {key}"},
    )


def client() -> httpx.AsyncClient:
    """Return a cached HTTP client carrying the Groq bearer token."""
    return _cached_client(api_key())


def _messages(system_prompt: str, turns: list[Turn]) -> list[dict[str, str]]:
    return [{"role": "system", "content": system_prompt}] + [
        {"role": turn.role, "content": turn.text} for turn in turns
    ]


def _strict(schema: JsonSchema) -> JsonSchema:
    """Close every object in `schema` to extra keys, as strict mode demands."""
    if schema.get("type") != "object":
        return schema
    properties = {
        name: _strict(value) for name, value in schema.get("properties", {}).items()
    }
    return {**schema, "properties": properties, "additionalProperties": False}


class GroqProvider:
    name = "groq"
    default_model = DEFAULT_MODEL
    default_requests_per_minute = 20

    async def generate_json(
        self,
        system_prompt: str,
        turns: list[Turn],
        schema: JsonSchema,
        model: str,
        temperature: float | None = None,
    ) -> Completion:
        """Generate a response constrained to `schema`, as JSON text.

        Strict schema enforcement is only available on Groq's gpt-oss models;
        other models will reject the request.

        Raises:
            TransientError: on a 429 or a 5xx from Groq.
            ProviderError: if the key is unset, the request is rejected, or the
                response cannot be read.
        """
        payload: JsonSchema = {
            "model": model,
            "messages": _messages(system_prompt, turns),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "promptwatch_response",
                    "strict": True,
                    "schema": _strict(schema),
                },
            },
            "stream": False,
        }
        if temperature is not None:
            payload["temperature"] = temperature

        try:
            response = await client().post("/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 429 or status >= 500:
                raise TransientError(f"Groq returned {status}") from exc
            raise ProviderError(f"Groq returned {status}: {exc.response.text}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"{type(exc).__name__}: {exc}") from exc

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"Groq returned no choices: {data!r}") from exc
        if not content:
            raise ProviderError(f"Groq returned empty content: {data!r}")

        usage = data.get("usage", {})
        return Completion(
            text=content,
            prompt_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )


def provider() -> GroqProvider:
    """Return the Groq provider."""
    return GroqProvider()
