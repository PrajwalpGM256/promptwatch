import os
from functools import lru_cache

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from promptwatch.provider import (
    Completion,
    JsonSchema,
    ProviderError,
    TransientError,
    Turn,
)

load_dotenv()

DEFAULT_MODEL = "gemini-3.5-flash-lite"


def api_key() -> str:
    """Read GEMINI_API_KEY from the environment.

    Raises:
        ProviderError: if it is not set.
    """
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ProviderError("GEMINI_API_KEY is not set. Add it to your .env file.")
    return key


@lru_cache(maxsize=1)
def _cached_client(key: str) -> genai.Client:
    return genai.Client(api_key=key)


def client() -> genai.Client:
    """Return a cached Gemini client so connections are reused across calls."""
    return _cached_client(api_key())


def _contents(turns: list[Turn]) -> list[types.Content]:
    return [
        types.Content(
            role="model" if turn.role == "assistant" else "user",
            parts=[types.Part(text=turn.text)],
        )
        for turn in turns
    ]


class GeminiProvider:
    name = "gemini"
    default_model = DEFAULT_MODEL
    default_requests_per_minute = 12
    default_concurrency = 5

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
            TransientError: on a 429 or a server-side failure.
            ProviderError: on any other API failure, or if the key is unset.
        """
        try:
            response = await client().aio.models.generate_content(
                model=model,
                contents=_contents(turns),
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=temperature,
                ),
            )
        except errors.ServerError as exc:
            raise TransientError(str(exc)) from exc
        except errors.ClientError as exc:
            if exc.code == 429:
                raise TransientError(str(exc)) from exc
            raise ProviderError(f"{type(exc).__name__}: {exc}") from exc
        except errors.APIError as exc:
            raise ProviderError(f"{type(exc).__name__}: {exc}") from exc

        if response.text is None:
            raise ProviderError(
                "Gemini returned no text; the response was empty or blocked"
            )

        usage = response.usage_metadata
        return Completion(
            text=response.text,
            prompt_tokens=(usage.prompt_token_count or 0) if usage else 0,
            output_tokens=(usage.candidates_token_count or 0) if usage else 0,
        )


def provider() -> GeminiProvider:
    """Return the Gemini provider."""
    return GeminiProvider()
