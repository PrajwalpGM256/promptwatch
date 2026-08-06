import json
import os
from functools import lru_cache
from typing import get_args

from dotenv import load_dotenv
from google import genai
from google.genai import types

from promptwatch.config import PromptConfig
from promptwatch.models import Category, ClassificationResult, EmailInput

load_dotenv()

_ALLOWED_CATEGORIES = get_args(Category)

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(_ALLOWED_CATEGORIES)},
        "summary": {"type": "string"},
    },
    "required": ["category", "summary"],
}


def _format_email(subject: str, body: str) -> str:
    return f"Subject: {subject}\nBody: {body}"


def _build_contents(
    prompt_config: PromptConfig, email: EmailInput
) -> list[types.Content]:
    """Replay the few-shot examples as prior turns, then append the email.

    Each example becomes a user turn holding the formatted email and a model
    turn holding the expected JSON, so the model continues an established
    pattern rather than following a described format.

    Returns:
        The turn list to send as `contents`.
    """
    contents: list[types.Content] = []
    for example in prompt_config.few_shot_examples:
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part(text=_format_email(example.subject, example.body))],
            )
        )
        contents.append(
            types.Content(
                role="model",
                parts=[
                    types.Part(
                        text=json.dumps(
                            {"category": example.category, "summary": example.summary}
                        )
                    )
                ],
            )
        )
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=_format_email(email.subject, email.body))],
        )
    )
    return contents


def _parse_result(text: str) -> ClassificationResult:
    """Parse the model's JSON response and validate it against the contract.

    Returns:
        A validated ClassificationResult.

    Raises:
        ValueError: if `text` is not valid JSON, omits a summary, or names a
            category outside the allowed Category values.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model response was not valid JSON: {text!r}") from exc

    category = data.get("category")
    summary = data.get("summary")
    if category not in _ALLOWED_CATEGORIES:
        raise ValueError(
            f"Model returned an off-contract category {category!r}; "
            f"expected one of {_ALLOWED_CATEGORIES}"
        )
    if not summary:
        raise ValueError(f"Model response is missing a summary: {data!r}")

    return ClassificationResult(category=category, summary=summary)


@lru_cache(maxsize=1)
def _get_client(api_key: str) -> genai.Client:
    """Return a cached Gemini client so connections are reused across calls."""
    return genai.Client(api_key=api_key)


def classify_email(
    prompt_config: PromptConfig,
    email: EmailInput,
    model: str = "gemini-2.5-flash",
) -> ClassificationResult:
    """Classify an email into a Category and summarize it, via Gemini.

    Builds the request from `prompt_config`: its system prompt as the system
    instruction and its few-shot examples as prior turns, with `email` as the
    final user turn. The model is constrained to JSON matching the
    ClassificationResult schema.

    Returns:
        A validated ClassificationResult.

    Raises:
        RuntimeError: if GEMINI_API_KEY is not set.
        ValueError: if the model's response is off-contract (see _parse_result).
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")

    client = _get_client(api_key)
    response = client.models.generate_content(
        model=model,
        contents=_build_contents(prompt_config, email),
        config=types.GenerateContentConfig(
            system_instruction=prompt_config.system_prompt,
            response_mime_type="application/json",
            response_schema=_RESPONSE_SCHEMA,
        ),
    )
    return _parse_result(response.text)


if __name__ == "__main__":
    config = PromptConfig.load("prompts/v2.yaml")
    email = EmailInput(
        subject="Interview invitation: Backend Engineer at Initech",
        body=(
            "Thanks for applying! We'd like to schedule a 30-minute phone "
            "screen for the Backend Engineer role. Are you available Thursday "
            "afternoon?"
        ),
    )
    result = classify_email(config, email)
    print(result)
