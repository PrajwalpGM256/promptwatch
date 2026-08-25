import json

from google.genai import types

from promptwatch.config import PromptConfig
from promptwatch.gemini import DEFAULT_MODEL, client
from promptwatch.models import ClassificationResult, EmailInput


def _response_schema(categories: list[str]) -> dict:
    """Build the JSON Schema constraining the model to `categories`.

    Derived from the prompt version rather than the `Category` type, so a
    version's allowed outputs cannot drift when the type is widened.
    """
    return {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": list(categories)},
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


def _parse_result(text: str, categories: list[str]) -> ClassificationResult:
    """Parse the model's JSON response and validate it against `categories`.

    Returns:
        A validated ClassificationResult.

    Raises:
        ValueError: if `text` is not valid JSON, omits a summary, or names a
            category this prompt version does not declare.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model response was not valid JSON: {text!r}") from exc

    category = data.get("category")
    summary = data.get("summary")
    if category not in categories:
        raise ValueError(
            f"Model returned an off-contract category {category!r}; "
            f"expected one of {categories}"
        )
    if not summary:
        raise ValueError(f"Model response is missing a summary: {data!r}")

    return ClassificationResult(category=category, summary=summary)


def _request(prompt_config: PromptConfig, email: EmailInput, model: str) -> dict:
    return {
        "model": model,
        "contents": _build_contents(prompt_config, email),
        "config": types.GenerateContentConfig(
            system_instruction=prompt_config.system_prompt,
            response_mime_type="application/json",
            response_schema=_response_schema(prompt_config.categories),
        ),
    }


def classify_email(
    prompt_config: PromptConfig,
    email: EmailInput,
    model: str = DEFAULT_MODEL,
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
    response = client().models.generate_content(
        **_request(prompt_config, email, model)
    )
    return _parse_result(response.text, prompt_config.categories)


async def classify_email_async(
    prompt_config: PromptConfig,
    email: EmailInput,
    model: str = DEFAULT_MODEL,
) -> tuple[ClassificationResult, types.GenerateContentResponseUsageMetadata | None]:
    """Async twin of `classify_email`, also returning token usage.

    Returns:
        The validated ClassificationResult and the response's usage metadata,
        which the eval runner records per case.

    Raises:
        RuntimeError: if GEMINI_API_KEY is not set.
        ValueError: if the model's response is off-contract (see _parse_result).
    """
    response = await client().aio.models.generate_content(
        **_request(prompt_config, email, model)
    )
    result = _parse_result(response.text, prompt_config.categories)
    return result, response.usage_metadata


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
