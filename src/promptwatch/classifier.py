import asyncio
import json
from collections.abc import Sequence

from promptwatch.config import PromptConfig
from promptwatch.models import ClassificationResult, EmailInput
from promptwatch.provider import Completion, JsonSchema, Provider, Turn


def _response_schema(categories: Sequence[str]) -> JsonSchema:
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


def _build_turns(prompt_config: PromptConfig, email: EmailInput) -> list[Turn]:
    """Replay the few-shot examples as prior turns, then append the email.

    Each example becomes a user turn holding the formatted email and an
    assistant turn holding the expected JSON, so the model continues an
    established pattern rather than following a described format.

    Returns:
        The turn list to send to the provider.
    """
    turns: list[Turn] = []
    for example in prompt_config.few_shot_examples:
        turns.append(
            Turn(role="user", text=_format_email(example.subject, example.body))
        )
        turns.append(
            Turn(
                role="assistant",
                text=json.dumps(
                    {"category": example.category, "summary": example.summary}
                ),
            )
        )
    turns.append(Turn(role="user", text=_format_email(email.subject, email.body)))
    return turns


def _parse_result(text: str, categories: Sequence[str]) -> ClassificationResult:
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


async def classify_email(
    provider: Provider,
    prompt_config: PromptConfig,
    email: EmailInput,
    model: str | None = None,
) -> tuple[ClassificationResult, Completion]:
    """Classify an email into a Category and summarize it.

    Builds the request from `prompt_config`: its system prompt as the system
    instruction and its few-shot examples as prior turns, with `email` as the
    final user turn. The model is constrained to JSON matching the
    ClassificationResult schema.

    Returns:
        A validated ClassificationResult and the raw Completion, whose token
        counts the eval runner records per case.

    Raises:
        ProviderError: if the provider is unreachable or misconfigured.
        ValueError: if the model's response is off-contract (see _parse_result).
    """
    completion = await provider.generate_json(
        system_prompt=prompt_config.system_prompt,
        turns=_build_turns(prompt_config, email),
        schema=_response_schema(prompt_config.categories),
        model=model or provider.default_model,
    )
    return _parse_result(completion.text, prompt_config.categories), completion


if __name__ == "__main__":
    import sys

    from promptwatch.provider import get_provider

    provider = get_provider(sys.argv[1] if len(sys.argv) > 1 else "ollama")
    config = PromptConfig.load("prompts/v2.yaml")
    email = EmailInput(
        subject="Interview invitation: Backend Engineer at Initech",
        body=(
            "Thanks for applying! We'd like to schedule a 30-minute phone "
            "screen for the Backend Engineer role. Are you available Thursday "
            "afternoon?"
        ),
    )
    result, completion = asyncio.run(classify_email(provider, config, email))
    print(f"{provider.name}/{provider.default_model}")
    print(result)
    print(f"tokens in {completion.prompt_tokens} out {completion.output_tokens}")
