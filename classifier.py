import json
import os
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


def classify_email(
    prompt_config: PromptConfig,
    email: EmailInput,
    model: str = "gemini-2.5-flash",
) -> ClassificationResult:
    """Classify an email into a Category and summarize it, via Gemini.

    Builds the request from `prompt_config`: its system prompt as the
    system instruction, its few-shot examples as prior user/model turns,
    and `email` as the final user turn. The model is constrained to return
    JSON matching the ClassificationResult schema.

    Returns:
        A validated ClassificationResult.

    Raises:
        RuntimeError: if GEMINI_API_KEY is not set.
        ValueError: if the model's response is not valid JSON, is missing
            required fields, or returns a category outside the allowed
            Category values.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")

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

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=prompt_config.system_prompt,
            response_mime_type="application/json",
            response_schema=_RESPONSE_SCHEMA,
        ),
    )

    try:
        data = json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model response was not valid JSON: {response.text!r}") from exc

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
