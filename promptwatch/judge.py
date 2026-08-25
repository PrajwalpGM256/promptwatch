import json
from pathlib import Path

import yaml
from google.genai import types
from pydantic import BaseModel, ConfigDict

from promptwatch.dataset import GoldenCase
from promptwatch.gemini import DEFAULT_MODEL, client

_SCORE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "integer"}},
    "required": ["score"],
}


class JudgeConfig(BaseModel):
    """A versioned judge prompt.

    Kept separate from PromptConfig: the judge declares no categories and no
    few-shot examples, and editing it silently reprices every historical score,
    so runs record which version graded them.
    """

    model_config = ConfigDict(extra="forbid")

    version: str
    timestamp: str
    system_prompt: str

    @classmethod
    def load(cls, path: str | Path) -> "JudgeConfig":
        """Load a judge prompt from YAML.

        Raises:
            ValueError: if the file is missing a required key or declares an
                unknown one.
        """
        with open(path, encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))


def _grading_request(case: GoldenCase, summary: str) -> str:
    facts = "\n".join(f"- {fact}" for fact in case.must_mention)
    return (
        f"EMAIL SUBJECT: {case.subject}\n"
        f"EMAIL BODY:\n{case.body}\n\n"
        f"KEY FACTS THE SUMMARY MUST CONVEY:\n{facts}\n\n"
        f"SUMMARY TO GRADE:\n{summary}"
    )


async def score_summary(
    judge_config: JudgeConfig,
    case: GoldenCase,
    summary: str,
    model: str = DEFAULT_MODEL,
) -> int:
    """Grade a generated summary against a golden case, 1 to 5.

    Runs at temperature 0 with a constrained schema, and grounds the score in
    the case's hand-labelled `must_mention` facts rather than open judgement.

    Returns:
        An integer score from 1 to 5.

    Raises:
        RuntimeError: if GEMINI_API_KEY is not set.
        ValueError: if the judge returns anything other than an integer 1-5.
    """
    response = await client().aio.models.generate_content(
        model=model,
        contents=_grading_request(case, summary),
        config=types.GenerateContentConfig(
            system_instruction=judge_config.system_prompt,
            response_mime_type="application/json",
            response_schema=_SCORE_SCHEMA,
            temperature=0,
        ),
    )

    try:
        data = json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"judge response was not valid JSON: {response.text!r}"
        ) from exc

    score = data.get("score")
    if not isinstance(score, int) or not 1 <= score <= 5:
        raise ValueError(f"judge returned an off-contract score: {data!r}")
    return score
