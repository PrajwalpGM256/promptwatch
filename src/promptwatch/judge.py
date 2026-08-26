import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from promptwatch.dataset import GoldenCase
from promptwatch.provider import Provider, Turn

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
    provider: Provider,
    judge_config: JudgeConfig,
    case: GoldenCase,
    summary: str,
    model: str | None = None,
) -> int:
    """Grade a generated summary against a golden case, 1 to 5.

    Runs at temperature 0 with a constrained schema, and grounds the score in
    the case's hand-labelled `must_mention` facts rather than open judgement.

    Returns:
        An integer score from 1 to 5.

    Raises:
        ProviderError: if the provider is unreachable or misconfigured.
        ValueError: if the judge returns anything other than an integer 1-5.
    """
    completion = await provider.generate_json(
        system_prompt=judge_config.system_prompt,
        turns=[Turn(role="user", text=_grading_request(case, summary))],
        schema=_SCORE_SCHEMA,
        model=model or provider.default_model,
        temperature=0,
    )

    try:
        data = json.loads(completion.text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"judge response was not valid JSON: {completion.text!r}"
        ) from exc

    score = data.get("score")
    if not isinstance(score, int) or not 1 <= score <= 5:
        raise ValueError(f"judge returned an off-contract score: {data!r}")
    return score
