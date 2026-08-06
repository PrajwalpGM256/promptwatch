from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator

from promptwatch.models import Category


class FewShotExample(BaseModel):
    subject: str
    body: str
    category: Category
    summary: str


class PromptConfig(BaseModel):
    """A single versioned prompt, including the category set it may return.

    `categories` is declared per version so a version stays reproducible:
    widening the `Category` type in code cannot retroactively change what an
    older prompt version is allowed to emit.
    """

    version: str
    timestamp: str
    categories: list[Category]
    system_prompt: str
    few_shot_examples: list[FewShotExample] = []

    @model_validator(mode="after")
    def _examples_use_declared_categories(self) -> "PromptConfig":
        undeclared = {e.category for e in self.few_shot_examples} - set(self.categories)
        if undeclared:
            raise ValueError(
                f"few-shot examples use categories this version does not declare: "
                f"{sorted(undeclared)}"
            )
        return self

    @classmethod
    def load(cls, path: str | Path) -> "PromptConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)