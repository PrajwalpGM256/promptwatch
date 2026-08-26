from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from promptwatch.models import Category


class FewShotExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

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
        """Load a versioned prompt from YAML.

        Raises:
            ValueError: if the file is missing a required key, declares an
                unknown one, or uses a category its few-shot examples do not.
        """
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)