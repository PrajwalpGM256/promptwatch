from pathlib import Path

import yaml
from pydantic import BaseModel

from models import Category


class FewShotExample(BaseModel):
    subject: str
    body: str
    category: Category
    summary: str


class PromptConfig(BaseModel):
    version: str
    timestamp: str
    system_prompt: str
    few_shot_examples: list[FewShotExample] = []

    @classmethod
    def load(cls, path: str | Path) -> "PromptConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)