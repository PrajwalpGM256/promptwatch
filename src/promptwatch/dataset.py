import json
import re
from collections import Counter
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, model_validator

from promptwatch.models import Category

Tag = Literal["ambiguous", "terse", "typos", "mixed_language", "sarcastic"]
Difficulty = Literal["easy", "medium", "hard"]
KeyFact = Annotated[str, StringConstraints(min_length=1, max_length=60)]

_ALLOWED_EMAIL_DOMAINS = frozenset({"example.com", "example.org"})
_EMAIL_DOMAIN = re.compile(r"[\w.+-]+@([\w-]+(?:\.[\w-]+)+)")


class GoldenCase(BaseModel):

    id: str = Field(pattern=r"^gc-\d{3}$")
    subject: str
    body: str = Field(min_length=1)
    expected_category: Category
    must_mention: list[KeyFact] = Field(min_length=2, max_length=4)
    difficulty: Difficulty
    tags: list[Tag] = []
    notes: str = Field(min_length=1)

    @model_validator(mode="after")
    def _addresses_are_redacted(self) -> "GoldenCase":
        leaked = {
            domain
            for domain in _EMAIL_DOMAIN.findall(f"{self.subject}\n{self.body}")
            if domain not in _ALLOWED_EMAIL_DOMAINS
        }
        if leaked:
            raise ValueError(
                f"unredacted email domain(s) {sorted(leaked)}: "
                f"replace addresses with *@example.com"
            )
        return self


class GoldenDataset(BaseModel):

    version: str
    timestamp: str
    categories: list[Category]
    cases: list[GoldenCase]

    @model_validator(mode="after")
    def _cases_are_consistent(self) -> "GoldenDataset":
        ids = [case.id for case in self.cases]
        duplicated = {id_ for id_ in ids if ids.count(id_) > 1}
        if duplicated:
            raise ValueError(f"duplicate case ids: {sorted(duplicated)}")

        labeled = {case.expected_category for case in self.cases}
        undeclared = labeled - set(self.categories)
        if undeclared:
            raise ValueError(
                "cases use categories this dataset does not declare: "
                f"{sorted(undeclared)}"
            )
        return self

    def by_id(self) -> dict[str, GoldenCase]:
        """Cases keyed by id, for joining against a run's results."""
        return {case.id: case for case in self.cases}

    def check_balance(self, min_per_category: int = 6, max_share: float = 0.30) -> None:
        """Raise unless the corpus is fit to evaluate against.

        Deliberately not a load-time validator: the file is legitimately
        unbalanced while labeling is still in progress.

        Raises:
            ValueError: if `max_share` is unsatisfiable for this many categories,
                or naming every category that is short or over-represented, so
                the message doubles as a labeling worklist.
        """
        even_share = 1 / len(self.categories)
        if max_share < even_share:
            raise ValueError(
                f"max_share={max_share} is unsatisfiable across "
                f"{len(self.categories)} categories (needs >= {even_share:.2f})"
            )

        counts = Counter(case.expected_category for case in self.cases)
        total = len(self.cases)
        max_allowed = int(total * max_share)
        gaps = []
        for category in self.categories:
            count = counts[category]
            if count < min_per_category:
                gaps.append(f"{category} is {count} (need {min_per_category})")
            elif count > max_allowed:
                gaps.append(f"{category} is {count}/{total} (max {max_allowed})")
        if gaps:
            raise ValueError("dataset is not eval-ready: " + "; ".join(gaps))

    @classmethod
    def load(cls, path: str | Path) -> "GoldenDataset":
        """Load and validate a dataset, checking `version` against the filename.

        Raises:
            ValueError: if the file is invalid or its version does not match
                the filename stem (e.g. golden_v2.json still declaring v1).
        """
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        dataset = cls(**data)
        expected_version = path.stem.removeprefix("golden_")
        if dataset.version != expected_version:
            raise ValueError(
                f"version {dataset.version!r} does not match filename {path.name!r}"
            )
        return dataset
