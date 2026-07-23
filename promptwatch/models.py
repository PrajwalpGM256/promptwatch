from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["application_ack", "rejection", "job_alert", "newsletter", "misc"]


@dataclass
class EmailInput:
    subject: str
    body: str


class ClassificationResult(BaseModel):
    category: Category
    summary: str = Field(min_length=1, max_length=200)