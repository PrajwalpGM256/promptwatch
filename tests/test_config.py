import pytest
import yaml
from pydantic import ValidationError

from promptwatch.config import FewShotExample, PromptConfig
from promptwatch.judge import JudgeConfig

EXAMPLE = {
    "subject": "Thanks for applying",
    "body": "We received your application.",
    "category": "application_ack",
    "summary": "The company confirms receipt.",
}


def config_with(**overrides) -> PromptConfig:
    base = {
        "version": "v1",
        "timestamp": "2026-08-25T00:00:00Z",
        "categories": ["application_ack", "misc"],
        "system_prompt": "Classify the email.",
        "few_shot_examples": [EXAMPLE],
    }
    return PromptConfig(**{**base, **overrides})


def test_valid_config_round_trips():
    assert config_with().categories == ["application_ack", "misc"]


def test_unknown_key_rejected_on_prompt_config():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        config_with(temperature=0.5)


def test_unknown_key_rejected_on_few_shot_example():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        FewShotExample(**EXAMPLE, weight=2)


def test_example_using_undeclared_category_rejected():
    with pytest.raises(ValidationError, match="does not declare"):
        config_with(categories=["misc"])


def test_shipped_prompts_declare_their_own_categories():
    v1 = PromptConfig.load("prompts/v1.yaml")
    v2 = PromptConfig.load("prompts/v2.yaml")
    assert "interview_invite" not in v1.categories
    assert "interview_invite" in v2.categories
    assert len(v1.categories) == 5
    assert len(v2.categories) == 6


def test_prompt_yaml_is_read_as_utf8(tmp_path):
    path = tmp_path / "v9.yaml"
    payload = {
        "version": "v9",
        "timestamp": "2026-08-25T00:00:00Z",
        "categories": ["misc"],
        "system_prompt": "Classify the email — carefully.",
        "few_shot_examples": [],
    }
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    assert "—" in PromptConfig.load(path).system_prompt


def test_judge_config_loads_and_rejects_unknown_keys():
    judge = JudgeConfig.load("prompts/judge_v1.yaml")
    assert judge.version == "v1"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        JudgeConfig(
            version="v1",
            timestamp="t",
            system_prompt="grade it",
            temperature=0,
        )
