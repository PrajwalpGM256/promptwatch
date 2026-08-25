import json

import pytest

from classifier import _build_contents, _parse_result, _response_schema
from promptwatch.config import PromptConfig
from promptwatch.models import EmailInput

CATEGORIES = ["application_ack", "rejection", "misc"]


def test_response_schema_restricts_the_enum():
    schema = _response_schema(CATEGORIES)
    assert schema["properties"]["category"]["enum"] == CATEGORIES
    assert schema["required"] == ["category", "summary"]


def test_response_schema_follows_the_prompt_not_the_type():
    narrow = _response_schema(["misc"])
    assert narrow["properties"]["category"]["enum"] == ["misc"]
    assert "interview_invite" not in narrow["properties"]["category"]["enum"]


def test_parse_result_accepts_a_declared_category():
    result = _parse_result('{"category": "misc", "summary": "A note."}', CATEGORIES)
    assert result.category == "misc"
    assert result.summary == "A note."


def test_parse_result_rejects_invalid_json():
    with pytest.raises(ValueError, match="not valid JSON"):
        _parse_result("not json", CATEGORIES)


def test_parse_result_rejects_undeclared_category():
    payload = '{"category": "interview_invite", "summary": "x"}'
    with pytest.raises(ValueError, match="off-contract category"):
        _parse_result(payload, CATEGORIES)


def test_parse_result_rejects_missing_summary():
    with pytest.raises(ValueError, match="missing a summary"):
        _parse_result('{"category": "misc"}', CATEGORIES)


def test_parse_result_rejects_overlong_summary():
    payload = json.dumps({"category": "misc", "summary": "x" * 201})
    with pytest.raises(Exception, match="at most 200 characters"):
        _parse_result(payload, CATEGORIES)


def test_build_contents_alternates_turns_and_ends_with_the_email():
    config = PromptConfig.load("prompts/v2.yaml")
    email = EmailInput(subject="Subject here", body="Body here")
    turns = _build_contents(config, email)

    expected = len(config.few_shot_examples) * 2 + 1
    assert len(turns) == expected
    assert [t.role for t in turns[:-1]] == ["user", "model"] * len(
        config.few_shot_examples
    )
    assert turns[-1].role == "user"
    assert "Subject here" in turns[-1].parts[0].text


def test_build_contents_renders_examples_as_json_answers():
    config = PromptConfig.load("prompts/v2.yaml")
    turns = _build_contents(config, EmailInput(subject="s", body="b"))
    answer = json.loads(turns[1].parts[0].text)
    assert answer["category"] == config.few_shot_examples[0].category
    assert answer["summary"] == config.few_shot_examples[0].summary
