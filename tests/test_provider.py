import pytest

from promptwatch.groq import _strict
from promptwatch.provider import get_provider, names


def test_every_registered_provider_loads():
    for name in names():
        provider = get_provider(name)
        assert provider.name == name
        assert provider.default_model


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="unknown provider"):
        get_provider("openai")


def test_local_provider_is_unpaced_and_hosted_ones_are_not():
    assert get_provider("ollama").default_requests_per_minute == 0
    assert get_provider("gemini").default_requests_per_minute > 0
    assert get_provider("groq").default_requests_per_minute > 0


def test_strict_closes_nested_objects():
    schema = {
        "type": "object",
        "properties": {
            "category": {"type": "string"},
            "meta": {"type": "object", "properties": {"n": {"type": "integer"}}},
        },
        "required": ["category"],
    }
    closed = _strict(schema)
    assert closed["additionalProperties"] is False
    assert closed["properties"]["meta"]["additionalProperties"] is False
    assert closed["required"] == ["category"]
