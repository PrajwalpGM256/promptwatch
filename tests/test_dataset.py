import json

import pytest
from conftest import ALL_CATEGORIES
from pydantic import ValidationError

from promptwatch.dataset import GoldenCase, GoldenDataset


def case_with(fields: dict, **overrides) -> GoldenCase:
    return GoldenCase(**{**fields, **overrides})


def test_valid_case_round_trips(golden_case):
    assert golden_case.id == "gc-001"
    assert golden_case.tags == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"id": "001"},
        {"id": "gc-1"},
        {"body": ""},
        {"must_mention": ["only one"]},
        {"must_mention": ["a", "b", "c", "d", "e"]},
        {"must_mention": ["x" * 61, "y"]},
        {"notes": ""},
        {"tags": ["mixed-language"]},
        {"difficulty": "trivial"},
    ],
)
def test_case_validators_reject(valid_case_fields, overrides):
    with pytest.raises(ValidationError):
        case_with(valid_case_fields, **overrides)


def test_blank_subject_is_allowed(valid_case_fields):
    assert case_with(valid_case_fields, subject="").subject == ""


def test_unredacted_address_rejected(valid_case_fields):
    with pytest.raises(ValidationError, match="unredacted email domain"):
        case_with(valid_case_fields, body="mail me at someone@gmail.com")


def test_sentence_ending_period_is_not_part_of_the_domain(valid_case_fields):
    case = case_with(valid_case_fields, body="Reply to noreply@example.com.")
    assert case.body.endswith("example.com.")


def dataset_with(cases: list[GoldenCase], categories=None) -> GoldenDataset:
    return GoldenDataset(
        version="v1",
        timestamp="2026-08-25T00:00:00Z",
        categories=categories or ALL_CATEGORIES,
        cases=cases,
    )


def test_duplicate_ids_rejected(golden_case):
    with pytest.raises(ValidationError, match="duplicate case ids"):
        dataset_with([golden_case, golden_case])


def test_undeclared_category_rejected(golden_case):
    with pytest.raises(ValidationError, match="does not declare"):
        dataset_with([golden_case], categories=["misc"])


def test_check_balance_reports_every_gap(golden_case):
    with pytest.raises(ValueError) as excinfo:
        dataset_with([golden_case]).check_balance()
    message = str(excinfo.value)
    assert "interview_invite is 1 (need 6)" in message
    assert "misc is 0 (need 6)" in message


def test_check_balance_rejects_unsatisfiable_share(golden_case):
    with pytest.raises(ValueError, match="unsatisfiable"):
        dataset_with([golden_case]).check_balance(max_share=0.1)


def test_check_balance_passes_when_even(valid_case_fields):
    cases = [
        case_with(
            valid_case_fields,
            id=f"gc-{i:03d}",
            expected_category=ALL_CATEGORIES[i % 6],
        )
        for i in range(1, 37)
    ]
    dataset_with(cases).check_balance()


def test_load_rejects_version_filename_mismatch(tmp_path, golden_case):
    payload = json.loads(dataset_with([golden_case]).model_dump_json())
    path = tmp_path / "golden_v2.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match filename"):
        GoldenDataset.load(path)


def test_load_accepts_matching_filename(tmp_path, golden_case):
    payload = json.loads(dataset_with([golden_case]).model_dump_json())
    path = tmp_path / "golden_v1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert len(GoldenDataset.load(path).cases) == 1


def test_shipped_dataset_loads_and_is_balanced():
    dataset = GoldenDataset.load("datasets/golden_v1.json")
    dataset.check_balance()
    assert len(dataset.cases) == 94
