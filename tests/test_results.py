import sqlite3

import pytest
from conftest import make_case_result, make_run

from promptwatch.results import connect, latest_run, load_run, save_run


@pytest.fixture
def connection(tmp_path):
    return connect(tmp_path / "runs.db")


def test_schema_is_created(connection):
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert tables == {"runs", "case_results"}


def test_run_round_trips(connection):
    run = make_run("r1", [make_case_result("gc-001", "misc", "misc")])
    save_run(connection, run)
    loaded = load_run(connection, "r1")
    assert loaded.run_id == "r1"
    assert loaded.cases[0].case_id == "gc-001"
    assert loaded.cases[0].category_match is True


def test_load_run_raises_for_unknown_id(connection):
    with pytest.raises(KeyError):
        load_run(connection, "nope")


def test_duplicate_run_id_rejected(connection):
    run = make_run("r1", [])
    save_run(connection, run)
    with pytest.raises(sqlite3.IntegrityError):
        save_run(connection, run)


@pytest.mark.parametrize(
    "columns,values",
    [
        (
            "run_id,case_id,status,expected_category,latency_ms,prompt_tokens,output_tokens",
            ("r1", "c1", "bogus", "misc", 0, 0, 0),
        ),
        (
            "run_id,case_id,status,expected_category,summary_score,latency_ms,"
            "prompt_tokens,output_tokens",
            ("r1", "c2", "scored", "misc", 9, 0, 0, 0),
        ),
        (
            "run_id,case_id,status,expected_category,latency_ms,prompt_tokens,output_tokens",
            ("missing", "c3", "scored", "misc", 0, 0, 0),
        ),
        (
            "run_id,case_id,status,expected_category,prompt_tokens,output_tokens",
            ("r1", "c4", "scored", "misc", 0, 0),
        ),
    ],
    ids=["bad status", "score out of range", "orphan case", "null latency"],
)
def test_database_constraints_reject(connection, columns, values):
    save_run(connection, make_run("r1", []))
    placeholders = ", ".join("?" * len(values))
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            f"INSERT INTO case_results ({columns}) VALUES ({placeholders})", values
        )


def test_deleting_a_run_cascades_to_its_cases(connection):
    save_run(connection, make_run("r1", [make_case_result("gc-001", "misc", "misc")]))
    assert connection.execute("SELECT COUNT(*) FROM case_results").fetchone()[0] == 1
    connection.execute("DELETE FROM runs WHERE run_id = 'r1'")
    connection.commit()
    assert connection.execute("SELECT COUNT(*) FROM case_results").fetchone()[0] == 0


def test_accuracy_excludes_unscored_cases():
    run = make_run(
        "r1",
        [
            make_case_result("a", "misc", "misc"),
            make_case_result("b", "misc", "rejection"),
            make_case_result("c", "interview_invite", None, status="out_of_contract"),
            make_case_result("d", "misc", None, status="error"),
        ],
    )
    assert run.category_accuracy == 0.5
    assert len(run.scored) == 2


def test_scored_ratio_ignores_out_of_contract():
    run = make_run(
        "r1",
        [make_case_result(f"c{i}", "misc", "misc") for i in range(8)]
        + [
            make_case_result(
                f"x{i}", "interview_invite", None, status="out_of_contract"
            )
            for i in range(2)
        ],
    )
    assert run.scored_ratio == 1.0
    assert len(run.attempted) == 8


def test_scored_ratio_counts_errors_against_the_run():
    run = make_run(
        "r1",
        [make_case_result("a", "misc", "misc")]
        + [make_case_result(f"e{i}", "misc", None, status="error") for i in range(3)],
    )
    assert run.scored_ratio == 0.25


def test_aggregates_and_counts():
    run = make_run(
        "r1",
        [
            make_case_result("a", "misc", "misc", summary_score=5),
            make_case_result("b", "misc", "misc", summary_score=3),
            make_case_result("c", "interview_invite", None, status="out_of_contract"),
        ],
    )
    assert run.mean_summary_score == 4
    assert run.total_tokens == 30
    assert run.count("out_of_contract") == 1
    assert set(run.by_id()) == {"a", "b", "c"}


def test_latest_run_filters_by_prompt_version(connection):
    save_run(connection, make_run("old-v1", [], prompt_version="v1"))
    save_run(connection, make_run("new-v2", [], prompt_version="v2"))
    assert latest_run(connection, "v1").run_id == "old-v1"
    assert latest_run(connection, "v9") is None
    assert latest_run(connection) is not None
