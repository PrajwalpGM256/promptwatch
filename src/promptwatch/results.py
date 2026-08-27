import sqlite3
from pathlib import Path
from statistics import mean
from typing import Literal

from pydantic import BaseModel, Field

from promptwatch.models import Category

CaseStatus = Literal["scored", "out_of_contract", "off_contract_output", "error"]

DEFAULT_DB = Path("runs.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    prompt_version  TEXT NOT NULL,
    provider        TEXT NOT NULL DEFAULT 'gemini',
    model           TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    judge_version   TEXT NOT NULL,
    judge_provider  TEXT NOT NULL DEFAULT 'none',
    judge_model     TEXT NOT NULL DEFAULT 'none',
    started_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS case_results (
    run_id            TEXT NOT NULL,
    case_id           TEXT NOT NULL,
    status            TEXT NOT NULL
                      CHECK (status IN ('scored', 'out_of_contract',
                                        'off_contract_output', 'error')),
    expected_category TEXT NOT NULL,
    actual_category   TEXT,
    category_match    INTEGER CHECK (category_match IN (0, 1)),
    summary           TEXT,
    summary_score     INTEGER CHECK (summary_score BETWEEN 1 AND 5),
    latency_ms        INTEGER NOT NULL,
    prompt_tokens     INTEGER NOT NULL,
    output_tokens     INTEGER NOT NULL,
    error             TEXT,
    PRIMARY KEY (run_id, case_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_runs_prompt ON runs(prompt_version, started_at);
"""

_CASE_COLUMNS = (
    "case_id, status, expected_category, actual_category, category_match, "
    "summary, summary_score, latency_ms, prompt_tokens, output_tokens, error"
)


class CaseResult(BaseModel):
    case_id: str
    status: CaseStatus
    expected_category: Category
    actual_category: Category | None = None
    category_match: bool | None = None
    summary: str | None = None
    summary_score: int | None = Field(default=None, ge=1, le=5)
    latency_ms: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


class RunResult(BaseModel):
    run_id: str
    prompt_version: str
    provider: str = "gemini"
    model: str
    dataset_version: str
    judge_version: str
    judge_provider: str = "none"
    judge_model: str = "none"
    started_at: str
    cases: list[CaseResult]

    @property
    def scored(self) -> list[CaseResult]:
        return [case for case in self.cases if case.category_match is not None]

    @property
    def attempted(self) -> list[CaseResult]:
        return [case for case in self.cases if case.status != "out_of_contract"]

    @property
    def scored_ratio(self) -> float:
        attempted = self.attempted
        return len(self.scored) / len(attempted) if attempted else 0.0

    @property
    def category_accuracy(self) -> float:
        scored = self.scored
        if not scored:
            return 0.0
        return sum(1 for case in scored if case.category_match) / len(scored)

    @property
    def mean_summary_score(self) -> float:
        scores = [c.summary_score for c in self.cases if c.summary_score is not None]
        return mean(scores) if scores else 0.0

    @property
    def mean_latency_ms(self) -> float:
        latencies = [c.latency_ms for c in self.cases if c.latency_ms]
        return mean(latencies) if latencies else 0.0

    @property
    def max_latency_ms(self) -> int:
        return max((c.latency_ms for c in self.cases), default=0)

    @property
    def total_tokens(self) -> int:
        return sum(c.prompt_tokens + c.output_tokens for c in self.cases)

    def count(self, status: CaseStatus) -> int:
        """Number of cases with the given status."""
        return sum(1 for case in self.cases if case.status == status)

    def by_id(self) -> dict[str, CaseResult]:
        """Cases keyed by case_id, for diffing against another run."""
        return {case.case_id: case for case in self.cases}


def _migrate(connection: sqlite3.Connection) -> None:
    columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(runs)")
    }
    with connection:
        if "provider" not in columns:
            connection.execute(
                "ALTER TABLE runs ADD COLUMN provider TEXT NOT NULL "
                "DEFAULT 'gemini'"
            )
        if "judge_provider" not in columns:
            connection.execute(
                "ALTER TABLE runs ADD COLUMN judge_provider TEXT NOT NULL "
                "DEFAULT ''"
            )
            connection.execute(
                "ALTER TABLE runs ADD COLUMN judge_model TEXT NOT NULL "
                "DEFAULT ''"
            )
            connection.execute(
                "UPDATE runs SET"
                " judge_provider = CASE WHEN judge_version = 'none'"
                "                  THEN 'none' ELSE provider END,"
                " judge_model = CASE WHEN judge_version = 'none'"
                "               THEN 'none' ELSE model END"
                " WHERE judge_provider = ''"
            )


def connect(path: str | Path = DEFAULT_DB) -> sqlite3.Connection:
    """Open the run database, creating the schema if absent.

    Older databases are migrated forward. The judge columns are backfilled
    from each run's own provider and model rather than a fixed value, because
    before the judge was separable it always ran on the backend under test.
    """
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA)
    _migrate(connection)
    return connection


def save_run(connection: sqlite3.Connection, run: RunResult) -> None:
    """Persist a run and its cases in one transaction.

    Raises:
        sqlite3.IntegrityError: if `run.run_id` already exists.
    """
    with connection:
        connection.execute(
            "INSERT INTO runs (run_id, prompt_version, provider, model, "
            "dataset_version, judge_version, judge_provider, judge_model, "
            "started_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run.run_id, run.prompt_version, run.provider, run.model,
             run.dataset_version, run.judge_version, run.judge_provider,
             run.judge_model, run.started_at),
        )
        connection.executemany(
            f"INSERT INTO case_results (run_id, {_CASE_COLUMNS}) "
            f"VALUES (?, {', '.join('?' * 11)})",
            [
                (run.run_id, c.case_id, c.status, c.expected_category,
                 c.actual_category, c.category_match, c.summary, c.summary_score,
                 c.latency_ms, c.prompt_tokens, c.output_tokens, c.error)
                for c in run.cases
            ],
        )


def load_run(connection: sqlite3.Connection, run_id: str) -> RunResult:
    """Load a run by id.

    Raises:
        KeyError: if no run with that id exists.
    """
    row = connection.execute(
        "SELECT * FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if row is None:
        raise KeyError(f"no run {run_id!r} in database")

    cases = connection.execute(
        f"SELECT {_CASE_COLUMNS} FROM case_results WHERE run_id = ? ORDER BY case_id",
        (run_id,),
    ).fetchall()
    return RunResult(**dict(row), cases=[CaseResult(**dict(c)) for c in cases])


def latest_run(
    connection: sqlite3.Connection,
    prompt_version: str | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> RunResult | None:
    """Most recent run matching every filter given.

    A run is only a baseline for the next one if the backend also matches, so
    callers looking for "the previous run of this prompt" must pass the
    provider and model too. Diffing across backends measures the swap, not the
    prompt.
    """
    filters = {
        "prompt_version": prompt_version,
        "provider": provider,
        "model": model,
    }
    clauses = [f"{column} = ?" for column, value in filters.items() if value]
    params = tuple(value for value in filters.values() if value)

    sql = "SELECT run_id FROM runs"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY started_at DESC, run_id DESC LIMIT 1"

    row = connection.execute(sql, params).fetchone()
    return load_run(connection, row["run_id"]) if row else None
