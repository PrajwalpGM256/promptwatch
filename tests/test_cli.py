import pytest
from conftest import make_case_result, make_run

from promptwatch.cli import EXIT_CODES, _diff, _parser, _runs, _summarise, main
from promptwatch.results import connect, save_run


def parse(*argv: str):
    return _parser().parse_args(argv)


def run_of(run_id: str, correct: int, wrong: int, **kwargs):
    cases = [make_case_result(f"c{i}", "misc", "misc") for i in range(correct)]
    cases += [
        make_case_result(f"w{i}", "misc", "rejection") for i in range(wrong)
    ]
    return make_run(run_id, cases, **kwargs)


@pytest.fixture
def db(tmp_path):
    return connect(tmp_path / "runs.db")


def test_run_defaults_to_the_local_backend():
    args = parse("run", "prompts/v2.yaml")
    assert args.provider == "ollama"
    assert args.model is None
    assert args.rpm is None
    assert args.limit is None
    assert args.skip_judge is False


def test_unknown_provider_is_rejected_by_the_parser():
    with pytest.raises(SystemExit):
        parse("run", "prompts/v2.yaml", "--provider", "openai")


def test_thresholds_are_configurable():
    args = parse("--warn", "0.01", "--critical", "0.05", "run", "prompts/v2.yaml")
    assert (args.warn, args.critical) == (0.01, 0.05)


def test_summarise_reports_the_backend():
    text = _summarise(run_of("r1", correct=3, wrong=1))
    assert "backend       gemini   model gemini-3.5-flash-lite" in text
    assert "accuracy      75.00%" in text


def test_runs_lists_saved_runs(db, capsys, monkeypatch):
    save_run(db, run_of("20260826T000000-v2", correct=2, wrong=0))
    monkeypatch.setattr("promptwatch.cli.connect", lambda path: db)

    assert _runs(parse("runs")) == 0

    out = capsys.readouterr().out
    assert "20260826T000000-v2" in out
    assert "gemini/gemini-3.5-flash-lite" in out


def test_runs_on_empty_database(db, capsys, monkeypatch):
    monkeypatch.setattr("promptwatch.cli.connect", lambda path: db)

    assert _runs(parse("runs")) == 0
    assert "no runs recorded" in capsys.readouterr().out


def test_diff_exits_zero_when_accuracy_holds(db, capsys, monkeypatch):
    save_run(db, run_of("base", correct=10, wrong=0))
    save_run(db, run_of("head", correct=10, wrong=0))
    monkeypatch.setattr("promptwatch.cli.connect", lambda path: db)

    assert _diff(parse("diff", "base", "head")) == EXIT_CODES["pass"]
    assert "head  vs  base" in capsys.readouterr().out


def test_diff_exits_two_on_a_critical_drop(db, capsys, monkeypatch):
    save_run(db, run_of("base", correct=10, wrong=0))
    save_run(db, run_of("head", correct=5, wrong=5))
    monkeypatch.setattr("promptwatch.cli.connect", lambda path: db)

    assert _diff(parse("diff", "base", "head")) == EXIT_CODES["critical"]


def test_main_dispatches_to_the_subcommand(db, monkeypatch):
    monkeypatch.setattr("promptwatch.cli.connect", lambda path: db)
    monkeypatch.setattr("sys.argv", ["promptwatch", "runs"])

    assert main() == 0
