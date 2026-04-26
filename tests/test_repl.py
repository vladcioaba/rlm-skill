"""Integration tests — drive rlm_repl.py via subprocess, no API needed."""

from __future__ import annotations

import json


def _json(s: str) -> dict:
    return json.loads(s)


# ---------- start --------------------------------------------------------

def test_start_single_file(run_cli, tiny_input):
    out, _, rc = run_cli("start", "--input", str(tiny_input), "--session", "t")
    assert rc == 0
    meta = _json(out)
    assert meta["session"] == "t"
    assert meta["is_directory"] is False
    assert meta["file_count"] == 1
    assert meta["context_total_chars"] > 0
    assert "line 0" in meta["context_prefix"]


def test_start_missing_file_errors(run_cli):
    _, err, rc = run_cli("start", "--input", "/does/not/exist", "--session", "x")
    assert rc != 0
    assert "not found" in err.lower()


def test_start_directory_with_glob(run_cli, fake_repo):
    out, _, rc = run_cli(
        "start", "--input", str(fake_repo),
        "--glob", "*.py,*.md", "--session", "d",
    )
    assert rc == 0
    meta = _json(out)
    assert meta["is_directory"] is True
    assert meta["file_count"] == 3  # 2 .py + 1 .md
    paths = [f["path"] for f in meta["files_preview"]]
    assert any(p.endswith("README.md") for p in paths)
    assert any(p.endswith("a.py") for p in paths)


def test_start_directory_prunes_ignored_dirs(run_cli, fake_repo):
    out, _, rc = run_cli(
        "start", "--input", str(fake_repo), "--session", "d2",
    )
    assert rc == 0
    meta = _json(out)
    # No files from .git or node_modules should appear.
    all_paths = [f["path"] for f in meta["files_preview"]]
    assert not any(".git" in p for p in all_paths)
    assert not any("node_modules" in p for p in all_paths)


def test_start_budgets_written_to_meta(run_cli, tiny_input):
    out, _, rc = run_cli(
        "start", "--input", str(tiny_input), "--session", "b",
        "--budget", "7", "--token-limit", "12345", "--token-warning", "6789",
    )
    assert rc == 0
    meta = _json(out)
    assert meta["budget_limit"] == 7
    assert meta["token_limit"] == 12345
    assert meta["token_warning"] == 6789


# ---------- exec ---------------------------------------------------------

def test_exec_persists_variables_between_calls(run_cli, tiny_input):
    run_cli("start", "--input", str(tiny_input), "--session", "p")
    run_cli("exec", "--session", "p", stdin="squad = [1, 2, 3]\nprint(len(squad))")
    out, _, rc = run_cli("exec", "--session", "p", stdin="print(sum(squad))")
    assert rc == 0
    report = _json(out)
    assert report["stdout"].strip() == "6"


def test_exec_truncates_large_stdout(run_cli, tiny_input, monkeypatch):
    monkeypatch.setenv("RLM_STDOUT_MAX", "64")
    run_cli("start", "--input", str(tiny_input), "--session", "s")
    out, _, _ = run_cli("exec", "--session", "s", stdin='print("x" * 500)')
    report = _json(out)
    assert report["stdout_truncated"] is True
    assert report["stdout_len"] == 501  # 500 xs + newline
    assert report["stdout_full_path"]  # path populated
    assert len(report["stdout"]) == 64


def test_exec_captures_error_without_crashing(run_cli, tiny_input):
    run_cli("start", "--input", str(tiny_input), "--session", "e")
    out, _, rc = run_cli("exec", "--session", "e", stdin="raise ValueError('boom')")
    assert rc == 0  # the driver itself must not crash
    report = _json(out)
    assert report["error"] is not None
    assert "ValueError" in report["error"]
    assert "boom" in report["error"]


def test_exec_final_direct(run_cli, tiny_input):
    run_cli("start", "--input", str(tiny_input), "--session", "f1")
    out, _, _ = run_cli("exec", "--session", "f1", stdin='FINAL("done")')
    report = _json(out)
    assert report["final"]["kind"] == "direct"
    assert report["final"]["preview"] == "done"

    # `final` subcommand returns the saved answer.
    body, _, rc = run_cli("final", "--session", "f1")
    assert rc == 0
    assert body == "done"


def test_exec_final_var(run_cli, tiny_input):
    run_cli("start", "--input", str(tiny_input), "--session", "f2")
    run_cli("exec", "--session", "f2", stdin='report = "A" * 50')
    out, _, _ = run_cli("exec", "--session", "f2", stdin='FINAL_VAR("report")')
    report = _json(out)
    assert report["final"]["kind"] == "var"
    assert report["final"]["name"] == "report"
    assert report["final"]["length"] == 50

    body, _, rc = run_cli("final", "--session", "f2")
    assert rc == 0
    assert body == "A" * 50


def test_exec_reports_budget_state(run_cli, tiny_input):
    run_cli("start", "--input", str(tiny_input), "--session", "bs",
            "--budget", "42", "--token-warning", "1000")
    out, _, _ = run_cli("exec", "--session", "bs", stdin='print("hi")')
    report = _json(out)
    assert report["budget"]["limit"] == 42
    assert report["budget"]["token_warning"] == 1000


def test_exec_surfaces_token_warning(run_cli, tiny_input):
    run_cli("start", "--input", str(tiny_input), "--session", "tw",
            "--token-warning", "100", "--token-limit", "1000")
    # Manually bump tokens_in past the warning threshold via the budget CLI.
    from pathlib import Path
    import os
    bp = Path(os.environ["RLM_SESSION_DIR"]) / "tw" / "budget.json"
    st = json.loads(bp.read_text())
    st["tokens_in"] = 500
    bp.write_text(json.dumps(st))

    out, _, _ = run_cli("exec", "--session", "tw", stdin="pass")
    report = _json(out)
    assert len(report["warnings"]) == 1
    assert "warning threshold 100" in report["warnings"][0]


# ---------- budget -------------------------------------------------------

def test_budget_inspect(run_cli, tiny_input):
    run_cli("start", "--input", str(tiny_input), "--session", "bi", "--budget", "11")
    out, _, rc = run_cli("budget", "--session", "bi")
    assert rc == 0
    state = _json(out)
    assert state["limit"] == 11


def test_budget_set_call_limit(run_cli, tiny_input):
    run_cli("start", "--input", str(tiny_input), "--session", "bs1")
    out, _, _ = run_cli("budget", "--session", "bs1", "--set", "999")
    state = _json(out)
    assert state["limit"] == 999


def test_budget_set_token_limit_and_warning(run_cli, tiny_input):
    run_cli("start", "--input", str(tiny_input), "--session", "bs2")
    out, _, _ = run_cli(
        "budget", "--session", "bs2",
        "--set-token-limit", "55555",
        "--set-token-warning", "22222",
    )
    state = _json(out)
    assert state["token_limit"] == 55555
    assert state["token_warning"] == 22222


def test_budget_clear_token_limit(run_cli, tiny_input):
    run_cli("start", "--input", str(tiny_input), "--session", "bc",
            "--token-limit", "5000")
    out, _, _ = run_cli("budget", "--session", "bc", "--clear-token-limit")
    state = _json(out)
    assert state["token_limit"] is None


# ---------- list / stop --------------------------------------------------

def test_list_reports_active_sessions(run_cli, tiny_input):
    run_cli("start", "--input", str(tiny_input), "--session", "l1")
    run_cli("start", "--input", str(tiny_input), "--session", "l2")
    out, _, _ = run_cli("list")
    data = _json(out)
    ids = {s["session"] for s in data["sessions"]}
    assert {"l1", "l2"} <= ids


def test_stop_removes_session(run_cli, tiny_input):
    run_cli("start", "--input", str(tiny_input), "--session", "rm")
    _, _, rc = run_cli("stop", "--session", "rm")
    assert rc == 0
    # `budget` on a stopped session should error.
    _, err, rc = run_cli("budget", "--session", "rm")
    assert rc != 0
    assert "no budget" in err.lower()


# ---------- context_files variable --------------------------------------

def test_exec_can_slice_individual_files_from_context(run_cli, fake_repo):
    run_cli("start", "--input", str(fake_repo), "--glob", "*.py",
            "--session", "cf")
    out, _, _ = run_cli("exec", "--session", "cf", stdin="""
paths = [f['path'] for f in context_files]
print('|'.join(sorted(paths)))
# slice out a.py
a = next(f for f in context_files if f['path'].endswith('a.py'))
snippet = context[a['start']:a['end']]
print('HAS_DEF_ADD' if 'def add' in snippet else 'MISSING')
""")
    report = _json(out)
    assert "a.py" in report["stdout"]
    assert "HAS_DEF_ADD" in report["stdout"]
