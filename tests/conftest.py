"""Shared fixtures for the rlm-skill test suite.

Most tests need an isolated session directory and a clean budget file. These
fixtures provide both, and also expose a helper to run the rlm_repl.py CLI as
a subprocess.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Make `import rlm_helper` / `import rlm_repl` work from tests.
REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_DIR = REPO_ROOT / "skills" / "rlm"
sys.path.insert(0, str(SKILL_DIR))


@pytest.fixture
def isolated_session_dir(tmp_path, monkeypatch):
    """Point RLM_SESSION_DIR at a per-test tmp dir so sessions don't collide."""
    d = tmp_path / "sessions"
    d.mkdir()
    monkeypatch.setenv("RLM_SESSION_DIR", str(d))
    return d


@pytest.fixture
def budget_file(tmp_path, monkeypatch):
    """Fresh budget.json pointed at by RLM_BUDGET_PATH. Returns the Path."""
    p = tmp_path / "budget.json"
    p.write_text(json.dumps({
        "calls": 0,
        "limit": 100,
        "tokens_in": 0,
        "tokens_out": 0,
        "cache_reads": 0,
        "cache_writes": 0,
        "token_limit": None,
        "token_warning": None,
        "warned_tokens": False,
    }))
    monkeypatch.setenv("RLM_BUDGET_PATH", str(p))
    return p


@pytest.fixture
def run_cli(isolated_session_dir, monkeypatch):
    """Callable that runs `rlm_repl.py <args>` and returns (stdout, stderr, rc).

    Optionally accepts stdin content. Inherits the monkeypatched env (incl.
    RLM_SESSION_DIR).
    """
    script = SKILL_DIR / "rlm_repl.py"

    def _run(*args: str, stdin: str | None = None):
        proc = subprocess.run(
            [sys.executable, str(script), *args],
            input=stdin,
            capture_output=True,
            text=True,
            env={**os.environ},
        )
        return proc.stdout, proc.stderr, proc.returncode

    return _run


@pytest.fixture
def tiny_input(tmp_path):
    """A small test input file. Returns its Path."""
    p = tmp_path / "input.txt"
    p.write_text("\n".join(f"line {i}" for i in range(100)))
    return p


@pytest.fixture
def fake_repo(tmp_path):
    """A directory with Python, Markdown, and an ignored .git subdir.
    Returns its Path.
    """
    root = tmp_path / "fake_repo"
    (root / "src").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / ".git").mkdir()
    (root / "node_modules").mkdir()
    (root / "src" / "a.py").write_text("def add(x, y):\n    return x + y\n")
    (root / "src" / "b.py").write_text("CONST = 42\n")
    (root / "docs" / "README.md").write_text("# Fake Repo\nTest fixture.\n")
    (root / ".git" / "HEAD").write_text("garbage\n")
    (root / "node_modules" / "junk.js").write_text("garbage\n")
    return root
