"""Recursive Language Model REPL driver.

Subcommands:
  start  --input PATH [--glob PATTERNS] [--session ID] [--budget N]
                                         create a session with `context` loaded
  exec   --session ID                    read Python code from stdin, exec it
  final  --session ID                    emit the FINAL answer
  stop   --session ID                    delete the session
  budget --session ID [--set N]          inspect or raise the call cap
  list                                   list active sessions

Design invariants (from Zhang/Kraska/Khattab 2026):
  - The loaded input NEVER prints back to stdout in full.
  - Only metadata + a bounded prefix of stdout ever leaves this process.
  - Sub-LLM calls go through rlm_helper.llm_query, not through this driver.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rlm_helper  # noqa: E402

SESSION_ROOT = Path(os.environ.get("RLM_SESSION_DIR", "/tmp/rlm-sessions"))
STDOUT_MAX = int(os.environ.get("RLM_STDOUT_MAX", "2048"))
CONTEXT_PREFIX_LEN = int(os.environ.get("RLM_CONTEXT_PREFIX_LEN", "800"))

DEFAULT_IGNORES = {
    ".git", ".hg", ".svn",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "node_modules", ".next", ".nuxt",
    ".venv", "venv", "env",
    "dist", "build", "target", ".gradle",
    ".DS_Store",
}

MAX_PER_FILE_BYTES = int(os.environ.get("RLM_MAX_PER_FILE_BYTES", str(4 * 1024 * 1024)))


# ---------- session I/O --------------------------------------------------

def _session_dir(sid: str) -> Path:
    p = SESSION_ROOT / sid
    p.mkdir(parents=True, exist_ok=True)
    return p


def _state_path(sid: str) -> Path:
    return _session_dir(sid) / "state.pkl"


def _budget_path(sid: str) -> Path:
    return _session_dir(sid) / "budget.json"


def _load_state(sid: str) -> dict:
    path = _state_path(sid)
    if not path.exists():
        raise SystemExit(f"No session named {sid!r}. Start one with: start --input PATH")
    with path.open("rb") as f:
        return pickle.load(f)


def _save_state(sid: str, ns: dict) -> None:
    clean: dict = {}
    for k, v in ns.items():
        if k.startswith("__"):
            continue
        if k in ("print",):
            continue
        if callable(v) and getattr(v, "__module__", "") == "rlm_helper":
            continue
        try:
            pickle.dumps(v)
        except Exception:
            continue
        clean[k] = v
    with _state_path(sid).open("wb") as f:
        pickle.dump(clean, f)


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, indent=2, default=str))
    sys.stdout.write("\n")


# ---------- input loading ------------------------------------------------

def _parse_globs(spec: str | None) -> list[str]:
    if not spec:
        return []
    return [p.strip() for p in spec.split(",") if p.strip()]


def _match_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def _walk_files(root: Path, include: list[str]) -> list[Path]:
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune ignored directories in-place.
        dirnames[:] = [d for d in dirnames if d not in DEFAULT_IGNORES]
        for fn in filenames:
            if fn in DEFAULT_IGNORES:
                continue
            if include and not _match_any(fn, include):
                continue
            out.append(Path(dirpath) / fn)
    out.sort()
    return out


def _load_context(input_arg: str, globs: list[str]) -> tuple[str, list[dict]]:
    """Returns (concatenated_content, file_index).

    For a single file: content is the file, index is a one-element list.
    For a directory: files are concatenated with boundary markers, index
    lists each file with its offset and length.
    """
    p = Path(input_arg).resolve()
    if not p.exists():
        raise SystemExit(f"Input not found: {p}")

    if p.is_file():
        content = _read_text_safe(p)
        return content, [{"path": str(p), "start": 0, "end": len(content), "chars": len(content)}]

    if not p.is_dir():
        raise SystemExit(f"Input is neither file nor directory: {p}")

    files = _walk_files(p, globs)
    if not files:
        raise SystemExit(
            f"No files matched under {p}"
            + (f" with globs {globs}" if globs else "")
        )

    parts: list[str] = []
    index: list[dict] = []
    offset = 0
    for f in files:
        try:
            if f.stat().st_size > MAX_PER_FILE_BYTES:
                continue
            text = _read_text_safe(f)
        except (UnicodeDecodeError, OSError):
            continue
        header = f"\n\n======== FILE: {f.relative_to(p)} ({len(text)} chars) ========\n"
        if not parts:
            header = header.lstrip("\n")
        parts.append(header)
        offset += len(header)
        start = offset
        parts.append(text)
        offset += len(text)
        index.append({
            "path": str(f.relative_to(p)),
            "abspath": str(f),
            "start": start,
            "end": offset,
            "chars": len(text),
        })
    return "".join(parts), index


def _read_text_safe(path: Path) -> str:
    with path.open("rb") as f:
        raw = f.read()
    # Best-effort decode; replace undecodable bytes rather than crashing.
    return raw.decode("utf-8", errors="replace")


# ---------- commands ----------------------------------------------------

def cmd_start(args: argparse.Namespace) -> None:
    globs = _parse_globs(args.glob)
    content, index = _load_context(args.input, globs)

    sid = args.session or hashlib.sha1(
        f"{args.input}:{time.time()}".encode()
    ).hexdigest()[:10]

    ns: dict = {
        "context": content,
        "context_files": index,
    }
    _save_state(sid, ns)

    # Initialize budget. Token fields default to env-var values (may be None).
    budget_state = {
        "calls": 0,
        "limit": int(args.budget),
        "tokens_in": 0,
        "tokens_out": 0,
        "token_limit": args.token_limit if args.token_limit is not None else rlm_helper.DEFAULT_TOKEN_LIMIT,
        "token_warning": args.token_warning if args.token_warning is not None else rlm_helper.DEFAULT_TOKEN_WARNING,
        "warned_tokens": False,
    }
    _budget_path(sid).write_text(json.dumps(budget_state))

    meta = {
        "session": sid,
        "input": str(Path(args.input).resolve()),
        "is_directory": Path(args.input).resolve().is_dir(),
        "context_total_chars": len(content),
        "context_prefix": content[:CONTEXT_PREFIX_LEN],
        "line_count": content.count("\n") + 1,
        "file_count": len(index),
        "files_preview": [
            {"path": f["path"], "chars": f["chars"]} for f in index[:20]
        ],
        "budget_limit": budget_state["limit"],
        "token_limit": budget_state["token_limit"],
        "token_warning": budget_state["token_warning"],
        "session_dir": str(_session_dir(sid)),
    }
    _emit(meta)


def cmd_exec(args: argparse.Namespace) -> None:
    sid = args.session
    ns = _load_state(sid)

    # Wire budget path into the helper for this exec.
    os.environ[rlm_helper.BUDGET_PATH_ENV] = str(_budget_path(sid))

    stdout_buf = io.StringIO()
    ns["print"] = rlm_helper.make_truncating_print(stdout_buf, STDOUT_MAX)
    ns["llm_query"] = rlm_helper.llm_query
    ns["llm_query_batch"] = rlm_helper.llm_query_batch
    ns["FINAL"] = rlm_helper.FINAL
    ns["FINAL_VAR"] = rlm_helper.FINAL_VAR

    code = sys.stdin.read()

    err = None
    try:
        compiled = compile(code, f"<rlm:{sid}>", "exec")
        exec(compiled, ns)
    except Exception:
        err = traceback.format_exc(limit=20)

    stdout_full = stdout_buf.getvalue()
    stdout_len = len(stdout_full)
    truncated = stdout_len > STDOUT_MAX
    stdout_display = stdout_full[:STDOUT_MAX]

    full_path = None
    if truncated:
        full_path = _session_dir(sid) / f"stdout-{int(time.time() * 1000)}.txt"
        full_path.write_text(stdout_full)

    final_payload = None
    final = ns.get(rlm_helper._FINAL_KEY)
    if isinstance(final, dict):
        if final.get("kind") == "direct":
            value = final["value"]
            fp = _session_dir(sid) / "final.txt"
            fp.write_text(value)
            final_payload = {
                "kind": "direct",
                "length": len(value),
                "saved_to": str(fp),
                "preview": value[:STDOUT_MAX],
            }
        elif final.get("kind") == "var":
            name = final["name"]
            value = ns.get(name)
            if value is None:
                err = (err or "") + f"\nFINAL_VAR: variable '{name}' is None"
            else:
                if not isinstance(value, str):
                    value = str(value)
                fp = _session_dir(sid) / "final.txt"
                fp.write_text(value)
                final_payload = {
                    "kind": "var",
                    "name": name,
                    "length": len(value),
                    "saved_to": str(fp),
                    "preview": value[:STDOUT_MAX],
                }

    ns.pop(rlm_helper._FINAL_KEY, None)
    _save_state(sid, ns)

    # Read back budget state to surface it.
    try:
        budget_state = json.loads(_budget_path(sid).read_text())
    except Exception:
        budget_state = None

    warnings: list[str] = []
    if budget_state:
        tw = budget_state.get("token_warning")
        total_tokens = budget_state.get("tokens_in", 0) + budget_state.get("tokens_out", 0)
        if tw is not None and total_tokens >= tw:
            warnings.append(
                f"token_usage {total_tokens} >= warning threshold {tw} "
                f"(limit: {budget_state.get('token_limit') or 'unset'})."
                " Consider converging soon — use FINAL with current results."
            )

    report = {
        "session": sid,
        "stdout_len": stdout_len,
        "stdout_truncated": truncated,
        "stdout": stdout_display,
        "stdout_full_path": str(full_path) if full_path else None,
        "error": err,
        "final": final_payload,
        "variables": _var_summary(ns),
        "budget": budget_state,
        "warnings": warnings,
    }
    _emit(report)


def _var_summary(ns: dict) -> list[dict]:
    out = []
    for k, v in ns.items():
        if k.startswith("__") or k in ("print",):
            continue
        if callable(v) and getattr(v, "__module__", "") == "rlm_helper":
            continue
        try:
            if isinstance(v, str):
                out.append({"name": k, "type": "str", "len": len(v)})
            elif isinstance(v, (list, tuple, dict, set)):
                out.append({"name": k, "type": type(v).__name__, "len": len(v)})
            else:
                out.append({"name": k, "type": type(v).__name__})
        except Exception:
            out.append({"name": k, "type": "?"})
    return out


def cmd_final(args: argparse.Namespace) -> None:
    sid = args.session
    fp = _session_dir(sid) / "final.txt"
    if not fp.exists():
        raise SystemExit(f"No final answer recorded for session {sid}.")
    sys.stdout.write(fp.read_text())


def cmd_stop(args: argparse.Namespace) -> None:
    sid = args.session
    import shutil
    d = SESSION_ROOT / sid
    if d.exists():
        shutil.rmtree(d)
    _emit({"stopped": sid})


def cmd_budget(args: argparse.Namespace) -> None:
    sid = args.session
    bp = _budget_path(sid)
    if not bp.exists():
        raise SystemExit(f"No budget file for session {sid}.")
    state = json.loads(bp.read_text())
    changed = False
    if args.set is not None:
        state["limit"] = int(args.set)
        changed = True
    if args.set_token_limit is not None:
        state["token_limit"] = int(args.set_token_limit)
        state["warned_tokens"] = False  # let the warning re-fire against the new limit
        changed = True
    if args.set_token_warning is not None:
        state["token_warning"] = int(args.set_token_warning)
        state["warned_tokens"] = False
        changed = True
    if args.clear_token_limit:
        state["token_limit"] = None
        changed = True
    if args.clear_token_warning:
        state["token_warning"] = None
        changed = True
    if changed:
        bp.write_text(json.dumps(state))
    _emit(state)


def cmd_list(_: argparse.Namespace) -> None:
    if not SESSION_ROOT.exists():
        _emit({"sessions": []})
        return
    sessions = []
    for d in sorted(SESSION_ROOT.iterdir()):
        if (d / "state.pkl").exists():
            stat = (d / "state.pkl").stat()
            entry = {"session": d.name, "updated": int(stat.st_mtime)}
            bp = d / "budget.json"
            if bp.exists():
                try:
                    entry["budget"] = json.loads(bp.read_text())
                except Exception:
                    pass
            sessions.append(entry)
    _emit({"sessions": sessions})


# ---------- entry point -------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(prog="rlm_repl")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("start")
    ps.add_argument("--input", required=True, help="File or directory path")
    ps.add_argument("--glob", default=None,
                    help="Comma-separated glob patterns for directory input, e.g. '*.py,*.md'")
    ps.add_argument("--session")
    ps.add_argument("--budget", default=str(rlm_helper.DEFAULT_BUDGET_LIMIT),
                    help="Max sub-LLM calls per session (default: 100)")
    ps.add_argument("--token-limit", type=int, default=None,
                    help="Hard cap on total tokens (input+output). Default: unlimited.")
    ps.add_argument("--token-warning", type=int, default=None,
                    help="Emit a warning once total tokens cross this threshold.")
    ps.set_defaults(func=cmd_start)

    pe = sub.add_parser("exec")
    pe.add_argument("--session", required=True)
    pe.set_defaults(func=cmd_exec)

    pf = sub.add_parser("final")
    pf.add_argument("--session", required=True)
    pf.set_defaults(func=cmd_final)

    pst = sub.add_parser("stop")
    pst.add_argument("--session", required=True)
    pst.set_defaults(func=cmd_stop)

    pb = sub.add_parser("budget")
    pb.add_argument("--session", required=True)
    pb.add_argument("--set", type=int, default=None,
                    help="Overwrite the call limit for this session")
    pb.add_argument("--set-token-limit", type=int, default=None,
                    help="Set a hard token cap")
    pb.add_argument("--set-token-warning", type=int, default=None,
                    help="Set (or reset) the token warning threshold")
    pb.add_argument("--clear-token-limit", action="store_true",
                    help="Remove the token cap")
    pb.add_argument("--clear-token-warning", action="store_true",
                    help="Remove the token warning threshold")
    pb.set_defaults(func=cmd_budget)

    pl = sub.add_parser("list")
    pl.set_defaults(func=cmd_list)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
