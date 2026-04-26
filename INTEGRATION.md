# Integrating rlm-skill into other projects

The default flow is "Claude Code session loads the skill via `~/.claude/skills/rlm/`," but the implementation is a thin Python layer that's reusable in three other modes:

1. **From another Claude Code skill** — delegate the chunk-and-query pattern to RLM via its CLI.
2. **As a library in a native Python LLM project** — `import rlm_helper` and use `llm_query` / `llm_query_batch` directly. No REPL needed.
3. **As a REPL pattern in non-Claude-Code orchestration** — drive `rlm_repl.py` as a subprocess from your own root-model loop.

This guide covers each, plus the stable API surface and the caveats.

---

## Stable API surface

These names are part of the public contract — semantics will not change without a major version bump:

```python
# Single call. Returns the response text.
rlm_helper.llm_query(
    prompt: str,
    system: str | None = None,      # cached if cache=True
    prefix: str | None = None,      # cached if cache=True (use for shared docs)
    model: str | None = None,       # default: claude-haiku-4-5
    max_tokens: int | None = None,  # default: 4096
    cache: bool = True,
) -> str

# Parallel batch. Same args, but `prompts` is iterable. Returns same-order list.
rlm_helper.llm_query_batch(
    prompts: Iterable[str],
    system=..., prefix=..., model=..., max_tokens=...,
    concurrency: int | None = None,  # default: 20
    cache: bool = True,
) -> list[str]

# Sentinels for the REPL driver. Outside the REPL they're inert.
rlm_helper.FINAL(answer: Any) -> None
rlm_helper.FINAL_VAR(var_name: str) -> None

# Exceptions you should catch.
rlm_helper.RLMError              # base class
rlm_helper.BudgetExceeded        # raised before a call that would exceed limits
```

Anything starting with an underscore (`_reserve`, `_FINAL_KEY`, etc.) is **internal** and may change. Provider internals live in `rlm_providers` and are similarly underscore-marked.

Relevant environment variables (one of the API-key vars required, the rest optional):

| Name | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` *or* `OPENAI_API_KEY` | *(one required)* | Picks the provider; can also use the generic `LLM_API_KEY` |
| `LLM_API_KEY` | *(unset)* | Generic key fallback used by either provider |
| `LLM_PROVIDER` | auto-detect | `anthropic` \| `openai`; explicit override |
| `LLM_BASE_URL` | OpenAI public | Endpoint for the OpenAI-compatible provider |
| `RLM_SUB_MODEL` | provider's fast tier | Default model for `llm_query[_batch]` |
| `RLM_SUB_MAX_TOKENS` | `4096` | Default per-call output cap |
| `RLM_CONCURRENCY` | `20` | Default batch parallelism |
| `RLM_BUDGET_PATH` | *(unset)* | Path to a JSON file for accounting; if unset, accounting is in-memory and lost on process exit |
| `RLM_BUDGET_LIMIT` | `100` | Default call cap when a fresh budget file is created |
| `RLM_TOKEN_LIMIT` | *(unset)* | Hard token cap |
| `RLM_TOKEN_WARNING` | *(unset)* | Soft warning threshold |

---

## Mode 1 — From another Claude Code skill

If your skill (`~/.claude/skills/your-skill/SKILL.md`) handles tasks that occasionally need to chew through huge inputs, point at the rlm CLI:

```markdown
For inputs larger than ~100K tokens, drive the rlm skill instead of reading
the file yourself:

  python3 ~/.claude/skills/rlm/rlm_repl.py start --input <path>
  # then iterate with `exec` and finalize with FINAL_VAR.

See ~/.claude/skills/rlm/SKILL.md for the full contract.
```

Nothing to install — once `install.sh` has been run, the rlm skill is globally available. Your skill just teaches Claude when to reach for it.

---

## Mode 2 — As a library in a Python LLM project

`rlm_helper` is plain Python with no runtime dependencies (stdlib + `urllib`). Drop it in via path or vendor the file. The functions you'll actually use are `llm_query` and `llm_query_batch`.

### Minimal worked example

Process a long markdown document into per-section summaries, with a cached rubric shared across all sub-calls:

```python
import os, sys
sys.path.insert(0, "/path/to/rlm-skill/skills/rlm")
import rlm_helper

os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."  # or OPENAI_API_KEY, or LLM_API_KEY + LLM_PROVIDER

doc = open("design-spec.md").read()
sections = doc.split("\n## ")            # naive but fine for the example

rubric = """# Review rubric
1. Identify any unstated assumptions.
2. Flag claims without evidence.
3. Note missing failure modes.
"""

# First call writes the rubric to cache; calls 2..N read from cache (~10× cheaper).
findings = rlm_helper.llm_query_batch(
    [f"<section>\n{s}\n</section>\nApply the rubric." for s in sections],
    system="You are a strict spec reviewer. Be terse.",
    prefix=rubric,
    concurrency=10,
)

for i, f in enumerate(findings):
    print(f"== section {i} ==\n{f}\n")
```

### Budget tracking from a script

If you want `BudgetExceeded` to fire before runaway cost, point `RLM_BUDGET_PATH` at any JSON file before importing or calling:

```python
import json, os, tempfile
budget = tempfile.mktemp(suffix=".json")
with open(budget, "w") as f:
    json.dump({
        "calls": 0, "limit": 50,
        "tokens_in": 0, "tokens_out": 0,
        "cache_reads": 0, "cache_writes": 0,
        "token_limit": 5_000_000, "token_warning": 1_000_000,
        "warned_tokens": False,
    }, f)
os.environ["RLM_BUDGET_PATH"] = budget

import rlm_helper
try:
    out = rlm_helper.llm_query("...")
except rlm_helper.BudgetExceeded as e:
    print(f"hit budget: {e}")
```

After the run, read `budget` to see total tokens, cache reads, cache writes.

### Error handling

```python
try:
    out = rlm_helper.llm_query("...")
except rlm_helper.BudgetExceeded:
    # Either raise the cap or accept and finalize with current results.
    ...
except rlm_helper.RLMError as e:
    # Anything else: missing API key, HTTP error, malformed response.
    log.error("rlm call failed: %s", e)
```

Note: there is currently **no automatic retry on 429/5xx**. If you're hitting rate limits, either lower `concurrency=` on `llm_query_batch` or wrap calls with your own backoff logic. (This is a known v0.2 item.)

---

## Mode 3 — As a REPL pattern from a non-Claude-Code orchestrator

If you want the full RLM paradigm — a persistent REPL where your "root model" writes Python that calls sub-LLMs over a `context` variable — and your root is *not* Claude Code, drive `rlm_repl.py` as a subprocess from your own orchestrator.

Your root model can be anything: another LLM, a deterministic planner, your own ReAct loop. The REPL handles all the "prompt-as-environment" plumbing.

```python
import json, subprocess, sys

REPL = ["python3", "/path/to/rlm-skill/skills/rlm/rlm_repl.py"]

def run(*args, stdin=None):
    p = subprocess.run([*REPL, *args], input=stdin,
                       capture_output=True, text=True)
    return json.loads(p.stdout) if p.stdout.strip().startswith("{") else p.stdout

# 1. Start a session over a 50MB log.
meta = run("start", "--input", "/var/log/big.log",
           "--session", "audit", "--budget", "200")
print("loaded:", meta["context_total_chars"], "chars")

# 2. Your root model decides what code to run. Here we hard-code one turn.
code = '''
import re
hits = [m.start() for m in re.finditer(r"(?i)error|exception", context)]
windows = list({context[max(0,p-500):p+1500] for p in hits})
diagnoses = llm_query_batch(
    [f"Diagnose: {w}" for w in windows[:50]],
    concurrency=10,
)
report = "\\n\\n".join(diagnoses)
FINAL_VAR("report")
'''
result = run("exec", "--session", "audit", stdin=code)
print("budget:", result["budget"])

# 3. Read the final answer.
if result["final"]:
    answer = run("final", "--session", "audit")
    print(answer)

run("stop", "--session", "audit")
```

The contract is documented in [`skills/rlm/SKILL.md`](skills/rlm/SKILL.md) — same JSON shapes, same invariants.

---

## Caveats and known limitations

- **Not pip-packaged.** Today you vendor the files or `sys.path.insert`. A future release may publish to PyPI; if you want this prioritized, open an issue.
- **Module name `rlm_helper` is generic.** If your project already has an `rlm_helper`, rename or vendor under a namespaced path.
- **Sync HTTP under the hood.** `llm_query_batch` uses a thread pool, not `asyncio`. Fine for typical workloads (each call is I/O-bound and the GIL releases on `urlopen`), but if you're mixing this with an async event loop you may want to wrap it in `asyncio.to_thread`.
- **No automatic retries on 429/5xx.** Yet — see "Error handling" above.
- **Depth=1 recursion only.** Sub-calls are plain LLM calls, not further RLMs. Matches the paper.
- **Budget races across processes.** `_budget_lock` is per-process. Two concurrent Python processes hitting the same `RLM_BUDGET_PATH` file may overcount or undercount slightly. Per-process use is race-free.
- **Stable surface stops at function signatures.** The on-disk JSON shapes (budget file, exec response) are also stable, but everything underscore-prefixed in the code is fair game to change.

---

## Roadmap

If you depend on this and want priority on any of these, open an issue:

- [ ] PyPI package + proper import path (`rlm_skill.helper`).
- [ ] Retry/backoff on 429/5xx with configurable policy.
- [ ] Cross-process file lock on `budget.json`.
- [ ] Async-native `llm_query_async` / `llm_query_batch_async`.
- [ ] Optional structured logging hook (per-call usage events).
- [ ] Streaming sub-call support for long outputs.
