---
name: rlm
description: Process a large input (file, log, transcript, concatenated codebase) against a query without loading it into your own context. Offloads the bulk into a Python REPL and distributes semantic work across cheap sub-LLM calls. Use when the input is >100K tokens, when a session is getting long and quality is slipping, or when a task requires touching most of the input (aggregate, pair-reason, transform chunk-by-chunk). Skip for small inputs (<50K tokens) where base-model quality is already fine.
tools: Bash, Read
---

# Recursive Language Model (RLM) skill

This skill implements the inference paradigm from *Recursive Language Models* (Zhang, Kraska, Khattab, MIT CSAIL 2026). The design goal is simple: **keep the orchestrating model's context window small and sharp**, and push the bulk of the material through a Python REPL where sub-LLM calls can process it in parallel slices.

## When to invoke

Use this skill when at least one of these is true:

- The input is a file or directory **larger than ~100K tokens** (~400KB of text).
- The task requires **touching most of the input** — aggregating, pairing, transforming chunk-by-chunk, exhaustive Q&A. Needle-in-haystack tasks can often be done faster with plain `grep` or `Read`.
- You're in a **long session where quality is degrading** and you'd like to isolate heavy per-chunk work from your main context.

Do **not** invoke this skill when:

- The input fits easily (under ~50K tokens) — the RLM overhead makes it slower and worse than just reading it directly.
- The task is a simple lookup that `grep`/`rg` solves cleanly.

## Hard invariants

1. **Never `Read`, `cat`, or otherwise pull the target input into your own context.** The whole point is that you don't see the bulk. You see metadata and short stdout prefixes.
2. **Never paste sub-LLM results back into the conversation manually.** Build them into REPL variables. Stitch them with more REPL code.
3. **Return the final answer through `FINAL(answer)` or `FINAL_VAR("var_name")`**, not by `print`ing it. `FINAL_VAR` is required for long answers — the driver saves it to disk and hands you a path.

## Usage

The helper scripts live next to this SKILL.md:

- `rlm_repl.py` — persistent REPL driver (start / exec / final / stop)
- `rlm_helper.py` — injected into the REPL: `llm_query`, `llm_query_batch`, `FINAL`, `FINAL_VAR`, truncating `print`
- `rlm_providers.py` — pluggable LLM backend (Anthropic native + OpenAI-compatible)

Set an LLM API key in your environment before starting:

- `ANTHROPIC_API_KEY=sk-ant-...` — Anthropic native (default; sub-calls use `claude-haiku-4-5`)
- `OPENAI_API_KEY=sk-...` — OpenAI native (sub-calls use `gpt-4o-mini`)
- `LLM_API_KEY=...` plus `LLM_PROVIDER=openai` and `LLM_BASE_URL=https://...` — any OpenAI-compatible endpoint (Azure OpenAI, OpenRouter, Ollama, LM Studio, GitHub Models)

Override the default model per call with `RLM_SUB_MODEL=...` when a sub-task needs more capability.

### Step 1 — Start a session

For a single file:

```bash
python3 ~/.claude/skills/rlm/rlm_repl.py start --input /absolute/path/to/input.txt
```

For a directory (all text files, filtered by globs):

```bash
python3 ~/.claude/skills/rlm/rlm_repl.py start \
    --input /absolute/path/to/repo \
    --glob '*.py,*.md'
```

Common ignores (`.git`, `node_modules`, `__pycache__`, `.venv`, `dist`, etc.) are pruned automatically. Directory inputs concatenate files with `======== FILE: <path> (<n> chars) ========` headers, and expose a `context_files` variable in the REPL — a list of `{path, start, end, chars}` dicts so you can slice a specific file out of `context`.

You can cap the sub-LLM budget for this session:

```bash
python3 ~/.claude/skills/rlm/rlm_repl.py start --input X.txt --budget 50
```

The response JSON includes `session`, `context_total_chars`, `line_count`, `file_count`, a short `context_prefix` (first ~800 chars), a `files_preview`, and the `budget_limit`. **Use the prefix and files_preview to plan your chunking strategy** — don't ask to see more of it.

### Step 2 — Execute code, iteratively

Write Python that slices `context` and calls sub-LLMs. Pipe it to `exec`:

```bash
python3 ~/.claude/skills/rlm/rlm_repl.py exec --session <SID> <<'PY'
# Chunk by newline, query in parallel.
lines = context.split("\n")
chunk_size = max(1, len(lines) // 50)
chunks = ["\n".join(lines[i:i+chunk_size]) for i in range(0, len(lines), chunk_size)]
answers = llm_query_batch(
    [f"Extract any mention of X from this chunk:\n{c}" for c in chunks],
    concurrency=20,
)
findings = [a for a in answers if a.strip() and "none" not in a.lower()[:20]]
print(f"kept {len(findings)}/{len(chunks)} chunks with findings")
PY
```

The response surfaces only `stdout_prefix` (up to 2KB), a list of REPL `variables` with names/types/sizes, any error, and the current `budget` (calls, limit, tokens_in, tokens_out). If `stdout_truncated` is true, the full text lives at `stdout_full_path` — read it sparingly, never in bulk.

**Prompt caching on sub-calls.** `llm_query` and `llm_query_batch` accept:

- `system="..."` — a system prompt that's automatically cached when reused across calls.
- `prefix="..."` — a user-message prefix prepended to each prompt, also cached.

Use `prefix` when many sub-calls share a large stable preamble (a spec, a reference document, a rubric). The first call pays the cache-write cost; the rest pay ~10× less for the shared portion:

```python
rubric = open("/tmp/rubric.md").read()  # e.g. 20K chars of rubric
answers = llm_query_batch(
    [f"<code>\n{chunk}\n</code>\nGrade against the rubric above." for chunk in chunks],
    system="You are a strict code reviewer.",
    prefix=f"# Rubric\n{rubric}\n\n",
    concurrency=10,
)
```

Pass `cache=False` to opt out for a specific call.

### Step 3 — Finalize

Either a short direct answer:

```bash
python3 ~/.claude/skills/rlm/rlm_repl.py exec --session <SID> <<'PY'
FINAL("The three mentions of X are in sections 2, 7, and 11.")
PY
```

Or a long answer built into a variable:

```bash
python3 ~/.claude/skills/rlm/rlm_repl.py exec --session <SID> <<'PY'
report = "\n\n".join(f"== Finding {i+1} ==\n{f}" for i, f in enumerate(findings))
FINAL_VAR("report")
PY
```

Then retrieve it:

```bash
python3 ~/.claude/skills/rlm/rlm_repl.py final --session <SID>
```

And when you're done:

```bash
python3 ~/.claude/skills/rlm/rlm_repl.py stop --session <SID>
```

## Strategy patterns

### Pattern A — one-shot chunk-and-aggregate

Good for: extract all mentions of X, summarize per-section, transform each chunk.

```python
chunks = [context[i:i+50_000] for i in range(0, len(context), 50_000)]
partials = llm_query_batch([f"<instruction>\n<chunk>\n{c}\n</chunk>" for c in chunks])
final = llm_query("Aggregate these partial answers:\n" + "\n---\n".join(partials))
FINAL(final)
```

### Pattern B — filter then deep-dive

Good for: the signal is sparse. Use code + cheap heuristics to narrow, then spend sub-LLM budget on the shortlist.

```python
import re
candidates = [m.start() for m in re.finditer(r"(?i)error|exception|failed", context)]
# Windows around each candidate
windows = list({context[max(0, p-500):p+1500] for p in candidates})
diagnoses = llm_query_batch([f"Diagnose this log window:\n{w}" for w in windows])
# keep only the substantive ones
kept = [d for d in diagnoses if len(d) > 200]
FINAL_VAR("kept")  # (after binding kept to a pretty-printed string)
```

### Pattern C — pairwise / quadratic

Good for: cross-referencing, consistency-checking, "which two sections conflict".

```python
sections = context.split("\n## ")
# O(n) per-section summaries first — don't do O(n²) full queries
summaries = llm_query_batch([f"One-sentence summary:\n{s}" for s in sections])
# Then O(n²) cheap reasoning over summaries only
pairs_prompt = "Given these section summaries, list pairs that contradict:\n" + \
               "\n".join(f"[{i}] {s}" for i, s in enumerate(summaries))
FINAL(llm_query(pairs_prompt, model="claude-sonnet-4-6"))
```

## Budget discipline

Each session has two independent budgets: **sub-call count** and **total tokens**. Either can warn or hard-stop you.

**Call cap (enforced).** Default 100 sub-calls per session, set via `--budget N` on `start` or `RLM_BUDGET_LIMIT`. `llm_query` / `llm_query_batch` raise `BudgetExceeded` before dispatching a call that would exceed it. The paper's Figure 3 shows 95th-percentile cost is ~5× the median; this cap is your guard against the tail.

**Token budget (opt-in).** Set at `start` or via env vars:

```bash
python3 ~/.claude/skills/rlm/rlm_repl.py start \
    --input X.txt \
    --token-warning 200000 \
    --token-limit 1000000
```

- `--token-warning N` — on each `exec` response that crosses the threshold, a `warnings` field is surfaced back to Claude. A one-shot notice is also written to stderr.
- `--token-limit N` — hard cap. Any subsequent `llm_query` raises `BudgetExceeded`.
- Cache-read tokens count as input tokens (that is the billable rate).

Inspect or change budgets at any time:

```bash
# inspect
python3 ~/.claude/skills/rlm/rlm_repl.py budget --session <SID>

# raise the call cap
python3 ~/.claude/skills/rlm/rlm_repl.py budget --session <SID> --set 200

# set / change / clear token thresholds
python3 ~/.claude/skills/rlm/rlm_repl.py budget --session <SID> --set-token-warning 500000
python3 ~/.claude/skills/rlm/rlm_repl.py budget --session <SID> --set-token-limit 2000000
python3 ~/.claude/skills/rlm/rlm_repl.py budget --session <SID> --clear-token-limit
```

Each `exec` response includes a `budget` object with `calls`, `limit`, `tokens_in`, `tokens_out`, `token_limit`, `token_warning`, `warned_tokens`, plus any active `warnings` strings.

**Operational tips:**
- Default concurrency is 20. Raise with `RLM_CONCURRENCY=40` for throughput; lower to 5 if you're rate-limited.
- When you see a token warning, prefer `FINAL` with current results over another round of sub-calls.
- If a buffer grows past 1MB, stream it to disk with `open(..., "w").write(buf)` inside the REPL rather than holding it in memory.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` *or* `OPENAI_API_KEY` | *(one required)* | Picks the provider; can also use the generic `LLM_API_KEY` |
| `LLM_API_KEY` | *(unset)* | Generic key fallback used by either provider |
| `LLM_PROVIDER` | auto-detect | `anthropic` \| `openai`; explicit override |
| `LLM_BASE_URL` | OpenAI public | Endpoint for the OpenAI-compatible provider (Azure / OpenRouter / Ollama / LM Studio / GitHub Models) |
| `RLM_SUB_MODEL` | provider's fast tier | Sub-LLM model id |
| `RLM_OPENAI_FAST` / `_BALANCED` / `_SMART` | gpt-4o-mini / gpt-4o / gpt-4o | OpenAI tier defaults; overrideable for Ollama / OpenRouter etc. |
| `RLM_SUB_MAX_TOKENS` | `4096` | Per sub-call output cap |
| `RLM_CONCURRENCY` | `20` | Parallel sub-calls in `llm_query_batch` |
| `RLM_STDOUT_MAX` | `2048` | Bytes of stdout surfaced to you per exec |
| `RLM_SESSION_DIR` | `/tmp/rlm-sessions` | Where sessions are stored |
| `RLM_CONTEXT_PREFIX_LEN` | `800` | Bytes of context prefix shown at `start` |
| `RLM_BUDGET_LIMIT` | `100` | Default max sub-calls per session |
| `RLM_TOKEN_LIMIT` | *(unset)* | Default hard cap on total tokens |
| `RLM_TOKEN_WARNING` | *(unset)* | Default token warning threshold |
| `RLM_MAX_PER_FILE_BYTES` | `4194304` | Skip files larger than this when loading a directory |

## What this skill does NOT do

- No deeper recursion (sub-calls are plain LLM calls, not further RLMs). The paper itself caps at depth 1.
- No streaming sub-call results — each `llm_query` blocks until done. Use `llm_query_batch` for parallelism.
- No automatic chunking — you decide how to slice `context`. Common strategies: by line, by regex, by fixed char-window with overlap, by markdown headers.
