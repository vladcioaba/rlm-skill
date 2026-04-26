# rlm-skill

A Claude Code skill that implements **Recursive Language Models** ([Zhang, Kraska, Khattab, 2026](https://arxiv.org/abs/2512.24601)) as a thin wrapper around a Python REPL — so the model never needs to load a huge input into its own context window.

**What problem this solves.** Large-context models degrade well before they hit their hard context limit — "context rot." RLMs sidestep this by keeping the orchestrating model's window small and distributing semantic work across cheaper sub-LLM calls operating on slices of the input. Instead of stuffing a 10MB log into the main session, the skill binds it to a Python variable and lets the model write code that loops, filters, chunks, and queries sub-LLMs programmatically.

## Install

```bash
git clone https://github.com/vladcioaba/rlm-skill.git
cd rlm-skill
./install.sh
```

`install.sh` symlinks `skills/rlm` into `~/.claude/skills/rlm/` so Claude Code discovers it automatically in any session.

Then configure an LLM provider (one of the below).

### LLM provider

The skill ships with two backends and auto-detects which one to use from the API key you've set. Override with `LLM_PROVIDER`.

```bash
# Anthropic native (default if ANTHROPIC_API_KEY is set)
export ANTHROPIC_API_KEY=sk-ant-...

# OpenAI native
export OPENAI_API_KEY=sk-...
export LLM_PROVIDER=openai          # only needed if you also have an Anthropic key

# Generic key (works for either provider; pair with LLM_PROVIDER)
export LLM_API_KEY=sk-...
export LLM_PROVIDER=openai          # or "anthropic"

# Azure OpenAI
export LLM_API_KEY=<azure-key>
export LLM_BASE_URL=https://<resource>.openai.azure.com/openai/v1
export LLM_PROVIDER=openai

# OpenRouter (proxies Claude / GPT / Gemini / Llama / Mistral under one API)
export LLM_API_KEY=sk-or-...
export LLM_BASE_URL=https://openrouter.ai/api/v1
export LLM_PROVIDER=openai

# GitHub Models (separate product from Copilot — has a free tier)
export LLM_API_KEY=<your-github-pat>
export LLM_BASE_URL=https://models.github.ai/inference
export LLM_PROVIDER=openai

# Local model via Ollama
ollama serve &
export LLM_API_KEY=ignored          # the SDK requires the var; Ollama doesn't check it
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_PROVIDER=openai
export RLM_OPENAI_FAST=llama3.1
export RLM_OPENAI_BALANCED=llama3.1:70b
export RLM_OPENAI_SMART=llama3.1:70b

# Local model via LM Studio
export LLM_API_KEY=ignored
export LLM_BASE_URL=http://localhost:1234/v1
export LLM_PROVIDER=openai
```

#### A note on GitHub Copilot specifically

Copilot is an editor product, not a programmable backend — there's no general-purpose chat completions API exposed to third-party apps. Closest alternatives:

1. **GitHub Models** (different product, link above) — OpenAI-compatible, free tier, lets you call GPT-4o, Llama, Phi, Mistral, etc. from the skill.
2. **OpenRouter** — gives you Claude / GPT / Gemini under one API.
3. **A native key** — Anthropic, OpenAI, etc. directly.

## Quick start

From a Claude Code session, point the skill at a file or directory:

> Summarize every error pattern in `/var/log/big.log` using the rlm skill.

Under the hood, Claude will:

1. Start a session: `rlm_repl.py start --input /var/log/big.log`
2. Iterate: `rlm_repl.py exec --session <SID>` with Python that chunks `context` and calls `llm_query_batch(...)` in parallel.
3. Finalize: `FINAL_VAR("report")` saves the answer to disk; `rlm_repl.py final` retrieves it.

You can also drive it manually — see [`skills/rlm/SKILL.md`](skills/rlm/SKILL.md) for the full contract.

## Features

- **Provider-pluggable** — Anthropic native + any OpenAI-compatible endpoint (OpenAI, Azure, OpenRouter, Ollama, LM Studio, GitHub Models). Auto-detected from env.
- **Persistent REPL** with `context` variable pre-loaded; the host model never sees the raw content, only metadata + bounded stdout.
- **Parallel sub-calls** via `llm_query_batch` (up to 20 concurrent by default).
- **Prompt caching** on system prompts and shared prefixes — ~10× cheaper across a batch on Anthropic; automatic on OpenAI when prefixes repeat.
- **Extended thinking** via `thinking_budget` — translates to Anthropic's `thinking` block or OpenAI's `reasoning_effort` (low/medium/high).
- **Directory loader** with sensible ignores (`.git`, `node_modules`, `__pycache__`, etc.) and glob filters.
- **Call + token budgets** with warning thresholds and hard caps to guard the tail-cost failure mode (paper Figure 3).
- **Zero runtime deps** — stdlib + urllib only.

## Files

```
skills/rlm/
├── SKILL.md              contract and usage instructions loaded by the host model
├── rlm_helper.py         llm_query, llm_query_batch, FINAL/FINAL_VAR, budget accounting
├── rlm_providers.py      AnthropicProvider + OpenAIProvider + auto-detect factory
└── rlm_repl.py           start / exec / final / stop / budget / list subcommands
install.sh                symlink into ~/.claude/skills/rlm
```

## Tests

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

# Unit + integration — no network, ~5 seconds:
python3 -m pytest tests/ -v

# Live — hits the API with a few small calls (<$0.01):
export ANTHROPIC_API_KEY=sk-ant-...   # or set up an OpenAI provider as above
python3 -m pytest tests/test_live.py -v
```

53 unit + integration tests; 5 opt-in live tests that auto-skip without an API key. Provider tests mock the HTTP layer so no key is needed for the offline suite. See [`tests/test_live.py`](tests/test_live.py) for what gets verified — notably that prompt-cache writes fire on the first call and cache reads on subsequent calls, and that budget accounting sums regular input + cache-read + cache-creation tokens correctly.

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` *or* `OPENAI_API_KEY` | *(one required)* | Picks the provider; can also use the generic `LLM_API_KEY` |
| `LLM_API_KEY` | *(unset)* | Generic key fallback used by either provider |
| `LLM_PROVIDER` | auto-detect | `anthropic` \| `openai`; explicit override |
| `LLM_BASE_URL` | OpenAI public | Endpoint for the OpenAI-compatible provider |
| `RLM_SUB_MODEL` | provider's fast tier | Default model id for sub-calls |
| `RLM_OPENAI_FAST` / `_BALANCED` / `_SMART` | gpt-4o-mini / gpt-4o / gpt-4o | OpenAI tier defaults; overrideable for Ollama / OpenRouter etc. |
| `RLM_BUDGET_LIMIT` | `100` | Default sub-call budget per session |
| `RLM_TOKEN_LIMIT` | *(unset)* | Hard token cap per session |
| `RLM_TOKEN_WARNING` | *(unset)* | Token warning threshold |
| `RLM_CONCURRENCY` | `20` | Parallel sub-calls in `llm_query_batch` |
| `RLM_STDOUT_MAX` | `2048` | Max bytes of stdout returned per `exec` call |
| `RLM_SESSION_DIR` | `/tmp/rlm-sessions` | Where session state is stored |

## Paper

Zhang, A.L., Kraska, T., Khattab, O. *Recursive Language Models.* arXiv:2512.24601, January 2026. [PDF](https://arxiv.org/pdf/2512.24601)

This implementation differs from the paper's reference code in a few ways:
- Pluggable provider — works against Anthropic native or any OpenAI-compatible endpoint, not just one vendor's API.
- Async parallelism via `concurrent.futures` for `llm_query_batch`.
- Prompt caching on by default for system prompts and shared prefixes (where supported).
- Depth=1 recursion only, matching the paper.

## License

MIT — see [`LICENSE`](LICENSE).
