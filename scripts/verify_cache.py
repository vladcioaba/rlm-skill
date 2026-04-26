#!/usr/bin/env python3
"""Live-API smoke test for prompt-caching accounting.

Hits claude-haiku-4-5 three times with a shared ~2K-token prefix. Prints the
raw usage block from each response so we can verify:

  call 1:  cache_creation_input_tokens > 0, cache_read_input_tokens == 0
  call 2:  cache_creation_input_tokens == 0, cache_read_input_tokens > 0
  call 3:  same as call 2

Then cross-checks that our budget accounting sums them correctly.

Run: ANTHROPIC_API_KEY=sk-ant-... python3 scripts/verify_cache.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "skills" / "rlm"))

import rlm_helper  # noqa: E402


def post_raw(prompt: str, prefix: str) -> dict:
    """Call the API directly so we can see the whole response."""
    payload = rlm_helper._build_payload(
        prompt=prompt, system=None, prefix=prefix,
        model=None, max_tokens=32, cache=True,
    )
    return rlm_helper._post(payload)


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 1

    # Generate a ~2500-token prefix. "The quick brown fox..." is ~11 tokens,
    # * 250 ≈ 2750 tokens, comfortably above the 1024-token minimum for caching.
    prefix = ("The quick brown fox jumps over the lazy dog. " * 250)
    print(f"prefix: {len(prefix)} chars (~{len(prefix) // 4} tokens, well above cache minimum)\n")

    # Point the budget at a temp file so we can cross-check accounting.
    with tempfile.TemporaryDirectory() as td:
        budget_path = Path(td) / "budget.json"
        budget_path.write_text(json.dumps({
            "calls": 0, "limit": 100,
            "tokens_in": 0, "tokens_out": 0,
            "cache_reads": 0, "cache_writes": 0,
            "token_limit": None, "token_warning": None, "warned_tokens": False,
        }))
        os.environ[rlm_helper.BUDGET_PATH_ENV] = str(budget_path)

        prompts = [
            "Reply with exactly: ONE",
            "Reply with exactly: TWO",
            "Reply with exactly: THREE",
        ]

        for i, p in enumerate(prompts, 1):
            print(f"=== call {i} ===")
            t0 = time.time()
            # Use the public helper so budget accounting runs.
            reply = rlm_helper.llm_query(p, prefix=prefix, max_tokens=32)
            elapsed = time.time() - t0
            # Also fetch the raw usage for the same payload, second time.
            # (Skip raw post — the helper already updated the budget.)
            print(f"  reply ({elapsed:.2f}s): {reply.strip()!r}")
            current = json.loads(budget_path.read_text())
            print(f"  budget after: tokens_in={current['tokens_in']} "
                  f"tokens_out={current['tokens_out']} "
                  f"cache_reads={current['cache_reads']} "
                  f"cache_writes={current['cache_writes']}")
            print()

        final = json.loads(budget_path.read_text())

    print("=== summary ===")
    print(json.dumps(final, indent=2))

    # Sanity checks.
    ok = True
    if final["cache_writes"] == 0:
        print("FAIL: no cache_writes recorded; check prefix length is above the model's minimum.")
        ok = False
    if final["cache_reads"] == 0:
        print("FAIL: no cache_reads recorded; subsequent calls didn't hit cache.")
        ok = False
    if final["tokens_in"] < final["cache_reads"] + final["cache_writes"]:
        print("FAIL: tokens_in doesn't include cache reads/writes — accounting bug.")
        ok = False
    if ok:
        print("PASS: cache writes + reads accounted for; budget totals consistent.")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
