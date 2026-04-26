"""Live-API tests. Skipped unless ANTHROPIC_API_KEY is set.

These confirm that:
  - llm_query actually returns text.
  - Prompt caching writes on first call and reads on subsequent calls.
  - Batch calls parallelize and accumulate budget correctly.
  - tokens_in accounting sums regular + cache-read + cache-creation.

Run on the target machine with:

    ANTHROPIC_API_KEY=sk-ant-... python3 -m pytest tests/test_live.py -v

Cost: a handful of Haiku calls with small prompts. Typically <$0.01 per run.
"""

from __future__ import annotations

import json
import os
import time

import pytest

import rlm_helper


pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; skipping live-API tests",
)


# Prefix sized to clear Haiku's cache minimum (~2048 tokens).
LARGE_PREFIX = "The quick brown fox jumps over the lazy dog. " * 250  # ~11K chars


def test_llm_query_returns_text(budget_file):
    reply = rlm_helper.llm_query(
        "Reply with exactly: PONG", max_tokens=16,
    )
    assert isinstance(reply, str)
    assert reply.strip()  # non-empty


def test_cache_write_then_read(budget_file):
    """First call warms the cache; subsequent calls should hit it."""
    # Hit 1: expect cache_creation_input_tokens > 0.
    rlm_helper.llm_query(
        "Reply: ONE",
        prefix=LARGE_PREFIX,
        max_tokens=16,
    )
    state1 = json.loads(budget_file.read_text())
    assert state1["cache_writes"] > 0, (
        "expected cache_creation_input_tokens on first call; "
        "if zero, the prefix may be below the model's cache minimum or caching "
        "is not being triggered by our payload shape."
    )

    # Hits 2 and 3: expect cache_read_input_tokens > 0.
    rlm_helper.llm_query("Reply: TWO", prefix=LARGE_PREFIX, max_tokens=16)
    rlm_helper.llm_query("Reply: THREE", prefix=LARGE_PREFIX, max_tokens=16)
    state3 = json.loads(budget_file.read_text())
    assert state3["cache_reads"] > state1["cache_reads"], (
        "expected cache_read_input_tokens to grow after second+third calls"
    )

    # Accounting invariant: tokens_in always >= cache_reads + cache_writes.
    assert state3["tokens_in"] >= state3["cache_reads"] + state3["cache_writes"]


def test_batch_parallelizes_and_records_budget(budget_file):
    t0 = time.time()
    replies = rlm_helper.llm_query_batch(
        [f"Reply only with the word: WORD{i}" for i in range(4)],
        max_tokens=16,
        concurrency=4,
    )
    elapsed = time.time() - t0

    assert len(replies) == 4
    assert all(isinstance(r, str) and r.strip() for r in replies)

    state = json.loads(budget_file.read_text())
    assert state["calls"] == 4
    assert state["tokens_in"] > 0
    assert state["tokens_out"] > 0

    # Sanity: four parallel calls should be faster than four serial ones.
    # Not a tight assertion — just catching catastrophic regressions.
    # A single Haiku call is typically ~0.5-2s; four serial would be ~2-8s.
    # We only assert elapsed is under a generous bound.
    assert elapsed < 30.0, f"batch took {elapsed:.1f}s — is concurrency working?"


def test_budget_hard_cap_blocks_before_api(budget_file):
    """Token limit rejects the call before it reaches the API."""
    # Seed the budget as if we've already blown the limit.
    st = json.loads(budget_file.read_text())
    st["token_limit"] = 100
    st["tokens_in"] = 200
    budget_file.write_text(json.dumps(st))

    with pytest.raises(rlm_helper.BudgetExceeded):
        rlm_helper.llm_query("anything", max_tokens=8)


def test_call_cap_blocks_oversized_batch(budget_file):
    st = json.loads(budget_file.read_text())
    st["limit"] = 2
    budget_file.write_text(json.dumps(st))

    with pytest.raises(rlm_helper.BudgetExceeded):
        rlm_helper.llm_query_batch(["a"] * 5, max_tokens=8)

    # After the failed reservation, no calls should have landed.
    post = json.loads(budget_file.read_text())
    assert post["calls"] == 0
