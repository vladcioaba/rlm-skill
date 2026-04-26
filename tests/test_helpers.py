"""Unit tests for rlm_helper — no network required."""

from __future__ import annotations

import io
import json

import pytest

import rlm_helper
import rlm_providers


# ---------- AnthropicProvider payload shape (formerly _build_payload) ---

def _capture_anthropic(monkeypatch) -> list:
    captured: list[dict] = []

    def fake_post(url, headers, payload, timeout=300.0):
        captured.append(payload)
        return {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1,
                      "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 0},
        }

    monkeypatch.setattr("rlm_providers._post", fake_post)
    return captured


def _anth() -> rlm_providers.AnthropicProvider:
    return rlm_providers.AnthropicProvider(api_key="sk-test-key")


def test_payload_minimal(monkeypatch):
    captured = _capture_anthropic(monkeypatch)
    _anth().chat(prompt="hi", system=None, prefix=None,
                 model="claude-x", max_tokens=4096,
                 thinking_budget=None, cache=True)
    p = captured[0]
    assert p["messages"] == [{"role": "user", "content": "hi"}]
    assert "system" not in p
    assert p["model"] == "claude-x"


def test_payload_system_cached_by_default(monkeypatch):
    captured = _capture_anthropic(monkeypatch)
    _anth().chat(prompt="hi", system="sys prompt", prefix=None,
                 model="claude-x", max_tokens=4096,
                 thinking_budget=None, cache=True)
    p = captured[0]
    assert p["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert p["system"][0]["text"] == "sys prompt"


def test_payload_system_not_cached_when_cache_false(monkeypatch):
    captured = _capture_anthropic(monkeypatch)
    _anth().chat(prompt="hi", system="sys", prefix=None,
                 model="claude-x", max_tokens=4096,
                 thinking_budget=None, cache=False)
    assert captured[0]["system"] == "sys"


def test_payload_prefix_cached_first_block_only(monkeypatch):
    captured = _capture_anthropic(monkeypatch)
    _anth().chat(prompt="q", system=None, prefix="long preamble",
                 model="claude-x", max_tokens=4096,
                 thinking_budget=None, cache=True)
    blocks = captured[0]["messages"][0]["content"]
    assert blocks[0]["text"] == "long preamble"
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert blocks[1]["text"] == "q"
    assert "cache_control" not in blocks[1]


def test_payload_prefix_not_cached_when_cache_false(monkeypatch):
    captured = _capture_anthropic(monkeypatch)
    _anth().chat(prompt="q", system=None, prefix="pre",
                 model="claude-x", max_tokens=4096,
                 thinking_budget=None, cache=False)
    blocks = captured[0]["messages"][0]["content"]
    assert "cache_control" not in blocks[0]


def test_thinking_budget_appears_in_payload(monkeypatch):
    captured = _capture_anthropic(monkeypatch)
    _anth().chat(prompt="hi", system=None, prefix=None,
                 model="claude-x", max_tokens=4096,
                 thinking_budget=8000, cache=True)
    assert captured[0]["thinking"] == {"type": "enabled", "budget_tokens": 8000}


def test_thinking_budget_bumps_max_tokens(monkeypatch):
    captured = _capture_anthropic(monkeypatch)
    _anth().chat(prompt="hi", system=None, prefix=None,
                 model="claude-x", max_tokens=512,
                 thinking_budget=4000, cache=True)
    assert captured[0]["max_tokens"] >= 4000 + 1024


# ---------- ChatResponse usage parsing ---------------------------------

def test_anthropic_usage_sums_all_three_input_kinds(monkeypatch):
    def fake_post(url, headers, payload, timeout=300.0):
        return {
            "content": [{"type": "text", "text": "x"}],
            "usage": {"input_tokens": 50,
                      "cache_read_input_tokens": 200,
                      "cache_creation_input_tokens": 800,
                      "output_tokens": 30},
        }
    monkeypatch.setattr("rlm_providers._post", fake_post)
    r = _anth().chat(prompt="x", system=None, prefix=None,
                     model="claude-x", max_tokens=4096,
                     thinking_budget=None, cache=True)
    assert r.tokens_in == 50 + 200 + 800
    assert r.tokens_out == 30
    assert r.cache_reads == 200
    assert r.cache_writes == 800


# ---------- Budget / _reserve -------------------------------------------

def test_reserve_under_limit(budget_file):
    state = rlm_helper._reserve(1)
    assert state["calls"] == 1
    state = rlm_helper._reserve(5)
    assert state["calls"] == 6


def test_reserve_at_limit_raises(budget_file):
    budget_file.write_text(json.dumps({
        "calls": 99, "limit": 100, "tokens_in": 0, "tokens_out": 0,
        "token_limit": None, "token_warning": None, "warned_tokens": False,
    }))
    with pytest.raises(rlm_helper.BudgetExceeded):
        rlm_helper._reserve(2)


def test_reserve_fails_fast_on_token_limit_already_exceeded(budget_file):
    budget_file.write_text(json.dumps({
        "calls": 0, "limit": 100,
        "tokens_in": 6000, "tokens_out": 0,
        "token_limit": 5000, "token_warning": None, "warned_tokens": False,
    }))
    with pytest.raises(rlm_helper.BudgetExceeded):
        rlm_helper._reserve(1)


def test_reserve_allows_when_token_limit_unset(budget_file):
    budget_file.write_text(json.dumps({
        "calls": 0, "limit": 100,
        "tokens_in": 10**9, "tokens_out": 0,
        "token_limit": None, "token_warning": None, "warned_tokens": False,
    }))
    rlm_helper._reserve(1)


def test_record_tokens_updates_totals(budget_file):
    rlm_helper._record_tokens(100, 20, cache_reads=50, cache_writes=30)
    state = json.loads(budget_file.read_text())
    assert state["tokens_in"] == 100
    assert state["tokens_out"] == 20
    assert state["cache_reads"] == 50
    assert state["cache_writes"] == 30


def test_record_tokens_fires_warning_once(budget_file, capsys):
    budget_file.write_text(json.dumps({
        "calls": 0, "limit": 100,
        "tokens_in": 0, "tokens_out": 0,
        "token_limit": 10000, "token_warning": 100, "warned_tokens": False,
    }))
    rlm_helper._record_tokens(150, 0)
    state = json.loads(budget_file.read_text())
    assert state["warned_tokens"] is True
    err = capsys.readouterr().err
    assert "token warning" in err.lower()

    rlm_helper._record_tokens(50, 0)
    err2 = capsys.readouterr().err
    assert err2 == ""


# ---------- FINAL / FINAL_VAR -------------------------------------------

def test_final_sets_sentinel_in_caller_globals():
    g: dict = {"FINAL": rlm_helper.FINAL}
    exec("FINAL('the answer')", g)  # noqa: S102
    assert rlm_helper._FINAL_KEY in g
    assert g[rlm_helper._FINAL_KEY]["kind"] == "direct"
    assert g[rlm_helper._FINAL_KEY]["value"] == "the answer"


def test_final_var_requires_existing_var():
    g: dict = {"FINAL_VAR": rlm_helper.FINAL_VAR}
    with pytest.raises(rlm_helper.RLMError):
        exec("FINAL_VAR('does_not_exist')", g, g)  # noqa: S102


def test_final_var_accepts_defined_var():
    g: dict = {"FINAL_VAR": rlm_helper.FINAL_VAR, "out": "hello"}
    exec("FINAL_VAR('out')", g, g)  # noqa: S102
    assert g[rlm_helper._FINAL_KEY] == {"kind": "var", "name": "out"}


# ---------- Truncating print --------------------------------------------

def test_truncating_print_writes_full_text_to_buffer():
    buf = io.StringIO()
    p = rlm_helper.make_truncating_print(buf, cap=10)
    p("hello", "world")
    assert buf.getvalue() == "hello world\n"


def test_truncating_print_handles_sep_and_end():
    buf = io.StringIO()
    p = rlm_helper.make_truncating_print(buf, cap=10)
    p("a", "b", "c", sep="-", end="!")
    assert buf.getvalue() == "a-b-c!"
