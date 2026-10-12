#!/usr/bin/env python3
"""Pure-python logic tests for the harness — runs in ~1s, no torch, no GPU.

Run this before spending a GPU session: it catches needle placement, scoring and
resume bugs that would otherwise cost you a 9-hour Kaggle window.

    python test_logic.py
"""
from __future__ import annotations

import os
import tempfile

import niah


class CharTokenizer:
    """1 char == 1 token, exactly reversible. Valid stand-in for the pure logic."""

    chat_template = "fake"

    def __call__(self, text, add_special_tokens=False, **kw):
        if isinstance(text, list):
            return {"input_ids": [[ord(c) for c in t] for t in text]}
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids, **kw):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "".join(m["content"] for m in msgs) + "<|assistant|>"


def test_sanity_guards() -> None:
    """The guards must block configurations that cannot measure YaRN."""
    import argparse
    import contextlib
    import io

    def cfg(maxpos):
        return argparse.Namespace(max_position_embeddings=maxpos)

    def mk(**kw):
        a = argparse.Namespace(
            model="Qwen/Qwen3-8B", arm="yarn", factor=4.0, original_max=32768,
            lengths="4096,8192,16384,32768,65536,131072", force=False,
        )
        a.__dict__.update(kw)
        return a

    def check(args, maxpos):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                niah.sanity_check(args, cfg(maxpos))
                return "ALLOW", buf.getvalue()
            except SystemExit:
                return "BLOCK", buf.getvalue()

    cases = [
        # (name, args, config max_position_embeddings, expected verdict, expected substring)
        ("Qwen3 official recipe", mk(), 40960, "ALLOW", "trap worth double-checking"),
        ("naive copy of config 40960", mk(original_max=40960), 40960, "BLOCK", "documented prompt window"),
        ("Qwen3.5 native 256k", mk(model="Qwen/Qwen3.5-4B"), 262144, "BLOCK", "INSIDE the trained window"),
        ("Qwen3 2507 long-context", mk(model="Qwen/Qwen3-4B-Instruct-2507"), 262144, "BLOCK", "INSIDE"),
        ("base arm inside window", mk(arm="base", lengths="4096,8192"), 40960, "BLOCK", "no extrapolation is exercised"),
        ("base arm past window", mk(arm="base"), 40960, "ALLOW", None),
        ("factor too small for 128k", mk(factor=2.0), 40960, "BLOCK", "exceeds the scaled range"),
        ("Qwen2.5-7B recipe", mk(model="Qwen/Qwen2.5-7B-Instruct"), 32768, "ALLOW", None),
        ("--force overrides", mk(original_max=40960, force=True), 40960, "ALLOW", None),
    ]
    for name, args, maxpos, want, needle in cases:
        got, out = check(args, maxpos)
        assert got == want, f"{name}: expected {want}, got {got}\n{out}"
        if needle:
            assert needle in out, f"{name}: missing {needle!r} in\n{out}"


def test_thinking_handling() -> None:
    assert niah.strip_thinking(" Thinking: reasoning 7391 </think>The answer is 7391.") == ("The answer is 7391.", True)
    assert niah.strip_thinking("The answer is 7391.") == ("The answer is 7391.", False)
    # The dangerous case: the model only *considered* the answer, then gave up.
    answer, had = niah.strip_thinking(" Thinking: maybe 7391 </think>I could not find it.")
    assert had and not niah.score_single(answer, "7391")["correct_word"], "an answer inside a think block must not score"


def main() -> None:
    tok = CharTokenizer()
    corpus_ids = tok("The quarterly report mentions ordinary logistics. " * 3000)["input_ids"]
    overhead = len(tok(tok.apply_chat_template([{"role": "system", "content": "s"}, {"role": "user", "content": "x"}], tokenize=False))["input_ids"])
    needle = niah.NEEDLE_POOL[1]

    # --- needle placement -------------------------------------------------
    positions = []
    for d in (0.0, 0.5, 1.0):
        case = niah.build_single_case(tok, corpus_ids, "yarn", "m", 20000, d, needle, "chat", overhead, verify_roundtrip=True)
        assert needle["text"] in case.prompt, f"needle vanished at depth {d}"
        assert case.expected == "7391" and case.needle_key == "berlin"
        assert case.n_tokens > 15000, case.n_tokens
        positions.append(case.prompt.index("Berlin"))
    assert positions[0] < positions[1] < positions[2], f"depth does not move the needle: {positions}"

    # --- multi-needle -----------------------------------------------------
    multi = niah.build_multi_case(tok, corpus_ids, "yarn", "m", 20000, [0.1, 0.5, 0.9], niah.NEEDLE_POOL[:3], "chat", overhead)
    assert all(nd["text"] in multi.prompt for nd in niah.NEEDLE_POOL[:3]), "multi: needle lost"

    # --- scoring ----------------------------------------------------------
    assert niah.score_single("The answer is 7391.", "7391")["correct_word"]
    assert not niah.score_single("The answer is 173910.", "7391")["correct_word"], "word-boundary scoring is broken"
    assert niah.score_single("173910", "7391")["correct_contains"], "lenient view should still match"
    assert niah.score_single("eat a sandwich and sit in Dolores Park", "eat a sandwich")["correct_word"]
    assert not niah.score_single("", "7391")["correct_word"]
    sm = niah.score_multi("7391 and blue-orchid-42", "7391 | blue-orchid-42 | 5566")
    assert (sm["n_hits"], sm["n_needles"]) == (2, 3) and abs(sm["recall"] - 2 / 3) < 1e-9
    assert sm["correct_word"] is False

    # --- rope_scaling wiring ---------------------------------------------
    assert niah.rope_scaling_for("base", 4, 32768, None, 32, 1) is None
    rs = niah.rope_scaling_for("yarn", 4, 32768, None, 32, 1)
    assert rs == {"type": "yarn", "factor": 4, "original_max_position_embeddings": 32768, "beta_fast": 32, "beta_slow": 1}, rs
    assert "beta_fast" not in niah.rope_scaling_for("linear", 4, 32768, None, 32, 1)

    # --- resumable writer -------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "r.jsonl")
        row = {"arm": "yarn", "length": 8192, "depth": 0.5, "needle_key": "berlin", "mode": "single"}
        niah.write_row(path, row)
        niah.write_row(path, row)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write('{"arm":"yarn","leng')  # killed mid-write
        keys = niah.completed_keys(path)
        assert keys == {("yarn", 8192, 0.5, "berlin", "single")}, keys

    test_sanity_guards()
    test_thinking_handling()

    print("ALL LOGIC TESTS PASSED")


if __name__ == "__main__":
    main()
