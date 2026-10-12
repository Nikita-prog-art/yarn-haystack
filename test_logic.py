#!/usr/bin/env python3
"""Pure-python logic tests for the harness — runs in ~1s, no torch, no GPU.

Run this before spending a GPU session: it catches needle placement, scoring and
resume bugs that would otherwise cost you a 9-hour Kaggle window.

    python test_logic.py
"""
from __future__ import annotations

import contextlib
import io
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


def cell_source(cell: dict) -> str:
    """Cell source regardless of whether it is stored as a string or a line list."""
    src = cell["source"]
    return src if isinstance(src, str) else "".join(src)


def find_code_cell(nb: dict, prefix: str) -> dict:
    return next(c for c in nb["cells"]
                if c["cell_type"] == "code" and cell_source(c).lstrip().startswith(prefix))


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
        # Advisory, not fatal: comparing base against yarn entirely inside the window
        # is how the short-context cost of static YaRN gets measured. Refusing it
        # killed wave1a's base arm while its yarn arm ran, halving the comparison.
        ("base arm inside window", mk(arm="base", lengths="4096,8192"), 40960, "ALLOW", "legitimate experiment"),
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


def test_notebook_install_cell() -> None:
    """Run the notebook's install cell against simulated images.

    Guards the fix for the pip conflict report: matplotlib must never be touched
    (ydata-profiling pins <=3.10), transformers must be upgraded only when too old,
    and the cell must hard-fail if it is still too old afterwards.
    """
    import importlib.metadata as md
    import json
    import subprocess
    import sys
    import types

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yarn_niah.ipynb")
    nb = json.load(open(path, encoding="utf-8"))
    cell = find_code_cell(nb, "# 2) Install")
    src = cell_source(cell)

    def run_cell(start_env, still_old_after_pip=False):
        env = dict(start_env)
        calls = []

        def fake_version(pkg):
            if pkg in env:
                return env[pkg]
            raise md.PackageNotFoundError(pkg)

        def fake_run(cmd, **kw):
            calls.append(cmd)
            for arg in cmd:
                if arg.startswith("transformers"):
                    env["transformers"] = "4.43.0" if still_old_after_pip else "4.52.4"
                elif arg in ("bitsandbytes", "accelerate", "datasets"):
                    env[arg] = "9.9.9"
            return types.SimpleNamespace(returncode=0)

        fake_torch = types.ModuleType("torch")
        fake_torch.__version__ = env.get("torch", "2.5.1")
        fake_torch.cuda = types.SimpleNamespace(
            is_available=lambda: True, device_count=lambda: 2,
            get_device_properties=lambda i: types.SimpleNamespace(name="T4", total_memory=16 * 2**30),
        )
        saved = (md.version, subprocess.run, sys.modules.get("torch"))
        md.version, subprocess.run, sys.modules["torch"] = fake_version, fake_run, fake_torch
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(src, "<install cell>", "exec"), {"__name__": "__main__"})
        finally:
            md.version, subprocess.run = saved[0], saved[1]
            if saved[2] is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = saved[2]
        requested = [a for c in calls for a in c if not a.startswith("-") and a != sys.executable]
        return requested

    kaggle = {"transformers": "4.43.0", "matplotlib": "3.8.4", "accelerate": "1.1.1",
              "datasets": "3.1.0", "numpy": "1.26.4", "torch": "2.5.1"}
    req = run_cell(kaggle)
    assert "transformers>=4.51" in req, req
    assert "bitsandbytes" in req, req
    assert not any(p.startswith("matplotlib") for p in req), f"matplotlib must not be touched: {req}"
    assert not any(p.split(">=")[0].strip() in ("accelerate", "datasets") for p in req), req

    modern = dict(kaggle, transformers="4.52.4", bitsandbytes="0.44.1")
    assert run_cell(modern) == [], f"nothing should be installed on a modern image: {run_cell(modern)}"

    # If pip reports success but transformers is still too old, the cell must refuse
    # to continue rather than run a YaRN arm that is secretly the control arm.
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            run_cell(kaggle, still_old_after_pip=True)
    except AssertionError as exc:
        assert "4.51" in str(exc), exc
    else:
        raise AssertionError("install cell did not hard-fail on transformers<4.51")


def test_placement() -> None:
    """Model placement: the bug where fp16 weights stayed on CPU and died in LayerNorm."""
    cases = [
        # (load_in_4bit, device_map, dtype, cuda, expected device, expected dtype, why)
        (False, None, "float16", True, "cuda", "float16", "the ordinary GPU path"),
        (False, None, "float16", False, "cpu", "float32", "no GPU: fp32, not a LayerNorm crash"),
        (False, None, "bfloat16", False, "cpu", "float32", "no GPU: fp32"),
        (False, None, "float32", False, "cpu", "float32", "no GPU, fp32 as asked"),
        (False, "auto", "float16", True, None, "float16", "accelerate places it, do not move it"),
        (True, None, "float16", True, None, "float16", "4-bit implies device_map=auto"),
    ]
    for li4, dm, dtype, cuda, want_dev, want_dtype, why in cases:
        dev, eff, _note = niah.resolve_placement(li4, dm, dtype, cuda)
        assert (dev, eff) == (want_dev, want_dtype), f"{why}: got {(dev, eff)}, want {(want_dev, want_dtype)}"
    # We must never claim a device for the accelerate-managed paths, or we would
    # move a sharded model onto one GPU.
    assert niah.resolve_placement(True, None, "float16", True)[0] is None
    assert niah.resolve_placement(False, "auto", "float16", True)[0] is None


def test_notebook_runner_cell() -> None:
    """The notebook runner must stream output AND put the child's error in the exception.

    Regression guard: subprocess.run(..., check=True) raises a bare
    CalledProcessError that hides the real error, which is the only thing you need.
    """
    import json

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yarn_niah.ipynb")
    nb = json.load(open(path, encoding="utf-8"))
    cell = find_code_cell(nb, "import subprocess, sys, os")
    src = cell_source(cell)
    # In the kernel notebooks the runner cell also carries its own niah(...) call;
    # in the interactive one the runner is standalone. Drop the call if present.
    if "\n# " in src:
        src = src[: src.index("\n# ")]
    # Drop the cell's `import subprocess` too: it would rebind the name and defeat
    # the fake below (which is how this test first failed, by running run.py for real).
    src = "\n".join(ln for ln in src.split("\n") if not ln.startswith("import subprocess"))

    seen = {}

    class FakeProc:
        def __init__(self, cmd, **kw):
            seen["cmd"] = cmd
            self.returncode = 1
            self.stdout = iter(["starting\n", "RuntimeError: the actual reason\n"])

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def wait(self):
            return self.returncode

    class FakeSubprocess:
        PIPE, STDOUT = -1, -2

        @staticmethod
        def Popen(cmd, **kw):
            return FakeProc(cmd, **kw)

    import collections, sys
    ns = {"WORK": "/tmp/x", "subprocess": FakeSubprocess,
          "sys": sys, "os": os, "collections": collections}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, "<runner>", "exec"), ns)
        try:
            ns["niah"]("--model", "m")
        except RuntimeError as exc:
            msg = str(exc)
        else:
            raise AssertionError("runner did not raise on a non-zero exit")

    assert "exited with 1" in msg, msg
    assert "the actual reason" in msg, f"the child's error is not in the exception: {msg}"
    # -u: without it the child's stdout is block-buffered through the pipe and a
    # long 128k run looks frozen.
    assert "-u" in seen["cmd"], f"child must run unbuffered: {seen['cmd']}"


def test_notebooks_are_not_glued() -> None:
    """Every notebook cell must survive nbformat's "" concatenation.

    Regression guard for a silent bug that cost a real kernel run: cells were
    stored as line lists whose elements had lost their trailing newline, so readers
    glued them. A cell beginning with a comment then became one long comment - it
    still parsed, the kernel reported COMPLETE, and nothing had run.

    The earlier version of this check compiled "\\n".join(...), which is not what
    nbformat readers do, so it passed while every notebook was broken. Join the way
    the consumer does.
    """
    import json

    root = os.path.dirname(os.path.abspath(__file__))
    paths = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in (".local", ".git", "__pycache__")]
        paths += [os.path.join(dirpath, f) for f in files if f.endswith(".ipynb")]
    assert paths, "no notebooks found to check"

    for path in sorted(paths):
        nb = json.load(open(path, encoding="utf-8"))
        rel = os.path.relpath(path, root)
        for i, cell in enumerate(nb.get("cells", [])):
            src = cell.get("source")
            if isinstance(src, list):
                assert len(src) <= 1 or all(s.endswith("\n") for s in src[:-1]), (
                    f"{rel} cell {i}: list source would be glued by nbformat - "
                    "store source as a single string (tools/normalize_notebooks.py)"
                )
                src = "".join(src)
            if cell.get("cell_type") == "code":
                try:
                    compile(src, f"{rel}:{i}", "exec")
                except SyntaxError as exc:
                    raise AssertionError(f"{rel} cell {i} does not compile: {exc}") from exc


def test_rope_scaling_shapes() -> None:
    """Both transformers generations must get the shape they expect.

    v4: rope_theta is a sibling attribute, rope_scaling a separate dict.
    v5: one rope_parameters dict holds rope_theta AND the scheme, and the scaling
    functions read rope_theta from inside it - so the entry must be MERGED, not
    replaced. Replacing it made Qwen2RotaryEmbedding die with a KeyError that
    mentions neither RoPE nor the config.
    """
    # base means "leave the shipped default alone"
    assert niah.rope_scaling_for("base", 4, 32768, None, 32, 1) is None
    assert niah.rope_scaling_for("base", 4, 32768, None, 32, 1, {"rope_theta": 1e6}) is None

    # v4 shape: no rope_theta inside the scaling dict
    rs4 = niah.rope_scaling_for("yarn", 4, 32768, None, 32, 1)
    assert rs4 == {"rope_type": "yarn", "factor": 4,
                   "original_max_position_embeddings": 32768,
                   "beta_fast": 32, "beta_slow": 1}, rs4
    assert "rope_theta" not in rs4
    assert "beta_fast" not in niah.rope_scaling_for("linear", 4, 32768, None, 32, 1)

    # v5 shape: merged into the existing dict, rope_theta preserved
    existing = {"rope_theta": 1_000_000.0, "rope_type": "default"}
    rs5 = niah.rope_scaling_for("yarn", 4, 32768, None, 32, 1, existing)
    assert rs5["rope_theta"] == 1_000_000.0, rs5
    assert rs5["rope_type"] == "yarn", rs5
    assert existing["rope_type"] == "default", "the original dict must not be mutated"

    # applying it: the three cases that differ between the generations
    class V4:
        rope_theta = 1e6
        rope_scaling = {"type": "default"}

    class V5:
        rope_parameters = {"rope_theta": 1e6, "rope_type": "default"}

    c = V4()
    assert niah.apply_rope_scaling(c, None) == "v4" and c.rope_scaling is None
    c = V4()
    assert niah.apply_rope_scaling(c, rs4) == "v4" and c.rope_scaling == rs4

    c = V5()
    assert niah.apply_rope_scaling(c, None) == "v5"
    # the bug: v5 must keep rope_theta when the base arm "clears" scaling
    assert c.rope_parameters == {"rope_theta": 1e6, "rope_type": "default"}, c.rope_parameters
    c = V5()
    assert niah.apply_rope_scaling(c, rs5) == "v5"
    assert c.rope_parameters["rope_theta"] == 1e6, c.rope_parameters
    assert c.rope_parameters["rope_type"] == "yarn", c.rope_parameters


def test_kernel_scripts() -> None:
    """The generated Kaggle scripts must be pushable and self-consistent.

    Guards the switch from notebooks to `kernel_type: "script"`: a stale metadata
    block would silently push a notebook that no longer exists, or a script whose
    pinned commit does not match the one it claims in its docstring.
    """
    import json
    import re

    root = os.path.dirname(os.path.abspath(__file__))
    folders = [d for d in sorted(os.listdir(os.path.join(root, "kaggle")))
               if os.path.isdir(os.path.join(root, "kaggle", d)) and not d.startswith("_")]
    assert folders, "no kernel folders generated - run kaggle/build_kernels.py"

    for slug in folders:
        folder = os.path.join(root, "kaggle", slug)
        meta_path = os.path.join(folder, "kernel-metadata.json")
        assert os.path.exists(meta_path), f"{slug}: missing kernel-metadata.json"
        meta = json.load(open(meta_path, encoding="utf-8"))

        assert meta["kernel_type"] == "script", f"{slug}: expected a script kernel, got {meta['kernel_type']}"
        assert meta["id"].endswith("/yarn-niah-" + slug), meta["id"]
        assert meta["machine_shape"] in ("NvidiaTeslaT4", "NvidiaTeslaP100"), meta["machine_shape"]
        assert meta["enable_gpu"] == "true" and meta["enable_internet"] == "true", (
            f"{slug}: GPU and internet are both required"
        )

        script = os.path.join(folder, meta["code_file"])
        assert os.path.exists(script), f"{slug}: code_file {meta['code_file']} does not exist"
        src = open(script, encoding="utf-8").read()
        try:
            compile(src, script, "exec")
        except SyntaxError as exc:
            raise AssertionError(f"{slug}: generated script does not compile: {exc}") from exc

        assert "@@" not in src, f"{slug}: unsubstituted template token left in the script"
        commit = re.search(r'^COMMIT = "([a-f0-9]{40})"$', src, flags=re.M)
        assert commit, f"{slug}: no pinned commit found"
        assert commit.group(1) in src, f"{slug}: commit not referenced"

        # Both arms, same grid: the ablation is worthless otherwise.
        assert '("base", [])' in src and '"yarn"' in src, f"{slug}: both arms must run"
        for flag in ("--lengths", "--depths", "--model"):
            assert flag in src, f"{slug}: missing {flag}"


def test_attention_setup() -> None:
    """On sm<80 the GQA flag is unusable, so expanding K/V must be forced.

    Measured on a real T4 by kaggle/_probe_attn: mem_efficient + enable_gqa raised
    "RuntimeError: Invalid backend", while mem_efficient without the flag peaked at
    0.10 GiB @8k and 0.76 GiB @64k - against 7.24 GiB for the math backend at just
    8k. That difference is the whole reason 128k is possible on a free T4.
    """
    import sys
    import types

    import attention_setup

    assert attention_setup.should_force_repeat_kv((7, 5)) is True
    assert attention_setup.should_force_repeat_kv((8, 0)) is False
    assert attention_setup.should_force_repeat_kv(None) is False

    # the arithmetic behind the budget guard
    assert abs(attention_setup.attention_budget_gib(12, 8192) - 1.5) < 0.01
    assert attention_setup.attention_budget_gib(12, 131072) > 300

    enabled_calls = {}

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_capability():
            return (7, 5)

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = FakeCuda()
    fake_torch.backends = types.SimpleNamespace(cuda=types.SimpleNamespace(
        enable_flash_sdp=lambda on: enabled_calls.__setitem__("flash", on),
        enable_mem_efficient_sdp=lambda on: enabled_calls.__setitem__("mem_efficient", on),
        enable_math_sdp=lambda on: enabled_calls.__setitem__("math", on),
    ))
    sdpa_mod = types.ModuleType("transformers.integrations.sdpa_attention")
    sdpa_mod.use_gqa_in_sdpa = lambda *a, **k: True      # as transformers ships it
    tr = types.ModuleType("transformers")
    tr.integrations = types.ModuleType("transformers.integrations")

    names = ("torch", "transformers", "transformers.integrations", "transformers.integrations.sdpa_attention")
    saved = {n: sys.modules.get(n) for n in names}
    saved_patched = attention_setup._PATCHED
    # attention_setup imports torch lazily, so injecting it into sys.modules is
    # enough for the call to pick up the fake.
    sys.modules.update({"torch": fake_torch, "transformers": tr,
                        "transformers.integrations": tr.integrations,
                        "transformers.integrations.sdpa_attention": sdpa_mod})
    try:
        info = attention_setup.configure_attention(allow_math=True, verbose=False)
    finally:
        attention_setup._PATCHED = saved_patched
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod

    assert info["forced_repeat_kv"] is True, info
    assert info["enabled"] == ["mem_efficient", "math"], info
    assert enabled_calls == {"flash": False, "mem_efficient": True, "math": True}, enabled_calls
    assert sdpa_mod.use_gqa_in_sdpa(None, None, None) is False, "the GQA flag was not disabled"


def test_no_undefined_names() -> None:
    """Names used inside the main functions must actually resolve.

    This is a poor man's pyflakes, and it exists because a NameError cost a real
    kernel run: `attention_setup` was imported inside load_model_and_tokenizer but
    used in run(), which is only reachable 90 seconds into a GPU session.
    """
    import ast
    import builtins

    root = os.path.dirname(os.path.abspath(__file__))
    tree = ast.parse(open(os.path.join(root, "niah.py"), encoding="utf-8").read())

    module_names = set(dir(builtins))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                module_names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module_names.add(node.name)
        elif isinstance(node, ast.Assign):
            module_names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            module_names.add(node.target.id)

    def unknown_in(fn_name: str) -> set[str]:
        fn = next((n for n in tree.body
                   if isinstance(n, ast.FunctionDef) and n.name == fn_name), None)
        if fn is None:
            return set()
        local = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                local.add(node.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                local.update((a.asname or a.name).split(".")[0] for a in node.names)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                local.add(node.name)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                local.add(node.name)
            elif isinstance(node, ast.comprehension) and isinstance(node.target, ast.Name):
                local.add(node.target.id)
        return {n.id for n in ast.walk(fn)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                and n.id not in local and n.id not in module_names}

    bad = {fn: sorted(unknown_in(fn))
           for fn in ("run", "load_model_and_tokenizer", "load_vllm", "generate_hf",
                      "generate_vllm", "build_single_case", "build_multi_case")}
    bad = {k: v for k, v in bad.items() if v}
    assert not bad, f"names that would raise NameError at runtime: {bad}"


def test_prefill_chunking() -> None:
    """Unchunked prefill activations are the second half of the 128k problem.

    Attention was fixed first, and the run then died inside down_proj instead: one
    131k prefill allocates every MLP intermediate at full length. The arithmetic
    below is what the automatic chunk size is chosen from.
    """
    import attention_setup

    class Qwen2_5_1_5B:
        intermediate_size, hidden_size = 8960, 1536

    class Qwen2_5_7B:
        intermediate_size, hidden_size = 18944, 3584

    per = attention_setup.bytes_per_token_activations(Qwen2_5_1_5B)
    assert 50_000 < per < 70_000, per
    # the number that explains the down_proj OOM
    assert 131072 * per / 2 ** 30 > 6, 131072 * per / 2 ** 30

    chunk = attention_setup.auto_prefill_chunk_size(Qwen2_5_1_5B)
    assert chunk, chunk
    assert chunk * per / 2 ** 30 < 3, (chunk, chunk * per / 2 ** 30)
    # a bigger model must not get a bigger chunk
    assert attention_setup.auto_prefill_chunk_size(Qwen2_5_7B) <= chunk

    # At long context the explicit per-chunk mask dominates: q != kv during chunked
    # prefill, so transformers materialises q x kv. chunk=32768 against 98304 keys
    # is exactly the 6.00 GiB allocation that OOMed on the T4.
    long_chunk = attention_setup.auto_prefill_chunk_size(Qwen2_5_1_5B, 131072)
    assert long_chunk < chunk, (long_chunk, chunk)
    mask_gib = long_chunk * 131072 * 2 / 2 ** 30
    assert mask_gib < 0.8, mask_gib
    assert 32768 * 98304 * 2 / 2 ** 30 > 5.9   # the observed failure, for the record

    # configs without the sizes must yield None rather than a bogus number
    assert attention_setup.bytes_per_token_activations(object()) == 0
    assert attention_setup.auto_prefill_chunk_size(object()) is None


def test_log_to_results_parser() -> None:
    """The log is the only working transport for results, so the parser matters.

    It broke twice in ways a live run exposed: an out-of-scope `args` inside parse(),
    and case lines being attributed to no arm at all because the arm lives in a
    separate "== ARM base" marker line rather than in the case line itself.
    """
    import json
    import subprocess
    import sys
    import tempfile

    root = os.path.dirname(os.path.abspath(__file__))
    entries = []

    def emit(text):
        entries.append({"stream_name": "stdout", "time": 1.0, "data": text + "\n"})

    emit("========================================================================\n== ARM base")
    emit("    [OK ] L=  4096 d=0.5  san_francisco  exp='eat a sandwich' pred='The best thing to do'")
    emit("    [MISS] L=131072 d=0.9  berlin         exp='7391' pred='cannot find it'")
    # repr() switches to double quotes when the prediction contains an apostrophe.
    # Matching only single quotes silently dropped these rows, and because a hedged
    # wrong answer is exactly the kind that contains "doesn't"/"it's", the loss
    # biased the recovered score upward. Regression for the wave1b recovery, where
    # the two missing rows were precisely the two base-arm failures.
    emit('    [MISS] L=131072 d=0.1  san_francisco  exp=\'eat a sandwich\' pred="it doesn\'t seem that San Francisco "')
    emit('    [MISS] L=131072 d=0.5  san_francisco  exp=\'eat a sandwich\' pred="it\'s not clear what the best thing "')
    emit("== ARM yarn")
    emit("    [OK ] L=131072 d=0.9  berlin         exp='7391' pred='the number is 7391'")
    # the structured path takes precedence when present
    emit("NIAH_ROW " + json.dumps({"arm": "yarn", "slug": "wave1a", "length": 8192,
                                   "depth": 0.1, "needle_key": "berlin",
                                   "expected": "7391", "correct_word": True}))
    emit("NIAH_MD | arm | 4k |")

    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "log.json")
        with open(raw, "w", encoding="utf-8") as fh:
            json.dump(entries, fh)
        out = os.path.join(tmp, "out")
        with open(raw, encoding="utf-8") as fh:
            r = subprocess.run([sys.executable, os.path.join(root, "tools", "log_to_results.py"),
                                "--outdir", out], stdin=fh, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr

        def rows(name):
            path = os.path.join(out, name)
            if not os.path.exists(path):
                return []
            return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

        base = rows("base_from_log.jsonl")
        yarn = rows("yarn_from_log.jsonl")
        assert len(base) == 4, [r["length"] for r in base]
        assert {r["length"] for r in base} == {4096, 131072}, base
        assert all(r["arm"] == "base" for r in base), base
        by_len = sorted(base, key=lambda r: (r["length"], r["depth"]))
        assert [r["correct_word"] for r in by_len] == [True, False, False, False], by_len
        # the two double-quoted rows must survive with their text intact, not just be counted
        hedged = [r for r in base if "doesn't" in r["prediction"] or "it's" in r["prediction"]]
        assert len(hedged) == 2, hedged

        # yarn's case line lands in the arm file ...
        assert len(yarn) == 1, yarn
        assert all(r["arm"] == "yarn" for r in yarn), yarn
        # ... and the structured row keeps its own arm_slug name, which is what makes
        # resuming exact rather than approximate
        structured = rows("yarn_wave1a.jsonl")
        assert len(structured) == 1 and structured[0]["slug"] == "wave1a", structured

        assert os.path.exists(os.path.join(out, "summary_from_log.md")), "NIAH_MD not written"


def test_device_map_offload_guard() -> None:
    """accelerate spills to CPU instead of failing, which is a silent wrong answer.

    `device_map="auto"` is how the 7B fp16 run is meant to reach both T4s, and the
    whole point of that run is memory. If the weights do not fit, accelerate does
    not raise: it puts the remainder on CPU and the run proceeds slowly, producing
    numbers indistinguishable from a real measurement. So it must raise here.
    """
    class Fake:
        def __init__(self, dmap):
            if dmap is not None:
                self.hf_device_map = dmap

    # no map at all (the plain .to(device) path) - nothing to check, nothing to say
    assert niah.report_device_map(Fake(None), "cuda") == {}

    # a healthy split across both T4s
    ok = {"model.embed_tokens": 0, "model.layers.0": 0, "model.layers.27": 1, "lm_head": 1}
    assert niah.report_device_map(Fake(ok), "device_map=auto") == ok

    # the failure this exists for: one layer landed on CPU
    bad = {"model.layers.0": 0, "model.layers.27": "cpu"}
    try:
        niah.report_device_map(Fake(bad), "device_map=auto")
    except RuntimeError as exc:
        assert "offloaded" in str(exc), exc
    else:
        raise AssertionError("a CPU-offloaded layer was accepted - the run would look fine and be meaningless")

    # disk offload is the same class of bug
    try:
        niah.report_device_map(Fake({"lm_head": "disk"}), "device_map=auto")
    except RuntimeError:
        pass
    else:
        raise AssertionError("disk offload accepted")

    # explicit opt-out still reports, for a deliberate CPU correctness check
    assert niah.report_device_map(Fake(bad), "device_map=auto", allow_cpu=True) == bad


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
    test_notebook_install_cell()
    test_placement()
    test_notebook_runner_cell()
    test_notebooks_are_not_glued()
    test_rope_scaling_shapes()
    test_kernel_scripts()
    test_attention_setup()
    test_no_undefined_names()
    test_prefill_chunking()
    test_log_to_results_parser()
    test_device_map_offload_guard()

    print("ALL LOGIC TESTS PASSED")


if __name__ == "__main__":
    main()
