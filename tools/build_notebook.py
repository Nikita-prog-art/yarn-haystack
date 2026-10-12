#!/usr/bin/env python3
"""Generate yarn_niah.ipynb, the interactive Kaggle/Colab driver.

This exists because hand-built notebooks in this repo were silently broken: cell
sources were stored as line lists whose elements had lost their trailing newline,
so nbformat readers glued them together. A cell starting with a comment then
became one long comment - it still parsed and the kernel reported COMPLETE, while
running nothing at all. Generating it means the format is right by construction,
and test_logic.py now fails if it ever is not.

Shared cell bodies (install, preflight, report) are imported from the kernel
builder so the two drivers cannot drift apart.

    python tools/build_notebook.py
"""
from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "kaggle"))

import build_kernels as bk  # noqa: E402  (path set above)

# --------------------------------------------------------------------------
# Cells unique to the interactive notebook. Its runner differs from the kernel's
# because here the experiment config is edited in a cell, not baked in at build
# time, and the same notebook must work on Kaggle and Colab.
# --------------------------------------------------------------------------

MD_HEADER = """# YaRN vs native RoPE on needle-in-a-haystack

Interactive driver for the harness in this repo, for a free Kaggle (2xT4) or
Colab (T4) session. Every run appends to a JSONL, so a killed session resumes
instead of restarting.

Run the cells in order. The kernel equivalents (`kaggle/*/`) run the same code
headlessly with the grid baked in; use those if you want unattended runs.
"""

CELL_CLONE = """# 1) Get the code. Works whether this repo sits in /kaggle/working, /content or
#    anywhere else, as long as run.py is reachable.
import os, sys, pathlib

CANDIDATES = [os.getcwd(), "/kaggle/working/yarn-haystack", "/content/yarn-haystack"]
REPO = os.environ.get("NIAH_REPO") or next(
    (p for p in CANDIDATES if pathlib.Path(p, "run.py").exists()), None
)
if REPO is None:
    raise SystemExit(
        "put run.py/niah.py/analyze.py somewhere reachable, or set NIAH_REPO.\\n"
        "  git clone https://github.com/Nikita-prog-art/yarn-haystack.git"
    )
os.chdir(REPO)
sys.path.insert(0, REPO)
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "/content"
print("repo:", REPO, "| work:", WORK)
"""

CELL_CONFIG = '''# 4) THE EXPERIMENT CONFIG. Change these lines, nothing else.
#
#    Wave 1 (one T4, cheapest path to a number):
#        MODEL = "Qwen/Qwen2.5-7B-Instruct"   # native 32k, 56 KiB KV/token
#    Wave 2 (needs 2xT4 + vLLM TP=2: Qwen3 KV is 144 KiB/token => 18 GiB at 128k):
#        MODEL = "Qwen/Qwen3-8B"              # native 32k (config says 40960), 144 KiB/token
#
#    Do NOT use Qwen3.5-*, Qwen3-*-Instruct-2507, Qwen3-Next-* or Qwen3-Coder-*:
#    they are natively 256k, so at <=128k there is no extrapolation failure for
#    YaRN to fix and the delta would be zero by construction. run.py blocks them.
MODEL    = "Qwen/Qwen2.5-7B-Instruct"
ORIG_MAX = 32768            # the TRAINED prompt window, NOT config.json's 40960
FACTOR   = 4.0              # -> 131072; Qwen advise matching the target (2.0 for 64k)
LENGTHS  = "4096,8192,16384,32768,65536,131072"
DEPTHS   = "0.0,0.1,0.25,0.5,0.75,0.9,1.0"
EXTRA    = ["--load-in-4bit", "--attn", "sdpa"]   # identical for BOTH arms, always
TAG      = MODEL.split("/")[-1].lower().replace(".", "") + f"_f{FACTOR:g}"
print(TAG, LENGTHS)
'''

CELL_RUNNER = '''import subprocess, sys, os, collections
RESULTS = f"{WORK}/results"; os.makedirs(RESULTS, exist_ok=True)

def niah(*args):
    """Run the harness as a subprocess, streaming its output live.

    -u is not decoration: without it the child's stdout is block-buffered through
    a pipe, so a 40-minute 128k case looks frozen. On failure the last 40 lines go
    INTO the exception, because a bare CalledProcessError hides the actual error.
    """
    cmd = [sys.executable, "-u", "run.py", *args]
    print("$", " ".join(cmd), flush=True)
    tail = collections.deque(maxlen=40)
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as p:
        for line in p.stdout:
            print(line, end="", flush=True)
            tail.append(line.rstrip())
        p.wait()
    if p.returncode:
        raise RuntimeError(f"run.py exited with {p.returncode}. Last lines of its output:\\n  "
                           + "\\n  ".join(tail))
'''

CELL_SMOKE = '''# 5) SMOKE TEST FIRST: tiny model, one length.
#    run.py raises if rope_scaling did not stick, but read the logged config too.
niah("--model", "Qwen/Qwen2.5-1.5B-Instruct", "--arm", "yarn", "--factor", "4",
     "--original-max", "32768", "--lengths", "4096", "--depths", "0.5",
     "--max-new-tokens", "24", "--out", f"{RESULTS}/smoke.jsonl")
'''

CELL_BASE = '''# 6) ARM 1 - native RoPE, extrapolated past its window. Expect collapse at 64k/128k.
#    Not a throwaway arm: yarn@128k - base@128k IS the YaRN effect.
niah("--model", MODEL, "--arm", "base",
     "--lengths", LENGTHS, "--depths", DEPTHS, "--max-new-tokens", "64",
     "--out", f"{RESULTS}/base_{TAG}.jsonl", *EXTRA)
'''

CELL_YARN = '''# 7) ARM 2 - same weights, YaRN. Byte-identical setup otherwise: same EXTRA, same grid.
niah("--model", MODEL, "--arm", "yarn", "--factor", str(FACTOR),
     "--original-max", str(ORIG_MAX),
     "--lengths", LENGTHS, "--depths", DEPTHS, "--max-new-tokens", "64",
     "--out", f"{RESULTS}/yarn_{TAG}.jsonl", *EXTRA)
'''

CELL_REPORT_I = '''# 8) Report: summary.md + heatmaps.png + CSVs, incl. the delta-vs-base table.
import glob, subprocess, sys, os
from IPython.display import Image, Markdown, display

res = sorted(glob.glob(f"{RESULTS}/*.jsonl"))
print("result files:", res)
if not res:
    raise SystemExit("no results yet - run the arms above")
subprocess.run([sys.executable, "analyze.py", *res, "--outdir", f"{WORK}/report_{TAG}"], check=True)
display(Markdown(open(f"{WORK}/report_{TAG}/summary.md", encoding="utf-8").read()))
if os.path.exists(f"{WORK}/report_{TAG}/heatmaps.png"):
    display(Image(f"{WORK}/report_{TAG}/heatmaps.png"))
'''

MD_FOOTER = """### Reading the result

- `base` at 64k/128k collapsing to ~0 while `yarn` holds up = YaRN doing its job.
- Check the **4k/8k/16k row**. Qwen's own card warns that static YaRN "may
  potentially degrade model performance" on short texts: a drop there is real, and
  it is the price of the 128k gain.
- Depth matters. NIAH tends to break in the middle before it breaks at the edges,
  so a flat-looking heatmap row at every length is worth more than one number.

### If 7B @ 128k OOMs

`--load-in-4bit --attn sdpa` needs ~11 GiB (NF4 weights + 7 GiB KV). If it still OOMs:
- `Qwen/Qwen2.5-3B-Instruct` in fp16 (~10.3 GiB at 128k): same protocol, weaker model;
- `--device-map auto` to shard across Kaggle's 2xT4 (slow, but it fits);
- `--backend vllm --tensor-parallel-size 2` (fastest, iff vLLM installs cleanly).
Never mix these flags across arms - a 4-bit `base` vs fp16 `yarn` measures quantization.

### Persisting across sessions

`/kaggle/working` survives a session stop. The JSONL is append-only and fsync'd,
so run.py skips finished cases automatically: just re-run the same cell. Sync with
`--upload <user>/yarn-niah` for belt and braces.
"""


def code(src):
    # A single string, not a list of lines: see the module docstring.
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.strip("\n")}


def markdown(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")}


def build() -> str:
    nb = {
        "cells": [
            markdown(MD_HEADER),
            code(CELL_CLONE),
            code(bk.CELL_INSTALL),
            code(CELL_CONFIG),
            code(bk.CELL_PREFLIGHT),
            code(CELL_RUNNER),
            code(CELL_SMOKE),
            code(CELL_BASE),
            code(CELL_YARN),
            code(CELL_REPORT_I),
            markdown(MD_FOOTER),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.12", "mimetype": "text/x-python",
                              "codemirror_mode": {"name": "ipython", "version": 3},
                              "pygments_lexer": "ipython3", "nbconvert_exporter": "python",
                              "file_extension": ".py"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,   # what Kaggle itself emits
    }
    path = os.path.join(REPO, "yarn_niah.ipynb")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    return path


if __name__ == "__main__":
    print("wrote", os.path.relpath(build(), REPO))
