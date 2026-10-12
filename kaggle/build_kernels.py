#!/usr/bin/env python3
"""Generate Kaggle kernel folders for the NIAH run.

Why a generator instead of checked-in notebooks compiled by hand: the kernel
notebook must pin the exact harness commit, and that pin changes every time the
harness changes. Generating keeps the three presets from drifting apart.

    python kaggle/build_kernels.py --commit <sha> [--user <kaggle-user>]

Produces, for each preset, a folder with kernel-metadata.json + <slug>.ipynb that
`kaggle kernels push -p <folder>` accepts.
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_URL = "https://github.com/Nikita-prog-art/yarn-haystack.git"

# Presets. Each runs BOTH arms with an identical grid, so the only difference
# between the two JSONLs is rope_scaling.
PRESETS = {
    "smoke": {
        "title": "yarn-niah smoke",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "lengths": "4096,131072",
        "depths": "0.25,0.5,0.75",
        "extra": [],
        "max_new_tokens": 32,
        "why": "1.5B, one cheap length and one expensive one: proves the whole "
               "path (clone, install, corpus, GPU, rope_scaling, report) in ~10 minutes.",
    },
    "wave1a": {
        "title": "yarn-niah wave1a short",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "lengths": "4096,8192,16384,32768",
        "depths": "0.0,0.1,0.25,0.5,0.75,0.9,1.0",
        "extra": ["--load-in-4bit", "--attn", "sdpa"],
        "max_new_tokens": 64,
        "why": "7B inside and at the edge of its 32k window: the short-context price "
               "of static YaRN, which Qwen's own card warns about.",
    },
    "wave1b": {
        "title": "yarn-niah wave1b long",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "lengths": "65536,131072",
        "depths": "0.1,0.5,0.9",
        "extra": ["--load-in-4bit", "--attn", "sdpa"],
        "max_new_tokens": 64,
        "why": "7B far past its window: 3 depths only, because prefill at 128k dominates "
               "and a kernel that times out returns nothing.",
    },
}

HEADER = '''# {title}

{why}

Preset `{slug}` on `{shape}`, harness pinned to `{commit}`.

| setting | value |
|---|---|
| model | `{model}` |
| lengths | `{lengths}` |
| depths | `{depths}` |
| extra flags | `{extra_pretty}` |
| max new tokens | {mnt} |

Both arms run the same grid, so the only variable is `rope_scaling`.
'''

CELL_CLONE = '''# 1) Harness at a pinned commit. Pinning matters: these numbers get compared
#    against a later run, and "main moved" is not an explanation anyone wants.
import os, subprocess, sys

REPO_URL = "{repo_url}"
COMMIT   = "{commit}"
WORK     = "/kaggle/working"
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)

if not os.path.isdir("yarn-haystack"):
    subprocess.run(["git", "clone", "-q", REPO_URL, "yarn-haystack"], check=True)
os.chdir("yarn-haystack")
subprocess.run(["git", "fetch", "-q", "--all"], check=True)
subprocess.run(["git", "checkout", "-q", COMMIT], check=True)
head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
assert head == COMMIT, f"harness is at {{head}}, expected {{COMMIT}}"
sys.path.insert(0, os.getcwd())
print("harness commit:", head)
'''

CELL_INSTALL = '''# 2) Install only what is missing. Kaggle ships a CUDA-matched torch, and
#    upgrading matplotlib/datasets provokes conflicts with preinstalled tools.
import importlib.metadata as md, subprocess, sys

def version(p):
    try:
        return md.version(p)
    except md.PackageNotFoundError:
        return None

def too_old(v, floor):
    return v is None or tuple(int(x) for x in v.split(".")[:2]) < floor

to_install = []
if too_old(version("transformers"), (4, 51)):
    to_install.append("transformers>=4.51")
for p in ("accelerate", "datasets", "bitsandbytes"):
    if version(p) is None:
        to_install.append(p)
if to_install:
    print("pip install:", " ".join(to_install), flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *to_install], check=True)

for p in ("transformers", "torch", "accelerate", "datasets", "bitsandbytes", "matplotlib"):
    print(f"  {p:14s} {version(p)}")

import torch
print(f"\\ntorch {torch.__version__} | cuda {torch.cuda.is_available()} | gpus {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    pr = torch.cuda.get_device_properties(i)
    print(f"  gpu{i}: {pr.name}, {pr.total_memory / 2**30:.1f} GiB")
assert torch.cuda.is_available(), "no CUDA: this kernel was pushed without a GPU"
_v = version("transformers")
assert _v and tuple(int(x) for x in _v.split(".")[:2]) >= (4, 51), f"need transformers>=4.51, have {_v}"
'''

CELL_PREFLIGHT = '''# 3) Preflight: internet (the usual cause of a one-second death) and the corpus.
import os, urllib.request
import niah

try:
    print("huggingface.co ->", urllib.request.urlopen("https://huggingface.co", timeout=20).status)
except Exception as exc:
    raise SystemExit(f"no internet in this kernel: {exc!r} (Kaggle needs phone verification for that)")

text = niah.load_corpus(None, os.path.join(os.getcwd(), "cache"))
print(f"corpus OK: {len(text)/1e6:.1f}M chars cached")
'''

CELL_RUN = '''# 4) Run both arms under the "{slug}" preset, streaming output live.
import collections, subprocess, sys

def niah(*args):
    """-u matters: through a pipe the child's stdout is block-buffered and a long
    128k case otherwise looks frozen. The tail goes into the exception, because a
    bare CalledProcessError hides the one thing worth reading."""
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
        print(f"!! exit {{p.returncode}} - keeping results collected so far")

COMMON = ["--model", "{model}", "--lengths", "{lengths}", "--depths", "{depths}",
          "--max-new-tokens", "{mnt}", *{extra}]

for arm, extra in (("base", []), ("yarn", ["--factor", "4", "--original-max", "32768"])):
    niah(*COMMON, "--arm", arm, *extra, "--out", f"results/{{arm}}_{slug}.jsonl")
'''

CELL_REPORT = '''# 5) Report. analyze.py is robust to a partial run, so this is worth doing even if
#    an arm was cut short: you get the heatmap for whatever finished.
import glob, subprocess, sys, os
from IPython.display import Image, Markdown, display

res = sorted(glob.glob("results/*.jsonl"))
print("result files:", res)
if not res:
    raise SystemExit("no results were produced - read the output above")
subprocess.run([sys.executable, "analyze.py", *res, "--outdir", "report"], check=True)
display(Markdown(open("report/summary.md", encoding="utf-8").read()))
if os.path.exists("report/heatmaps.png"):
    display(Image("report/heatmaps.png"))
print("\\nfiles under /kaggle/working (fetched by `kaggle kernels output`):")
for root, _dirs, files in os.walk("results"):
    for f in sorted(files):
        print("  ", os.path.join(root, f))
'''


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.strip("\n")}


def markdown(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")}


def build(slug: str, preset: dict, commit: str, user: str, shape: str) -> str:
    folder = os.path.join(HERE, slug)
    os.makedirs(folder, exist_ok=True)

    fmt = dict(slug=slug, commit=commit, repo_url=REPO_URL, shape=shape, **preset)
    # `extra` must stay a LIST for the code cell (`*{extra}`), while the header
    # table wants something readable. Conflating them produced `*--load-in-4bit ...`.
    fmt["extra_pretty"] = " ".join(preset["extra"]) if preset["extra"] else "(none)"
    fmt["mnt"] = str(preset["max_new_tokens"])
    nb = {
        "cells": [
            markdown(HEADER.format(**fmt)),
            code(CELL_CLONE.format(**fmt)),
            code(CELL_INSTALL),
            code(CELL_PREFLIGHT),
            code(CELL_RUN.format(**fmt)),
            code(CELL_REPORT),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }

    nb_path = os.path.join(folder, f"{slug}.ipynb")
    with open(nb_path, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)

    meta = {
        # Kernel slugs are global per user, so prefix them: "smoke" alone is a
        # collision waiting to happen.
        "id": f"{user}/yarn-niah-{slug}", "title": preset["title"],
        "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": "true",          # a half-finished run is not for the world
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",     # mandatory: model + corpus downloads
        "machine_shape": shape,        # NvidiaTeslaT4 == 2xT4, 32 GiB total
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [], "model_sources": [],
    }
    with open(os.path.join(folder, "kernel-metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return nb_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True, help="harness commit the kernels must run")
    ap.add_argument("--user", default="nikita-prog-art", help="Kaggle username (metadata id)")
    ap.add_argument("--shape", default="NvidiaTeslaT4", choices=["NvidiaTeslaT4", "NvidiaTeslaP100"])
    ap.add_argument("--only", default=None, help="build a single preset")
    args = ap.parse_args()

    if len(args.commit) != 40:
        raise SystemExit(f"--commit must be a full 40-char sha, got {args.commit!r}")

    for slug, preset in PRESETS.items():
        if args.only and slug != args.only:
            continue
        path = build(slug, preset, args.commit, args.user, args.shape)
        print(f"built {slug:8s} -> {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
