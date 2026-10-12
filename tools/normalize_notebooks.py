#!/usr/bin/env python3
"""Normalize nbformat cell sources so they survive concatenation.

The bug this prevents is nasty because it is usually SILENT. In nbformat a cell's
`source` may be a list of lines, and readers concatenate it with "" - so any
element that misses its trailing "\n" glues onto the next one. A cell starting
with a comment then becomes one long comment: it still parses, it just does
nothing. A kernel pushed like that reports COMPLETE with zero output, and you go
hunting for a GPU problem that does not exist.

Storing `source` as a single string (what Kaggle's own notebooks do) removes the
class of bug entirely.

    python tools/normalize_notebooks.py [--check] [paths...]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

SKIP_DIRS = {".local", ".git", "__pycache__"}


def notebooks(paths: list[str]) -> list[str]:
    if paths:
        found = []
        for p in paths:
            found.extend(glob.glob(p, recursive=True) if any(ch in p for ch in "*?[") else [p])
        return sorted(p for p in found if p.endswith(".ipynb"))
    out = []
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        out.extend(os.path.join(root, f) for f in files if f.endswith(".ipynb"))
    return sorted(out)


def glues(src) -> bool:
    """True if a list source would be glued together by an nbformat reader."""
    if isinstance(src, str):
        return False
    return len(src) > 1 and not all(s.endswith("\n") for s in src[:-1])


def normalize(path: str) -> bool:
    with open(path, encoding="utf-8") as fh:
        nb = json.load(fh)
    changed = False
    for cell in nb.get("cells", []):
        src = cell.get("source")
        if isinstance(src, list):
            # Join the way readers do, then store as one string: no ambiguity left.
            cell["source"] = "".join(src)
            changed = True
    if changed:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(nb, fh, ensure_ascii=False, indent=1)
            fh.write("\n")
    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--check", action="store_true", help="report only, non-zero exit if any file needs fixing")
    args = ap.parse_args()

    bad = []
    for path in notebooks(args.paths):
        with open(path, encoding="utf-8") as fh:
            nb = json.load(fh)
        if any(glues(c.get("source")) for c in nb.get("cells", [])):
            bad.append(path)

    if args.check:
        for p in bad:
            print(f"NEEDS FIXING  {p}")
        print(f"\n{len(bad)} notebook(s) would be glued when read")
        sys.exit(1 if bad else 0)

    for path in notebooks(args.paths):
        if normalize(path):
            print(f"normalized  {path}")
    print("done")


if __name__ == "__main__":
    main()
