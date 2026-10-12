#!/usr/bin/env python3
"""Rebuild results/*.jsonl from a Kaggle kernel's log.

Why this exists: `kaggle kernels output` returns nothing for these runs (tried
with --file-pattern, -o, and explicit versions) while `kernels logs` works fine.
The results only exist inside the kernel container, so if the output endpoint is
unusable the data has to travel in the log.

Two paths, in order of preference:

1. `NIAH_ROW <json>` lines, which the kernel scripts now print - exact rows, with
   every field.
2. `[OK ]` / `[MISS]` case lines, which the harness has always printed. Enough to
   rebuild the numbers that matter (arm, length, depth, needle, correctness, the
   prediction), so logs from before the transport existed are still recoverable.

    kaggle/from_log <owner/slug[/version]> [outdir]      # fetches, then pipes here
    cat log.json | python tools/log_to_results.py --outdir results
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROW_PREFIX = "NIAH_ROW "
MD_PREFIX = "NIAH_MD "

# [OK ] L=  4096 d=0.25 san_francisco  exp='eat a sandwich' pred='The best...'
CASE_RE = re.compile(
    r"\[(?P<flag>OK |MISS)\]\s+L=\s*(?P<length>\d+)\s+d=(?P<depth>[\d.]+)\s+"
    r"(?P<needle>\S+)\s+exp='(?P<expected>.*?)'\s+pred='(?P<pred>.*)'\s*$"
)


def log_lines(raw: str):
    """Yield stdout lines from a kernel log (JSON array of stream entries)."""
    raw = raw.strip()
    if not raw:
        return
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError:
        yield from raw.split("\n")
        return
    if isinstance(entries, list):
        for entry in entries:
            if entry.get("stream_name") == "stdout":
                for line in entry.get("data", "").split("\n"):
                    yield line
    else:
        yield from raw.split("\n")


def parse(lines) -> tuple[dict[str, list[dict]], list[str]]:
    rows: dict[str, list[dict]] = {}
    markdown: list[str] = []
    for line in lines:
        if line.startswith(MD_PREFIX):
            markdown.append(line[len(MD_PREFIX):])
            continue
        if line.startswith(ROW_PREFIX):
            payload = line[len(ROW_PREFIX):]
            try:
                row = json.loads(payload)
            except json.JSONDecodeError:
                continue
            key = f"{row.get('arm', 'unknown')}_{row.get('slug', row.get('mode', 'run'))}.jsonl"
            rows.setdefault(key, []).append(row)
            continue
        m = CASE_RE.search(line)
        if m:
            key = f"{m.group('needle')}"  # placeholder, fixed up by caller
            payload = {
                "arm": args.arm, "length": int(m.group("length")), "depth": float(m.group("depth")),
                "needle_key": m.group("needle"), "expected": m.group("expected"),
                "prediction": m.group("pred"), "correct_word": m.group("flag").strip() == "OK",
                "source": "parsed from case line",
            }
            rows.setdefault("__case_lines__", []).append(payload)
    return rows, markdown


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--arm", default=None, help="arm for rows recovered from case lines (they do not carry it)")
    args = ap.parse_args()

    rows, markdown = parse(log_lines(sys.stdin.read()))
    os.makedirs(args.outdir, exist_ok=True)

    case_rows = rows.pop("__case_lines__", [])
    if case_rows:
        # Case lines do not say which arm they belong to; the caller supplies it.
        rows.setdefault(f"{args.arm or 'unknown'}_from_log.jsonl", []).extend(case_rows)

    if not rows and not markdown:
        print("no NIAH_ROW / case lines found in this log", file=sys.stderr)
        sys.exit(1)

    total = 0
    for name, items in sorted(rows.items()):
        path = os.path.join(args.outdir, name)
        with open(path, "a", encoding="utf-8") as fh:
            for row in items:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        total += len(items)
        print(f"  {name}: {len(items)} row(s)")
    if markdown:
        path = os.path.join(args.outdir, "summary_from_log.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(markdown) + "\n")
        print(f"  summary_from_log.md: {len(markdown)} line(s)")
    print(f"recovered {total} row(s) from the log")


if __name__ == "__main__":
    main()
