#!/usr/bin/env python3
"""Aggregate NIAH run JSONL files into the actual deliverable:
   accuracy vs (context length, needle depth) per arm, plus the YaRN delta.

Usage
-----
python analyze.py results/base.jsonl results/yarn_f4.jsonl --outdir report
python analyze.py 'results/*.jsonl' --outdir report          # shell expands it
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def load(paths: list[str]) -> list[dict]:
    rows = []
    for pattern in paths:
        for path in glob.glob(pattern) or [pattern]:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return rows


def arm_label(row: dict) -> str:
    if row["arm"] == "base":
        return "base (native RoPE)"
    f = row.get("factor")
    return f"{row['arm']} f={f:g}" if f else row["arm"]


def pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--outdir", default="report")
    ap.add_argument("--metric", default="correct_word", choices=["correct_word", "correct_contains", "recall"])
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = load(args.jsonl)
    if not rows:
        raise SystemExit("no rows loaded — check the glob and that the runs produced output")
    print(f"[info] {len(rows)} rows, arms={sorted({arm_label(r) for r in rows})}, "
          f"models={sorted({r['model'] for r in rows})}")

    metric = args.metric

    # ---- 1. per-arm aggregate by length -------------------------------
    by_arm_len: dict[tuple[str, int], list[float]] = defaultdict(list)
    # ---- 2. per-arm heatmap by (length, depth) ------------------------
    by_arm_cell: dict[tuple[str, int, float], list[float]] = defaultdict(list)
    for r in rows:
        m = float(r.get(metric, 0.0))
        by_arm_len[(arm_label(r), r["length"])].append(m)
        if r.get("depth", -1) >= 0:
            by_arm_cell[(arm_label(r), r["length"], r["depth"])].append(m)

    lengths = sorted({r["length"] for r in rows})
    arms = sorted({arm_label(r) for r in rows})
    depths = sorted({r["depth"] for r in rows if r.get("depth", -1) >= 0})

    # CSV: accuracy vs length
    with open(os.path.join(args.outdir, "accuracy_by_length.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", *[f"{l}" for l in lengths], "mean"])
        for arm in arms:
            cells = [by_arm_len.get((arm, l), []) for l in lengths]
            means = [sum(c) / len(c) if c else float("nan") for c in cells]
            valid = [m for m in means if m == m]
            w.writerow([arm, *[f"{m:.3f}" if m == m else "" for m in means], f"{sum(valid) / len(valid):.3f}" if valid else ""])

    # CSV: heatmap
    with open(os.path.join(args.outdir, "heatmap.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "length", *[f"d={d}" for d in depths], "mean"])
        for arm in arms:
            for l in lengths:
                cells = [by_arm_cell.get((arm, l, d), []) for d in depths]
                means = [sum(c) / len(c) if c else float("nan") for c in cells]
                valid = [m for m in means if m == m]
                w.writerow([arm, l, *[f"{m:.3f}" if m == m else "" for m in means],
                            f"{sum(valid) / len(valid):.3f}" if valid else ""])

    # ---- 3. the number everybody actually asks for: YaRN delta --------
    base_arm = next((a for a in arms if a.startswith("base")), None)
    lines = []
    lines.append(f"# NIAH results — metric `{metric}`\n")
    lines.append("## Accuracy by context length\n")
    header = "| arm | " + " | ".join(f"{l // 1024}k" for l in lengths) + " | mean |"
    lines.append(header)
    lines.append("|" + "---|" * (len(lengths) + 2))
    for arm in arms:
        cells = []
        for l in lengths:
            c = by_arm_len.get((arm, l), [])
            cells.append(pct(sum(c) / len(c)) if c else "  n/a")
        allc = [v for l in lengths for v in by_arm_len.get((arm, l), [])]
        lines.append(f"| {arm} | " + " | ".join(cells) + f" | {pct(sum(allc) / len(allc)) if allc else 'n/a'} |")

    if base_arm:
        lines.append("\n## Delta vs base (positive = YaRN helps)\n")
        lines.append("| length | " + " | ".join(a for a in arms if a != base_arm) + " |")
        lines.append("|" + "---|" * len([a for a in arms if a != base_arm]) + "---|")
        for l in lengths:
            b = by_arm_len.get((base_arm, l), [])
            bm = sum(b) / len(b) if b else None
            cells = []
            for arm in arms:
                if arm == base_arm:
                    continue
                c = by_arm_len.get((arm, l), [])
                cm = sum(c) / len(c) if c else None
                cells.append(f"{(cm - bm) * 100:+.1f} pp" if (bm is not None and cm is not None) else "n/a")
            lines.append(f"| {l} | " + " | ".join(cells) + " |")

    md = "\n".join(lines) + "\n"
    with open(os.path.join(args.outdir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print("\n" + md)

    # ---- 4. heatmaps --------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] matplotlib/numpy unavailable ({exc}); skipping plots")
        return

    import math

    n = len(arms)
    cols = min(3, n)
    rows_n = math.ceil(n / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(5.2 * cols, 4.4 * rows_n), squeeze=False)
    vmax = 1.0
    for idx, arm in enumerate(arms):
        ax = axes[idx // cols][idx % cols]
        grid = np.full((len(depths), len(lengths)), np.nan)
        for i, d in enumerate(depths):
            for j, l in enumerate(lengths):
                c = by_arm_cell.get((arm, l, d), [])
                if c:
                    grid[i, j] = sum(c) / len(c)
        im = ax.imshow(grid, vmin=0, vmax=vmax, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(lengths)), [f"{l // 1024}k" for l in lengths])
        ax.set_yticks(range(len(depths)), [f"{d:g}" for d in depths])
        ax.set_xlabel("context length")
        ax.set_ylabel("needle depth")
        ax.set_title(arm, fontsize=10)
        for i in range(len(depths)):
            for j in range(len(lengths)):
                if not math.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j] * 100:.0f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    for k in range(n, rows_n * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle(f"NIAH accuracy — {metric}", fontsize=12)
    fig.tight_layout()
    out = os.path.join(args.outdir, "heatmaps.png")
    fig.savefig(out, dpi=130)
    print(f"[done] wrote heatmaps.png, heatmap.csv, accuracy_by_length.csv, summary.md -> {args.outdir}/")


if __name__ == "__main__":
    main()
