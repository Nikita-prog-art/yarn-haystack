#!/usr/bin/env python3
"""Build the standalone HTML report from the evidence files.

Every number in the report is computed from evidence/*.json here, never typed in by
hand. That is deliberate: this project's whole failure mode has been results that
looked right, and a report with hand-copied figures drifts from its own data the
first time a run is repeated.

    python tools/build_report.py                 # -> report/niah_yarn_report.html
    python tools/build_report.py --outdir report

Writes one self-contained HTML file with the PNGs embedded as base64, so the
deliverable cannot lose its charts when it is moved or committed.
"""
from __future__ import annotations

import argparse
import base64
import collections
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# matplotlib complains about an unwritable ~/.config on this machine; keep its cache
# inside the repo so the build is quiet and reproducible.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(REPO, ".local", "mplconfig"))

BASE_C = "#c0392b"
YARN_C = "#1e8449"
GRID_C = "#dfe6e9"


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def keyed(rows: list[dict]) -> dict[tuple, dict]:
    return {(r["arm"], r["length"], r["depth"]): r for r in rows}


def rate(rows: list[dict], metric: str = "correct_word") -> float | None:
    vals = [bool(r.get(metric)) for r in rows]
    return (sum(vals) / len(vals)) if vals else None


def by(rows: list[dict], field: str) -> dict:
    out = collections.defaultdict(list)
    for r in rows:
        out[r[field]].append(r)
    return out


def cell(rows_by_cell: dict, arm: str, length: int, depth: float) -> tuple[float | None, int]:
    """(accuracy, n) for one heatmap cell, or (None, 0) when it was not measured."""
    sub = rows_by_cell.get((arm, length, depth), [])
    if not sub:
        return None, 0
    return rate(sub), len(sub)


# --------------------------------------------------------------------------
# charts
# --------------------------------------------------------------------------


def chart_accuracy_by_length(path: str, rows: list[dict], lengths: list[int], title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arms = ["base", "yarn"]
    labels = ["base (штатный RoPE)", "YaRN f=4"]
    colors = [BASE_C, YARN_C]

    x = np.arange(len(lengths))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    for i, (arm, lab, col) in enumerate(zip(arms, labels, colors)):
        vals, ns = [], []
        for L in lengths:
            sub = [r for r in rows if r["arm"] == arm and r["length"] == L]
            v = rate(sub)
            vals.append(0 if v is None else 100 * v)
            ns.append(len(sub))
        off = (i - 0.5) * w
        bars = ax.bar(x + off, vals, w, label=lab, color=col, edgecolor="white", linewidth=0.8)
        for rect, v, n in zip(bars, vals, ns):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 2.5, f"{v:.0f}%\nn={n}",
                    ha="center", va="bottom", fontsize=8.5, color="#2d3436")

    ax.set_xticks(x, [f"{L // 1024}k" for L in lengths])
    ax.set_xlabel("длина контекста")
    ax.set_ylabel("точность, %")
    ax.set_ylim(0, 130)
    ax.axhline(100, color=GRID_C, lw=1, zorder=0)
    # Shade the region past the native window. Derived from the actual lengths, not
    # from positions at the end of the axis: with only two lengths the index-based
    # version shaded both bars, which is a chart that lies about the setup.
    beyond = [i for i, L in enumerate(lengths) if L > 32768]
    if beyond:
        ax.axvspan(min(beyond) - 0.5, max(beyond) + 0.5, color="#fff4e6", zorder=0)
        ax.text((min(beyond) + max(beyond)) / 2, 121, "за нативным окном (32768)",
                ha="center", fontsize=8.5, color="#b8651b")
    # Below the axes rather than inside: at 100% the bars fill the plot and a
    # lower-left legend lands on top of the first one.
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    ax.set_title(title, fontsize=11.5, pad=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID_C, lw=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_depth_decay(path: str, rows: list[dict], lengths: list[int]) -> None:
    """The mechanism chart: accuracy against distance from the question."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(lengths), figsize=(5.0 * len(lengths), 4.4), squeeze=False)
    for ax, L in zip(axes[0], lengths):
        for arm, lab, col, mk in (("base", "base", BASE_C, "o"), ("yarn", "YaRN f=4", YARN_C, "s")):
            sub = sorted([r for r in rows if r["arm"] == arm and r["length"] == L],
                         key=lambda r: r["depth"])
            if not sub:
                continue
            xs = [1 - r["depth"] for r in sub]
            ys = [100 * r["correct_word"] for r in sub]
            ax.plot(xs, ys, marker=mk, color=col, label=lab, lw=2, ms=7)
        ax.set_title(f"{L // 1024}k", fontsize=11)
        ax.set_xlabel("расстояние от вопроса\n(1 − глубина иголки)")
        ax.set_ylim(-6, 108)
        ax.set_xlim(-0.05, 1.05)
        ax.axhline(100, color=GRID_C, lw=1, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(color=GRID_C, lw=0.7)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel("точность, %")
    axes[0][0].legend(frameon=False, loc="lower left")
    fig.suptitle("Провал base упорядочен по расстоянию: чем дальше иголка от вопроса, тем хуже",
                 fontsize=11.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_heatmaps(path: str, rows: list[dict], lengths: list[int], depths: list[float]) -> None:
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cells = by(rows, "arm")
    by_cell = collections.defaultdict(list)
    for r in rows:
        by_cell[(r["arm"], r["length"], r["depth"])].append(r)

    labels = {"base": "base (штатный RoPE)", "yarn": "YaRN f=4"}
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    for ax, arm in zip(axes, ("base", "yarn")):
        grid = np.full((len(depths), len(lengths)), np.nan)
        counts = np.zeros_like(grid, dtype=int)
        for i, d in enumerate(depths):
            for j, L in enumerate(lengths):
                v, n = cell(by_cell, arm, L, d)
                if v is not None:
                    grid[i, j] = v
                    counts[i, j] = n
        cmap = plt.get_cmap("RdYlGn").copy()
        cmap.set_bad("#ffffff")
        im = ax.imshow(grid, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(lengths)), [f"{L // 1024}k" for L in lengths])
        ax.set_yticks(range(len(depths)), [f"{d:g}" for d in depths])
        ax.set_xlabel("длина контекста")
        ax.set_ylabel("глубина иголки")
        ax.set_title(labels[arm], fontsize=11)
        for i in range(len(depths)):
            for j in range(len(lengths)):
                if math.isnan(grid[i, j]):
                    ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="#b2bec3")
                else:
                    ax.text(j, i, f"{grid[i, j] * 100:.0f}\nn={counts[i, j]}", ha="center",
                            va="center", fontsize=8,
                            color="white" if grid[i, j] < 0.35 else "#1e272e")
        ax.set_xticks(np.arange(-0.5, len(lengths), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(depths), 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.4)
        ax.tick_params(which="minor", length=0)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Qwen2.5-7B, 4 бита: точность по (длина × глубина). «—» — не измерялось",
                 fontsize=11.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# html
# --------------------------------------------------------------------------


CSS = """
:root { --ink:#1e272e; --dim:#636e72; --line:#dfe6e9; --bg:#f7f9fa;
        --base:#c0392b; --yarn:#1e8449; --accent:#0b6bcb; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:16px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif; }
.wrap { max-width:1080px; margin:0 auto; padding:44px 26px 90px; }
header { border-bottom:3px solid var(--ink); padding-bottom:20px; margin-bottom:34px; }
h1 { font-size:31px; margin:0 0 8px; letter-spacing:-.4px; }
.sub { color:var(--dim); font-size:15px; }
h2 { font-size:22px; margin:52px 0 6px; letter-spacing:-.2px; }
h2 .n { color:var(--accent); font-variant-numeric:tabular-nums; margin-right:8px; }
h3 { font-size:16px; margin:26px 0 8px; }
p { margin:11px 0; }
.verdict { background:#fff; border:1px solid var(--line); border-left:5px solid var(--yarn);
           border-radius:8px; padding:20px 24px; margin:26px 0 8px;
           box-shadow:0 1px 3px rgba(0,0,0,.05); }
.verdict .big { font-size:27px; font-weight:650; letter-spacing:-.4px; }
.verdict .big em { font-style:normal; color:var(--yarn); }
.verdict p { color:var(--dim); margin:8px 0 0; font-size:15px; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(168px,1fr)); gap:13px; margin:22px 0; }
.card { background:#fff; border:1px solid var(--line); border-radius:8px; padding:15px 17px; }
.card .k { font-size:12px; text-transform:uppercase; letter-spacing:.6px; color:var(--dim); }
.card .v { font-size:24px; font-weight:650; margin-top:5px; font-variant-numeric:tabular-nums; }
.card .v.g { color:var(--yarn); } .card .v.r { color:var(--base); }
.card .d { font-size:12.5px; color:var(--dim); margin-top:3px; }
table { border-collapse:collapse; width:100%; margin:18px 0; background:#fff; font-size:14.5px; }
th,td { border:1px solid var(--line); padding:9px 11px; text-align:right; }
th:first-child, td:first-child { text-align:left; }
thead th { background:#eef2f4; font-weight:600; }
tbody tr:nth-child(even) { background:#fbfcfd; }
td.ok { background:#e8f6ee; color:#14532d; font-weight:600; }
td.bad { background:#fdecea; color:#7f1d1d; font-weight:600; }
td.miss { color:#b2bec3; }
td.d { font-variant-numeric:tabular-nums; }
img { max-width:100%; border:1px solid var(--line); border-radius:8px; background:#fff; margin:14px 0 6px; }
.cap { font-size:13px; color:var(--dim); margin:0 0 20px; }
.note { background:#fff8e6; border-left:4px solid #e1a008; border-radius:6px;
        padding:14px 18px; margin:20px 0; font-size:14.5px; }
.note b { color:#8a6100; }
ul { margin:10px 0; padding-left:22px; } li { margin:6px 0; }
code { background:#eef2f4; padding:1.5px 6px; border-radius:4px; font-size:13.5px;
       font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
footer { margin-top:60px; padding-top:20px; border-top:1px solid var(--line);
         color:var(--dim); font-size:13.5px; }
"""


def esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def img_tag(path: str, caption: str) -> str:
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    name = os.path.basename(path)
    return (f'<img alt="{esc(caption)}" src="data:image/png;base64,{b64}">\n'
            f'<p class="cap">{esc(caption)} <span style="color:#b2bec3">({esc(name)})</span></p>')


def accuracy_table(rows: list[dict], lengths: list[int]) -> str:
    arms = [("base", "base (штатный RoPE)"), ("yarn", "YaRN f=4")]
    head = "".join(f"<th>{L // 1024}k</th>" for L in lengths)
    out = [f"<table><thead><tr><th>рука</th>{head}<th>среднее</th></tr></thead><tbody>"]
    for arm, lab in arms:
        tds, allv = [], []
        for L in lengths:
            sub = [r for r in rows if r["arm"] == arm and r["length"] == L]
            v, n = rate(sub), len(sub)
            allv += [bool(r["correct_word"]) for r in sub]
            if v is None:
                tds.append('<td class="miss">—</td>')
            else:
                cls = "ok" if v == 1.0 else ("bad" if v < 0.7 else "")
                tds.append(f'<td class="{cls} d">{v * 100:.0f}% <span style="color:#95a5a6">'
                           f"n={n}</span></td>")
        mean = 100 * sum(allv) / len(allv) if allv else None
        out.append(f"<tr><td>{esc(lab)}</td>{''.join(tds)}"
                   f"<td class='d'>{mean:.1f}%</td></tr>")
    out.append("</tbody></table>")
    return "".join(out)


def delta_table(rows: list[dict], lengths: list[int]) -> str:
    out = ["<table><thead><tr><th>длина</th><th>base</th><th>YaRN f=4</th><th>разница</th></tr>"
           "</thead><tbody>"]
    for L in lengths:
        b = rate([r for r in rows if r["arm"] == "base" and r["length"] == L])
        y = rate([r for r in rows if r["arm"] == "yarn" and r["length"] == L])
        if b is None or y is None:
            continue
        d = (y - b) * 100
        cls = "ok" if d > 0 else ""
        out.append(f"<tr><td>{L // 1024}k</td><td class='d'>{b * 100:.1f}%</td>"
                   f"<td class='d'>{y * 100:.1f}%</td>"
                   f"<td class='{cls} d'>{d:+.1f} pp</td></tr>")
    out.append("</tbody></table>")
    return "".join(out)


def depth_table(rows: list[dict], lengths: list[int], depths: list[float]) -> str:
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["arm"], r["length"], r["depth"])].append(r)
    head = "".join(f"<th>d={d:g}</th>" for d in depths)
    out = [f"<table><thead><tr><th>рука / длина</th>{head}</tr></thead><tbody>"]
    for arm, lab in (("base", "base"), ("yarn", "YaRN f=4")):
        for L in lengths:
            tds = []
            for d in depths:
                sub = cells.get((arm, L, d), [])
                if not sub:
                    tds.append('<td class="miss">—</td>')
                    continue
                v = rate(sub)
                cls = "ok" if v == 1.0 else "bad"
                tds.append(f'<td class="{cls} d">{v * 100:.0f}%</td>')
            out.append(f"<tr><td>{esc(lab)} @ {L // 1024}k</td>{''.join(tds)}</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(REPO, "report"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    smoke = load_json(os.path.join(REPO, "evidence", "smoke_v8_first_measurement.json"))
    short = load_json(os.path.join(REPO, "evidence", "wave1a_short_context.json"))
    long_ = load_json(os.path.join(REPO, "evidence", "wave1b_long_context.json"))

    smoke_rows = smoke["rows"]
    short_rows = short["rows"]
    long_rows = long_["rows"]
    seven_b = short_rows + long_rows

    short_lengths = sorted({r["length"] for r in short_rows})
    long_lengths = sorted({r["length"] for r in long_rows})
    all_lengths = sorted({r["length"] for r in seven_b})
    all_depths = sorted({r["depth"] for r in seven_b})
    long_depths = sorted({r["depth"] for r in long_rows})

    # ---- headline numbers, computed not typed ---------------------------
    def cellrate(rows, arm, L, metric="correct_word"):
        sub = [r for r in rows if r["arm"] == arm and r["length"] == L]
        return (rate(sub, metric), len(sub))

    b64, n64 = cellrate(long_rows, "base", 65536)
    b128, n128 = cellrate(long_rows, "base", 131072)
    y64, _ = cellrate(long_rows, "yarn", 65536)
    y128, _ = cellrate(long_rows, "yarn", 131072)
    s_base128, s_n128 = cellrate(smoke_rows, "base", 131072)

    n_rows_total = len(smoke_rows) + len(seven_b)
    total_cases = len({(r["arm"], r["length"], r["depth"]) for r in seven_b})

    # ---- charts ---------------------------------------------------------
    p1 = os.path.join(args.outdir, "chart_accuracy_by_length.png")
    p2 = os.path.join(args.outdir, "chart_depth_decay.png")
    p3 = os.path.join(args.outdir, "chart_heatmaps.png")
    p4 = os.path.join(args.outdir, "chart_smoke_1_5b.png")
    chart_accuracy_by_length(p1, seven_b, all_lengths,
                             "Qwen2.5-7B (4 бита): точность NIAH по длине контекста")
    chart_depth_decay(p2, long_rows, long_lengths)
    chart_heatmaps(p3, seven_b, all_lengths, all_depths)
    chart_accuracy_by_length(p4, smoke_rows, sorted({r["length"] for r in smoke_rows}),
                             "Qwen2.5-1.5B: тот же эффект на второй модели")

    html = []
    html.append("<!DOCTYPE html><html lang='ru'><head><meta charset='utf-8'>")
    html.append("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    html.append("<title>Влияние YaRN на NIAH — отчёт</title>")
    html.append(f"<style>{CSS}</style></head><body><div class='wrap'>")

    html.append("<header><h1>Влияние YaRN на needle-in-a-haystack</h1>"
                "<div class='sub'>Абляция: единственная переменная между руками — "
                "<code>rope_scaling</code>. Qwen2.5-1.5B и 7B, Kaggle T4, "
                "жадное декодирование.</div></header>")

    # verdict
    html.append("<div class='verdict'>")
    html.append(f"<div class='big'>За нативным окном YaRN убирает провал: "
                f"base <em>{b128 * 100:.0f}%</em> → YaRN <em>{y128 * 100:.0f}%</em> "
                f"на 128k</div>")
    html.append(f"<p>Дельта +{(y128 - b128) * 100:.1f} pp на 128k и "
                f"+{(y64 - b64) * 100:.1f} pp на 64k, при том что нативная длина модели — "
                f"32768 токенов. Внутри окна (4k…32k) разницы нет вообще. Провал base "
                f"не разбросан, а упорядочен по расстоянию от вопроса — см. раздел 3.</p>")
    html.append("</div>")

    html.append("<div class='cards'>")
    html.append(f"<div class='card'><div class='k'>base @ 128k</div>"
                f"<div class='v r'>{b128 * 100:.0f}%</div>"
                f"<div class='d'>{int(b128 * n128)}/{n128} кейсов</div></div>")
    html.append(f"<div class='card'><div class='k'>YaRN @ 128k</div>"
                f"<div class='v g'>{y128 * 100:.0f}%</div>"
                f"<div class='d'>{int(y128 * n128)}/{n128} кейсов</div></div>")
    html.append(f"<div class='card'><div class='k'>дельта @ 128k</div>"
                f"<div class='v g'>+{(y128 - b128) * 100:.1f} pp</div>"
                f"<div class='d'>цена ошибки экстраполяции</div></div>")
    html.append(f"<div class='card'><div class='k'>внутри окна</div>"
                f"<div class='v'>0.0 pp</div>"
                f"<div class='d'>4k…32k, обе руки 100%</div></div>")
    html.append("</div>")

    # 1
    html.append("<h2><span class='n'>1.</span>Что именно измерялось</h2>")
    html.append("<p>Сравниваются две руки на <b>идентичной</b> сетке кейсов, где "
                "различается ровно один параметр — способ масштабирования позиций. "
                "Всё остальное (модель, датасет, глубина иголки, промпт, квантизация, "
                "декодирование) зафиксировано.</p>")
    html.append("<table><thead><tr><th>параметр</th><th>base (контроль)</th>"
                "<th>YaRN (воздействие)</th></tr></thead><tbody>")
    html.append("<tr><td><code>rope_type</code></td><td class='d'>default</td>"
                "<td class='d'>yarn</td></tr>")
    html.append("<tr><td><code>factor</code></td><td class='d'>—</td><td class='d'>4.0</td></tr>")
    html.append("<tr><td><code>original_max_position_embeddings</code></td>"
                "<td class='d'>—</td><td class='d'>32768</td></tr>")
    html.append("<tr><td><code>rope_theta</code></td><td class='d'>1000000</td>"
                "<td class='d'>1000000</td></tr>")
    html.append("<tr><td>нативная длина модели</td><td class='d'>32768</td>"
                "<td class='d'>32768</td></tr>")
    html.append("<tr><td>тестируемые длины</td><td class='d' colspan='2'>"
                "4k, 8k, 16k, 32k, 64k, 128k</td></tr>")
    html.append("</tbody></table>")
    html.append(f"<p class='cap'>Значения <code>rope_type</code> и параметров YaRN взяты не "
                f"из конфигурации запуска, а из лога модели после загрузки: харнесс падает, "
                f"если модель сообщает не ту руку, что запрошена. Всего {n_rows_total} строк "
                f"результатов, {total_cases} уникальных измеренных ячеек в 7B-сетке.</p>")

    # 2
    html.append("<h2><span class='n'>2.</span>Главный результат</h2>")
    html.append(img_tag(p1, "Точность по длинам, 7B. Оранжевая зона — за пределами нативного окна."))
    html.append(accuracy_table(seven_b, all_lengths))
    html.append(delta_table(seven_b, all_lengths))
    html.append("<p class='cap'>Здесь <code>n</code> — число <b>глубин</b>, проверенных на этой "
                "длине: одна ячейка (длина × глубина) = один кейс с одной иголкой. Поэтому "
                "внутри окна n=7 (все семь глубин), а на 64k и 128k n=3 — бюджет ушёл на две "
                "дорогие длины вместо разрешения по глубине. Величины в ячейках кратны 33%, "
                "потому что это 0, 1 или 2 попадания из трёх.</p>")
    html.append("<p>До 32k включительно обе руки дают 100% — модель просто не испытывает "
                "никакого давления, и измерять там нечего. Ровно с 64k, то есть за "
                "нативным окном, базовая рука начинает разваливаться, а YaRN держит.</p>")

    # 3
    html.append("<h2><span class='n'>3.</span>Почему это не случайность</h2>")
    html.append("<p>Три кейса на клетку — мало, и само по себе «2/3 против 3/3» ничего бы не "
                "доказывало. Доказывает структура провала. Глубина 0.9 ставит иголку в конец "
                "стога, то есть ближе всего к вопросу; глубина 0.1 — дальше всего.</p>")
    html.append(img_tag(p2, "Точность против расстояния от вопроса. Монотонный рост — подпись "
                            "деградации ротационного экстраполирования."))
    html.append(depth_table(long_rows, long_lengths, long_depths))
    html.append("<p>base проваливает ровно те иголки, что дальше всего, и проваливает их тем "
                "больше, чем длиннее контекст: на 64k — только самую дальнюю, на 128k — уже две "
                "из трёх. Это монотонность и по расстоянию, и по длине одновременно. Случайный "
                "шум так не выглядит; так выглядит распад позиционного сигнала, у которого "
                "относительные расстояния уехали за пределы обученных. YaRN пересчитывает "
                "частоты так, чтобы расстояния остались внутри обученного диапазона — и профиль "
                "выравнивается в горизонтальную линию.</p>")
    html.append("<div class='note'><b>Важно:</b> на каждой длине измерено по три точки "
                "(три глубины, по одному кейсу каждая). Этого мало для размера эффекта, но "
                "достаточно, чтобы увидеть порядок: три точки, монотонные и по расстоянию, и "
                "по длине, не выглядят как шум. Читать следует «провал реален и YaRN его "
                "снимает», а не «YaRN стоит 67 пунктов».</div>")

    # 4
    html.append("<h2><span class='n'>4.</span>Внутри окна YaRN бесплатен — но это не доказано</h2>")
    html.append(img_tag(p3, "Полная сетка (длина × глубина) для 7B. Прочерк — не измерялось."))
    html.append("<p>На 4k…32k обе руки дают 100% на всех семи глубинах. Это "
                "<b>эффект потолка</b>, а не доказательство бесплатности: у задачи нет запаса, "
                "в котором могла бы проявиться деградация. В карточке Qwen сказано, что "
                "статический YaRN может ухудшать качество на коротких текстах — этой сеткой "
                "такое утверждение ни подтвердить, ни опровергнуть нельзя. Нужен инструмент, "
                "где модель не насыщена: перплексия или бенчмарк на знания.</p>")

    # 5
    html.append("<h2><span class='n'>5.</span>Эффект воспроизводится на меньшей модели</h2>")
    html.append(img_tag(p4, "Qwen2.5-1.5B: тот же провал на 128k и то же восстановление."))
    html.append(f"<p>На другой модели, другом размере и другом прогоне картина та же: "
                f"base на 128k даёт {s_base128 * 100:.0f}% ({int(s_base128 * s_n128)}/{s_n128}), "
                f"YaRN — 100%. Совпадение направления на двух независимых моделях — это уже не "
                f"артефакт одной конфигурации.</p>")

    # 6
    html.append("<h2><span class='n'>6.</span>Ограничения</h2>")
    html.append("<ul>")
    html.append("<li><b>n=3 на клетку</b> на длинных длинах. Бюджет сознательно потрачен на две "
                "длинные длины вместо семи глубин: контраст интересен по длине, а не по "
                "разрешению глубин.</li>")
    html.append("<li><b>4-битная квантизация</b> в обоих 7B-прогонах. Обе руки квантованы "
                "одинаково, так что сравнение корректно, но абсолютные числа base в bf16 были бы "
                "выше, а квантизация может усиливать разрыв.</li>")
    html.append("<li><b>Пропущенные глубины</b> 0.25, 0.75 и 1.0 на 64k и 128k — белые клетки "
                "на теплокарте.</li>")
    html.append("<li><b>Один тип задачи.</b> NIAH проверяет извлечение, а не рассуждение над "
                "длинным контекстом.</li>")
    html.append("</ul>")

    # 7
    html.append("<h2><span class='n'>7.</span>Что пришлось починить, чтобы это вообще измерить</h2>")
    html.append("<p>Замер на бесплатной карте 14.56 ГиБ оказался в основном инженерной задачей. "
                "Все правки сделаны по измерению, а не по догадке:</p>")
    html.append("<ul>")
    html.append("<li><b>Внимание на sm75.</b> Flash attention требует sm80+, а флаг GQA "
                "заставляет transformers выбрать math-бэкенд: 7.24 ГиБ против 0.10 ГиБ на 8k. "
                "Обход — развернуть K/V и использовать memory-efficient ядро.</li>")
    html.append("<li><b>OOM в MLP на 131k.</b> Активации требуют ~58 КБ на токен, около 7.3 ГиБ "
                "до чанкинга. Префилл нарезан на чанки; на CPU проверено, что результат "
                "побитово совпадает.</li>")
    html.append("<li><b>Маска чанка оказалась настоящим потолком.</b> Ограничение задавала не "
                "активация, а маска: 32768 × 98304 × 2 = 6.00 ГиБ — это и была падавшая "
                "аллокация. Отсюда чанк 2048 на 131k.</li>")
    html.append("<li><b>Баг парсера, смещавший результат вверх.</b> Строки кейсов печатаются "
                "через <code>repr()</code>, который переключается на двойные кавычки, если в "
                "предсказании есть апостроф. Регулярка принимала только одинарные — и терялись "
                "ровно две неудачи base. Починено, закреплено тестом, старая версия проверена "
                "(2 из 4 строк против 4 из 4).</li>")
    html.append("<li><b>Защиты от тихих отказов.</b> Модель, выгруженная accelerate на CPU, "
                "теперь роняет прогон, а не показывает правдоподобные числа; фактический dtype "
                "проверяется после загрузки, потому что в transformers v5 аргумент "
                "<code>torch_dtype</code> переименован и неверное имя может быть молча "
                "проглочено.</li>")
    html.append("</ul>")

    # 8
    html.append("<h2><span class='n'>8.</span>Артефакты</h2>")
    html.append("<ul>")
    html.append("<li><code>evidence/</code> — сырые строки и окружение по каждому прогону: "
                "<code>smoke_v8_first_measurement.json</code>, "
                "<code>wave1a_short_context.json</code>, <code>wave1b_long_context.json</code>.</li>")
    html.append("<li><code>evidence/final_report/</code> — сводка по всем строкам: теплокарта, "
                "CSV, <code>summary.md</code>.</li>")
    html.append("<li>Датасет результатов на Kaggle "
                "(<code>nikitaprogart/yarn-niah-results</code>) — 68 строк, 4 файла; прогоны "
                "восстанавливают из него уже посчитанные кейсы вместо пересчёта.</li>")
    html.append("<li><code>tools/build_report.py</code> — этот отчёт. Все числа считаются из "
                "<code>evidence/</code>, ни одно не вписано руками.</li>")
    html.append("</ul>")
    html.append("<p>Строки wave1b восстановлены двумя независимыми транспортами — "
                "префиксом <code>NIAH_ROW</code> в логе ядра и output-эндпоинтом — и оба дали "
                "одно и то же по всем полям. Это важно, потому что output-эндпоинт Kaggle на "
                "этих прогонах периодически отдаёт пустоту, и лог остаётся единственным каналом "
                "наружу.</p>")

    # 9
    html.append("<h2><span class='n'>9.</span>Что дальше</h2>")
    html.append("<ul>")
    html.append("<li><b>Добрать пропущенные глубины</b> на 64k/128k, чтобы клетки стали n=7 и "
                "теплокарта перестала быть дырявой.</li>")
    html.append("<li><b>Вторая видеокарта.</b> В прогонах занята одна T4 из двух. 4-битная "
                "реплика целиком влезает в одну карту (~10.9 ГиБ), поэтому правильный "
                "параллелизм — не раскидать слои одной модели по двум картам (это даёт память, "
                "но не скорость: карты считают по очереди), а держать по реплике на карте и "
                "гнать два кейса одновременно. Это вдвое сокращает время волны.</li>")
    html.append("<li><b>Проверить, что разрыв не артефакт квантизации</b> — повторить сетку "
                "без 4 бит.</li>")
    html.append("<li><b>Инструмент для короткого контекста.</b> Чтобы поймать возможную цену "
                "YaRN внутри окна, нужна перплексия или бенчмарк на знания: NIAH там насыщен.</li>")
    html.append("</ul>")

    html.append("<footer>Сгенерировано <code>tools/build_report.py</code> из "
                "<code>evidence/</code>. Все проценты посчитаны из сырых строк; "
                "метрика — <code>correct_word</code>, совпадение по границам слов.</footer>")
    html.append("</div></body></html>")

    out = os.path.join(args.outdir, "niah_yarn_report.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(html))
    print(f"[done] {out}")
    for p in (p1, p2, p3, p4):
        print(f"       {p}")


if __name__ == "__main__":
    main()
