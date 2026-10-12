# yarn-haystack

Measure **what YaRN actually buys you** on needle-in-a-haystack, on free-tier GPUs
(Kaggle 2×T4 / Colab T4).

> **Results report:** [`report/niah_yarn_report.html`](report/niah_yarn_report.html) —
> results, charts, limitations and conclusions. It is generated from `evidence/` by
> `tools/build_report.py`, so no number in it is typed by hand.

## The one design decision that makes this a real experiment

The only thing that changes between arms is `rope_scaling`. Same weights, same
tokenizer, same haystack sentence order, same needle offset, greedy decoding.
Anything else and you are measuring two differences at once.

| arm | `rope_scaling` | what it means |
|---|---|---|
| `base` | none | native RoPE of the checkpoint (Qwen2.5: 32k, `rope_theta=1e6`) |
| `yarn` | `{type: yarn, factor: 4, original_max_position_embeddings: 32768}` | 131k effective |
| `linear` | `{type: linear, ...}` | cheap reference point |
| `dynamic` | `{type: dynamic, ...}` | NTK-by-parts reference point |

`base` is the interesting arm, not a throwaway: at 64k/128k it is *pure
extrapolation* past the trained window. The YaRN effect is exactly
`yarn@128k − base@128k`, and `yarn@4k − base@4k` is the price you pay in
short-context quality. Both numbers matter; only reporting the first is how
people fool themselves.

> Why Qwen2.5 / Qwen3 and not the newest Qwen: **the ablation needs the model to
> actually fail without YaRN.** Anything natively long-context has no failure for
> YaRN to fix at ≤128k, so the "effect" you measure is zero by construction — the
> worst possible outcome, because it looks like a clean negative result. The
> shipped configs, verified against the Hub:

| model | `rope_scaling` in config | trained prompt window | KV @128k, fp16 | usable here? |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | none | 32768 | 7.0 GiB (56 KiB/tok) | **yes** — cheapest path to a number |
| Qwen2.5-3B-Instruct | none | 32768 | 4.5 GiB | yes, fp16 on one T4 |
| **Qwen3-8B** | **none** | **32768** (config says 40960) | **18.0 GiB (144 KiB/tok)** | yes, but needs 2×T4 at 128k |
| Qwen3-1.7B / 0.6B | none | 32768 (config 40960) | 14.0 GiB | yes, tight on one T4 |
| Qwen3-4B-Instruct-2507 | none | 262144 | 18.0 GiB | **no** — inside window |
| Qwen3.5-0.8B … 397B | none | 262144 | ~4–8 GiB (hybrid linear attn) | **no** — inside window |
| Qwen3-Next-80B, Qwen3-Coder | none | 262144 | — | **no** — inside window |

Two things worth knowing before you pick a model, both from Qwen's own cards:

1. **Qwen3's config says `max_position_embeddings: 40960`, but the card says
   "32,768 natively and 131,072 tokens with YaRN"** — the extra 8192 is output
   space reserved out of the same budget. The official YaRN recipe is
   `factor: 4.0, original_max_position_embeddings: 32768`. Copying 40960 out of
   `config.json` (the obvious thing to do) builds `inv_freq` over positions the
   model never trained on. `run.py` blocks exactly this.
2. **Static YaRN is static**: the card states it "remains constant regardless of
   input length, **potentially impacting performance on shorter texts**". So the
   `yarn@4k − base@4k` delta is a documented, expected cost — measure it, don't
   be surprised by it. Qwen's guidance is to match the factor to the target
   (2.0 for 64k), which makes a cheap `factor ∈ {2, 3, 4}` sweep a natural second
   axis at the same prefill cost.

Switching models is one line (`MODEL` in the notebook); the protocol is unchanged.

### Guardrails, because the failure mode is a plausible-looking wrong number

`run.py` refuses to start (before downloading weights) when the setup cannot
answer the question: a natively-256k model, `original_max` not matching the
documented prompt window, a `base` arm that never leaves the trained window, or
a test length beyond `original_max × factor`. `--force` overrides. See
`test_logic.py` for the cases.

## Quickstart

```bash
pip install -r requirements.txt

# smoke first — the harness raises if rope_scaling did not actually stick
python run.py --model Qwen/Qwen2.5-1.5B-Instruct --arm yarn --factor 4 \
  --original-max 32768 --lengths 8192 --depths 0.5 --max-new-tokens 24 \
  --out results/smoke.jsonl

python run.py --model Qwen/Qwen2.5-7B-Instruct --arm base --load-in-4bit \
  --lengths 4096,8192,16384,32768,65536,131072 --out results/base.jsonl
python run.py --model Qwen/Qwen2.5-7B-Instruct --arm yarn --factor 4 --original-max 32768 --load-in-4bit \
  --lengths 4096,8192,16384,32768,65536,131072 --out results/yarn_f4.jsonl

python analyze.py results/*.jsonl --outdir report   # summary.md + heatmaps.png + CSVs
```

Or open `yarn_niah.ipynb` and run it top to bottom on Kaggle/Colab.

## Cost

Grid = 6 lengths × 7 depths × 1 needle = 42 cases per arm; 2 arms = 84 generations.

| model | prefill for the whole 2-arm grid | footprint @128k | verdict on one T4 |
|---|---|---|---|
| Qwen2.5-1.5B | ~6 min of T4 FLOPs | 6.3 GiB fp16 | yes |
| Qwen2.5-3B | ~13 min | 10.3 GiB fp16 | yes |
| Qwen2.5-7B | ~33 min | 11.3 GiB (NF4 + KV) | yes, with `--load-in-4bit` |
| Qwen3-8B | ~36 min | 22.2 GiB (NF4 + KV) | **no** → 64k only, or 2×T4 |
| Qwen3-1.7B | ~9 min | 14.9 GiB (NF4 + KV) | tight |

Compute is not the binding constraint — memory is, and Qwen3's 144 KiB/token KV
cache is what makes "take a newer Qwen" cost more than it looks at a glance.
Naive HF decoding at 128k is the other tax; `--backend vllm --tensor-parallel-size 2`
on Kaggle's two T4s removes both problems if vLLM installs cleanly (see `SETUP.md`).

## Files

| file | role |
|---|---|
| `niah.py` | corpus, case construction, deterministic scoring, arm/config wiring, sanity guards, generation, resumable JSONL writer |
| `run.py` | CLI for one arm |
| `analyze.py` | JSONL → `summary.md` + `heatmaps.png` + CSVs, incl. the YaRN delta table |
| `test_logic.py` | torch-free tests: needle placement, scoring, resume, sanity guards, thinking handling |
| `yarn_niah.ipynb` | Kaggle/Colab driver |
| `SETUP.md` | platform specifics, session-splitting plan, install risks |

## First measurement (smoke, n=3 — direction, not an effect size)

Qwen2.5-1.5B-Instruct, Kaggle T4 (sm75), greedy, one needle at three depths:

| arm | 4096 | 131072 |
|---|---|---|
| base (shipped config, extrapolating) | 3/3 | **1/3** |
| yarn (factor 4.0, original_max 32768) | 3/3 | **3/3** |

The only difference between the two rows is `rope_scaling`. The direction is what
the protocol predicts: `base` is far outside its 32k window at 128k and collapses
there, YaRN holds. Three samples per cell cannot support a magnitude claim, and the
4k row shows no short-context penalty either - with n=3 that is an absence of
evidence, not evidence of absence.

Raw rows and the full environment are in `evidence/smoke_v8_first_measurement.json`.

### wave1a — short contexts, Qwen2.5-7B (4k to 32k, 7 depths each)

| arm | 4k | 8k | 16k | 32k |
|---|---|---|---|---|
| base (shipped config) | 7/7 | 7/7 | 7/7 | 7/7 |
| yarn (factor 4.0) | 7/7 | 7/7 | 7/7 | 7/7 |

No penalty at any length - but read this as a **ceiling effect, not a free lunch**.
Both arms are perfect at every depth, so NIAH at these lengths has no headroom to
show a degradation a 7B model would have to be near a boundary to exhibit. Qwen's
own card warns that static YaRN "may potentially degrade model performance" on short
texts; that claim is not refutable or confirmable with a task the model already
saturates. Perplexity or a knowledge benchmark would be the instrument for it.

Two runs produced this number, not one: the first ran only the yarn arm because an
over-strict sanity check refused the base arm, and the second did not restore the
first's rows as intended (the results dataset version was still being created when
the kernel was pushed). Correctness was unaffected - `run.py` skips only whole cases
it has already written - but it is why the pair took 17 + 32 minutes instead of ~35
for the pair or ~17 with a working carry-over.

Raw rows: `evidence/wave1a_short_context.json`, report in `evidence/wave1a_report/`.

Running this at all on a 14.6 GiB T4 required three separate memory fixes, each
measured rather than guessed - see `attention_setup.py` and `SETUP.md`:
attention must use the memory-efficient kernel with K/V expanded instead of the GQA
flag (0.76 GiB at 64k vs 7.24 GiB for math at 8k), and the prefill must be chunked
to 2048 at 131k because the per-chunk mask, not the activations, is what sets the
bound (32768 x 98304 x 2 = 6.00 GiB was the failing allocation).

### wave1b — past the native window, Qwen2.5-7B (64k and 128k)

This is the wave the whole setup was built for. The model's native window is 32768,
so at 64k and 128k it is operating outside anything it was trained on and the
treatment has something to repair:

| arm | 64k | 128k | mean |
|---|---|---|---|
| base (shipped config, extrapolating) | 2/3 | **1/3** | 50.0% |
| yarn (factor 4.0, original_max 32768) | 3/3 | **3/3** | 100.0% |
| delta | +33.3 pp | **+66.7 pp** | +50.0 pp |

The failures are not scattered, which is why three samples per cell still says
something. Broken down by needle depth:

| arm | length | 0.1 | 0.5 | 0.9 |
|---|---|---|---|---|
| base | 64k | **fail** | ok | ok |
| base | 128k | **fail** | **fail** | ok |
| yarn | 64k | ok | ok | ok |
| yarn | 128k | ok | ok | ok |

Depth 0.9 places the needle nearest the end of the haystack, i.e. closest to the
question. `base` fails exactly the needles that are far away, and fails more of them
as the context grows — the signature of distance-based rotary extrapolation decay,
not of a model that got unlucky. YaRN rescales the frequencies so the relative
distances stay inside the trained range, and retrieval goes flat.

Look at the full heatmap rather than the means: `evidence/final_report/heatmaps.png`
shows the two red cells in the base panel and a solid green panel for YaRN. The
unmeasured depths in the 64k/128k columns (0.25, 0.75, 1.0) are white — wave1b
deliberately spent its budget on three depths at two long lengths rather than seven
depths, because the interesting contrast is length, not mid-depth resolution.

Three caveats, stated plainly:

- **n=3 per cell.** The direction is consistent across two models (1.5B in the smoke
  run, 7B here) and monotone in both distance and length, but this is not an effect
  size. Read it as "the failure is real and YaRN removes it", not "YaRN is worth 67
  points".
- **4-bit quantization.** Both arms used the same `--load-in-4bit`, so the comparison
  is controlled, but the absolute base-arm numbers would likely be somewhat higher in
  bf16. Quantization makes long-context retrieval harder, so it may amplify the gap.
- **Only `rope_scaling` differs.** Verified in the log: the base arm reports
  `rope={'rope_theta': 1000000.0, 'rope_type': 'default'}` and the yarn arm reports
  `rope={'rope_theta': 1000000.0, 'rope_type': 'yarn', 'factor': 4.0,
  'original_max_position_embeddings': 32768, 'beta_fast': 32.0, 'beta_slow': 1.0}`,
  same model, same dtype, same attention, same chunking.

Raw rows: `evidence/wave1b_long_context.json`, consolidated report over all 68 rows
in `evidence/final_report/` (heatmaps, per-length CSV, `summary.md`).

## Validity checklist

- `--load-in-4bit`, `--dtype`, `--attn`, `--prompt-style`, `--enable-thinking` must be **identical across arms**.
- `--depths` must be the same string for both arms; depths are fractions of the haystack, which is what makes them comparable across lengths.
- 1 needle × 1 depth is not a result, it is an anecdote. Report the full depth×length heatmap; NIAH failure is usually depth-localised (lost-in-the-middle), not uniform.
- Answers are checked at word boundaries, so `7391` does not match inside `173910`.
- The answer is taken from **after** the last thinking block. An answer the model only considered internally is not a hit — otherwise Qwen3's default reasoning mode inflates every score.
- Qwen3 needs `transformers>=4.51`; older versions silently drop `original_max_position_embeddings` from `rope_scaling`, turning the treatment arm into a copy of the control.
- Empty/truncated preds from a killed session are visible in the JSONL, not silently scored as 0 — filter on `ts` if in doubt.
- Recovered rows are cross-checked across transports when both work. The wave1b run was recovered twice — from `NIAH_ROW` lines in the kernel log and from the kernel output endpoint — and the two agreed on all 12 rows and every field. When only the log path works, the case-line parser is a lossy fallback: it only sees the truncated prediction, and it silently dropped rows whose `repr()`-quoted prediction switched to double quotes, which biased the score upward (fixed, with a regression test in `test_log_to_results_parser`).
- Run `python test_logic.py` before a GPU session. It is one second and it catches the failures above.

