# Running on free Kaggle / Colab

## What you get

| | Kaggle Notebooks | Colab free |
|---|---|---|
| GPU | 2× T4 16 GiB (or P100 16 GiB) | 1× T4 16 GiB |
| GPU quota | ~30 h/week | soft daily cap, no guarantee |
| Session cap | ~9 h (idle 20 min) | ~4 h, disconnects without warning |
| Persistent disk | `/kaggle/working` (~20 GiB) survives stop | only Drive (mount it) |
| Internet | **off by default** — flip it on in the notebook settings | on |
| Install freedom | free, but torch is pinned to the image's CUDA | same |

Both are enough. The whole 2-arm grid is well under an hour of T4 time — the
sessions are limited by *disconnects*, not quota. Hence: append-only JSONL,
`fsync` after every row, skip completed cases on re-run. You already have that;
just don't clear `/kaggle/working` between sessions.

## Session plan (lose-safe)

1. **Session A** — install, run the 1.5B smoke test (2 min). Confirm the log line
   `arm=yarn rope_scaling={'type': 'yarn', ...}` before anything else. If it says
   `None`, fix that and stop; a long run on a no-op toggle is wasted GPU.
2. **Session B** — `base` arm, all lengths. Longest cases last: the resumable
   writer means a kill at 128k costs you one case, not the run.
3. **Session C** — `yarn` arm, same grid. Must be the same `--lengths/--depths`.
4. **Session D** — `analyze.py`, write the report. Cheap, CPU-only.

Sync out at the end of each session with `--upload <user>/yarn-niah` (needs
`huggingface-cli login` once), or just don't clear `/kaggle/working`.

## Install

The notebook installs selectively rather than with a blanket `pip install -U`,
and you should do the same by hand:

```bash
# Kaggle/Colab already ship a CUDA-matched torch. Do not reinstall it.
# Only transformers has a real version floor.
pip install -q -U "transformers>=4.51"
# then, only if missing:
pip install -q accelerate datasets bitsandbytes
```

### pip prints dependency conflicts — ignore them

After any install, pip re-checks the *whole* environment and reports every
conflict it finds, including pre-existing ones from the image. On Kaggle you will
typically see:

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
bigframes 2.48.0 requires google-cloud-bigquery-storage<3.0.0,>=2.30.0, which is not installed.
ydata-profiling 4.18.4 requires matplotlib<=3.10,>=3.5, but you have matplotlib 3.11.2 which is incompatible.
```

This header says `ERROR`, but the install succeeded. Both lines are harmless here:

- **bigframes** is Google's BigQuery DataFrame client, preinstalled in the Kaggle
  image with a missing transitive dependency. It existed before you ran anything;
  pip merely noticed it. The harness never imports it.
- **ydata-profiling** is Kaggle's EDA helper. It was fine until we asked for
  `-U matplotlib`. The fix is to not upgrade matplotlib at all — `analyze.py`
  uses only `subplots` / `imshow` / `colorbar` / `tight_layout` / `savefig`,
  which have been stable for a decade, so whatever the image ships is fine.

Neither package is imported anywhere in this repo, so neither can affect a score.
Check it instead of trusting it — this returns nothing:

```bash
grep -rn "import bigframes\|import ydata\|from bigframes\|from ydata" --include="*.py" .
```

Kaggle and Colab also scope `pip install` to the current session — a new session
starts from the pristine image, so this report disappears on its own.

`bitsandbytes` is only needed for `--load-in-4bit`. NF4 works on T4 (sm75).

Gated models (Llama) need `HF_TOKEN` in the environment / Colab secrets.

## Attention backend

Use `--attn sdpa` and move on. It is part of torch, has memory-efficient and
flash kernels, and never breaks.

`flash-attn` is a genuine speedup at 128k prefill but `pip install flash-attn
--no-build-isolation` routinely spends 20–40 minutes compiling, and then may not
match the session's torch/CUDA. Not worth it under time pressure; try it only if
you have a spare session.

## vLLM (the big win, and the big risk)

```bash
pip install vllm
python run.py ... --backend vllm --tensor-parallel-size 2 \
  --max-model-len 131072 --gpu-memory-utilization 0.90
```

On Kaggle's 2×T4 this gives you 32 GiB, so 7B fp16 at 128k fits without
quantization, and vLLM's automatic prefix caching reuses the long haystack
prefix across depths. Prefill gets paged + flash-attended, decode stops
re-reading the whole KV per token.

Risk: the preinstalled torch/CUDA version may fight the vLLM wheel. If `pip
install` starts pulling a new torch, **stop** — that install can eat your session
budget and break the HF path too. Fall back to `--backend hf`.

## If you OOM at 64k/128k

Order of preference:

1. `--load-in-4bit --attn sdpa` (7B: ~11.3 GiB at 128k). Use it for **every** arm.
2. `--model Qwen/Qwen2.5-3B-Instruct --dtype float16` (~10.3 GiB). Weaker model,
   same protocol — still shows the YaRN effect, and it is the honest answer if
   you cannot afford 7B.
3. `--device-map auto` to shard 7B fp16 across 2×T4. Works, slow, and
   cross-device KV shuffling at 128k is painful.
4. `--backend vllm --tensor-parallel-size 2`.

Never mix these across arms. A 4-bit `base` vs fp16 `yarn` comparison measures
quantization, not YaRN.

## Choosing the model (Qwen3 vs Qwen3.5)

Verified against the Hub configs, not from memory:

| family | trained window | usable for this ablation? |
|---|---|---|
| Qwen2.5-1.5B/3B/7B/14B | 32768 | yes — cheapest, `--load-in-4bit` for 7B+ |
| **Qwen3-0.6B…32B, Qwen3-30B-A3B** | **32768** (config says 40960) | **yes** — newer model, `rope_scaling: null` |
| Qwen3-4B/30B-A3B-Instruct-**2507** | 262144 | no — at ≤128k there is nothing to fix |
| Qwen3.5-0.8B…397B | 262144 | no — same reason |
| Qwen3-Next-80B, Qwen3-Coder-30B | 262144 | no — same reason |

"Native length is bigger" is only half the problem with Qwen3.5. The other half:
it is a **hybrid** architecture (`full_attention_interval: 4` — only 1 layer in 4
is full attention, the rest are linear attention), so `rope_scaling` is not even
the mechanism that governs its long-context behaviour. Measuring "the effect of
YaRN" there is ill-defined.

`run.py` now refuses these configurations outright, so the trap is closed.

Two Qwen3-specific things that will silently ruin a run:

- **`transformers>=4.51`.** Older versions log `Unrecognized keys in rope_scaling
  ... {'original_max_position_embeddings'}` and drop it, so your "YaRN" arm runs
  with a default `original_max` and stops being YaRN. Kaggle images sometimes
  ship older versions — the `pip install -U` in the notebook is not optional.
- **Thinking mode is on by default** and Qwen's card says *"DO NOT use greedy
  decoding"* in it — greedy + thinking is a documented repetition trap, and a
  NIAH answer buried in a reasoning block is not the answer. The harness passes
  `enable_thinking=False` and scores only text after the last thinking block.
  `--enable-thinking` turns it back on if you want to study that separately.

### Qwen3 KV cache is the real cost

Qwen3 dense models have 8 KV heads (Qwen2.5-7B has 4), so KV per token is
144 KiB vs 56 KiB — 2.6×. At 128k that is 18 GiB of KV alone:

| model | KV/token | @64k | @128k | NF4 weights + KV @128k |
|---|---|---|---|---|
| Qwen2.5-7B | 56 KiB | 3.5 GiB | 7.0 GiB | 11.3 GiB — fits one T4 |
| Qwen3-8B | 144 KiB | 9.0 GiB | 18.0 GiB | 22.2 GiB — needs 2×T4 |
| Qwen3-1.7B | 112 KiB | 7.0 GiB | 14.0 GiB | 14.9 GiB — tight on one T4 |
| Qwen3.5-4B | ~32 KiB (hybrid linear attn) | — | ~4 GiB | fits easily, but see above |

So on Colab's single T4, Qwen3-8B reaches 64k, not 128k. Kaggle's 2×T4 with
`--backend vllm --tensor-parallel-size 2` is the clean path to Qwen3-8B @128k.
Recommended order: get a number with Qwen2.5-7B first, then repeat with Qwen3-8B
on Kaggle to show the result is not a Qwen2.5 quirk.

## Troubleshooting the smoke test

`run.py` failing inside the notebook used to surface as a bare
`CalledProcessError: ... returned non-zero exit status 1`, which tells you
nothing. The runner cell now streams the child's output live and, on failure,
puts its **last 40 lines into the exception**, so the actual error is right
there. If you still only see the exit code, run the command by hand:

```bash
python -u run.py --model Qwen/Qwen2.5-1.5B-Instruct --arm yarn --factor 4 \
  --original-max 32768 --lengths 4096 --depths 0.5 --max-new-tokens 24 \
  --out results/smoke.jsonl
```

The two causes that have nothing to do with the harness:

1. **No internet.** Off by default in Kaggle notebooks, and both the tokenizer
   and the corpus download need it. Symptom: dies within a second with
   `Could not load a corpus ... or enable internet and pip install datasets`, or a
   connection error from `AutoTokenizer`. Fix: notebook settings → Internet → On.
   The preflight cell checks this before anything heavy starts.
2. **No CUDA / fp16 on CPU.** Half-precision on CPU dies in kernels such as
   `LayerNormKernelImpl not implemented for 'Half'`. `resolve_placement()` now
   warns and switches to float32 on CPU, so a CPU session runs instead of
   crashing — but a CPU run is a correctness check only, never a timing or memory
   measurement. Check the logged line:

   ```
   [info] device=cuda dtype=float16
   ```

   If it says `device=cpu`, you are on the wrong accelerator and every number
   below 32k will take forever and mean little.

Both are checked up front by the notebook's `PREFLIGHT` cell. Run it before the
smoke test whenever a session misbehaves.

## Colab CLI (optional, for A100 / no queue)

`google-colab-cli` is installed into this repo's tool dir and driven through
`colab/cl`. The wrapper exists because the CLI writes unconditionally to
`~/.config/colab-cli` (log, `token.json`, `sessions.json`, history) and reads
`~/.colab-cli-oauth-config.json`, with no environment override, while `$HOME` is
read-only here - so every command, including `readme`, died in `os.makedirs`:

    OSError: [Errno 30] Read-only file system: '/home/nikita/.config/colab-cli'

`os.path.expanduser` honours `$HOME`, so the wrapper points `HOME` at `.local/home/`
(gitignored) and relocates all of it at once. The OAuth client config is bundled in
the package, so none is needed.

Auth is interactive and needs a browser and your Google account:

```bash
./colab/cl status      # prints a URL, then asks for the authorization code
```

### Measured: Colab free is not a way around the memory ceiling

Probed on a live free-tier session (`colab/cl exec`):

| | Kaggle | Colab free |
|---|---|---|
| GPU | Tesla T4, 14.56 GiB, sm75 | Tesla T4, **14.56 GiB**, sm75 |
| torch | 2.11.0+cu128 | 2.11.0+**cu130** |
| transformers | 5.16.1 | **5.18.0** |
| disk free | ~20 GiB | 70 GiB |
| compute units | n/a | **0.00** |

So the same 14.56 GiB ceiling applies, and the 2048-token prefill chunk is needed
either way. A100/L4/H100 exist as `--gpu` variants but need compute units, of which
this account has none. My earlier note here claimed Colab would remove the ceiling;
that was wrong, and the probe is why it is now written down.

vLLM fails on both platforms, for different reasons - `libcudart.so.13` missing on
Kaggle (CUDA 12.8), and on Colab `TypeError: Config() got an unexpected keyword
argument 'deprecated'` against transformers 5.18. Fixing the second means pinning a
transformers that vLLM accepts, which is a separate experiment, not a quick win.

Colab's remaining advantage is the absence of a GPU queue; its disadvantage is a
daily quota and shorter sessions.

## Kaggle-specific gotchas

- **Turn internet on** in the notebook settings, or the corpus download and the
  HF model pull both fail.
- In `/kaggle/working` you have ~20 GiB; Qwen2.5-7B fp16 is 15 GiB of that. Point
  `HF_HOME` at `/kaggle/temp` (bigger, faster, but wiped when the session ends)
  and keep only the JSONL in `/kaggle/working`.
- Kaggle mounts datasets read-only under `/kaggle/input`. Uploading this repo as
  a dataset is the cleanest code distribution — copy it into `/kaggle/working`
  before running so the results land on a writable path.
- The first `run.py` call downloads wikitext-103 (the `--cache-dir` copy makes
  subsequent sessions instant).

## Corpus

Default is wikitext-103 (`--cache-dir` caches a single 4 MB text blob). The
classic Kamradt haystack is Paul Graham essays; any long English text works, as
long as it is *never summarizable* — repetitive filler is what makes retrieval
the only way to answer, which is the point of the benchmark. Pass
`--corpus /path/to/text.txt` to substitute.
