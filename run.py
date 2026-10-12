#!/usr/bin/env python3
"""CLI: run one arm of the YaRN-vs-native NIAH ablation.

Examples
--------
# native RoPE, extrapolated far past its 32k training window
python run.py --model Qwen/Qwen2.5-7B-Instruct --arm base \
    --lengths 4096,8192,16384,32768,65536,131072 --out results/base.jsonl

# same weights, YaRN factor 4 -> 131072
python run.py --model Qwen/Qwen2.5-7B-Instruct --arm yarn --factor 4 \
    --original-max 32768 --lengths 4096,8192,16384,32768,65536,131072 \
    --out results/yarn_f4.jsonl

# smoke test: 1 length, 2 depths, tiny output budget
python run.py --model Qwen/Qwen2.5-1.5B-Instruct --arm yarn --factor 4 \
    --lengths 8192 --depths 0.5 --max-new-tokens 32 --out results/smoke.jsonl
"""
from __future__ import annotations

import argparse

from niah import run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NIAH ablation of RoPE scaling (YaRN vs native).")
    g = p.add_argument_group("experiment")
    g.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    g.add_argument("--arm", default="yarn", choices=["base", "yarn", "linear", "dynamic"])
    g.add_argument("--factor", type=float, default=4.0, help="YaRN scaling factor -> effective context = original_max * factor")
    g.add_argument("--original-max", type=int, default=32768, help="must match the model's max_position_embeddings")
    g.add_argument("--attention-factor", type=float, default=None, help="YaRN mscale; default = transformers' own computation")
    g.add_argument("--beta-fast", type=float, default=32.0)
    g.add_argument("--beta-slow", type=float, default=1.0)
    g.add_argument("--lengths", default="4096,8192,16384,32768,65536,131072", help="target context lengths in tokens, comma-separated")
    g.add_argument("--depths", default="0.0,0.1,0.25,0.5,0.75,0.9,1.0", help="needle depth as a fraction of the haystack")
    g.add_argument("--needles-per-case", type=int, default=1, help="how many different needles to test per (length,depth)")
    g.add_argument("--n-needles", type=int, default=5)
    g.add_argument("--mode", default="single", choices=["single", "multi"])
    g.add_argument("--prompt-style", default="chat", choices=["chat", "raw"])
    g.add_argument("--enable-thinking", action="store_true",
                   help="Qwen3/Qwen3.5 reason before answering; off by default, because the model card"
                        " warns greedy decoding in thinking mode degenerates, and we score the final answer")
    g.add_argument("--force", action="store_true", help="proceed even if the sanity check says the setup cannot measure YaRN")
    g.add_argument("--seed", type=int, default=0)

    h = p.add_argument_group("runtime")
    h.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    h.add_argument("--attn", default=None, choices=[None, "sdpa", "flash_attention_2", "eager"])
    h.add_argument("--allow-math-sdpa", action="store_true",
                   help="permit SDPA's math backend, which materialises the full attention matrix "
                        "and therefore OOMs at long context on sm75 (T4)")
    h.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    h.add_argument("--load-in-4bit", action="store_true", help="NF4; needed for 7B @128k on a 16GB T4. Use the SAME flag for every arm.")
    h.add_argument("--device-map", default=None, help="e.g. auto; use 'auto' to shard across Kaggle's 2xT4")
    h.add_argument("--hf-batch", type=int, default=1)
    h.add_argument("--max-new-tokens", type=int, default=64)
    h.add_argument("--max-model-len", type=int, default=None, help="vLLM only; must be >= max --lengths")
    h.add_argument("--tensor-parallel-size", type=int, default=1)
    h.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    h.add_argument("--trust-remote-code", action="store_true")

    i = p.add_argument_group("io")
    i.add_argument("--out", default="results/run.jsonl", help="JSONL; appended to and resumable after a session kill")
    i.add_argument("--corpus", default=None, help="path to a .txt haystack corpus; if omitted, download wikitext-103")
    i.add_argument("--cache-dir", default="./cache")
    i.add_argument("--upload", default=None, help="HF dataset repo id to sync the JSONL to, e.g. me/yarn-niah")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.max_model_len is None:
        args.max_model_len = max(int(x) for x in args.lengths.split(","))
    run(args)


if __name__ == "__main__":
    main()
