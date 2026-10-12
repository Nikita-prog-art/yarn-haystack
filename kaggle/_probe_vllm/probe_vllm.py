#!/usr/bin/env python3
"""Probe: can vLLM be installed and used on Kaggle's T4 image?

HF SDPA cannot do 128k on a T4: flash needs sm80+, the memory-efficient kernel is
unavailable here, and the math backend materialises [heads, q, kv]. vLLM uses paged
attention with a backend that does work on sm75, so it is the candidate for the
# can_use_flash_attention()/can_use_efficient_attention() take an SDPAParams object,
# they are not zero-argument probes. kaggle/_probe_attn tests them properly.
long lengths. This probe answers the install question before the real run depends
on it.
"""
import os
import subprocess
import sys
import time

os.environ.setdefault("HF_HOME", "/kaggle/temp/hf")
print("python", sys.version.split()[0], flush=True)

import torch
print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i}: {p.name} {p.total_memory / 2**30:.1f} GiB")
print(f"  capability: {torch.cuda.get_device_capability()}")

print("\ninstalling vllm...", flush=True)
t0 = time.time()
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"],
                   capture_output=True, text=True)
print(f"pip rc={r.returncode} after {time.time() - t0:.0f}s")
if r.returncode:
    print("STDERR tail:\n", r.stderr[-4000:])
    raise SystemExit("vllm install FAILED")

import vllm
print("vllm", vllm.__version__)
print("attention backends available:", getattr(vllm, "platforms", None) and "n/a")

from vllm import LLM, SamplingParams

print("\nconstructing LLM (Qwen2.5-0.5B, 8k) ...", flush=True)
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", max_model_len=8192,
          dtype="float16", gpu_memory_utilization=0.60)
print("LLM constructed OK")

sp = SamplingParams(temperature=0.0, max_tokens=16)
out = llm.generate([{"prompt": "The capital of France is"}], sp, use_tqdm=False)
print("short gen:", repr(out[0].outputs[0].text[:80]))

# The real question: does a long prompt fit and complete?
print("\nlong-context test: 32768 tokens ...", flush=True)
long_prompt = "The quick brown fox jumps over the lazy dog. " * 8000
n_tok = len(llm.get_tokenizer()(long_prompt)["input_ids"])
print(f"  prompt is {n_tok} tokens")
t0 = time.time()
out = llm.generate([{"prompt": long_prompt + "\n\nWhat animal jumps over the lazy dog?"}],
                   SamplingParams(temperature=0.0, max_tokens=16), use_tqdm=False)
dt = time.time() - t0
print(f"  32k prefill+gen OK in {dt:.1f}s: {out[0].outputs[0].text[:80]!r}")
print(f"  peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

print("\nVLLM PROBE OK - attention backend used by vLLM:", flush=True)
print("  env VLLM_ATTENTION_BACKEND =", os.environ.get("VLLM_ATTENTION_BACKEND", "(auto)"))
