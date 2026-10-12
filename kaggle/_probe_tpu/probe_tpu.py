#!/usr/bin/env python3
"""Can the harness run on Kaggle's TPU v3-8 (machine_shape Tpu1VmV38)?

The GPU path needed three memory fixes because a T4 has 14.56 GiB. A v3-8 has
8 x 16 GiB = 128 GiB of HBM, so the whole class of problem disappears: no 4-bit
quantisation, no 2048-token prefill chunk, no attention-backend workaround.

What is NOT obvious is whether the harness can use it. niah.resolve_placement picks
"cuda" or "cpu" only, so on a TPU VM it would silently choose CPU and merely look
slow instead of failing. And the scientific prerequisite is separate: YaRN must
actually change inv_freq under XLA, or the whole comparison is void.

Every step is wrapped, because on the vLLM probe an IPython traceback hid the real
error behind a TypeError inside ultratb.
"""
import os
import sys


def step(label, fn):
    print(f"--- {label}", flush=True)
    try:
        return fn()
    except BaseException as exc:
        print(f"    FAILED: {type(exc).__name__}: {str(exc)[:400]}", flush=True)
        return None


print("python", sys.version.split()[0], flush=True)
print("PJRT_DEVICE =", os.environ.get("PJRT_DEVICE"), flush=True)
print("TPU_NAME    =", os.environ.get("TPU_NAME"), flush=True)
print("XRT_TPU_CONFIG =", os.environ.get("XRT_TPU_CONFIG"), flush=True)


def torch_info():
    import torch
    print(f"    torch {torch.__version__} | cuda_available {torch.cuda.is_available()}", flush=True)
    import torch_xla
    import torch_xla.core.xla_model as xm
    print(f"    torch_xla {torch_xla.__version__}", flush=True)
    print(f"    xla_device() -> {xm.xla_device()}", flush=True)
    print(f"    device count -> {xm.xrt_world_size()}", flush=True)


step("torch + torch_xla", torch_info)

dev = step("acquire XLA device",
           lambda: __import__("torch_xla.core.xla_model", fromlist=["xla_device"]).xla_device())

# The prerequisite: base vs yarn must produce different inv_freq on this stack.
def rope_check():
    import copy
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    import niah

    cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    print(f"    default rope_parameters: {cfg.rope_parameters}", flush=True)

    base = copy.deepcopy(cfg)
    niah.apply_rope_scaling(base, niah.rope_scaling_for("base", 4.0, 32768, None, 32, 1,
                                                        base.rope_parameters))
    emb_base = Qwen2RotaryEmbedding(config=base)

    yarn = copy.deepcopy(cfg)
    entry = niah.rope_scaling_for("yarn", 4.0, 32768, None, 32, 1, yarn.rope_parameters)
    niah.apply_rope_scaling(yarn, entry)
    emb_yarn = Qwen2RotaryEmbedding(config=yarn)

    same = torch.allclose(emb_base.inv_freq, emb_yarn.inv_freq)
    print(f"    base rope_type={emb_base.rope_type!r} yarn rope_type={emb_yarn.rope_type!r}", flush=True)
    print(f"    attention_scaling yarn={float(emb_yarn.attention_scaling):.4f}", flush=True)
    print(f"    inv_freq identical base vs yarn: {same}", flush=True)
    if same:
        raise RuntimeError("YaRN did NOT change inv_freq - the ablation would be void")


step("YaRN changes inv_freq (scientific prerequisite)", rope_check)

# The real question: a forward pass and a generation on the TPU.
def tiny_generate():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
                                                 dtype=torch.bfloat16)
    model = model.to(dev)
    model.eval()
    ids = tok("The capital of France is", return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=12, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    print(f"    generated: {text[:60]!r}", flush=True)


step("forward + generate on TPU (0.5B, bf16)", tiny_generate)


def prefill_speed():
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
                                                 dtype=torch.bfloat16).to(dev).eval()
    ids = tok("The quick brown fox jumps over the lazy dog. " * 2000,
              return_tensors="pt").input_ids.to(dev)
    print(f"    prompt {ids.shape[1]} tokens", flush=True)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=8, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    print(f"    {ids.shape[1]} token prefill + 8 gen in {time.time()-t0:.1f}s", flush=True)


step("prefill speed (8k tokens)", prefill_speed)

print("\nPROBE DONE", flush=True)
