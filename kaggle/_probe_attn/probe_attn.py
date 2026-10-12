#!/usr/bin/env python3
"""Which SDPA kernel can actually run a long causal attention on this GPU?

transformers passes `enable_gqa=True` whenever there is no attention mask and
head_dim <= 256, and that flag is only honoured by the flash kernel (sm80+). On a
T4 that can leave the math backend, which materialises [heads, q, kv] - hundreds of
GiB at 128k. Whether the memory-efficient kernel can rescue us depends on two
things this probe answers directly: does it exist on sm75, and does it accept the
GQA flag. No model download needed.
"""
import torch
import torch.nn.functional as F

print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()}")
print("capability:", torch.cuda.get_device_capability(),
      "(flash needs major >= 8)")


def probe(label, heads_q, heads_kv, seq, enable_gqa, backends, mask=None):
    """Run one SDPA call with a restricted backend set and report what happens."""
    torch.backends.cuda.enable_flash_sdp("flash" in backends)
    torch.backends.cuda.enable_mem_efficient_sdp("mem_efficient" in backends)
    torch.backends.cuda.enable_math_sdp("math" in backends)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    q = torch.randn(1, heads_q, seq, 128, dtype=torch.float16, device="cuda")
    k = torch.randn(1, heads_kv, seq, 128, dtype=torch.float16, device="cuda")
    v = torch.randn(1, heads_kv, seq, 128, dtype=torch.float16, device="cuda")
    kw = {"enable_gqa": True} if enable_gqa else {}
    try:
        F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None, **kw)
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"  [OK  ] {label:52s} peak={peak:6.2f} GiB")
        return True
    except RuntimeError as exc:
        msg = str(exc).split("\n")[0][:70]
        print(f"  [FAIL] {label:52s} {type(exc).__name__}: {msg}")
        return False
    finally:
        del q, k, v
        torch.cuda.empty_cache()


S = 8192
print(f"\n=== seq={S}, head_dim=128, GQA 12q/2kv ===")
probe("math + enable_gqa", 12, 2, S, True, {"math"})
probe("mem_efficient + enable_gqa", 12, 2, S, True, {"mem_efficient"})
probe("mem_efficient, no gqa (repeat_kv)", 12, 12, S, False, {"mem_efficient"})
probe("flash + enable_gqa", 12, 2, S, True, {"flash"})
probe("mem_efficient + explicit mask", 12, 12, S, False, {"mem_efficient"},
      mask=torch.zeros(1, 1, S, S, dtype=torch.float16, device="cuda"))

print("\n=== how far does each backend go? (no gqa, so repeat_kv) ===")
for seq in (16384, 32768, 65536):
    ok = probe(f"mem_efficient, no gqa, seq={seq}", 12, 12, seq, False, {"mem_efficient"})
    if not ok:
        break

print("\n=== summary that matters ===")
print("mem_efficient without GQA is usable on this GPU:", flush=True)
probe("confirmation: seq=16384, no gqa", 12, 12, 16384, False, {"mem_efficient"})
print("\nNOTE: if the no-gqa mem_efficient rows say OK, transformers can use them by")
print("forcing repeat_kv instead of enable_gqa (use_gqa_in_sdpa -> False).")
