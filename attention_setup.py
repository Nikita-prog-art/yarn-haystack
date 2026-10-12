"""Attention backend configuration for long-context evaluation on small GPUs.

Why this module exists, stated as measurements rather than folklore. On a Kaggle
Tesla T4 (sm75, torch 2.11) a probe of `scaled_dot_product_attention` with
head_dim=128 and GQA 12q/2kv gave:

    math + enable_gqa                    OK     peak 7.24 GiB @ seq=8192
    mem_efficient + enable_gqa           FAIL   RuntimeError: Invalid backend
    mem_efficient, no gqa (repeat_kv)    OK     peak 0.10 GiB @ seq=8192
    flash + enable_gqa                   FAIL   flash needs sm80+
    mem_efficient, no gqa, seq=65536     OK     peak 0.76 GiB

Two conclusions drive the code below.

1. `enable_gqa=True` is the blocker, not the device. transformers passes that flag
   whenever there is no attention mask and head_dim <= 256 (see
   `use_gqa_in_sdpa`), and on sm75 only the math backend honours it. Math
   materialises [heads, q_len, kv_len]: 7.2 GiB already at 8k, and hundreds of GiB
   at 128k, which surfaces as "Tried to allocate 16.00 GiB" and points at memory
   instead of at the kernel choice.

2. Forcing the expand path (`repeat_kv`) makes the memory-efficient kernel usable
   and it scales linearly - 0.76 GiB at 64k, so roughly 1.5 GiB at 128k. The cost
   is a transient per-layer expansion of K and V; the KV cache itself stays
   GQA-compressed, so the cache size is unaffected.

So: on sm75 we force the expand path, which is the difference between "128k is
impossible on a free T4" and "128k costs about 1.5 GiB".
"""
from __future__ import annotations

# torch is imported lazily inside the functions that need it: this module must stay
# importable without torch installed, because test_logic.py (the torch-free test
# suite) imports it to check the backend-selection logic.

# Set once we have patched transformers, so repeated calls are cheap and the log
# line can tell the truth about what happened.
_PATCHED = False


def should_force_repeat_kv(capability: tuple[int, int] | None) -> bool:
    """True when flash attention is unavailable, i.e. `enable_gqa` cannot be used.

    Flash needs sm80+. Below that, GQA-aware SDPA falls back to a kernel that
    either refuses the flag or materialises the full attention matrix.
    """
    return capability is not None and capability[0] < 8


def force_repeat_kv_in_sdpa() -> bool:
    """Make transformers expand K/V instead of passing `enable_gqa`.

    Patches the name inside `transformers.integrations.sdpa_attention`, which is
    where the call is made unqualified. Returns True if the patch is in place.
    """
    global _PATCHED
    import transformers.integrations.sdpa_attention as sdpa_mod  # noqa: PLC0415

    sdpa_mod.use_gqa_in_sdpa = lambda *_args, **_kwargs: False
    _PATCHED = True
    return True


def configure_attention(allow_math: bool = True, verbose: bool = True) -> dict:
    """Enable every usable SDPA kernel and dodge the sm75 GQA trap.

    `allow_math` stays True by default on purpose: disabling math turned a working
    4k run into "RuntimeError: Invalid backend", because on this device the
    remaining kernels cannot take the GQA flag. Better to keep math as a last
    resort and refuse only the lengths that cannot possibly fit.
    """
    import torch

    out: dict = {"capability": None, "enabled": [], "forced_repeat_kv": False, "device": "cpu"}
    if not torch.cuda.is_available():
        if verbose:
            print("[info] no CUDA: attention backend selection skipped")
        return out

    capability = torch.cuda.get_device_capability()
    major = capability[0]
    flash = major >= 8

    torch.backends.cuda.enable_flash_sdp(flash)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(bool(allow_math))

    forced = False
    if not flash:
        forced = force_repeat_kv_in_sdpa()

    out.update(capability=capability, device=f"sm{major}{capability[1]}", forced_repeat_kv=forced)
    out["enabled"] = [n for n, on in (("flash", flash), ("mem_efficient", True),
                                      ("math", bool(allow_math))) if on]
    if verbose:
        print(f"[info] attention: device={out['device']} backends={out['enabled']} "
              f"repeat_kv_forced={forced}")
        if forced:
            print("[info] sm<80 has no flash attention, so the GQA flag is unusable; "
                  "expanding K/V lets the memory-efficient kernel run (measured: "
                  "0.76 GiB at 64k vs 7.24 GiB for math at 8k)")
    return out


def attention_budget_gib(num_heads: int, length: int, dtype_bytes: int = 2) -> float:
    """Peak bytes for ONE materialised attention matrix, in GiB.

    Only the math backend materialises it, but that is exactly the fallback when
    nothing better is available.
    """
    return num_heads * length * length * dtype_bytes / 2 ** 30


def check_attention_budget(model, lengths: list[int], allow_math: bool = True,
                           verbose: bool = True) -> None:
    """Refuse lengths that cannot fit, naming the reason and the way out.

    An OOM deep inside SDPA says "Tried to allocate 16.00 GiB" and mentions neither
    the kernel nor the flag, so say the arithmetic up front.
    """
    import torch

    if not torch.cuda.is_available():
        return
    heads = getattr(getattr(model, "config", None), "num_attention_heads", None)
    if not heads or not lengths:
        return

    flash = torch.cuda.get_device_capability()[0] >= 8
    if flash or _PATCHED:
        return  # a real kernel is in use; lengths are limited by KV cache, not L^2

    worst_len = max(lengths)
    worst = attention_budget_gib(heads, worst_len)
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    safe_len = int((0.25 * total * 2 ** 30 / (heads * 2)) ** 0.5)
    message = (
        f"no memory-efficient attention kernel is active and {worst_len} tokens would need "
        f"~{worst:.0f} GiB for a single attention matrix (device total {total:.1f} GiB). "
        f"Lengths that fit with the math backend: up to about {safe_len} tokens.\n"
        "  Fix: make the memory-efficient kernel usable (it refuses enable_gqa on sm<80; "
        "attention_setup.force_repeat_kv_in_sdpa()), or use --backend vllm, or shorten --lengths."
    )
    if allow_math:
        if verbose:
            print(f"[warn] {message}")
        return
    raise SystemExit("Refusing to run: " + message)

def bytes_per_token_activations(config) -> int:
    """Rough peak activation cost of one token through a transformer layer.

    The dominant terms are the MLP's gate/up projections and their product, each
    [tokens, intermediate_size] in fp16. For Qwen2.5-1.5B (intermediate 8960) that
    is ~54 KB per token, so a single 131k prefill allocates ~7 GiB of MLP buffers
    before the KV cache and weights - which is exactly how the 128k smoke run died
    inside down_proj rather than inside attention.
    """
    inter = (getattr(config, "intermediate_size", None)
             or getattr(config, "moe_intermediate_size", None) or 0)
    hidden = getattr(config, "hidden_size", None) or 0
    if not inter or not hidden:
        return 0
    return int(3 * inter * 2 + 2 * hidden * 2)


def auto_prefill_chunk_size(config, budget_gib: float = 2.5) -> int | None:
    """A chunk size that keeps per-layer activations inside `budget_gib`.

    transformers v5 chunks the prefill when `generation_config.prefill_chunk_size`
    is set, which bounds every intermediate tensor to `chunk` rows instead of the
    whole context. Returns None when the config does not expose the sizes needed.
    """
    per_token = bytes_per_token_activations(config)
    if not per_token:
        return None
    chunk = int(budget_gib * 2 ** 30 / per_token)
    # Round down to a power of two, and never below a useful minimum.
    power = 1 << max(9, chunk.bit_length() - 1)
    return max(512, min(power, 32768))


def configure_prefill_chunking(model, requested: int | None, lengths: list[int],
                               verbose: bool = True) -> int | None:
    """Bound prefill activations by chunking long contexts.

    Attention was only half the 128k problem. The other half: a single unchunked
    prefill allocates every intermediate at full length, so for Qwen2.5-1.5B the MLP
    buffers alone are ~7 GiB and the run dies inside down_proj - after attention has
    already been fixed, which is what made that failure look like a new problem.

    transformers v5 chunks prefill when `generation_config.prefill_chunk_size` is
    set. Verified on real 5.16.1 with Qwen2.5-0.5B: identical output chunked and
    unchunked, so this is a memory change, not a numerics change.
    """
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is None:
        return None
    per_token = bytes_per_token_activations(model.config)

    if requested:
        gen_cfg.prefill_chunk_size = int(requested)
        if verbose:
            print(f"[info] prefill_chunk_size={requested} (explicit)")
        return int(requested)

    longest = max(lengths) if lengths else 0
    chunk = auto_prefill_chunk_size(model.config)
    if chunk is None or not longest or chunk >= longest:
        return None
    gen_cfg.prefill_chunk_size = chunk
    if verbose:
        print(f"[info] prefill_chunk_size={chunk} (auto): longest case {longest} tokens would "
              f"need ~{longest * per_token / 2 ** 30:.1f} GiB of activations unchunked, "
              f"~{chunk * per_token / 2 ** 30:.2f} GiB chunked")
    return chunk
