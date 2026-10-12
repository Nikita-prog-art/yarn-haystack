"""
NIAH (needle-in-a-haystack) harness: measure the effect of YaRN vs native RoPE.

Design principle: the ONLY thing that differs between arms is `rope_scaling`.
Same weights, same tokenizer, same haystack, same needle position, same decode
settings. That is what makes the measurement an actual ablation of YaRN.

Arms
----
  base    : model config as shipped (Qwen2.5 => native RoPE, 32k, rope_theta=1e6, no scaling)
  yarn    : {"type": "yarn", "factor": F, "original_max_position_embeddings": OMAX}
  linear  : {"type": "linear", ...}          (reference / cheap baseline)
  dynamic : {"type": "dynamic", ...}         (NTK-by-parts; reference)

At lengths <= OMAX every arm sees essentially the same positions, so a difference
there is an artifact. The interesting region is OMAX < L <= OMAX*F: `base` is
pure extrapolation, `yarn` is the interpolation/attention-temperature scheme.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from typing import Any

import attention_setup  # sibling module; torch is imported lazily inside it

# --------------------------------------------------------------------------
# Needles. Each needle must contain a unique, non-guessable answer string so a
# "hit" cannot come from world knowledge or from a neighbouring needle.
# --------------------------------------------------------------------------
NEEDLE_POOL: list[dict[str, str]] = [
    {
        "key": "san_francisco",
        "text": "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.",
        "answer": "eat a sandwich",
        "question": "What is the best thing to do in San Francisco?",
    },
    {
        "key": "berlin",
        "text": "The special magic number for the Berlin facility is 7391.",
        "answer": "7391",
        "question": "What is the special magic number for the Berlin facility?",
    },
    {
        "key": "tokyo",
        "text": "The secret passphrase stored in the Tokyo vault is blue-orchid-42.",
        "answer": "blue-orchid-42",
        "question": "What is the secret passphrase stored in the Tokyo vault?",
    },
    {
        "key": "lisbon",
        "text": "The launch code for the Lisbon project is 5566.",
        "answer": "5566",
        "question": "What is the launch code for the Lisbon project?",
    },
    {
        "key": "oslo",
        "text": "The name of the Oslo project lead is Marta Lindqvist.",
        "answer": "Marta Lindqvist",
        "question": "Who leads the Oslo project?",
    },
]

SYSTEM_PROMPT = "You are a helpful assistant that answers questions based on the provided context."

# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------


def load_corpus(path: str | None, cache_dir: str, min_chars: int = 4_000_000) -> str:
    """Return a large plain-text haystack source.

    Order of preference:
      1. explicit --corpus file
      2. cached copy in cache_dir
      3. HuggingFace `wikitext` (wikitext-103-raw-v1) — needs network + `datasets`
    """
    if path:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read()

    os.makedirs(cache_dir, exist_ok=True)
    cached = os.path.join(cache_dir, "haystack_corpus.txt")
    if os.path.exists(cached) and os.path.getsize(cached) >= min_chars:
        with open(cached, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read()

    errors: list[str] = []
    for repo, name, split in (
        ("Salesforce/wikitext", "wikitext-103-raw-v1", "train"),
        ("wikitext", "wikitext-103-raw-v1", "train"),
        ("Salesforce/wikitext", "wikitext-103-v1", "train"),
    ):
        try:
            from datasets import load_dataset

            ds = load_dataset(repo, name, split=split)
            chunks, total = [], 0
            for row in ds:
                t = row["text"]
                if not t.strip():
                    continue
                chunks.append(t)
                total += len(t)
                if total >= min_chars:
                    break
            text = "\n".join(chunks)
            with open(cached, "w", encoding="utf-8") as fh:
                fh.write(text)
            return text
        except Exception as exc:  # noqa: BLE001 - we want the full list of failures
            errors.append(f"{repo}/{name}:{split} -> {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "Could not load a corpus. Pass --corpus /path/to/text.txt, or enable "
        "internet and `pip install datasets`. Tried:\n  " + "\n  ".join(errors)
    )


# --------------------------------------------------------------------------
# Case construction
# --------------------------------------------------------------------------


@dataclass
class Case:
    arm: str
    model: str
    length: int  # target context length in tokens
    depth: float  # 0.0 = very start, 1.0 = very end
    needle_key: str
    question: str
    expected: str
    prompt: str
    n_tokens: int
    haystack_sha: str


# Qwen3 emits these around its reasoning block. Answer text lives AFTER the close.
_THINK_CLOSE = ("</think>", "</think>", " Response:", " Response\n")


def strip_thinking(text: str) -> tuple[str, bool]:
    """Return (text_after_last_thinking_block, had_block).

    Scoring the raw text would "succeed" on an answer the model only considered
    internally, so we score the final answer. Kept deliberately dumb: if no close
    marker is present we score the whole string.
    """
    had = False
    for marker in _THINK_CLOSE:
        if marker in text:
            had = True
            text = text.rsplit(marker, 1)[-1]
    return text.strip(), had


def _chat_prompt(tokenizer, haystack: str, question: str, style: str, enable_thinking: bool = False) -> str:
    if style == "chat" and getattr(tokenizer, "chat_template", None):
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{haystack}\n\n{question}"},
        ]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if not enable_thinking:
            # Qwen3/Qwen3.5 think by default. The model card explicitly warns that
            # greedy decoding in thinking mode degrades into repetition, and a NIAH
            # answer buried in a reasoning block is not the thing we want to score.
            try:
                return tokenizer.apply_chat_template(msgs, enable_thinking=False, **kwargs)
            except TypeError:
                pass  # tokenizer has no thinking switch; plain chat template is fine
        return tokenizer.apply_chat_template(msgs, **kwargs)
    return f"{haystack}\n\nQuestion: {question}\nAnswer:"


def build_single_case(
    tokenizer,
    corpus_ids: list[int],
    arm: str,
    model_name: str,
    length: int,
    depth: float,
    needle: dict[str, str],
    style: str,
    overhead: int,
    verify_roundtrip: bool = False,
    enable_thinking: bool = False,
) -> Case:
    """Insert one needle at `depth` of a haystack that tokenizes to ~`length`."""
    needle_ids = tokenizer(needle["text"], add_special_tokens=False)["input_ids"]
    budget = max(length - overhead - len(needle_ids), 16)
    if budget >= len(corpus_ids):
        raise ValueError(f"corpus too small for length={length} (need {budget} tokens)")

    # Position the needle on a token boundary, then splice at text level.
    # Tokenizing the result is *approximately* length tokens; we report the exact
    # count in the case so analysis never has to guess.
    split_at = int(round(depth * (budget - len(needle_ids))))
    split_at = max(0, min(split_at, budget - len(needle_ids)))
    head = tokenizer.decode(corpus_ids[:split_at])
    tail = tokenizer.decode(corpus_ids[split_at:budget])
    haystack = f"{head}\n\n{needle['text']}\n\n{tail}"

    prompt = _chat_prompt(tokenizer, haystack, needle["question"], style, enable_thinking)
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if verify_roundtrip:
        rt = tokenizer.decode(ids[:200])  # cheap sanity check only
        assert rt.strip(), "tokenizer decode/encode roundtrip produced empty text"

    return Case(
        arm=arm,
        model=model_name,
        length=length,
        depth=depth,
        needle_key=needle["key"],
        question=needle["question"],
        expected=needle["answer"],
        prompt=prompt,
        n_tokens=len(ids),
        haystack_sha=hashlib.sha256(haystack.encode("utf-8")).hexdigest()[:16],
    )


def build_multi_case(
    tokenizer,
    corpus_ids: list[int],
    arm: str,
    model_name: str,
    length: int,
    depths: list[float],
    needles: list[dict[str, str]],
    style: str,
    overhead: int,
    enable_thinking: bool = False,
) -> Case:
    """All needles in ONE haystack, ONE question asking for all of them.

    Rationale: prefill dominates the cost at long context. This turns 5 prefills
    into 1. Report it as a secondary metric (recall of the needle set), because it
    conflates retrieval with instruction-following.
    """
    n = min(len(depths), len(needles))
    depths, needles = depths[:n], needles[:n]
    needle_tokens = sum(len(tokenizer(nd["text"], add_special_tokens=False)["input_ids"]) for nd in needles)
    budget = max(length - overhead - needle_tokens, 16)

    body = tokenizer.decode(corpus_ids[:budget])
    # Insert from the END so earlier insertions do not shift later offsets.
    order = sorted(range(n), key=lambda i: depths[i], reverse=True)
    for i in order:
        pos = int(round(depths[i] * len(body)))
        pos = max(0, min(pos, len(body)))
        body = f"{body[:pos]}\n\n{needles[i]['text']}\n\n{body[pos:]}"

    listing = " ".join(f"({i + 1}) {nd['question']}" for i, nd in enumerate(needles))
    question = (
        "I hid several facts inside the text above. Answer each of the following, "
        f"in the same order, one line per answer:\n{listing}"
    )
    prompt = _chat_prompt(tokenizer, body, question, style, enable_thinking)
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    return Case(
        arm=arm,
        model=model_name,
        length=length,
        depth=-1.0,
        needle_key="+".join(nd["key"] for nd in needles),
        question=question,
        expected=" | ".join(nd["answer"] for nd in needles),
        prompt=prompt,
        n_tokens=len(ids),
        haystack_sha=hashlib.sha256(body.encode("utf-8")).hexdigest()[:16],
    )


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

_PUNCT = re.compile(r"[^\w\s\-.]", re.UNICODE)


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = _PUNCT.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .")


def score_single(prediction: str, expected: str) -> dict[str, Any]:
    p, e = normalize(prediction), normalize(expected)
    contains = e in p
    # For answers that are single tokens/numbers, contains is too lenient
    # (e.g. "7391" inside "173910"); require a token-boundary match too.
    word_hit = bool(re.search(rf"(?<!\w){re.escape(e)}(?!\w)", p))
    return {"correct_contains": contains, "correct_word": word_hit}


def score_multi(prediction: str, expected: str) -> dict[str, Any]:
    p = normalize(prediction)
    answers = [normalize(a) for a in expected.split("|")]
    hits = [bool(re.search(rf"(?<!\w){re.escape(a)}(?!\w)", p)) for a in answers]
    return {
        "correct_word": all(hits),
        "recall": sum(hits) / max(len(hits), 1),
        "n_hits": sum(hits),
        "n_needles": len(hits),
    }


# --------------------------------------------------------------------------
# Model loading (the part where YaRN is actually switched on)
# --------------------------------------------------------------------------


def rope_scaling_for(
    arm: str,
    factor: float,
    omax: int,
    attn_factor: float | None,
    beta_fast: float,
    beta_slow: float,
    existing: dict | None = None,
) -> dict | None:
    """Build the RoPE scaling entry for `arm`, or None for the untouched default.

    Two transformers generations have to be satisfied, and the difference is not
    cosmetic:

    * v4 keeps `rope_theta` as its own config attribute and `rope_scaling` as a
      sibling dict. A `rope_theta` key inside `rope_scaling` only earns an
      "unrecognized keys" warning there, so it is omitted when `existing` is None.
    * v5 merged both into one `rope_parameters` dict (`rope_scaling` is now just a
      property alias over it) and its scaling functions read `rope_theta` from
      INSIDE that dict. So the entry must be merged into the existing dict, not
      replace it: replacing drops `rope_theta`, and the model then dies inside
      Qwen2RotaryEmbedding with a KeyError that names neither RoPE nor the config.

    `rope_type` is used for the discriminator because v4.45+ and v5 both accept it,
    whereas v5 dropped the old `type` spelling.
    """
    if arm == "base":
        return None
    entry: dict[str, Any] = {
        "rope_type": arm,
        "factor": factor,
        "original_max_position_embeddings": omax,
    }
    if arm == "yarn":
        if attn_factor is not None:
            entry["attention_factor"] = attn_factor
        entry["beta_fast"] = beta_fast
        entry["beta_slow"] = beta_slow
    if existing:
        merged = dict(existing)
        merged.update(entry)
        return merged
    return entry


def apply_rope_scaling(cfg, rs: dict | None) -> str:
    """Install `rs` on the config in the shape this transformers version expects.

    Returns "v5" or "v4". Factored out and unit-tested because this is exactly
    where the rename bit us: assigning None to `rope_scaling` on v5 clears
    `rope_parameters`, which also carries `rope_theta`.
    """
    v5 = hasattr(cfg, "rope_parameters")
    if rs is None:
        if not v5:
            cfg.rope_scaling = None
    elif v5:
        cfg.rope_parameters = rs
    else:
        cfg.rope_scaling = rs
    return "v5" if v5 else "v4"


def sanity_check(args, cfg) -> None:
    """Refuse to waste a GPU session on a configuration that cannot answer the question.

    These are the failure modes that produce plausible-looking numbers which mean
    nothing — the expensive kind of wrong.
    """
    native = int(getattr(cfg, "max_position_embeddings", 0) or 0)
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    longest = max(lengths) if lengths else 0
    effective = int(args.original_max * args.factor)
    problems: list[str] = []

    # 1. Some checkpoints are already long-context native, with the scaling baked
    #    into pretraining. Inside their window there is no extrapolation to fix, so
    #    the ablation has no contrast to measure.
    NATIVE_LONG = ("Qwen3-4B-Instruct-2507", "Qwen3-30B-A3B-Instruct-2507", "Qwen3.5-", "Qwen3-Next-", "Qwen3-Coder-")
    if native >= longest and any(tag in args.model for tag in NATIVE_LONG):
        problems.append(
            f"{args.model} is natively {native} tokens and long-context trained. Longest test length is "
            f"{longest}, i.e. INSIDE the trained window — `base` has no extrapolation failure for YaRN to fix. "
            "Use the original Qwen3/Qwen2.5 sizes (native 32k) instead."
        )

    # 2. Positions past the scaled range are undefined; YaRN does not invent them.
    if args.arm != "base" and longest > effective:
        problems.append(f"longest test length {longest} exceeds the scaled range original_max*factor = {effective}.")

    # 3. `original_max_position_embeddings` must be the window the model was TRAINED
    #    on, which is not always `config.max_position_embeddings`. Qwen3 configs say
    #    40960, but the card says "32,768 natively" — the extra 8192 is output space
    #    reserved out of the same budget. Copying 40960 out of config.json yields a
    #    silently wrong inv_freq. Equality is the trap here, so `>` alone misses it.
    KNOWN_PROMPT_WINDOW = (("Qwen3-", 32768), ("Qwen2.5-", 32768))
    documented = next((w for tag, w in KNOWN_PROMPT_WINDOW if tag in args.model), None)
    if args.arm != "base" and documented is not None and args.original_max != documented:
        problems.append(
            f"--original-max {args.original_max} does not match the documented prompt window {documented} for "
            f"{args.model}. config.json's max_position_embeddings ({native}) reserves space for output tokens, so "
            "using it here builds inv_freq over positions the model was never trained to attend."
        )
    elif args.arm != "base" and native and args.original_max > native:
        problems.append(
            f"--original-max {args.original_max} > config max_position_embeddings {native}. Pass the window the "
            "model was TRAINED on, which is not always the config value."
        )

    # 4. No contrast: everything inside the native window, so YaRN is a no-op.
    if args.arm == "base" and native and longest <= native:
        problems.append(
            f"arm=base with longest length {longest} <= native window {native}: no extrapolation is exercised, "
            "so there is nothing for the YaRN arm to differ from."
        )

    print(f"[info] native window={native}  original_max={args.original_max}  factor={args.factor}  "
          f"-> scaled range={effective}  test lengths={lengths}")
    if args.arm != "base" and args.original_max != native:
        print(f"[note] original_max ({args.original_max}) != config max_position_embeddings ({native}). Expected for "
              "Qwen3 (config reserves 40960 = 32768 prompt + 8192 output, and the official YaRN recipe uses 32768) — "
              "but it is exactly the trap worth double-checking.")
    for p in problems:
        print(f"[WARN] {p}")
    if problems and not getattr(args, "force", False):
        raise SystemExit(
            "Refusing to run: this configuration cannot measure what you think it measures. "
            "Fix the arguments, or pass --force if you really mean it."
        )


def tf_version() -> str:
    try:
        import transformers
        return transformers.__version__
    except Exception:  # noqa: BLE001
        return "?"


def resolve_placement(load_in_4bit: bool, device_map: str | None, dtype_name: str, cuda_available: bool) -> tuple[str | None, str, str | None]:
    """Decide where the model runs and in what precision.

    Returns (device_to_move_to_or_None, effective_dtype_name, note).

    Passing `device_map` (or 4-bit, which implies `device_map="auto"`) hands
    placement to accelerate, so we must not move the model ourselves.

    Without either, `from_pretrained` silently leaves the weights on CPU. With a
    half precision dtype that dies in CPU kernels (`LayerNormKernelImpl not
    implemented for 'Half'`), which is an obscure way to find out you forgot to
    move the model. CPU also cannot run this benchmark at any useful speed, so
    fall back to fp32 and say so.
    """
    if load_in_4bit or device_map:
        return None, dtype_name, None
    if not cuda_available:
        if dtype_name in ("float16", "bfloat16"):
            return "cpu", "float32", (
                f"no CUDA available: using float32 on CPU instead of {dtype_name}. "
                "This is a correctness check, not a timing or memory measurement."
            )
        return "cpu", dtype_name, "no CUDA available: running on CPU."
    return "cuda", dtype_name, None


def load_model_and_tokenizer(args):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    # Fail before the multi-GB weight download, not after.
    sanity_check(args, cfg)

    # CRITICAL: the scaling must be on the config BEFORE the model is built,
    # otherwise the rotary embedding is constructed with the old inv_freq and the
    # flag is a silent no-op.
    existing = getattr(cfg, "rope_parameters", None) if hasattr(cfg, "rope_parameters") else None
    rs = rope_scaling_for(args.arm, args.factor, args.original_max, args.attention_factor,
                          args.beta_fast, args.beta_slow, existing)
    fam = apply_rope_scaling(cfg, rs)
    if getattr(cfg, "use_sliding_window", False) and getattr(cfg, "sliding_window", None):
        # A sliding window would silently truncate attention and make a long-context
        # comparison meaningless.
        print(f"[warn] disabling sliding_window={cfg.sliding_window} for a fair NIAH comparison")
        cfg.use_sliding_window = False

    device, dtype_name, note = resolve_placement(
        args.load_in_4bit, args.device_map, args.dtype, torch.cuda.is_available()
    )
    if note:
        print(f"[warn] {note}")

    kwargs: dict[str, Any] = {"config": cfg, "torch_dtype": getattr(torch, dtype_name)}
    if args.attn:
        kwargs["attn_implementation"] = args.attn
    if args.load_in_4bit:
        if not torch.cuda.is_available():
            raise SystemExit("--load-in-4bit needs a CUDA device; bitsandbytes has no CPU fallback here.")
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    elif args.device_map:
        kwargs["device_map"] = args.device_map

    if args.trust_remote_code:
        kwargs["trust_remote_code"] = True

    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.eval()
    if device is not None:
        # Without this the model stays on CPU while we feed it CUDA-shaped hopes.
        model = model.to(device)
    where = device if device is not None else f"device_map={args.device_map or 'auto'}"
    print(f"[info] device={where} dtype={dtype_name}")
    # On sm<80 the GQA flag forces the math backend, which OOMs at long context;
    # this picks usable kernels and, where needed, expands K/V instead. See
    # attention_setup's docstring for the measured numbers.
    attention_setup.configure_attention(allow_math=True, verbose=True)

    applied = getattr(model.config, "rope_scaling", None)
    print(f"[info] transformers={tf_version()} ({fam}) arm={args.arm} rope={applied} "
          f"max_position_embeddings={model.config.max_position_embeddings}")
    if rs is not None:
        # Compare the discriminator, not the whole dict: v5 may normalise keys.
        got = (applied or {}).get("rope_type") or (applied or {}).get("type")
        if got != args.arm:
            raise RuntimeError(
                f"rope scaling was not applied: requested rope_type={args.arm!r}, model reports "
                f"{got!r} (full entry: {applied}). Refusing to run - this would silently measure "
                "the wrong thing."
            )
    return model, tok


def load_vllm(args):
    """vLLM backend: batched prefill + automatic prefix caching.

    Prefix caching matters here: all depths for one length share a long haystack
    prefix, but a *different question*, so the shared prefix is reused.
    """
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    sanity_check(args, AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code))
    rs = rope_scaling_for(args.arm, args.factor, args.original_max, args.attention_factor, args.beta_fast, args.beta_slow)
    kw: dict[str, Any] = dict(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        trust_remote_code=args.trust_remote_code,
    )
    if args.load_in_4bit:
        kw["quantization"] = "bitsandbytes"
        kw["load_format"] = "bitsandbytes"
    try:
        llm = LLM(rope_scaling=rs, **kw)
    except TypeError:
        llm = LLM(hf_overrides=({"rope_scaling": rs} if rs else {}), **kw)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    return llm, tok, sp


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def generate_hf(model, tok, prompts: list[str], max_new_tokens: int, batch: int) -> list[str]:
    import torch

    outs: list[str] = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i : i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        n_in = int(enc["input_ids"].shape[1])
        with torch.inference_mode():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tok.pad_token_id,
            )
        for row in gen:
            outs.append(tok.decode(row[n_in:], skip_special_tokens=True))
    return outs


def generate_vllm(llm, sp, prompts: list[str]) -> list[str]:
    reqs = [{"prompt": p} for p in prompts]
    res = llm.generate(reqs, sp, use_tqdm=False)
    return [r.outputs[0].text for r in res]


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def completed_keys(path: str) -> set[tuple]:
    keys: set[tuple] = set()
    if not os.path.exists(path):
        return keys
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated final line from a killed session
            keys.add((row["arm"], row["length"], row["depth"], row["needle_key"], row.get("mode", "single")))
    return keys


def write_row(path: str, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())  # survive a Colab/Kaggle kill


def run(args) -> None:
    random.seed(args.seed)
    corpus = load_corpus(args.corpus, args.cache_dir)

    if args.backend == "vllm":
        llm, tok, sp = load_vllm(args)
    else:
        model, tok = load_model_and_tokenizer(args)

    corpus_ids = tok(corpus, add_special_tokens=False)["input_ids"]
    print(f"[info] corpus: {len(corpus_ids)} tokens, {len(corpus)} chars")

    et = bool(getattr(args, "enable_thinking", False))
    overhead = len(tok(_chat_prompt(tok, "", "x", args.prompt_style, et), add_special_tokens=False)["input_ids"])
    done = completed_keys(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    depths = [float(x) for x in args.depths.split(",") if x.strip()]
    needles = NEEDLE_POOL[: args.n_needles]

    if args.backend != "vllm":
        # Refuse (or loudly warn about) lengths that cannot fit, before downloading
        # anything or burning a GPU session on a guaranteed OOM.
        attention_setup.check_attention_budget(
            model, lengths, allow_math=bool(getattr(args, "allow_math_sdpa", False))
        )
        attention_setup.configure_prefill_chunking(
            model, getattr(args, "prefill_chunk_size", None), lengths
        )

    for length in lengths:
        cases: list[Case] = []
        if args.mode == "single":
            for d in depths:
                for nd in needles[: args.needles_per_case]:
                    cases.append(
                        build_single_case(tok, corpus_ids, args.arm, args.model, length, d, nd,
                                          args.prompt_style, overhead, enable_thinking=et)
                    )
        else:
            cases.append(build_multi_case(tok, corpus_ids, args.arm, args.model, length, depths, needles,
                                          args.prompt_style, overhead, enable_thinking=et))

        todo = [c for c in cases if (c.arm, c.length, c.depth, c.needle_key, args.mode) not in done]
        if not todo:
            print(f"[skip] L={length}: already complete")
            continue
        print(f"[run ] L={length}: {len(todo)}/{len(cases)} cases, prompt_tokens~{todo[0].n_tokens}")

        t0 = time.time()
        if args.backend == "vllm":
            preds = generate_vllm(llm, sp, [c.prompt for c in todo])
        else:
            preds = generate_hf(model, tok, [c.prompt for c in todo], args.max_new_tokens, args.hf_batch)
        dt = time.time() - t0

        for case, pred in zip(todo, preds):
            answer, had_think = strip_thinking(pred)
            sc = score_single(answer, case.expected) if args.mode == "single" else score_multi(answer, case.expected)
            row = asdict(case)
            row.pop("prompt")
            row.update(sc)
            row.update(
                mode=args.mode,
                prediction=pred,
                answer=answer,
                had_think_block=had_think,
                enable_thinking=et,
                seed=args.seed,
                max_new_tokens=args.max_new_tokens,
                factor=args.factor,
                original_max=args.original_max,
                dtype=args.dtype,
                load_in_4bit=bool(args.load_in_4bit),
                backend=args.backend,
                attn=args.attn,
                elapsed_s_total=round(dt, 2),
                ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            write_row(args.out, row)
            flag = "OK " if sc.get("correct_word") else "MISS"
            warn = " [THINK]" if had_think else ""
            print(f"    [{flag}] L={case.length:>6} d={case.depth:<4} {case.needle_key:<14} "
                  f"exp={case.expected[:28]!r} pred={answer[:70]!r}{warn}")
        if args.upload:
            _upload(args.out, args.upload)
    print(f"[done] results -> {args.out}")


def _upload(local: str, remote: str) -> None:
    """Push the JSONL to an HF dataset repo (works from Kaggle/Colab alike)."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_file(path_or_fileobj=local, path_in_repo=os.path.basename(local), repo_id=remote, repo_type="dataset")
        print(f"[sync] uploaded {local} -> {remote}")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] upload failed (results are still on local disk): {exc}")
