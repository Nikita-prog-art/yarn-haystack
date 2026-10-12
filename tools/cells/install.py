# Notebook cell fragment, read as text by tools/build_notebook.py.
# It is NOT a module: importing it would execute against names that only
# exist inside a notebook session.

# 2) Install only what is missing. Kaggle ships a CUDA-matched torch, and
#    upgrading matplotlib/datasets provokes conflicts with preinstalled tools.
import importlib.metadata as md, subprocess, sys

def version(p):
    try:
        return md.version(p)
    except md.PackageNotFoundError:
        return None

def too_old(v, floor):
    return v is None or tuple(int(x) for x in v.split(".")[:2]) < floor

to_install = []
if too_old(version("transformers"), (4, 51)):
    to_install.append("transformers>=4.51")
for p in ("accelerate", "datasets", "bitsandbytes"):
    if version(p) is None:
        to_install.append(p)
if to_install:
    print("pip install:", " ".join(to_install), flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *to_install], check=True)

for p in ("transformers", "torch", "accelerate", "datasets", "bitsandbytes", "matplotlib"):
    print(f"  {p:14s} {version(p)}")

import torch
print(f"\ntorch {torch.__version__} | cuda {torch.cuda.is_available()} | gpus {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    pr = torch.cuda.get_device_properties(i)
    print(f"  gpu{i}: {pr.name}, {pr.total_memory / 2**30:.1f} GiB")
assert torch.cuda.is_available(), "no CUDA: this kernel was pushed without a GPU"
_v = version("transformers")
assert _v and tuple(int(x) for x in _v.split(".")[:2]) >= (4, 51), f"need transformers>=4.51, have {_v}"
