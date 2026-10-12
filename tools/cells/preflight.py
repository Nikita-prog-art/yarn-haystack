# Notebook cell fragment, read as text by tools/build_notebook.py.
# It is NOT a module: importing it would execute against names that only
# exist inside a notebook session.

# 3) Preflight: internet (the usual cause of a one-second death) and the corpus.
import os, urllib.request
import niah

try:
    print("huggingface.co ->", urllib.request.urlopen("https://huggingface.co", timeout=20).status)
except Exception as exc:
    raise SystemExit(f"no internet in this kernel: {exc!r} (Kaggle needs phone verification for that)")

text = niah.load_corpus(None, os.path.join(os.getcwd(), "cache"))
print(f"corpus OK: {len(text)/1e6:.1f}M chars cached")
