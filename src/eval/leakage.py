"""Train/test leakage and memorization audit.
For each generated output we find the longest span (in characters) that also occurs verbatim
somewhere in the training responses, using hashed character n-grams:
- ``overlap_rate``: share of an output's 32-char windows found in the training set.
- ``max_span``: the longest run of consecutive overlapping windows, converted to characters.
The base model never saw our training data, so its numbers are the *chance* baseline (common
phrases); Kitsune's excess over the base is the memorization signal we report.
"""
from __future__ import annotations
import hashlib
from collections.abc import Iterable
import numpy as np
N = 32

def _hash(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little", signed=True)

if __name__ == "__main__":
    import json
    from pathlib import Path

    ja = run(
        "data/processed/train.jsonl", "reports/generations", ["base", "kitsune-sft", "kitsune", "teacher"]
    )
    Path("reports/leakage.json").write_text(json.dumps(ja, indent=2), encoding="utf-8")
    en = run(
        "data/processed_en/train.jsonl", "reports/generations_en", ["base-en", "kitsune-en-sft", "kitsune-en"]
    )
    Path("reports/leakage_en.json").write_text(json.dumps(en, indent=2), encoding="utf-8")
    print(json.dumps({"ja": ja, "en": en}, indent=1))
