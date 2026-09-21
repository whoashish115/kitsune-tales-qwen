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


def _windows(text: str, n: int = N) -> list[int]:
    s = "".join(text.split())
    return [_hash(s[i : i + n]) for i in range(len(s) - n + 1)]


class NgramIndex:
    """Sorted array of hashed n-grams from a corpus, queried with ``np.isin``."""

    def __init__(self, texts: Iterable[str], n: int = N) -> None:
        self.n = n
        hs: list[int] = []
        for t in texts:
            hs.extend(_windows(t, n))
        self.arr = np.unique(np.asarray(hs, dtype=np.int64))

    def audit(self, text: str) -> dict[str, float]:
        w = np.asarray(_windows(text, self.n), dtype=np.int64)
        if len(w) == 0:
            return {"overlap_rate": 0.0, "max_span": 0.0}
        # self.arr is sorted and unique: binary search instead of np.isin (which re-sorts ~10M hashes per call).
        pos = np.minimum(np.searchsorted(self.arr, w), len(self.arr) - 1)
        hit = self.arr[pos] == w
        best = run = 0
        for h in hit:
            run = run + 1 if h else 0
            best = max(best, run)
        return {"overlap_rate": float(hit.mean()), "max_span": float(best + self.n - 1 if best else 0)}


def run(train_path: str, gen_dir: str, systems: list[str]) -> dict[str, dict[str, float]]:
    """Audit every system's seed-0 test outputs against the training responses."""
    import gzip
    import json

    from kitsune.schema import read_jsonl

    idx = NgramIndex(r["response"] for r in read_jsonl(train_path))
    out: dict[str, dict[str, float]] = {}
    for s in systems:
        with gzip.open(f"{gen_dir}/{s}.jsonl.gz", "rt", encoding="utf-8") as f:
            rows = [json.loads(x) for x in f if x.strip()]
        a = [idx.audit(r["text"]) for r in rows if r["suite"] == "test" and r["seed"] == 0]
        spans = np.array([x["max_span"] for x in a])
        out[s] = {
            "n": len(a),
            "overlap_rate_mean": float(np.mean([x["overlap_rate"] for x in a])),
            "max_span_median": float(np.median(spans)),
            "max_span_p99": float(np.percentile(spans, 99)),
            "max_span_max": float(spans.max()),
            "share_span_ge_100": float((spans >= 100).mean()),
        }
    return out


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
