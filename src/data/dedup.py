"""Exact and near-duplicate removal (MinHash LSH over character shingles)."""
from __future__ import annotations
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar
from datasketch import MinHash, MinHashLSH
from kitsune.schema import normalize_text
T = TypeVar("T")

def shingles(text: str, k: int = 5) -> set[str]:
    """Character k-shingles of the whitespace-free, NFKC-normalized text."""
    s = "".join(normalize_text(text).split())
    if len(s) <= k:
        return {s} if s else set()
    return {s[i : i + k] for i in range(len(s) - k + 1)}

def jaccard(a: set[str], b: set[str]) -> float:
    """Exact Jaccard similarity of two sets (1.0 for two empty sets)."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)

def minhash(text: str, k: int = 5, num_perm: int = 128, seed: int = 1) -> MinHash:
    raise NotImplementedError

@dataclass(frozen=True)
class DedupReport:
    kept: int
    exact_dupes: int
    near_dupes: int
