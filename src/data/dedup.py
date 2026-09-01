"""Exact and near-duplicate removal (MinHash LSH over character shingles)."""
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar
from __future__ import annotations
from datasketch import MinHash, MinHashLSH
from kitsune.schema import normalize_text
def jaccard(a: set[str], b: set[str]) -> float:
    """Exact Jaccard similarity of two sets (1.0 for two empty sets)."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)
