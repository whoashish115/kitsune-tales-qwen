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
    m = MinHash(num_perm=num_perm, seed=seed)
    for sh in shingles(text, k):
        m.update(sh.encode("utf-8"))
    return m


@dataclass(frozen=True)
class DedupReport:
    kept: int
    exact_dupes: int
    near_dupes: int


def dedup(
    items: Sequence[T],
    key: Callable[[T], str],
    threshold: float = 0.7,
    k: int = 5,
    num_perm: int = 128,
    seed: int = 1,
) -> tuple[list[T], DedupReport, dict[int, int]]:
    """Remove exact then near-duplicates, keeping the first occurrence (input order is the priority).

    Near-duplicate candidates from LSH are confirmed with exact shingle Jaccard ≥ ``threshold``
    to avoid false positives from MinHash estimation error.

    Returns:
        (kept items, report, mapping dropped_index -> index of the kept item it duplicates)
    """
    seen_exact: dict[str, int] = {}
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    shingle_cache: dict[int, set[str]] = {}
    kept: list[T] = []
    dup_of: dict[int, int] = {}
    exact = near = 0
    for i, it in enumerate(items):
        text = key(it)
        norm = "".join(normalize_text(text).split())
        if norm in seen_exact:
            dup_of[i] = seen_exact[norm]
            exact += 1
            continue
        sh = shingles(text, k)
        mh = MinHash(num_perm=num_perm, seed=seed)
        for s in sh:
            mh.update(s.encode("utf-8"))
        cands = lsh.query(mh)
        match = next((int(c) for c in cands if jaccard(sh, shingle_cache[int(c)]) >= threshold), None)
        if match is not None:
            dup_of[i] = match
            near += 1
            continue
        seen_exact[norm] = i
        shingle_cache[i] = sh
        lsh.insert(str(i), mh)
        kept.append(it)
    return kept, DedupReport(kept=len(kept), exact_dupes=exact, near_dupes=near), dup_of


class TitleIndex:
    """Precomputed bigram shingles of reference titles, for fast repeated ``is_near`` checks."""

    def __init__(self, titles: Sequence[str], threshold: float = 0.6) -> None:
        self.threshold = threshold
        self.exact = {"".join(normalize_text(t).split()) for t in titles}
        self.shingles = [shingles(t, 2) for t in titles]

    def is_near(self, title: str) -> bool:
        if "".join(normalize_text(title).split()) in self.exact:
            return True
        s = shingles(title, 2)
        return any(jaccard(s, o) >= self.threshold for o in self.shingles)


def title_is_near(title: str, others: Sequence[str], threshold: float = 0.6) -> bool:
    """True if ``title`` exactly matches or is a near-duplicate (char-bigram Jaccard) of any in ``others``.

    Used to keep held-out test titles (and close paraphrases) out of the training data.
    """
    t = shingles(title, 2)
    tn = "".join(normalize_text(title).split())
    for o in others:
        if tn == "".join(normalize_text(o).split()) or jaccard(t, shingles(o, 2)) >= threshold:
            return True
    return False
