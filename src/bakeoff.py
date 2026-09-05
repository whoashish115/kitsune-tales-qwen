"""Summaries for the D-001 base-model bake-off (zero-shot; prompts disjoint from the test set)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from kitsune.eval.metrics import output_metrics


def summarize(rows: list[dict]) -> dict[str, Any]:
    """Mean rule-based metrics per system, plus tokens/s, from bake-off generation rows."""
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        p = r["prompt"]
        m = output_metrics(r["text"], p["format"], p["genres"], p["title"], p.get("passage")).__dict__
        by[r["system"]].append(
            m | {"n_tokens": r["n_tokens"], "truncated": float(r["finish_reason"] == "length")}
        )
    out: dict[str, Any] = {}
    for sysname, ms in by.items():
        keys = [k for k in ms[0] if isinstance(ms[0][k], int | float)]
        out[sysname] = {k: float(np.nanmean([m[k] for m in ms])) for k in keys} | {"n": len(ms)}
    return out
