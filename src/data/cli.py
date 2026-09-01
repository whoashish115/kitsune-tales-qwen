"""Data CLI: freeze the held-out prompt sets.
python -m kitsune.data.cli freeze-test        # writes + hashes data/test_prompts.jsonl, data/eval_policy_prompts.jsonl
python -m kitsune.data.cli verify-test        # fails if the frozen files changed
python -m kitsune.data.cli freeze-test-en     # English variant (D-024): data/test_prompts_en.jsonl + policy suite
python -m kitsune.data.cli build-test-set-en --raw data/raw/full_en
"""
from __future__ import annotations
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from kitsune.data.policy import eval_policy_prompts
from kitsune.data.seeds import build_test_prompts
from kitsune.schema import file_sha256, write_jsonl
DATA = Path("data")
TEST = DATA / "test_prompts.jsonl"
POLICY = DATA / "eval_policy_prompts.jsonl"
HASHES = DATA / "test_prompts.sha256"

def freeze_test(per_cell: int = 10, seed: int = 20260929, force: bool = False) -> dict[str, str]:
    if HASHES.exists() and not force:
        raise SystemExit(
            f"{HASHES} exists: the test set is frozen. Use --force only to rebuild an identical copy."
        )
    write_jsonl(TEST, [p.to_dict() for p in build_test_prompts(per_cell, seed)])
    write_jsonl(POLICY, [asdict(p) for p in eval_policy_prompts()])
    hashes = {TEST.name: file_sha256(TEST), POLICY.name: file_sha256(POLICY)}
    if force and HASHES.exists():
        old = json.loads(HASHES.read_text(encoding="utf-8"))
        if {k: old.get(k) for k in hashes} != hashes:
            raise SystemExit(f"rebuilt test set differs from the frozen hashes: {old} vs {hashes}")
    HASHES.write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    return hashes

TEST_SET = DATA / "test_set.jsonl"
TEST_SET_HASH = DATA / "test_set.sha256"

def build_test_set(raw_dir: Path) -> dict[str, object]:
    """Attach frozen passages to the 90 held-out 続き prompts and write data/test_set.jsonl (+ SHA-256).

    For each 続き prompt, candidate source stories are tried in order; the first whose passage and
    reference continuation pass the purity/safety/artifact filters is used. The reference continuation is
    stored for inspection only (it is never used for training).
    """
    import random

    from kitsune.data.filters import f_japanese_purity, f_latin_leak, f_no_artifacts, f_safety_rule
    from kitsune.data.generate import split_for_continuation
    from kitsune.schema import read_jsonl

    if TEST_SET_HASH.exists():
        raise SystemExit(f"{TEST_SET_HASH} exists: the full test set is already frozen.")
    verify_test()
    cands: dict[str, list[dict]] = {}
    for f in sorted(raw_dir.glob("test_passages_*.jsonl*")):
        for r in read_jsonl(f):
            pid = r["id"].split(":")[0]
            cands.setdefault(pid, []).append(r)
    rng = random.Random(90)
    out, missing = [], []
    for p in read_jsonl(TEST):
        row = dict(p)
        if p["format"] == "続き":
            chosen = None
            for r in sorted(cands.get(p["id"], []), key=lambda r: r["id"]):
                if r.get("finish_reason") == "length":
                    continue
                split = split_for_continuation(r["text"].strip(), rng)
                checks = (f_japanese_purity, f_safety_rule, f_no_artifacts, f_latin_leak)
                # Only the passage is shown to the evaluated models; the reference continuation is kept
                if split and all(f(split[0]).passed for f in checks):
                    chosen = (split, r, all(f(split[1]).passed for f in checks))
                    break
            if chosen is None:
                missing.append(p["id"])
                continue
            (passage, ref), r, ref_clean = chosen
            row |= {
                "passage": passage,
                "reference_continuation": ref if ref_clean else None,
                "passage_source": r["id"],
                "passage_generator": r["generator"],
            }
        out.append(row)
    if missing:
        raise SystemExit(f"no valid passage for {len(missing)} prompts: {missing[:10]}")
    write_jsonl(TEST_SET, out)
    h = {TEST_SET.name: file_sha256(TEST_SET)}
    TEST_SET_HASH.write_text(json.dumps(h, indent=2) + "\n", encoding="utf-8")
    return {"n": len(out), "n_passages": sum(1 for r in out if r.get("passage")), **h}
