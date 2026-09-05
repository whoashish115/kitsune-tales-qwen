"""Build the processed dataset: raw generations + labels → filtered, deduplicated, split JSONL.

    python -m kitsune.data.pipeline build --raw data/raw --out data/processed --reports reports/data
Every dropped sample keeps its first failing reason, so ``stats.json`` and the funnel plot
show exactly what was removed and why.
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from kitsune.data.dedup import dedup
from kitsune.data.filters import FilterOutcome, clean_generation, f_title_clean, run_rule_filters
from kitsune.data.generate import split_for_continuation
from kitsune.data.labels import Labels, parse_labels
from kitsune.data.policy import REDIRECT_PREFIX, PolicyPrompt, refusal_text, train_policy_prompts
from kitsune.prompts import StoryRequest, build_user_prompt
from kitsune.schema import Record, file_sha256, make_record, read_jsonl, write_jsonl
from kitsune.taxonomy import GENRES
SYNTHETIC_LICENSE = "apache-2.0 (generated with an Apache-2.0 model; see DATA_CARD.md)"
SEED_LICENSE = "apache-2.0 (written for this project)"

def _candidate_from_generation(g: dict, rng: random.Random, lang: str = "ja") -> Candidate:
    if lang == "en":
        return _candidate_en(g, rng)
    kind = g["kind"]
    meta = {k: g[k] for k in ("sampling", "n_tokens", "finish_reason") if k in g}
    title_for_clean = (g.get("meta", {}).get("seed") or g.get("meta", {}).get("policy") or {}).get(
        "title", ""
    )
    text, cleaned = clean_generation(g["text"], title_for_clean)
    if cleaned:
        meta["cleaned"] = True
    g = {**g, "text": text}
    if "knobs" in g.get("meta", {}):
        meta["knobs"] = g["meta"]["knobs"]
    if kind == "offgenre":
        p = PolicyPrompt(**g["meta"]["policy"])
        c = Candidate(
            g["id"], g["generator"], kind, ["ハイファンタジー"], p.title, p.format, p.user_prompt(),
            REDIRECT_PREFIX + "\n\n" + g["text"].strip(),
            meta=meta | {"policy_kind": "offgenre", "requested_genres": p.genres_text},
        )  # fmt: skip
    else:
        seed = g["meta"]["seed"]
        genres, title = seed["genres"], seed["title"]
        fmt = seed["format"]
        meta |= {"title_source": seed.get("title_source", "template")}
        if kind == "source":
            split = split_for_continuation(g["text"].strip(), rng)
            if split is None:
                c = Candidate(g["id"], g["generator"], kind, genres, title, "続き", "", "", meta=meta)
                c.drop_reason = "split:no_valid_cut"
                return c
            passage, cont = split
            prompt = build_user_prompt(StoryRequest(genres, title, "続き", passage))
            c = Candidate(g["id"], g["generator"], kind, genres, title, "続き", prompt, cont, passage, meta)
        else:
            prompt = build_user_prompt(StoryRequest(genres, title, fmt))
            c = Candidate(
                g["id"], g["generator"], kind, genres, title, fmt, prompt, g["text"].strip(), meta=meta
            )
    c.gen_key = g.get("gen_key") or g["generator"]
    if g.get("finish_reason") == "length":
        c.drop_reason = "generation:truncated"
    return c

def _candidate_en(g: dict, rng: random.Random) -> Candidate:
    raise NotImplementedError

def _pick_labels(
    sample_id: str, generator: str, labels: dict[str, dict[str, Labels | None]]
) -> tuple[Labels | None, str]:
    """Prefer a label from a model other than the sample's generator (cross), else a self-label."""
    by = labels.get(sample_id, {})
    cross = [(lab, who) for who, lab in by.items() if who != generator]
    for lab, who in cross:
        if lab is not None:
            return lab, f"cross:{who}"
    if generator in by and by[generator] is not None:
        return by[generator], f"self:{generator}"
    return None, "none"

def refusal_records(policy: list[PolicyPrompt], variants: int = 3, seed: int = 5) -> list[Record]:
    """Templated refusals for disallowed training prompts, with genre/format variants."""
    rng = random.Random(seed)
    out: list[Record] = []
    fantasy_texts = ["異世界転生", "魔王と勇者", "冒険者ギルド", "魔法学園", "魔法少女", "ダークファンタジー"]
    for p in policy:
        if not p.expects_refusal:
            continue
        for v in range(variants):
            q = PolicyPrompt(
                p.id + f"-v{v}",
                p.kind,
                rng.choice(fantasy_texts),
                p.title,
                rng.choice(["あらすじ", "短編", "短編"]),
            )
            out.append(
                make_record(
                    id=q.id,
                    genres=["ハイファンタジー"],
                    title=q.title,
                    format=q.format,
                    prompt=q.user_prompt(),
                    response=refusal_text(p.kind),
                    source="seed",
                    generator="template:policy.refusal_text",
                    license=SEED_LICENSE,
                    filters_passed=["template"],
                    meta={"policy_kind": p.kind, "requested_genres": q.genres_text},
                )
            )
    return out

def refusal_records_en(variants: int = 3, seed: int = 5) -> list[Record]:
    """English templated refusals for disallowed training prompts (D-024)."""
    from kitsune import en

    rng = random.Random(seed)
    out: list[Record] = []
    for p in en.train_policy_prompts_en():
        if p["kind"] not in ("sexual", "real_person", "existing_ip", "hate"):
            continue
        for v in range(variants):
            q = dict(
                p,
                id=p["id"] + f"-v{v}",
                genres_text=rng.choice(en._FANTASY_GENRES_EN),
                format=rng.choice(["あらすじ", "短編", "短編"]),
            )
            out.append(
                make_record(
                    id=q["id"],
                    genres=["ハイファンタジー"],
                    title=q["title"],
                    format=q["format"],
                    prompt=en.policy_user_prompt_en(q),
                    response=en.refusal_text_en(p["kind"]),
                    language="en",
                    source="seed",
                    generator="template:en.refusal_text_en",
                    license=SEED_LICENSE,
                    filters_passed=["template"],
                    meta={"policy_kind": p["kind"], "requested_genres": q["genres_text"]},
                )
            )
    return out
