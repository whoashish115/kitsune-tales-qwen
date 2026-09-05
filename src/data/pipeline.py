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

@dataclass
class Candidate:
    """A sample on its way through the pipeline."""

    sample_id: str
    generator: str  # model@revision, recorded in the Record for provenance
    kind: str
    genres: list[str]
    title: str
    format: str
    prompt: str
    response: str
    passage: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    outcomes: list[FilterOutcome] = field(default_factory=list)
    drop_reason: str | None = None
    gen_key: str = ""  # short generator key ("gen1"/"gen2") used to tell cross-labels from self-labels

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
    from kitsune import en

    kind = g["kind"]
    meta = {k: g[k] for k in ("sampling", "n_tokens", "finish_reason") if k in g}
    src = g.get("meta", {}).get("seed") or g.get("meta", {}).get("policy") or {}
    text, cleaned = en.clean_generation_en(g["text"], src.get("title", ""))
    if cleaned:
        meta["cleaned"] = True
    rebalanced = en.rebalance_names_en(text, g["id"], src.get("title", ""))
    if rebalanced != text:
        meta["names_rebalanced"] = True
        text = rebalanced
    if "knobs" in g.get("meta", {}):
        meta["knobs"] = g["meta"]["knobs"]
    if kind == "offgenre":
        p = g["meta"]["policy"]
        c = Candidate(
            g["id"], g["generator"], kind, ["ハイファンタジー"], p["title"], p["format"], en.policy_user_prompt_en(p),
            en.REDIRECT_PREFIX_EN + "\n\n" + text.strip(),
            meta=meta | {"policy_kind": "offgenre", "requested_genres": p["genres_text"]},
        )  # fmt: skip
    else:
        genres, title, fmt = src["genres"], src["title"], src["format"]
        meta |= {"title_source": src.get("title_source", "template")}
        if kind == "source":
            split = en.split_for_continuation_en(text.strip(), rng)
            if split is None:
                c = Candidate(g["id"], g["generator"], kind, genres, title, "続き", "", "", meta=meta)
                c.drop_reason = "split:no_valid_cut"
                c.gen_key = g.get("gen_key") or g["generator"]
                return c
            passage, cont = split
            c = Candidate(
                g["id"],
                g["generator"],
                kind,
                genres,
                title,
                "続き",
                en.build_user_prompt_en(genres, title, "続き", passage),
                cont,
                passage,
                meta,
            )
        else:
            c = Candidate(
                g["id"],
                g["generator"],
                kind,
                genres,
                title,
                fmt,
                en.build_user_prompt_en(genres, title, fmt),
                text.strip(),
                meta=meta,
            )
    c.gen_key = g.get("gen_key") or g["generator"]
    if g.get("finish_reason") == "length":
        c.drop_reason = "generation:truncated"
    return c

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

def filter_candidates(
    cands: list[Candidate],
    labels: dict[str, dict[str, Labels | None]],
    require_label: bool = True,
    min_quality: int = 3,
    lang: str = "ja",
) -> list[Candidate]:
    """Apply rule filters then LLM labels; sets ``drop_reason`` on failures. Returns survivors."""
    kept = []
    for c in cands:
        if c.drop_reason:
            continue
        if lang == "en":
            from kitsune.en import run_rule_filters_en

            outs = run_rule_filters_en(c.response, c.format, c.genres, c.title, c.passage, kind=c.kind)
        else:
            outs = run_rule_filters(c.response, c.format, c.genres, c.title, c.passage)
            if c.kind != "offgenre":
                outs = [f_title_clean(c.title), *outs]
            if c.kind == "offgenre":  # requested genres are off-taxonomy by construction
                outs = [o for o in outs if o.name != "tag_consistency"]
        c.outcomes = outs
        fail = next((o for o in outs if not o.passed), None)
        if fail:
            c.drop_reason = f"{fail.name}:{fail.reason.split(':')[0] or 'fail'}"
            continue
        lab, src = _pick_labels(c.sample_id, c.gen_key, labels)
        c.meta["label_source"] = src
        if lab is None:
            if require_label:
                c.drop_reason = "llm_label:missing_or_invalid"
                continue
        else:
            c.meta["labels"] = lab.__dict__
            if not lab.passes(c.format if c.kind != "offgenre" else "続き", min_quality):
                why = (
                    "not_fantasy" if not lab.fantasy
                    else "not_general_audience" if not lab.general_audience
                    else "real_or_ip" if lab.real_person_or_existing_ip
                    else "low_quality" if lab.quality < min_quality
                    else "tag_mismatch"
                )  # fmt: skip
                c.drop_reason = f"llm_label:{why}"
                continue
        kept.append(c)
    return kept

def to_record(c: Candidate, generator_license: str = SYNTHETIC_LICENSE, lang: str = "ja") -> Record:
    return make_record(
        id=c.sample_id,
        genres=c.genres,
        title=c.title,
        format=c.format,
        prompt=c.prompt,
        response=c.response,
        language=lang,
        source="synthetic",
        generator=c.generator,
        license=generator_license,
        filters_passed=[o.name for o in c.outcomes] + (["llm_label"] if "labels" in c.meta else []),
        meta=c.meta | ({"passage_chars": len(c.passage)} if c.passage else {}),
    )

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

def stratified_split(
    records: list[Record], val_frac: float, seed: int = 3
) -> tuple[list[Record], list[Record]]:
    """Validation = ``val_frac`` of *story* records per (primary genre, format) cell. Policy records stay in train."""
    rng = random.Random(seed)
    cells: dict[tuple[str, str], list[Record]] = defaultdict(list)
    train: list[Record] = []
    for r in records:
        if "policy_kind" in r.meta:
            train.append(r)
        else:
            cells[(r.genres[0], r.format)].append(r)
    val: list[Record] = []
    for key in sorted(cells):
        rs = cells[key][:]
        rng.shuffle(rs)
        k = max(1, round(len(rs) * val_frac)) if len(rs) >= 10 else 0
        val += rs[:k]
        train += rs[k:]
    rng.shuffle(train)
    return train, val

def write_inspection_sample(records: list[Record], path: Path, n: int = 100, seed: int = 99) -> None:
    """A seeded random sample for the manual-inspection gate (notes are added by hand below each item)."""
    raise NotImplementedError
