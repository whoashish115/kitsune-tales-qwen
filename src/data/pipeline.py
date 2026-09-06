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


def build(
    raw_dir: Path,
    out_dir: Path,
    reports_dir: Path,
    val_frac: float = 0.03,
    require_label: bool = True,
    lang: str = "ja",
) -> dict:
    """Run the full pipeline; writes train/val JSONL, stats.json, plots and an inspection sample."""
    rng = random.Random(1234)
    gens = [g for f in sorted(raw_dir.glob("gen_*.jsonl*")) for g in read_jsonl(f)]
    labels: dict[str, dict[str, Labels | None]] = defaultdict(dict)
    for f in sorted(raw_dir.glob("labels_*.jsonl*")):
        for row in read_jsonl(f):
            m = row.get("meta", {})
            sid, who = row.get("sample_id") or m["sample_id"], row.get("labeler") or m["labeler"]
            labels[sid][who] = parse_labels(row["text"])

    cands = [
        _candidate_from_generation(g, rng, lang)
        for g in gens
        if g["kind"] in {"synopsis", "story", "source", "offgenre"}
    ]
    survivors = filter_candidates(cands, labels, require_label=require_label, lang=lang)
    kept, rep, dup_of = dedup(survivors, key=lambda c: c.response, threshold=0.7)
    for i, c in enumerate(survivors):
        if i in dup_of:
            c.drop_reason = "dedup:near_or_exact"
    refusals = refusal_records_en() if lang == "en" else refusal_records(train_policy_prompts())
    records = [to_record(c, lang=lang) for c in kept] + refusals
    train, val = stratified_split(records, val_frac)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)

    drop = Counter(c.drop_reason for c in cands if c.drop_reason)
    stats = {
        "n_generations": len(gens),
        "n_candidates": len(cands),
        "n_after_filters": len(survivors),
        "dedup": rep.__dict__,
        "n_kept_synthetic": len(kept),
        "n_refusal_templates": len(records) - len(kept),
        "n_train": len(train),
        "n_val": len(val),
        "drop_reasons": dict(drop.most_common()),
        "by_generator_kept": dict(Counter(c.gen_key for c in kept)),
        "by_generator_total": dict(Counter(c.gen_key for c in cands)),
        "drop_reasons_by_generator": {
            k: dict(Counter(c.drop_reason for c in cands if c.gen_key == k and c.drop_reason).most_common())
            for k in sorted({c.gen_key for c in cands})
        },
        "gen_tokens_mean_kept": (sum(c.meta.get("n_tokens", 0) for c in kept) / len(kept)) if kept else None,
        "label_sources": dict(Counter(c.meta.get("label_source", "none") for c in kept)),
        "cells_train": {
            f"{g}|{f}": n for (g, f), n in sorted(Counter((r.genres[0], r.format) for r in train).items())
        },
        "title_source_kept": dict(Counter(c.meta.get("title_source", "policy") for c in kept)),
        "sha256": {
            "train.jsonl": file_sha256(out_dir / "train.jsonl"),
            "val.jsonl": file_sha256(out_dir / "val.jsonl"),
        },
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    write_inspection_sample(train, reports_dir / "inspection_sample.md", n=100)
    plot_stats(cands, train, reports_dir, lang)
    return stats


def write_inspection_sample(records: list[Record], path: Path, n: int = 100, seed: int = 99) -> None:
    """A seeded random sample for the manual-inspection gate (notes are added by hand below each item)."""
    rng = random.Random(seed)
    pick = rng.sample(records, min(n, len(records)))
    lines = [f"# Manual inspection sample ({len(pick)} random training records, seed {seed})", ""]
    for i, r in enumerate(pick, 1):
        lines += [
            f"## {i}. `{r.id}` | {r.format} | {', '.join(r.genres)} | {r.generator}",
            f"**Title:** {r.title}",
            "",
            "```text",
            r.response[:1200] + ("…" if len(r.response) > 1200 else ""),
            "```",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_stats(cands: list[Candidate], train: list[Record], out: Path, lang: str = "ja") -> None:
    """Filter funnel (what was removed and why), length histograms per format, genre×format counts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    jp = [
        f.name
        for f in font_manager.fontManager.ttflist
        if any(
            k in f.name for k in ("Noto Sans CJK", "Noto Sans JP", "IPAexGothic", "Yu Gothic", "MS Gothic")
        )
    ]
    if jp:
        plt.rcParams["font.family"] = jp[0]

    drop = Counter(c.drop_reason for c in cands if c.drop_reason)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(drop) + 1)))
    names = [k for k, _ in drop.most_common()][::-1]
    ax.barh(names, [drop[k] for k in names], color="#4C72B0")
    ax.set_xlabel("samples removed")
    ax.set_title(f"Filter funnel: {len(cands)} candidates → {len(cands) - sum(drop.values())} kept")
    fig.tight_layout()
    fig.savefig(out / "filter_funnel.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, fmt in zip(axes, ["あらすじ", "短編", "続き"], strict=True):
        if lang == "en":
            from kitsune.en import count_words

            ls = [count_words(r.response) for r in train if r.format == fmt and "policy_kind" not in r.meta]
        else:
            ls = [
                sum(1 for ch in r.response if not ch.isspace())
                for r in train
                if r.format == fmt and "policy_kind" not in r.meta
            ]
        ax.hist(ls, bins=30, color="#55A868")
        ax.set_title(f"{fmt} (n={len(ls)})")
        ax.set_xlabel("words" if lang == "en" else "characters")
    fig.tight_layout()
    fig.savefig(out / "length_hist.png", dpi=150)
    plt.close(fig)

    fmts = ["あらすじ", "短編", "続き"]
    grid = [
        [
            sum(1 for r in train if r.genres[0] == g and r.format == f and "policy_kind" not in r.meta)
            for f in fmts
        ]
        for g in GENRES
    ]
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(grid, cmap="Blues")
    ax.set_xticks(range(3), fmts)
    ax.set_yticks(range(len(GENRES)), GENRES)
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    ax.set_title("train examples: primary genre × format")
    fig.tight_layout()
    fig.savefig(out / "genre_format_grid.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--raw", type=Path, default=Path("data/raw"))
    b.add_argument("--out", type=Path, default=Path("data/processed"))
    b.add_argument("--reports", type=Path, default=Path("reports/data"))
    b.add_argument("--val-frac", type=float, default=0.03)
    b.add_argument("--no-require-label", action="store_true")
    b.add_argument("--lang", choices=["ja", "en"], default="ja")
    a = ap.parse_args()
    stats = build(a.raw, a.out, a.reports, a.val_frac, require_label=not a.no_require_label, lang=a.lang)
    print(json.dumps({k: v for k, v in stats.items() if k != "cells_train"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
