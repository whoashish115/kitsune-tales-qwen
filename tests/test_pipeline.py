"""End-to-end CPU test of the dataset pipeline on hand-written fixtures."""
from __future__ import annotations
import json
import random
from pathlib import Path
from kitsune.data.generate import split_for_continuation
from kitsune.data.policy import REDIRECT_PREFIX, train_policy_prompts
from kitsune.data.seeds import SeedPrompt, build_test_prompts, build_train_prompts
from kitsune.fixtures import GENRES, STORY, SYNOPSIS, TITLE
from kitsune.schema import Record, read_jsonl, write_jsonl
from kitsune.taxonomy import count_chars
GOOD_LABEL = json.dumps(
    {
        "fantasy": True,
        "general_audience": True,
        "real_person_or_existing_ip": False,
        "genre_match": 2,
        "title_match": 2,
        "quality": 4,
    }
)

def _gen(i: str, kind: str, fmt: str, text: str, gen: str = "genA", finish: str = "stop") -> dict:
    seed = SeedPrompt(i, GENRES, TITLE, fmt).to_dict()
    return {
        "id": i,
        "kind": kind,
        "text": text,
        "finish_reason": finish,
        "generator": f"{gen}-model@abc123",  # model@revision, as modal_app.data_run writes it
        "gen_key": gen,
        "meta": {"seed": seed, "knobs": {"pov": "三人称"}},
    }

def _lab(sid: str, who: str, text: str) -> dict:
    """A label row in the format modal_app.data_run writes."""
    raise NotImplementedError

def test_offgenre_redirect_prefix_survives(tmp_path: Path) -> None:
    from kitsune.data.pipeline import build

    policy = next(p for p in train_policy_prompts() if p.kind == "offgenre" and p.format == "短編")
    raw = tmp_path / "raw"
    g = {
        "id": "o1",
        "kind": "offgenre",
        "text": STORY,
        "finish_reason": "stop",
        "generator": "genA",
        "meta": {"policy": policy.__dict__},
    }
    write_jsonl(raw / "gen_a.jsonl", [g])
    write_jsonl(raw / "labels_a.jsonl", [{"sample_id": "o1", "labeler": "genB", "text": GOOD_LABEL}])
    build(raw, tmp_path / "out", tmp_path / "rep")
    rows = [Record(**r) for r in read_jsonl(tmp_path / "out" / "train.jsonl")]
    red = [r for r in rows if r.meta.get("policy_kind") == "offgenre"]
    assert len(red) == 1 and red[0].response.startswith(REDIRECT_PREFIX)
    assert policy.genres_text in red[0].prompt
