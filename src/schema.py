"""Record schema for the processed dataset (JSONL, one record per line)."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

from kitsune.taxonomy import FORMATS, GENRES
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Format = Literal["あらすじ", "短編", "続き"]
Source = Literal["real", "synthetic", "seed"]


def normalize_text(s: str) -> str:
    """NFKC + collapse whitespace; used for hashing and exact dedup."""
    s = unicodedata.normalize("NFKC", s)
    return " ".join(s.split())


def content_hash(prompt: str, response: str) -> str:
    """SHA-256 of the normalized prompt and response."""
    h = hashlib.sha256()
    h.update(normalize_text(prompt).encode("utf-8"))
    h.update(b"\x00")
    h.update(normalize_text(response).encode("utf-8"))
    return h.hexdigest()


class Record(BaseModel):
    """One training/validation example (see spec section 6)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    genres: list[str] = Field(min_length=1, max_length=3)
    title: str = Field(min_length=1, max_length=120)
    format: Format
    prompt: str = Field(min_length=1)
    response: str = Field(min_length=1)
    language: Literal["ja", "en"] = "ja"
    source: Source
    generator: str = Field(min_length=1)
    license: str = Field(min_length=1)
    filters_passed: list[str] = Field(default_factory=list)
    hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    # Optional provenance: sampling params, style knobs, parent story id for 続き, policy kind.
    meta: dict = Field(default_factory=dict)

    @field_validator("genres")
    @classmethod
    def _genres_in_taxonomy(cls, v: list[str]) -> list[str]:
        bad = [g for g in v if g not in GENRES]
        if bad:
            raise ValueError(f"genres not in taxonomy: {bad}")
        if len(set(v)) != len(v):
            raise ValueError("duplicate genres")
        return v

    @model_validator(mode="after")
    def _hash_matches(self) -> Record:
        if self.hash != content_hash(self.prompt, self.response):
            raise ValueError("hash does not match prompt/response")
        if self.format not in FORMATS:
            raise ValueError("bad format")
        return self


def make_record(**kw: object) -> Record:
    """Build a :class:`Record`, computing ``hash`` from prompt and response."""
    kw = dict(kw)
    kw["hash"] = content_hash(str(kw["prompt"]), str(kw["response"]))
    return Record(**kw)  # type: ignore[arg-type]


def write_jsonl(path: str | Path, rows: Iterable[BaseModel | dict]) -> int:
    """Write rows as UTF-8 JSONL (non-ASCII kept). Returns the number of rows."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            d = r.model_dump() if isinstance(r, BaseModel) else r
            f.write(json.dumps(d, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield dicts from a UTF-8 JSONL file (``.gz`` supported)."""
    p = Path(path)
    if p.suffix == ".gz":
        import gzip

        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's bytes (used to freeze the test set and fingerprint datasets)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
