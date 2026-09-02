"""LLM labels for data filtering (the "small judge" half of the fantasy/safety/consistency filters).

A labeler model reads (request, story) and returns strict JSON. Labels from a model that did
not write the sample ("cross" labels) are preferred over self-labels; see D-011.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Final

from kitsune.data.generate import GenJob

LABEL_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "fantasy": {"type": "boolean"},
        "general_audience": {"type": "boolean"},
        "real_person_or_existing_ip": {"type": "boolean"},
        "genre_match": {"type": "integer", "minimum": 0, "maximum": 2},
        "title_match": {"type": "integer", "minimum": 0, "maximum": 2},
        "quality": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": [
        "fantasy",
        "general_audience",
        "real_person_or_existing_ip",
        "genre_match",
        "title_match",
        "quality",
    ],
    "additionalProperties": False,
}

LABEL_SYSTEM: Final = (
    "あなたは出版社の厳格な編集者です。与えられた依頼と本文を読み、指定のJSON形式だけで評価を返してください。"
)

_LABEL_USER: Final = """依頼:
ジャンル: {genres}
タイトル: {title}
形式: {fmt}

本文:
<<<
{text}
>>>

次の項目を判定し、JSONのみを出力してください。
- fantasy: 本文が剣と魔法などのファンタジー作品になっているか (true/false)
- general_audience: 性的描写・過度な残虐描写・自傷の描写がなく、全年齢向けか (true/false)
- real_person_or_existing_ip: 実在の人物、または既存の漫画・アニメ・ゲーム・小説のキャラクターや作品名が登場するか (true/false)
- genre_match: 指定ジャンルの要素の反映度 (0=反映なし, 1=一部, 2=すべて明確)
- title_match: タイトルの内容の反映度 (0=無関係, 1=部分的, 2=明確)
- quality: 日本語の文章と物語としての完成度 (1〜5, 5が最高)"""


def label_job(sample_id: str, genres: list[str], title: str, fmt: str, text: str, labeler: str) -> GenJob:
    """A deterministic (temperature 0) labeling request for one sample."""
    return GenJob(
        id=f"{sample_id}:label:{labeler}",
        kind="label",
        messages=[
            {"role": "system", "content": LABEL_SYSTEM},
            {
                "role": "user",
                "content": _LABEL_USER.format(genres=", ".join(genres), title=title, fmt=fmt, text=text),
            },
        ],
        sampling={"temperature": 0.0},
        max_tokens=120,
        meta={"sample_id": sample_id, "labeler": labeler},
    )


@dataclass(frozen=True)
class Labels:
    fantasy: bool
    general_audience: bool
    real_person_or_existing_ip: bool
    genre_match: int
    title_match: int
    quality: int

    def passes(self, fmt: str, min_quality: int = 3) -> bool:
        """The keep rule used by the pipeline (title match is not required for 続き)."""
        return (
            self.fantasy
            and self.general_audience
            and not self.real_person_or_existing_ip
            and self.genre_match >= 1
            and (fmt == "続き" or self.title_match >= 1)
            and self.quality >= min_quality
        )


def parse_labels(text: str) -> Labels | None:
    """Parse the labeler's JSON (tolerates code fences and surrounding text). ``None`` if invalid."""
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return Labels(
            fantasy=bool(d["fantasy"]),
            general_audience=bool(d["general_audience"]),
            real_person_or_existing_ip=bool(d["real_person_or_existing_ip"]),
            genre_match=int(d["genre_match"]),
            title_match=int(d["title_match"]),
            quality=int(d["quality"]),
        )
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None


def structured_output_kwargs(schema: dict[str, Any] = LABEL_SCHEMA) -> dict[str, Any]:
    """SamplingParams kwargs that force JSON matching ``schema`` in the pinned vLLM (GPU image only).

    vLLM renamed guided decoding to structured outputs; both spellings are tried, and each candidate is
    validated by constructing a SamplingParams. If neither works, returns {} and the labeler's free-text
    JSON is parsed tolerantly by :func:`parse_labels` (invalid labels are counted, never guessed).
    """
    from vllm import SamplingParams

    candidates = []
    try:
        from vllm.sampling_params import StructuredOutputsParams

        candidates.append({"structured_outputs": StructuredOutputsParams(json=schema)})
    except (ImportError, TypeError):
        pass
    try:
        from vllm.sampling_params import GuidedDecodingParams

        candidates.append({"guided_decoding": GuidedDecodingParams(json=schema)})
    except (ImportError, TypeError):
        pass
    for kw in candidates:
        try:
            SamplingParams(max_tokens=8, **kw)
            return kw
        except (TypeError, ValueError):
            continue
    return {}
