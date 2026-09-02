"""Synthetic data generation with an open-weight generator served by vLLM.
CPU side (unit-tested): building generator requests with varied style knobs and sampling
parameters, parsing brainstormed titles, and splitting stories into 続き pairs.
GPU side (``run_vllm_chat``): runs inside the Modal GPU image only.
"""
from __future__ import annotations
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Final
from kitsune.data.policy import PolicyPrompt
from kitsune.data.seeds import SeedPrompt
from kitsune.taxonomy import CONTINUATION_PASSAGE_RANGE, FORMATS, count_chars
GEN_SYSTEM: Final = (
    "あなたは日本のライトノベル作家です。全年齢向けのオリジナル・ファンタジー作品だけを書きます。"
    "性的な描写、実在の人物、既存作品のキャラクターや固有名詞は絶対に使いません。"
    "出力は本文のみとし、タイトル、見出し、前置き、後書き、注釈は書かないでください。"
    "中国語の表現や簡体字は使わず、自然な日本語（日本の漢字）だけで書いてください。"
)
POV: Final = ("一人称（俺）", "一人称（私）", "一人称（僕）", "一人称（わたくし）", "三人称")
TONE: Final = ("コミカル", "シリアス", "切ない", "熱血", "ほのぼの", "ミステリアス", "爽快", "しんみり")
OPENING: Final = ("台詞から始める", "情景描写から始める", "主人公の独白から始める", "事件の最中から始める")
STYLE: Final = ("会話を多めに", "地の文を丁寧に", "テンポよく短い段落で", "心情描写を厚めに")
PROTAGONIST: Final = (
    "少年", "少女", "青年", "女性", "老騎士", "元冒険者の中年", "見習いの若者", "人ならざる存在",
)  # fmt: skip
# Probe finding (D-017): simplified-Chinese contamination of Qwen3.6 rises steeply with temperature
# (0.7: 5 %, 0.8: 22 %, 0.9: 29 %, 1.0: 60 %). Temperatures are capped at 0.75; diversity comes from
# the style knobs, top-p and presence penalty.
SAMPLING_GRID: Final = (
    {"temperature": 0.6, "top_p": 0.95, "presence_penalty": 0.8},
    {"temperature": 0.65, "top_p": 0.95, "presence_penalty": 0.5},
    {"temperature": 0.7, "top_p": 0.9, "presence_penalty": 1.0},
    {"temperature": 0.75, "top_p": 0.9, "presence_penalty": 0.5},
)
_LENGTH_ASK: Final[dict[str, str]] = {
    "あらすじ": "物語全体のあらすじを250〜450字で書いてください。結末まで含めてください。",
    "短編": "900〜1400字の短編小説を書いてください。起承転結のある、一話で完結する物語にしてください。",
    "source": "1100〜1500字の短編小説の冒頭から中盤までを書いてください。物語が続いていく形で構いません。",
}

@dataclass
class GenJob:
    """One generator call. ``kind`` ∈ {synopsis, story, source, offgenre, titles, test_passage}."""

    id: str
    kind: str
    messages: list[dict[str, str]]
    sampling: dict[str, Any]
    max_tokens: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

def _knobs(rng: random.Random) -> dict[str, str]:
    return {
        "pov": rng.choice(POV),
        "tone": rng.choice(TONE),
        "opening": rng.choice(OPENING),
        "style": rng.choice(STYLE),
        "protagonist": rng.choice(PROTAGONIST),
    }

def _story_user(genres: list[str], title: str, ask: str, k: dict[str, str]) -> str:
    return (
        f"ジャンル: {', '.join(genres)}\n"
        f"タイトル: {title}\n\n"
        f"条件:\n"
        f"- {ask}\n"
        f"- 視点: {k['pov']}、トーン: {k['tone']}、主人公: {k['protagonist']}\n"
        f"- {k['opening']}。{k['style']}。\n"
        f"- タイトルとすべてのジャンルの要素を物語にはっきり反映させてください。\n"
        f"- 登場人物や地名はすべてオリジナルにしてください。日本語のみで書いてください。"
    )

def story_job(seed: SeedPrompt, rng: random.Random, kind: str | None = None) -> GenJob:
    """Generator request for a seed. 続き seeds get a longer ``source`` story that is split later."""
    raise NotImplementedError

def offgenre_job(p: PolicyPrompt, rng: random.Random, n: int = 0) -> GenJob:
    """Transpose a non-fantasy request into fantasy (the response gets ``REDIRECT_PREFIX`` later)."""
    raise NotImplementedError

_TITLE_LINE = re.compile(r"^\s*(?:[-・*●]|\d+[.)．、]|[（(]?\d+[)）])?\s*[「『]?(.+?)[」』]?\s*$")

def parse_titles(text: str) -> list[str]:
    """Extract clean titles (one per line) from a brainstorm response."""
    out = []
    for line in text.splitlines():
        m = _TITLE_LINE.match(line)
        if not m:
            continue
        t = m.group(1).strip()
        if 4 <= len(t) <= 40 and not t.endswith(("：", ":")) and "タイトル" not in t:
            out.append(t)
    return list(dict.fromkeys(out))

_SENT_END = re.compile(r"(?<=[。！？!?」』])")

def sentences(text: str) -> list[str]:
    """Split into sentences at Japanese sentence ends, keeping closing brackets attached."""
    parts = [p for p in _SENT_END.split(text) if p]
    merged: list[str] = []
    for p in parts:
        if merged and p and p[0] in "」』）)":
            merged[-1] += p
        else:
            merged.append(p)
    return merged
