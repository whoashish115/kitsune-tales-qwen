"""Synthetic data generation with an open-weight generator served by vLLM.

CPU side (unit-tested): building generator requests with varied style knobs and sampling
parameters, parsing brainstormed titles, and splitting stories into 続き pairs.
GPU side (``run_vllm_chat``): runs inside the cloud GPU image only.
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
    kind = kind or {"あらすじ": "synopsis", "短編": "story", "続き": "source"}[seed.format]
    ask_key = {"synopsis": "あらすじ", "story": "短編", "source": "source", "test_passage": "source"}[kind]
    k = _knobs(rng)
    sampling = dict(rng.choice(SAMPLING_GRID))
    max_tokens = {"synopsis": 700, "story": 2000, "source": 2200, "test_passage": 2200}[kind]
    return GenJob(
        id=f"{seed.id}:{kind}",
        kind=kind,
        messages=[
            {"role": "system", "content": GEN_SYSTEM},
            {"role": "user", "content": _story_user(seed.genres, seed.title, _LENGTH_ASK[ask_key], k)},
        ],
        sampling=sampling,
        max_tokens=max_tokens,
        meta={"seed": seed.to_dict(), "knobs": k},
    )


def offgenre_job(p: PolicyPrompt, rng: random.Random, n: int = 0) -> GenJob:
    """Transpose a non-fantasy request into fantasy (the response gets ``REDIRECT_PREFIX`` later)."""
    k = _knobs(rng)
    ask = _LENGTH_ASK["あらすじ" if p.format == "あらすじ" else "短編"]
    user = (
        f"次の依頼は本来ファンタジーではありません。\n元の依頼: ジャンル「{p.genres_text}」、タイトル「{p.title}」\n\n"
        f"この依頼の核となる題材や感情を、剣と魔法のファンタジー世界に置き換えて書いてください。\n"
        f"- {ask}\n- 視点: {k['pov']}、トーン: {k['tone']}。\n"
        f"- 現代の固有名詞や実在の地名は使わず、ファンタジーの舞台にしてください。"
    )
    return GenJob(
        id=f"{p.id}:offgenre:{n}",
        kind="offgenre",
        messages=[{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": user}],
        sampling=dict(rng.choice(SAMPLING_GRID)),
        max_tokens=2000,
        meta={"policy": asdict(p), "knobs": k},
    )


def titles_job(genre: str, examples: list[str], rng: random.Random, i: int) -> GenJob:
    """Ask the generator to brainstorm 20 new light-novel titles for ``genre``."""
    ex = "\n".join(f"- {e}" for e in examples)
    user = (
        f"ジャンル「{genre}」のライトノベルのタイトルを、オリジナルで20個考えてください。\n"
        f"参考（この表現をそのまま使わないこと）:\n{ex}\n\n"
        f"条件: 実在の人物・既存作品の名前は使わない。全年齢向け。1行に1タイトル、番号や記号なし、タイトルだけを出力。"
    )
    return GenJob(
        id=f"titles:{genre}:{i}",
        kind="titles",
        messages=[{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": user}],
        sampling={"temperature": 1.0, "top_p": 0.95, "presence_penalty": 1.0},
        max_tokens=900,
        meta={"genre": genre},
    )


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


def split_for_continuation(
    story: str,
    rng: random.Random,
    passage_range: tuple[int, int] = CONTINUATION_PASSAGE_RANGE,
    cont_range: tuple[int, int] = (FORMATS["続き"].target_min, FORMATS["続き"].target_max),
) -> tuple[str, str] | None:
    """Cut a story into (passage, continuation) at sentence boundaries.

    Among the sentence boundaries whose prefix length lies in ``passage_range``, pick the one
    closest to a uniformly drawn target. The continuation takes the following whole sentences up
    to ``cont_range[1]`` characters. Returns ``None`` if no valid cut exists.
    """
    sents = sentences(story)
    target = rng.randint(*passage_range)
    cuts: list[tuple[int, int]] = []  # (index after sentence, prefix length)
    acc = 0
    for i, s in enumerate(sents):
        acc += count_chars(s)
        if passage_range[0] <= acc <= passage_range[1]:
            cuts.append((i + 1, acc))
        if acc > passage_range[1]:
            break
    if not cuts:
        return None
    cut = min(cuts, key=lambda c: abs(c[1] - target))[0]
    cont: list[str] = []
    n = 0
    for s in sents[cut:]:
        if n + count_chars(s) > cont_range[1]:
            break
        cont.append(s)
        n += count_chars(s)
    if n < cont_range[0]:
        return None
    return "".join(sents[:cut]).strip(), "".join(cont).strip()


# ----------------------------------------------------------------------------- GPU side


class ChatEngine:
    """A vLLM engine loaded once and reused for several passes (titles, stories, labels).

    Imported lazily: vllm exists only in the GPU image.
    """

    def __init__(
        self,
        model: str,
        revision: str,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.92,
        chat_template_kwargs: dict | None = None,
        text_only: bool = True,
        **llm_kwargs: Any,
    ) -> None:
        from vllm import LLM

        self.model, self.revision = model, revision
        self.chat_template_kwargs = (
            chat_template_kwargs if chat_template_kwargs is not None else {"enable_thinking": False}
        )
        base = {
            "model": model,
            "revision": revision,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enable_prefix_caching": True,
            **llm_kwargs,
        }
        # Multimodal checkpoints (Qwen3.5/3.6, Gemma 4): skip the vision tower if the pinned vLLM allows it.
        attempts: list[dict[str, Any]] = (
            [{"language_model_only": True}, {"limit_mm_per_prompt": {"image": 0, "video": 0}}, {}]
            if text_only
            else [{}]
        )
        errors = []
        for extra in attempts:
            try:
                self.llm = LLM(**base, **extra)
                self.load_kwargs = extra
                break
            except (TypeError, ValueError) as e:  # unknown kwarg / unsupported option → try the next form
                errors.append(f"{extra}: {e}")
        else:
            raise RuntimeError("vLLM failed to load " + model + ": " + " | ".join(errors))

    def chat(self, jobs: list[GenJob], seed: int = 0, extra_sampling: dict | None = None) -> list[dict]:
        """Run ``jobs``; returns [{id, text, finish_reason, n_tokens, n_prompt_tokens}] in input order."""
        from vllm import SamplingParams

        params = [
            SamplingParams(max_tokens=j.max_tokens, seed=seed + i, **j.sampling, **(extra_sampling or {}))
            for i, j in enumerate(jobs)
        ]
        outs = self.llm.chat(
            [j.messages for j in jobs], params, use_tqdm=True, chat_template_kwargs=self.chat_template_kwargs
        )
        res = []
        for j, o in zip(jobs, outs, strict=True):
            c = o.outputs[0]
            res.append(
                {
                    "id": j.id,
                    "text": c.text,
                    "finish_reason": c.finish_reason,
                    "n_tokens": len(c.token_ids),
                    "n_prompt_tokens": len(o.prompt_token_ids or []),
                }
            )
        return res
