"""Synthetic data generation with an open-weight generator served by vLLM.
CPU side (unit-tested): building generator requests with varied style knobs and sampling
parameters, parsing brainstormed titles, and splitting stories into 続き pairs.
GPU side (``run_vllm_chat``): runs inside the Modal GPU image only.
"""
from __future__ import annotations
from kitsune.data.policy import PolicyPrompt
from kitsune.data.seeds import SeedPrompt
from kitsune.taxonomy import CONTINUATION_PASSAGE_RANGE, FORMATS, count_chars
POV: Final = ("一人称（俺）", "一人称（私）", "一人称（僕）", "一人称（わたくし）", "三人称")
TONE: Final = ("コミカル", "シリアス", "切ない", "熱血", "ほのぼの", "ミステリアス", "爽快", "しんみり")
OPENING: Final = ("台詞から始める", "情景描写から始める", "主人公の独白から始める", "事件の最中から始める")
STYLE: Final = ("会話を多めに", "地の文を丁寧に", "テンポよく短い段落で", "心情描写を厚めに")
PROTAGONIST: Final = (
    "少年", "少女", "青年", "女性", "老騎士", "元冒険者の中年", "見習いの若者", "人ならざる存在",
)  # fmt: skip
SAMPLING_GRID: Final = (
    {"temperature": 0.6, "top_p": 0.95, "presence_penalty": 0.8},
    {"temperature": 0.65, "top_p": 0.95, "presence_penalty": 0.5},
    {"temperature": 0.7, "top_p": 0.9, "presence_penalty": 1.0},
    {"temperature": 0.75, "top_p": 0.9, "presence_penalty": 0.5},
)
GEN_SYSTEM: Final = (
    "あなたは日本のライトノベル作家です。全年齢向けのオリジナル・ファンタジー作品だけを書きます。"
    "性的な描写、実在の人物、既存作品のキャラクターや固有名詞は絶対に使いません。"
    "出力は本文のみとし、タイトル、見出し、前置き、後書き、注釈は書かないでください。"
    "中国語の表現や簡体字は使わず、自然な日本語（日本の漢字）だけで書いてください。"
)
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
