"""Pairwise LLM-as-judge with position swapping, plus known-answer validation of the judge itself.
Protocol (D-004, D-010):
- Every pair is judged twice, (A=x, B=y) and (A=y, B=x). A system *wins* a pair only if it wins
  both orders; disagreement between orders counts as a tie and is reported as the
  position-inconsistency rate (Zheng et al. 2023; Wang et al. 2023).
- The rubric tells the judge not to reward length.
- Before any judge result is reported, the judge must pass a known-answer test: an intact story
  vs a deliberately corrupted copy (paragraph shuffle, loop, Chinese contamination, wrong story,
  truncation). Its accuracy there is reported next to the pairwise results.
"""
from __future__ import annotations
import random
import re
from dataclasses import dataclass
from typing import Final
from kitsune.data.generate import GenJob, sentences
JUDGE_SYSTEM: Final = "あなたはライトノベル新人賞の審査員です。公平かつ厳密に、二つの作品のどちらが依頼に対して優れているかを判定します。"
JUDGE_USER: Final = """依頼:
{request}
【作品A】
<<<
{a}
>>>
【作品B】
<<<
{b}
>>>
以下の5つの観点で両作品を比較してください。
1. 一貫性: 物語や文章が破綻なくつながっているか
2. 文体: ライトノベルらしい読みやすく魅力的な文体か
3. 独創性: ありきたりでない発想や展開があるか
4. 依頼への適合: 指定されたジャンル・タイトル・形式（長さの目安を含む）に沿っているか
5. 日本語の自然さ: 誤字、不自然な表現、他言語の混入がないか
注意: 提示された順番に影響されないこと。長いというだけで高く評価しないこと。
最後の行に必ず「判定: A」「判定: B」「判定: 引き分け」のいずれかだけを書いてください。"""
_VERDICT = re.compile(r"判定\s*[:：]\s*(A|B|Ａ|Ｂ|引き分け)")

# DPO labels (teacher, non-thinking): with the full rubric the teacher wrote ~1,000+ token analyses and 87 % of
# the first Japanese label run hit the 1,024-token cap before the verdict line. Labels therefore ask for one
# sentence per criterion and get a 2,048-token cap. The evaluation judge keeps the full, validated prompt.
BRIEF_JA: Final = (
    "\n\n各観点の評価は一文ずつ、全体で300字以内に簡潔にまとめてから、最後の行に判定を書いてください。"
)

def judge_job(
    pair_id: str, order: str, request: str, a: str, b: str, max_tokens: int = 3072, brief: bool = False
) -> GenJob:
    """One judge call. ``order`` is "xy" or "yx" and is carried in the id for re-assembly."""
    return GenJob(
        id=f"{pair_id}|{order}",
        kind="judge",
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": JUDGE_USER.format(request=request, a=a, b=b) + (BRIEF_JA if brief else ""),
            },
        ],
        sampling={"temperature": 0.0},
        max_tokens=max_tokens,
        meta={"pair_id": pair_id, "order": order},
    )

def combine(v_xy: str | None, v_yx: str | None) -> PairOutcome:
    """Combine verdicts from (A=x,B=y) and (A=y,B=x)."""
    if v_xy is None or v_yx is None:
        return PairOutcome("invalid", False)
    first = {"A": "x", "B": "y", "tie": "tie"}[v_xy]
    second = {"A": "y", "B": "x", "tie": "tie"}[v_yx]
    if first == second:
        return PairOutcome(first, True)
    return PairOutcome("tie", False)
