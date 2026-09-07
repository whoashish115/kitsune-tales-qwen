"""Merge a LoRA adapter into the base weights and verify the merge.
Verification (spec section 5): on a fixed prompt set, the merged model must match
adapter-on-base (1) token-for-token under greedy decoding and (2) within a small logit
tolerance on a teacher-forced batch. The report is written next to the merged weights and
copied to ``reports/merge_check.json``.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
from kitsune.prompts import StoryRequest, build_user_prompt, render_prompt
CHECK_REQUESTS = [
    StoryRequest(["異世界転生", "冒険者ギルド"], "追放された剣士は辺境で最強になる", "あらすじ"),
    StoryRequest(["悪役令嬢・転生"], "断罪された令嬢は薬草園で幸せになる", "短編"),
    StoryRequest(["魔王と勇者", "スローライフ"], "引退した魔王は湖畔で喫茶店を開く", "あらすじ"),
    StoryRequest(["魔法少女"], "魔法少女ルミナは今日も遅刻する", "短編"),
    StoryRequest(["ダークファンタジー"], "灰燼の王冠", "あらすじ"),
    StoryRequest(["ハイファンタジー", "魔法学園"], "星詠みの巫女と銀の塔", "短編"),
    StoryRequest(["スローライフ"], "北の雪国で始める薬師のスローライフ", "あらすじ"),
    StoryRequest(["冒険者ギルド"], "Fランク冒険者の荷物持ちは竜と暮らす", "短編"),
]
