"""Prompt format: build, parse and tokenize chat examples.
The user turn is a small key/value block (see ``docs/DECISIONS.md`` D-007)::

    ジャンル: 異世界転生, 冒険者ギルド, ハイファンタジー
    タイトル: 追放された剣士は二度目の人生で最強になる
    形式: 短編
For ``続き`` a ``本文:`` field follows, with the passage on the next lines.
All models are prompted in non-thinking mode (``enable_thinking=False``).
"""
import re
from dataclasses import dataclass, field
from typing import Any, Protocol
from __future__ import annotations
from kitsune.taxonomy import FORMATS, normalize_genres
CHAT_TEMPLATE_KWARGS: dict[str, Any] = {"enable_thinking": False}
_GENRE_SEP = re.compile(r"[,、，/／・]\s*|\s+")
