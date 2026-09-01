"""Prompt format: build, parse and tokenize chat examples.

The user turn is a small key/value block (see ``docs/DECISIONS.md`` D-007)::

    ジャンル: 異世界転生, 冒険者ギルド, ハイファンタジー
    タイトル: 追放された剣士は二度目の人生で最強になる
    形式: 短編

For ``続き`` a ``本文:`` field follows, with the passage on the next lines.
All models are prompted in non-thinking mode (``enable_thinking=False``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from kitsune.taxonomy import FORMATS, normalize_genres

SYSTEM_PROMPT = (
    "あなたは全年齢向けのオリジナル・ファンタジー小説を書く作家です。"
    "指定されたジャンル・タイトル・形式に従い、日本語のライトノベルらしい文体で執筆してください。"
    "性的な内容、実在の人物、既存作品のキャラクターは扱いません。"
    "ファンタジー以外の依頼は、ファンタジー作品として書き直します。"
)

CHAT_TEMPLATE_KWARGS: dict[str, Any] = {"enable_thinking": False}

_GENRE_SEP = re.compile(r"[,、，/／・]\s*|\s+")


@dataclass(frozen=True)
class StoryRequest:
    """A structured request. ``passage`` is required for ``続き`` and forbidden otherwise."""

    genres: list[str]
    title: str
    format: str
    passage: str | None = None
    extra: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.format not in FORMATS:
            raise ValueError(f"unknown format {self.format!r}; expected one of {list(FORMATS)}")
        if (self.format == "続き") != (self.passage is not None and self.passage.strip() != ""):
            raise ValueError("a passage is required for 続き and not allowed for other formats")
        if not self.title.strip():
            raise ValueError("title must be non-empty")
        object.__setattr__(self, "genres", normalize_genres(self.genres))


def build_user_prompt(req: StoryRequest) -> str:
    """Render a request into the canonical user-turn text."""
    lines = [
        f"ジャンル: {', '.join(req.genres)}",
        f"タイトル: {req.title.strip()}",
        f"形式: {req.format}",
    ]
    if req.format == "続き":
        assert req.passage is not None
        lines.append("本文:")
        lines.append(req.passage.strip())
    return "\n".join(lines)


def parse_user_prompt(text: str) -> StoryRequest:
    """Parse a user-turn text back into a :class:`StoryRequest` (inverse of :func:`build_user_prompt`).

    Genre separators may be ``,``, ``、`` or spaces, and aliases are normalized.

    Raises:
        ValueError: on missing fields or unknown genres/formats.
    """
    fields: dict[str, str] = {}
    lines = text.strip().splitlines()
    passage_lines: list[str] | None = None
    for line in lines:
        if passage_lines is not None:
            passage_lines.append(line)
            continue
        m = re.match(r"^\s*(ジャンル|タイトル|形式|本文)\s*[:：]\s*(.*)$", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if key == "本文":
            passage_lines = [val] if val else []
        else:
            fields[key] = val
    missing = [k for k in ("ジャンル", "タイトル", "形式") if k not in fields]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    # Split on separators but keep the taxonomy's own "・" inside 悪役令嬢・転生.
    raw = fields["ジャンル"].replace("悪役令嬢・転生", "悪役令嬢転生")
    genres = [g for g in _GENRE_SEP.split(raw) if g]
    passage = "\n".join(passage_lines).strip() if passage_lines is not None else None
    return StoryRequest(
        genres=genres, title=fields["タイトル"], format=fields["形式"], passage=passage or None
    )


def build_messages(user_prompt: str, response: str | None = None, system: str = SYSTEM_PROMPT) -> list[dict]:
    """Chat messages in the OpenAI/HF format. ``response`` is appended as the assistant turn if given."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}]
    if response is not None:
        msgs.append({"role": "assistant", "content": response})
    return msgs


class _ChatTokenizer(Protocol):
    def apply_chat_template(self, conversation: list[dict], **kwargs: Any) -> Any: ...
    def __call__(self, text: str, **kwargs: Any) -> Any: ...


def gemma_prompt(user_prompt: str, system: str = SYSTEM_PROMPT) -> str:
    """Tokenizer-free Gemma 4 non-thinking generation prompt, identical to ``render_prompt`` with the pinned
    tokenizer (checked in tests against prompts rendered by the real tokenizer)."""
    return f"<bos><|turn>system\n{system}<turn|>\n<|turn>user\n{user_prompt}<turn|>\n<|turn>model\n"


def render_prompt(tokenizer: _ChatTokenizer, user_prompt: str, system: str = SYSTEM_PROMPT) -> str:
    """The exact text fed to the model at inference time (generation prompt, non-thinking)."""
    return tokenizer.apply_chat_template(
        build_messages(user_prompt, system=system),
        tokenize=False,
        add_generation_prompt=True,
        **CHAT_TEMPLATE_KWARGS,
    )


def render_completion(
    tokenizer: _ChatTokenizer, user_prompt: str, response: str, system: str = SYSTEM_PROMPT
) -> str:
    """The assistant-side text the model must learn to produce, ending with the end-of-turn token.

    Computed as ``full_conversation[len(prompt):]`` so it matches the chat template exactly.
    Raises ``ValueError`` if the template does not render the prompt as a prefix of the full
    conversation (in which case assistant-only masking would be wrong).
    """
    prompt = render_prompt(tokenizer, user_prompt, system)
    full = tokenizer.apply_chat_template(
        build_messages(user_prompt, response, system=system), tokenize=False, **CHAT_TEMPLATE_KWARGS
    )
    if not full.startswith(prompt):
        raise ValueError(
            "chat template does not render the generation prompt as a prefix of the conversation"
        )
    completion = full[len(prompt) :].rstrip("\n")
    if response.strip() not in completion:
        raise ValueError("response text not found in rendered completion")
    return completion


def tokenize_example(
    tokenizer: _ChatTokenizer, user_prompt: str, response: str, max_length: int, system: str = SYSTEM_PROMPT
) -> dict[str, list[int]]:
    """Tokenize one SFT example with loss on the assistant turn only.

    Prompt and completion are tokenized separately and concatenated, which is exactly what
    happens at inference time (prompt tokens, then generated tokens). Prompt positions get
    label ``-100``. Examples longer than ``max_length`` raise ``ValueError`` rather than
    being silently truncated (the data pipeline filters them earlier).
    """
    prompt_ids = tokenizer(render_prompt(tokenizer, user_prompt, system), add_special_tokens=False)[
        "input_ids"
    ]
    completion_ids = tokenizer(
        render_completion(tokenizer, user_prompt, response, system), add_special_tokens=False
    )["input_ids"]
    input_ids = list(prompt_ids) + list(completion_ids)
    if len(input_ids) > max_length:
        raise ValueError(f"example has {len(input_ids)} tokens > max_length={max_length}")
    labels = [-100] * len(prompt_ids) + list(completion_ids)
    return {"input_ids": input_ids, "labels": labels}
