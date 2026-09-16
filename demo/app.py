"""Kitsune Tales playground (Hugging Face Space, free CPU tier): two 4-bit GGUF models via llama.cpp, plus galleries.
Two models, one per language (D-024), chosen by the tab:
    日本語  → kitsune-tales-e4b-jp  (original Japanese fantasy light novels)
    English → kitsune-tales-e4b-en  (English fantasy with Japanese anime / light-novel themes)
Costs $0 in Modal credits: the models run on the Space's own CPU and each loads on first use. Live generation
is slow (a few tokens/s on 2 vCPU), so the gallery tabs always work instantly.
Environment (all optional):
    KITSUNE_GGUF_JP / KITSUNE_GGUF_EN      local .gguf paths (otherwise downloaded from the Hub)
    KITSUNE_GGUF_REPO_JP / _EN             Hub repos with the GGUFs
    KITSUNE_GGUF_FILE_JP / _EN             file names inside those repos
    KITSUNE_GPU_LAYERS                     layers to offload to a GPU (-1 = all; default 0 = CPU only)
    KITSUNE_SHARE=1                        also serve a public gradio.live link (Colab)
"""
from __future__ import annotations
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
HERE = Path(__file__).resolve().parent
# In the Space, the minimal kitsune package is copied next to app.py; in the repo it is the installed package.
sys.path.insert(0, str(HERE))
from kitsune import en, versions  # noqa: E402
from kitsune.data.filters import f_prompt_safety, f_real_or_copyrighted, f_safety_rule  # noqa: E402
from kitsune.data.policy import refusal_kind, refusal_text  # noqa: E402
from kitsune.prompts import SYSTEM_PROMPT, StoryRequest, build_user_prompt  # noqa: E402
from kitsune.taxonomy import FORMATS, GENRE_EN, GENRES, UnknownGenreError  # noqa: E402
MODELS = {
    "jp": {
        "repo": os.environ.get("KITSUNE_GGUF_REPO_JP", versions.HF_GGUF_REPO),
        "file": os.environ.get("KITSUNE_GGUF_FILE_JP", f"{versions.MODEL_SLUG}-Q4_K_M.gguf"),
        "local": os.environ.get("KITSUNE_GGUF_JP"),
        "system": SYSTEM_PROMPT,
    },
    "en": {
        "repo": os.environ.get("KITSUNE_GGUF_REPO_EN", versions.HF_GGUF_REPO_EN),
        "file": os.environ.get("KITSUNE_GGUF_FILE_EN", f"{versions.MODEL_SLUG_EN}-Q4_K_M.gguf"),
        "local": os.environ.get("KITSUNE_GGUF_EN"),
        "system": en.SYSTEM_PROMPT_EN,
    },
}
POLICY_MD = """**Content policy.** Both models write *original, general-audience* fantasy only.
They refuse sexual content, real people, existing copyrighted characters or fan fiction, and hateful content.
Requests outside fantasy are rewritten as fantasy. Requests are screened before generation and outputs after it.
Model outputs are fiction and may contain mistakes or repetition; see the model cards for known limitations."""

def chat_prompt(user: str, system: str = SYSTEM_PROMPT) -> str:
    """Exactly the non-thinking Gemma 4 generation prompt used in training, minus ``<bos>``.

    llama.cpp adds BOS itself for Gemma GGUFs; the test checks ``"<bos>" + chat_prompt(u) == render_prompt(tok, u)``.
    """
    return f"<|turn>system\n{system}<turn|>\n<|turn>user\n{user}<turn|>\n<|turn>model\n"

def screen_request(genres: list[str], title: str, fmt: str, passage: str) -> tuple[str | None, str | None]:
    """Validate and screen a Japanese request. Returns (user_prompt, None) or (None, message to show)."""
    raise NotImplementedError

def screen_request_en(genres: list[str], title: str, fmt: str, passage: str) -> tuple[str | None, str | None]:
    """Validate and screen an English request (``fmt`` is a canonical format key)."""
    if not genres:
        return None, "Choose 1 to 3 genres."
    if len(genres) > 3:
        return None, "Choose at most 3 genres."
    if not title.strip():
        return None, "Enter a title."
    if fmt == "続き" and not passage.strip():
        return None, "Input error: a continuation needs a passage to continue."
    for text in (title, passage or ""):
        check = en.screen_request_text_en(text)
        if not check.passed:
            return None, en.refusal_text_en(en.refusal_kind_en(check.reason, text))
    try:
        user = en.build_user_prompt_en(genres, title.strip(), fmt, passage.strip() if fmt == "続き" else None)
    except (ValueError, KeyError, UnknownGenreError) as e:
        return None, f"Input error: {e}"
    return user, None

def screen_output(text: str) -> str:
    """Replace a Japanese output that trips the safety or real-person/IP filters (should be rare)."""
    if not f_safety_rule(text).passed or not f_real_or_copyrighted(text).passed:
        return "（安全フィルタにより出力を非表示にしました。条件を変えてもう一度お試しください。）"
    return text

def screen_output_en(text: str) -> str:
    raise NotImplementedError

_LLMS: dict[str, object] = {}

DEFAULTS = {
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 50,
    "min_p": 0.0,
    "repeat_penalty": 1.05,
    "presence_penalty": 0.0,
    "max_tokens": 900,
    "seed": -1,
}
SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repeat_penalty",
    "presence_penalty",
    "max_tokens",
    "seed",
)

def _length(lang: str, text: str) -> int:
    if lang == "jp":
        from kitsune.taxonomy import count_chars

        return count_chars(text)
    return en.count_words(text)

def generate_live(
    lang: str,
    genres: list[str],
    title: str,
    fmt: str,
    passage: str,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    repeat_penalty: float,
    presence_penalty: float,
    max_tokens: int,
    seed: int,
) -> Iterator[tuple[str, str, str | None]]:
    """Stream a story. Yields (text, stats markdown, path of a .txt download or None)."""
    import tempfile
    import time

    user, msg = _screen(lang, genres, title, fmt, passage)
    if msg is not None:
        yield msg, "", None
        return
    t0, out, n_tok, finish = time.time(), "", 0, "length"
    for chunk in get_llm(lang)(
        chat_prompt(user, MODELS[lang]["system"]),
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        min_p=float(min_p),
        repeat_penalty=float(repeat_penalty),
        presence_penalty=float(presence_penalty),
        seed=None if int(seed) < 0 else int(seed),
        stop=["<turn|>", "<eos>"],
        stream=True,
    ):
        choice = chunk["choices"][0]
        out += choice["text"]
        n_tok += 1
        finish = choice.get("finish_reason") or finish
        if n_tok % 8 == 0:
            yield out, f"{n_tok} tokens, {time.time() - t0:.0f} s", None
    out = screen_output(out) if lang == "jp" else screen_output_en(out)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(f"{title}\n\n{out}\n")
    yield out, stats_line(lang, fmt, out, n_tok, time.time() - t0, finish), f.name
