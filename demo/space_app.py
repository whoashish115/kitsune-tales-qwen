"""Kitsune Tales playground: Hugging Face Space on ZeroGPU.
Both released models run in bf16 with transformers; each request borrows a GPU for its generation only
(``@spaces.GPU``). Requests are screened before generation and outputs after it, exactly as in demo/app.py.

    Write     genres, title, format and passage; every sampling setting; streamed output with its length against the
              requested range; the exact prompt; a .txt download.
    Explore   all 270 held-out test prompts per language (seed 0), released model next to the base model.
    About     headline results with sample sizes, links, limitations.
from __future__ import annotations
Environment: HF_TOKEN is needed only while the model repositories are private.
"""
import json
import os
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from threading import Thread
import gradio as gr
from kitsune import en, versions
from kitsune.data.filters import f_prompt_safety, f_real_or_copyrighted, f_safety_rule
from kitsune.data.policy import refusal_kind, refusal_text
from kitsune.prompts import SYSTEM_PROMPT, StoryRequest, build_user_prompt
from kitsune.taxonomy import FORMATS, GENRES, UnknownGenreError, count_chars
# --------------------------------------------------------------------------------------------- models
def load(lang: str):  # type: ignore[no-untyped-def]
    if lang not in _MODELS:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(REPOS[lang], token=TOKEN)
        model = AutoModelForCausalLM.from_pretrained(REPOS[lang], token=TOKEN, torch_dtype=torch.bfloat16)
        model.to("cuda" if torch.cuda.is_available() or "spaces" in globals() else "cpu").eval()
        _MODELS[lang] = (tok, model)
    return _MODELS[lang]

LOAD_ERROR: dict[str, str] = {}
NO_ACCESS = (
    "The model weights could not be loaded, so live writing is unavailable. While the model repositories are private, "
    "the Space needs a read token as the secret HF_TOKEN (Settings > Variables and secrets). Explore and About still work."
)


def band(lang: str, fmt: str) -> tuple[int, int, str]:
    spec = FORMATS[fmt] if lang == "jp" else en.FORMATS_EN[fmt]
    return spec.target_min, spec.target_max, ("字" if lang == "jp" else "words")
