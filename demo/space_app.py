"""Kitsune Tales playground: Hugging Face Space on ZeroGPU.
Both released models run in bf16 with transformers; each request borrows a GPU for its generation only
(``@spaces.GPU``). Requests are screened before generation and outputs after it, exactly as in demo/app.py.

    Write     genres, title, format and passage; every sampling setting; streamed output with its length against the
              requested range; the exact prompt; a .txt download.
    Explore   all 270 held-out test prompts per language (seed 0), released model next to the base model.
    About     headline results with sample sizes, links, limitations.
Environment: HF_TOKEN is needed only while the model repositories are private.
"""
from __future__ import annotations
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
try:
    import spaces  # ZeroGPU
except ImportError:  # local run without ZeroGPU: plain decorator

    class spaces:
        @staticmethod
        def GPU(*_a, **_k):
            return lambda f: f

HERE = Path(__file__).resolve().parent
TOKEN = os.environ.get("HF_TOKEN")
REPOS = {"jp": versions.HF_MODEL_REPO, "en": versions.HF_MODEL_REPO_EN}
SLUG = {"jp": versions.MODEL_SLUG, "en": versions.MODEL_SLUG_EN}
SYSTEM = {"jp": SYSTEM_PROMPT, "en": en.SYSTEM_PROMPT_EN}
DEFAULTS = {
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.05,
    "max_new_tokens": 1200,
    "seed": -1,
}
SITE = "https://kitsune-tales-qwen.vercel.app"
GITHUB = "https://github.com/whoashish115/kitsune-tales-qwen"
COLAB = "https://colab.research.google.com/github/whoashish115/kitsune-tales-qwen/blob/main/notebooks/playground.ipynb"
COLLECTION = "https://huggingface.co/collections/whoashish115/kitsune-tales-6abd4e61de4896bb86692bc1"
DATASETS = "https://huggingface.co/datasets?search=kitsune-tales"

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
if os.environ.get("SPACE_ID"):  # load both at start-up so GPU time is spent on generation only
    for _lang in ("jp", "en"):
        try:
            load(_lang)
        except (
            OSError
        ) as e:  # private repo without HF_TOKEN, or a network error: keep the rest of the app running
            LOAD_ERROR[_lang] = str(e).splitlines()[0]
            print(f"[load] {REPOS[_lang]}: {LOAD_ERROR[_lang]}")

def chat_prompt(user: str, system: str) -> str:
    """The Gemma 4 turn format used in training (the tokenizer adds <bos>)."""
    return f"<|turn>system\n{system}<turn|>\n<|turn>user\n{user}<turn|>\n<|turn>model\n"



def screen_output(lang: str, text: str) -> str:
    if lang == "jp" and (not f_safety_rule(text).passed or not f_real_or_copyrighted(text).passed):
        return "（安全フィルタにより出力を非表示にしました。条件を変えてもう一度お試しください。）"
    if lang == "en" and en.unsafe_output_en(text):
        return "(The safety filter hid this output. Please try a different request.)"
    return text

def band(lang: str, fmt: str) -> tuple[int, int, str]:
    spec = FORMATS[fmt] if lang == "jp" else en.FORMATS_EN[fmt]
    return spec.target_min, spec.target_max, ("字" if lang == "jp" else "words")

# --------------------------------------------------------------------------------------------- generation

@spaces.GPU(duration=120)
def _generate(
    lang: str, prompt: str, temperature: float, top_p: float, top_k: int, rep: float, max_new: int, seed: int
):  # type: ignore[no-untyped-def]
    import torch
    from transformers import TextIteratorStreamer

    tok, model = load(lang)
    if seed >= 0:
        torch.manual_seed(int(seed))
    ids = tok(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(
        **ids,
        streamer=streamer,
        max_new_tokens=int(max_new),
        do_sample=temperature > 0,
        temperature=max(float(temperature), 1e-5),
        top_p=float(top_p),
        top_k=int(top_k),
        repetition_penalty=float(rep),
    )
    Thread(target=model.generate, kwargs=kwargs).start()
    yield from streamer

def write(
    lang: str, genres: list[str], title: str, fmt: str, passage: str,
    temperature: float, top_p: float, top_k: int, rep: float, max_new: int, seed: int,
) -> Iterator[tuple[str, str, str, str | None]]:  # fmt: skip
    """Stream a story: (text, stats, exact prompt, .txt path)."""
    if lang in LOAD_ERROR:
        yield NO_ACCESS, "", "", None
        return
    user, msg = screen(lang, genres, title, fmt, passage)
    if msg is not None:
        yield msg, "", "", None
        return
    prompt = chat_prompt(user, SYSTEM[lang])
    t0, out = time.time(), ""
    for piece in _generate(lang, prompt, temperature, top_p, top_k, rep, max_new, seed):
        out += piece
        yield out, f"writing… {time.time() - t0:.0f} s", prompt, None
    out = screen_output(lang, out)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(f"{title}\n\n{out}\n")
    yield out, f"{verdict(lang, fmt, out)} · {time.time() - t0:.0f} s", prompt, f.name

OUTPUTS = {
    lang: json.loads((HERE / "data" / f"outputs_{lang}.json").read_text(encoding="utf-8"))
    for lang in ("jp", "en")
}
META = json.loads((HERE / "data" / "meta.json").read_text(encoding="utf-8"))

def genre_label(lang: str, g: str) -> str:
    return g if lang == "jp" else en.GENRE_NAME_EN[g]
