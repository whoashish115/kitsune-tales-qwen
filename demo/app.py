"""Kitsune Tales playground (Hugging Face Space, free CPU tier): two 4-bit GGUF models via llama.cpp, plus galleries.

Two models, one per language (D-024), chosen by the tab:
    日本語  → kitsune-tales-e4b-jp  (original Japanese fantasy light novels)
    English → kitsune-tales-e4b-en  (English fantasy with Japanese anime / light-novel themes)

Costs nothing to run: the models run on the Space's own CPU and each loads on first use. Live generation
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
    if not genres:
        return None, "ジャンルを1〜3個選んでください。"
    if len(genres) > 3:
        return None, "ジャンルは3個までです。"
    if not title.strip():
        return None, "タイトルを入力してください。"
    for text in (title, passage or ""):
        check = f_prompt_safety(text)
        if not check.passed:
            return None, refusal_text(refusal_kind(check.reason, text))
    try:
        req = StoryRequest(genres, title.strip(), fmt, passage.strip() if fmt == "続き" else None)
    except (ValueError, UnknownGenreError) as e:
        return None, f"入力エラー: {e}"
    return build_user_prompt(req), None


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
    if en.unsafe_output_en(text):
        return "(The safety filter hid this output. Please try a different request.)"
    return text


_LLMS: dict[str, object] = {}


def get_llm(lang: str):  # type: ignore[no-untyped-def]
    """Load the language's GGUF on first use (memory-mapped, so both fit in the free tier's RAM)."""
    if lang not in _LLMS:
        from llama_cpp import Llama

        m = MODELS[lang]
        path = m["local"]
        if not path:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(m["repo"], m["file"])
        _LLMS[lang] = Llama(
            model_path=path,
            n_ctx=4096,
            n_threads=os.cpu_count() or 2,
            n_gpu_layers=int(os.environ.get("KITSUNE_GPU_LAYERS", "0")),  # -1 offloads every layer (Colab T4)
            verbose=False,
        )
    return _LLMS[lang]


# Defaults are the decoding settings behind every number in the report (configs/eval.yaml).
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


def _screen(
    lang: str, genres: list[str], title: str, fmt: str, passage: str
) -> tuple[str | None, str | None]:
    if lang == "jp":
        return screen_request(genres, title, fmt, passage)
    return screen_request_en(genres, title, fmt, passage)


def _band(lang: str, fmt: str) -> tuple[int, int, str]:
    spec = FORMATS[fmt] if lang == "jp" else en.FORMATS_EN[fmt]
    return spec.target_min, spec.target_max, ("字" if lang == "jp" else "words")


def _length(lang: str, text: str) -> int:
    if lang == "jp":
        from kitsune.taxonomy import count_chars

        return count_chars(text)
    return en.count_words(text)


def target_hint(lang: str, fmt: str) -> str:
    lo, hi, unit = _band(lang, fmt)
    return f"目標の長さ: {lo:,}〜{hi:,}{unit}" if lang == "jp" else f"Target length: {lo:,} to {hi:,} {unit}"


def stats_line(lang: str, fmt: str, text: str, n_tok: int, secs: float, finish: str) -> str:
    lo, hi, unit = _band(lang, fmt)
    n = _length(lang, text)
    where = (
        "within the target"
        if lo <= n <= hi
        else ("shorter than the target" if n < lo else "longer than the target")
    )
    rate = n_tok / secs if secs > 0 else 0.0
    return f"**{n:,} {unit}**, {where} ({lo:,} to {hi:,}). {n_tok} tokens in {secs:.0f} s ({rate:.1f} tok/s), finish reason: {finish}."


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


def show_prompt(lang: str, genres: list[str], title: str, fmt: str, passage: str) -> str:
    user, msg = _screen(lang, genres, title, fmt, passage)
    return msg if msg is not None else chat_prompt(user, MODELS[lang]["system"])


def load_gallery(lang: str) -> list[dict]:
    p = HERE / f"gallery_{lang}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def random_request(lang: str) -> tuple[list[str], str, str, str]:
    """A held-out test request from the gallery, to try the model on something it was evaluated on."""
    import random

    gallery = load_gallery(lang) or [
        {"genres": ["異世界転生"], "title": "追放された剣士", "format": "あらすじ"}
    ]
    g = random.choice(gallery)
    return g["genres"], g["title"], g["format"], g.get("passage") or ""


GALLERY_NOTE = (
    "Pre-generated by the released model on held-out prompts (not cherry-picked: seeded random draw "
    "stratified by genre × format; see REPORT.md). Always available, even when live generation is slow."
)


def _gallery_tab(gr, demo, lang: str) -> None:  # type: ignore[no-untyped-def]
    gallery = load_gallery(lang)
    gr.Markdown(GALLERY_NOTE)
    if not gallery:
        gr.Markdown("_Gallery not built yet._")
        return
    if lang == "jp":
        labels = [f"{g['format']}｜{g['title']}" for g in gallery]
    else:
        labels = [f"{en.FORMAT_NAME_EN[g['format']]} | {g['title']}" for g in gallery]
    pick = gr.Dropdown(labels, value=labels[0], label="作品" if lang == "jp" else "Story")
    meta = gr.Markdown()
    body = gr.Textbox(lines=18, label="本文" if lang == "jp" else "Text")
    trans = gr.Textbox(lines=8, label="English translation (by the maintainer)", visible=lang == "jp")

    def show(label: str) -> tuple[str, str, str]:
        g = gallery[labels.index(label)]
        if lang == "jp":
            head = f"**ジャンル:** {', '.join(g['genres'])}　**形式:** {g['format']}"
        else:
            names = ", ".join(en.GENRE_NAME_EN[x] for x in g["genres"])
            head = f"**Genres:** {names}  **Format:** {en.FORMAT_NAME_EN[g['format']]}"
        return head, g["text"], g.get("translation_en", "")

    pick.change(show, pick, [meta, body, trans])
    demo.load(show, pick, [meta, body, trans])


EXAMPLES = {
    "jp": [
        [["悪役令嬢・転生", "スローライフ"], "断罪された令嬢は辺境で薬草園を営む", "あらすじ", ""],
        [["魔王と勇者"], "引退した魔王は湖畔で喫茶店を開く", "短編", ""],
        [["魔法少女", "魔法学園"], "魔法少女ルミナは今日も遅刻する", "あらすじ", ""],
    ],
    "en": [
        [
            ["スローライフ", "ハイファンタジー"],
            "A Kicked-Out Summoner Wants a Quiet Life in the Frontier",
            "続き",
            None,
        ],
        [
            ["悪役令嬢・転生", "スローライフ"],
            "The Condemned Lady Runs an Herb Garden on the Frontier",
            "あらすじ",
            "",
        ],
        [["魔王と勇者"], "The Retired Demon Lord Opens a Lakeside Cafe", "短編", ""],
        [["魔法少女", "魔法学園"], "Magical Girl Lumina Is Late Again Today", "あらすじ", ""],
    ],
}


def examples(lang: str) -> list[list]:
    """Example requests; a passage of None is filled from the gallery entry with the same title."""
    by_title = {g["title"]: g for g in load_gallery(lang)}
    out = []
    for genres, title, fmt, passage in EXAMPLES[lang]:
        if passage is None:
            passage = (by_title.get(title) or {}).get("passage") or ""
            if not passage:
                continue
        out.append([genres, title, fmt, passage])
    return out


def _live_tab(gr, lang: str) -> None:  # type: ignore[no-untyped-def]
    jp = lang == "jp"
    gr.Markdown(
        "4-bit GGUF on the Space's free CPU: about 1 to 3 minutes for a short story. The defaults are the evaluation settings."
    )
    with gr.Row():
        with gr.Column(scale=1, min_width=320):
            if jp:
                info = " / ".join(f"{g}={GENRE_EN[g]}" for g in GENRES)
                genres = gr.CheckboxGroup(
                    list(GENRES), value=["異世界転生", "冒険者ギルド"], label="ジャンル（1〜3）", info=info
                )
                title = gr.Textbox(value="追放された剣士は二度目の人生で最強になる", label="タイトル")
                fmt = gr.Radio(list(FORMATS), value="あらすじ", label="形式")
            else:
                genres = gr.CheckboxGroup(
                    [(en.GENRE_NAME_EN[g], g) for g in GENRES],
                    value=["スローライフ", "ハイファンタジー"],
                    label="Genres (1 to 3)",
                )
                title = gr.Textbox(
                    value="A Kicked-Out Summoner Wants a Quiet Life in the Frontier", label="Title"
                )
                fmt = gr.Radio([(en.FORMAT_NAME_EN[f], f) for f in FORMATS], value="あらすじ", label="Format")
            hint = gr.Markdown(target_hint(lang, "あらすじ"))
            passage = gr.Textbox(
                lines=5,
                label="本文（「続き」のときだけ使用）" if jp else "Passage (continuation only)",
                placeholder="続きを書いてほしい文章" if jp else "Text to continue",
            )
            with gr.Accordion("Sampling / サンプリング", open=False):
                temperature = gr.Slider(
                    0.1, 1.5, value=DEFAULTS["temperature"], step=0.05, label="temperature"
                )
                top_p = gr.Slider(0.5, 1.0, value=DEFAULTS["top_p"], step=0.01, label="top-p")
                top_k = gr.Slider(0, 200, value=DEFAULTS["top_k"], step=1, label="top-k (0 = off)")
                min_p = gr.Slider(0.0, 0.3, value=DEFAULTS["min_p"], step=0.01, label="min-p")
                repeat_penalty = gr.Slider(
                    1.0, 1.5, value=DEFAULTS["repeat_penalty"], step=0.01, label="repetition penalty"
                )
                presence_penalty = gr.Slider(
                    0.0, 1.5, value=DEFAULTS["presence_penalty"], step=0.05, label="presence penalty"
                )
                max_tokens = gr.Slider(
                    128, 3000, value=DEFAULTS["max_tokens"], step=32, label="max new tokens"
                )
                seed = gr.Number(value=DEFAULTS["seed"], precision=0, label="seed (-1 = random)")
                reset = gr.Button("Reset to evaluation settings", size="sm")
            with gr.Row():
                btn = gr.Button("生成する" if jp else "Write", variant="primary")
                stop = gr.Button("Stop", variant="stop")
                rnd = gr.Button("テストから選ぶ" if jp else "Random test request")
        with gr.Column(scale=2):
            out = gr.Textbox(lines=26, label="出力" if jp else "Output")
            stats = gr.Markdown()
            dl = gr.File(label="Download (.txt)")
            with gr.Accordion("Prompt sent to the model", open=False):
                peek = gr.Button("Show the exact prompt", size="sm")
                prompt_box = gr.Code(label="System and user turns (Gemma 4 chat format)")
    lang_s = gr.State(lang)
    sampling = [temperature, top_p, top_k, min_p, repeat_penalty, presence_penalty, max_tokens, seed]
    request = [genres, title, fmt, passage]
    run = btn.click(generate_live, [lang_s, *request, *sampling], [out, stats, dl])
    stop.click(None, cancels=[run])
    fmt.change(lambda f: target_hint(lang, f), fmt, hint)
    rnd.click(lambda: random_request(lang), None, request)
    peek.click(show_prompt, [lang_s, *request], prompt_box)
    reset.click(lambda: [DEFAULTS[k] for k in SAMPLING_KEYS], None, sampling)
    ex = examples(lang)
    if ex:
        gr.Examples(ex, request, label="Examples")


def build_ui():  # type: ignore[no-untyped-def]
    import gradio as gr

    with gr.Blocks(title="Kitsune Tales") as demo:
        gr.Markdown(
            "# Kitsune Tales playground\nLoRA fine-tunes of Gemma 4 E4B that write original fantasy light-novel fiction: "
            f"**`{versions.MODEL_SLUG}`** in Japanese and **`{versions.MODEL_SLUG_EN}`** in English."
        )
        gr.Markdown(POLICY_MD)
        with gr.Tab(f"日本語 · {versions.MODEL_SLUG}"):
            with gr.Tab("生成 / Live"):
                _live_tab(gr, "jp")
            with gr.Tab("ギャラリー / Gallery"):
                _gallery_tab(gr, demo, "jp")
        with gr.Tab(f"English · {versions.MODEL_SLUG_EN}"):
            with gr.Tab("Live"):
                _live_tab(gr, "en")
            with gr.Tab("Gallery"):
                _gallery_tab(gr, demo, "en")
    return demo


if __name__ == "__main__":
    build_ui().queue(default_concurrency_limit=1).launch(share=os.environ.get("KITSUNE_SHARE") == "1")
