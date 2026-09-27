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

_MODELS: dict[str, tuple] = {}


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


# --------------------------------------------------------------------------------------------- requests


def screen(lang: str, genres: list[str], title: str, fmt: str, passage: str) -> tuple[str | None, str | None]:
    """Validate and screen a request. Returns (user prompt, None) or (None, message)."""
    jp = lang == "jp"
    if not genres:
        return None, "ジャンルを1〜3個選んでください。" if jp else "Choose one to three genres."
    if len(genres) > 3:
        return None, "ジャンルは3個までです。" if jp else "Choose at most three genres."
    if not title.strip():
        return None, "タイトルを入力してください。" if jp else "Enter a title."
    if fmt == "続き" and not passage.strip():
        return None, "「続き」には本文が必要です。" if jp else "A continuation needs a passage."
    for text in (title, passage or ""):
        if jp:
            check = f_prompt_safety(text)
            if not check.passed:
                return None, refusal_text(refusal_kind(check.reason, text))
        else:
            check = en.screen_request_text_en(text)
            if not check.passed:
                return None, en.refusal_text_en(en.refusal_kind_en(check.reason, text))
    try:
        if jp:
            return build_user_prompt(
                StoryRequest(genres, title.strip(), fmt, passage.strip() if fmt == "続き" else None)
            ), None
        return en.build_user_prompt_en(
            genres, title.strip(), fmt, passage.strip() if fmt == "続き" else None
        ), None
    except (ValueError, KeyError, UnknownGenreError) as e:
        return None, f"Input error: {e}"


def screen_output(lang: str, text: str) -> str:
    if lang == "jp" and (not f_safety_rule(text).passed or not f_real_or_copyrighted(text).passed):
        return "（安全フィルタにより出力を非表示にしました。条件を変えてもう一度お試しください。）"
    if lang == "en" and en.unsafe_output_en(text):
        return "(The safety filter hid this output. Please try a different request.)"
    return text


def band(lang: str, fmt: str) -> tuple[int, int, str]:
    spec = FORMATS[fmt] if lang == "jp" else en.FORMATS_EN[fmt]
    return spec.target_min, spec.target_max, ("字" if lang == "jp" else "words")


def length(lang: str, text: str) -> int:
    return count_chars(text) if lang == "jp" else en.count_words(text)


def verdict(lang: str, fmt: str, text: str) -> str:
    lo, hi, unit = band(lang, fmt)
    n = length(lang, text)
    where = "within" if lo <= n <= hi else ("below" if n < lo else "above")
    return f"**{n:,} {unit}**, {where} the requested {lo:,} to {hi:,}"


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


# --------------------------------------------------------------------------------------------- explore data

OUTPUTS = {
    lang: json.loads((HERE / "data" / f"outputs_{lang}.json").read_text(encoding="utf-8"))
    for lang in ("jp", "en")
}
META = json.loads((HERE / "data" / "meta.json").read_text(encoding="utf-8"))


def fmt_label(lang: str, k: str) -> str:
    return k if lang == "jp" else en.FORMAT_NAME_EN[k]


def genre_label(lang: str, g: str) -> str:
    return g if lang == "jp" else en.GENRE_NAME_EN[g]


def explore_choices(lang: str, genre: str, fmt: str, q: str) -> list[tuple[str, str]]:
    out = []
    for x in OUTPUTS[lang]:
        if genre and genre not in x["genres"]:
            continue
        if fmt and x["format"] != fmt:
            continue
        if q and q.lower() not in x["title"].lower():
            continue
        ok = x["band"][0] <= x["lk"] <= x["band"][1]
        out.append(
            (f"{x['id']} · {fmt_label(lang, x['format'])} · {x['title']}  {'✓' if ok else '✗'}", x["id"])
        )
    return out


def explore_show(lang: str, pid: str) -> tuple[str, str, str, str, str, str, str]:
    x = next((i for i in OUTPUTS[lang] if i["id"] == pid), None)
    if x is None:
        return "", "", "", "", "", "", ""
    head = f"### {x['title']}\n{' · '.join(genre_label(lang, g) for g in x['genres'])} · {fmt_label(lang, x['format'])} · `{x['id']}`"
    lo, hi = x["band"]
    unit = "字" if lang == "jp" else "words"

    def mark(n: int) -> str:
        return f"{'✓' if lo <= n <= hi else '✗'} {n:,} {unit} (requested {lo:,} to {hi:,})"

    return head, x["passage"], x["k"], mark(x["lk"]), x["b"], mark(x["lb"]), x["tr"]


# --------------------------------------------------------------------------------------------- UI

# Palette of the Kitsune site and the earlier playground: violet-grey neutrals, deep indigo selection states, and the
# logo violet (#5848f8) reserved for the primary action. Light and dark values mirror each other.
NEUTRAL = gr.themes.Color(
    c50="#f7f7fa",
    c100="#f0eff6",
    c200="#e4e2ec",
    c300="#c8c5d3",
    c400="#9a97a6",
    c500="#77747f",
    c600="#5a5864",
    c700="#47464f",
    c800="#302f37",
    c900="#1c1b22",
    c950="#121216",
)
PRIMARY = gr.themes.Color(
    c50="#f1f0ff",
    c100="#e6e3ff",
    c200="#cdc8ff",
    c300="#aaa2fb",
    c400="#8478f5",
    c500="#5848f8",
    c600="#4a3aea",
    c700="#3e2fcf",
    c800="#2d2677",
    c900="#241f5c",
    c950="#1d1170",
)
THEME = gr.themes.Base(
    primary_hue=PRIMARY,
    secondary_hue=PRIMARY,
    neutral_hue=NEUTRAL,
    font=[gr.themes.GoogleFont("Roboto"), gr.themes.GoogleFont("Noto Sans JP"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Roboto Mono"), "ui-monospace", "monospace"],
    radius_size="lg",
).set(
    body_background_fill="#f4f4f8",
    body_background_fill_dark="#121216",
    background_fill_primary="#ffffff",
    background_fill_primary_dark="#1c1b22",
    background_fill_secondary="#f0eff6",
    background_fill_secondary_dark="#24232b",
    block_background_fill="#ffffff",
    block_background_fill_dark="#1c1b22",
    block_border_color="#e4e2ec",
    block_border_color_dark="#302f37",
    border_color_primary="#e4e2ec",
    border_color_primary_dark="#302f37",
    input_background_fill="#ffffff",
    input_background_fill_dark="#1c1b22",
    input_border_color="#c8c5d3",
    input_border_color_dark="#48464f",
    color_accent="#5848f8",
    color_accent_soft="#e6e3ff",
    color_accent_soft_dark="#2d2677",
    checkbox_label_background_fill="transparent",
    checkbox_label_background_fill_dark="transparent",
    checkbox_label_background_fill_selected="#e6e3ff",
    checkbox_label_background_fill_selected_dark="#2d2677",
    checkbox_label_text_color_selected="#1d1170",
    checkbox_label_text_color_selected_dark="#e4e0ff",
    checkbox_background_color_selected="#5848f8",
    checkbox_background_color_selected_dark="#6a5cff",
    button_primary_background_fill="#5848f8",
    button_primary_background_fill_dark="#5848f8",
    button_primary_background_fill_hover="#4637e0",
    button_primary_background_fill_hover_dark="#6a5cff",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="transparent",
    button_secondary_background_fill_dark="transparent",
    button_secondary_background_fill_hover="#f0eff6",
    button_secondary_background_fill_hover_dark="#24232b",
    button_secondary_border_color="#c8c5d3",
    button_secondary_border_color_dark="#48464f",
    button_secondary_text_color="#4a3aea",
    button_secondary_text_color_dark="#cdc8ff",
    slider_color="#5848f8",
    slider_color_dark="#8478f5",
    block_label_background_fill="#f0eff6",
    block_label_background_fill_dark="#24232b",
)

CSS = """
.gradio-container{max-width:1280px!important}
#brand img{border-radius:0}
#side{background:#ecebf4}
.dark #side{background:#18171d}
.story textarea{font-size:15px!important;line-height:1.9!important}
footer{display:none!important}
"""


def build() -> gr.Blocks:
    lang0 = "jp"
    with gr.Blocks(title="Kitsune Tales Playground") as demo:
        lang = gr.State(lang0)
        with gr.Sidebar(open=True, width=300, elem_id="side"):
            gr.Image(
                str(HERE / "logo.png"),
                show_label=False,
                container=False,
                height=72,
                elem_id="brand",
                interactive=False,
            )
            gr.Markdown(
                "## Kitsune Tales\nPlayground for two Gemma 4 E4B fine-tunes that write original fantasy light-novel fiction."
            )
            model = gr.Radio(
                [
                    (f"{versions.MODEL_SLUG} · Japanese · SFT", "jp"),
                    (f"{versions.MODEL_SLUG_EN} · English · SFT + DPO", "en"),
                ],
                value=lang0,
                label="Model",
            )
            if LOAD_ERROR:
                gr.Markdown(f"**Live writing is off.** {NO_ACCESS}")
            gr.Markdown(
                f"[Site]({SITE}) · [Models]({COLLECTION}) · [Datasets]({DATASETS}) · [GitHub]({GITHUB}) · "
                f"[Report]({GITHUB}/blob/main/REPORT.md) · [Colab]({COLAB})\n\nRuns on ZeroGPU: each request borrows a GPU for its "
                "generation only. Requests and outputs are screened (general-audience fantasy only)."
            )
        with gr.Tab("Write"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5, min_width=320):
                    genres = gr.CheckboxGroup(
                        [(f"{g} {en.GENRE_NAME_EN[g]}", g) for g in GENRES],
                        value=["異世界転生", "冒険者ギルド"],
                        label="Genres (one to three)",
                    )
                    title = gr.Textbox(value="追放された剣士は二度目の人生で最強になる", label="Title")
                    fmt = gr.Radio(
                        [(f"{k} {en.FORMAT_NAME_EN[k]}", k) for k in FORMATS],
                        value="あらすじ",
                        label="Format",
                    )
                    target = gr.Markdown(
                        f"Target length: {band('jp', 'あらすじ')[0]} to {band('jp', 'あらすじ')[1]} 字"
                    )
                    passage = gr.Textbox(lines=5, label="Passage to continue (continuation only)")
                    with gr.Accordion("Sampling", open=False):
                        temperature = gr.Slider(
                            0.0, 1.5, DEFAULTS["temperature"], step=0.05, label="Temperature"
                        )
                        top_p = gr.Slider(0.5, 1.0, DEFAULTS["top_p"], step=0.01, label="Top-p")
                        top_k = gr.Slider(0, 200, DEFAULTS["top_k"], step=1, label="Top-k (0 = off)")
                        rep = gr.Slider(
                            1.0, 1.5, DEFAULTS["repetition_penalty"], step=0.01, label="Repetition penalty"
                        )
                        max_new = gr.Slider(
                            128, 2400, DEFAULTS["max_new_tokens"], step=32, label="Max new tokens"
                        )
                        seed = gr.Number(DEFAULTS["seed"], precision=0, label="Seed (-1 = random)")
                        gr.Markdown("Defaults are the evaluation settings behind every reported number.")
                    with gr.Row():
                        go = gr.Button("Write", variant="primary")
                        stop = gr.Button("Stop", variant="stop")
                    with gr.Row():
                        rnd = gr.Button("Random held-out request", variant="secondary")
                        gr.Button("Open in Colab", variant="secondary", link=COLAB)
                with gr.Column(scale=7, min_width=360):
                    out = gr.Textbox(label="Story", lines=24, buttons=["copy"], elem_classes="story")
                    stats = gr.Markdown()
                    dl = gr.File(label="Download (.txt)")
                    with gr.Accordion("Exact prompt sent to the model", open=False):
                        prompt = gr.Code(label="Gemma 4 chat format")
            inputs = [lang, genres, title, fmt, passage, temperature, top_p, top_k, rep, max_new, seed]
            run = go.click(write, inputs, [out, stats, prompt, dl])
            stop.click(None, cancels=[run])

            def on_fmt(lang_v: str, f: str) -> str:
                lo, hi, unit = band(lang_v, f)
                return f"Target length: {lo:,} to {hi:,} {unit}"

            fmt.change(on_fmt, [lang, fmt], target)

            def on_random(lang_v: str):  # type: ignore[no-untyped-def]
                import random

                x = random.choice(OUTPUTS[lang_v])
                return x["genres"], x["title"], x["format"], x["passage"]

            rnd.click(on_random, lang, [genres, title, fmt, passage])

        with gr.Tab("Explore outputs"):
            gr.Markdown(
                "All 270 held-out test prompts for the selected language (seed 0), frozen before any training data existed. ✓ = within the requested length."
            )
            with gr.Row():
                f_genre = gr.Dropdown(
                    [("All genres", "")] + [(g, g) for g in GENRES], value="", label="Genre"
                )
                f_fmt = gr.Dropdown(
                    [("All formats", "")] + [(k, k) for k in FORMATS], value="", label="Format"
                )
                f_q = gr.Textbox(label="Search titles")
            pick = gr.Dropdown(
                explore_choices(lang0, "", "", ""), value=OUTPUTS[lang0][0]["id"], label="Prompt"
            )
            head = gr.Markdown()
            with gr.Accordion("Passage given to continue", open=False):
                ppass = gr.Textbox(show_label=False, lines=4)
            with gr.Row(equal_height=False):
                with gr.Column():
                    k_len = gr.Markdown()
                    k_txt = gr.Textbox(label="Kitsune (released)", lines=18, elem_classes="story")
                with gr.Column():
                    b_len = gr.Markdown()
                    b_txt = gr.Textbox(label="Gemma 4 E4B (base)", lines=18, elem_classes="story")
            tr = gr.Textbox(label="English translation", lines=8, visible=True)

            def refilter(lang_v: str, g: str, f: str, q: str):  # type: ignore[no-untyped-def]
                ch = explore_choices(lang_v, g, f, q)
                return gr.update(choices=ch, value=ch[0][1] if ch else None)

            def show(lang_v: str, pid: str):  # type: ignore[no-untyped-def]
                h, p, k, kl, b, bl, t = explore_show(lang_v, pid)
                return h, p, kl, k, bl, b, gr.update(value=t, visible=bool(t))

            for c in (f_genre, f_fmt, f_q):
                c.change(refilter, [lang, f_genre, f_fmt, f_q], pick)
            pick.change(show, [lang, pick], [head, ppass, k_len, k_txt, b_len, b_txt, tr])
            demo.load(show, [lang, pick], [head, ppass, k_len, k_txt, b_len, b_txt, tr])

        with gr.Tab("About"):
            rows = "\n".join(f"| {r[0]} | {r[1]} % | **{r[2]} %** | {r[3]} |" for r in META["glance"])
            gr.Markdown(
                f"""### Base model vs Kitsune

| Measured on held-out prompts | Base | Kitsune | n |
|---|---:|---:|---:|
{rows}

n = prompts × seeds. On equal-length openings an LLM judge rates Kitsune and the base model about the same (JP +0.12,
EN −0.03 net preference; both 95 % CIs include 0). Full results in the [site]({SITE}) and the
[technical report]({GITHUB}/blob/main/REPORT.md).

### Models
- **{versions.MODEL_SLUG}**: Japanese light-novel prose, SFT. [Weights](https://huggingface.co/{REPOS["jp"]}) · [LoRA](https://huggingface.co/{versions.HF_ADAPTER_REPO}) · [GGUF](https://huggingface.co/{versions.HF_GGUF_REPO})
- **{versions.MODEL_SLUG_EN}**: English prose with Japanese anime themes, SFT + DPO. [Weights](https://huggingface.co/{REPOS["en"]}) · [LoRA](https://huggingface.co/{versions.HF_ADAPTER_REPO_EN}) · [GGUF](https://huggingface.co/{versions.HF_GGUF_REPO_EN})

### Limitations
- No human evaluation; quality rests on rule-based metrics and one LLM judge.
- Everything was learned from two larger models, including their clichés.
- Safety filtering is lexicon-based and misses paraphrases.
- Titles that ask the model to drop fantasy are followed more often than by the base model.
"""
            )

        def switch(lang_v: str):  # type: ignore[no-untyped-def]
            jp = lang_v == "jp"
            lo, hi, unit = band(lang_v, "あらすじ")
            ch = explore_choices(lang_v, "", "", "")
            return (
                lang_v,
                gr.update(choices=[((f"{g} {en.GENRE_NAME_EN[g]}" if jp else en.GENRE_NAME_EN[g]), g) for g in GENRES], value=["異世界転生", "冒険者ギルド"] if jp else ["スローライフ", "ハイファンタジー"]),
                "追放された剣士は二度目の人生で最強になる" if jp else "A Kicked-Out Summoner Wants a Quiet Life in the Frontier",
                gr.update(choices=[((f"{k} {en.FORMAT_NAME_EN[k]}" if jp else en.FORMAT_NAME_EN[k]), k) for k in FORMATS], value="あらすじ"),
                f"Target length: {lo:,} to {hi:,} {unit}",
                "",
                gr.update(choices=[("All genres", "")] + [(genre_label(lang_v, g), g) for g in GENRES], value=""),
                gr.update(choices=[("All formats", "")] + [(fmt_label(lang_v, k), k) for k in FORMATS], value=""),
                "",
                gr.update(choices=ch, value=ch[0][1]),
            )  # fmt: skip

        model.change(switch, model, [lang, genres, title, fmt, target, passage, f_genre, f_fmt, f_q, pick])
    return demo


if __name__ == "__main__":
    build().queue(default_concurrency_limit=2).launch(theme=THEME, css=CSS)
