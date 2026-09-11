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
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from __future__ import annotations
POLICY_MD = """**Content policy.** Both models write *original, general-audience* fantasy only.
They refuse sexual content, real people, existing copyrighted characters or fan fiction, and hateful content.
Requests outside fantasy are rewritten as fantasy. Requests are screened before generation and outputs after it.
Model outputs are fiction and may contain mistakes or repetition; see the model cards for known limitations."""

def chat_prompt(user: str, system: str = SYSTEM_PROMPT) -> str:
    """Exactly the non-thinking Gemma 4 generation prompt used in training, minus ``<bos>``.

    llama.cpp adds BOS itself for Gemma GGUFs; the test checks ``"<bos>" + chat_prompt(u) == render_prompt(tok, u)``.
    """
    raise NotImplementedError

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
