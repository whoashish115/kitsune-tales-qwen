"""Single source of truth for pinned versions and model revisions.

The Modal images in ``modal_app.py`` install exactly these versions, and
``tests/test_versions.py`` checks that ``pyproject.toml`` agrees with them.
Model revisions are full Hugging Face commit hashes, resolved on 2026-09-29.
"""

from __future__ import annotations

PYTHON_VERSION = "3.12"

# GPU image (Linux, CUDA). torch is the version vllm==0.30.0 pins.
GPU_PACKAGES: dict[str, str] = {
    "torch": "2.13.0",
    "vllm": "0.30.0",
    "transformers": "5.17.0",
    "trl": "1.14.1",
    "peft": "0.21.1",
    "accelerate": "1.15.0",
    "datasets": "5.0.1",
    "flash-linear-attention": "0.5.2",
    "lm-eval": "0.4.13",
    "wandb": "0.30.0",
    "huggingface-hub": "1.33.0",
    "pydantic": "2.13.5",
    "pyyaml": "6.0.3",
    "datasketch": "2.0.0",
}

# Models. "revision" is the exact Hub commit used for every run.
# D-001a (bake-off, 2026-09-29): the base switched from Qwen3.5-4B to Gemma 4 E4B under the pre-registered rule.
BASE_MODEL = "google/gemma-4-E4B-it"
BASE_REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"
BASE_END_OF_TURN = "<turn|>"  # the chat template's end-of-turn marker; a generation stop token (id 106)

ALT_BASE_MODEL = "Qwen/Qwen3.5-4B"  # former choice; kept as an extra baseline
ALT_BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

GENERATOR_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"  # D-003 primary synthetic-data generator
GENERATOR_REVISION = "95a723d08a9490559dae23d0cff1d9466213d989"

GENERATOR2_MODEL = "google/gemma-4-26B-A4B-it"  # D-003 secondary generator (style diversity)
GENERATOR2_REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"

JUDGE_MODEL = "llm-jp/llm-jp-4-32b-a3b-thinking"  # D-004 pairwise judge
JUDGE_REVISION = "30b90b28539c2d02f23b732a10c846b814730a50"

# Naming (D-015, revised after D-001a): "kitsune-tales" + size. Gemma 4 E4B stores 7.52B text parameters, 2.90B of
# which are per-layer embedding lookup tables; it computes like a 4.62B model. Google's "E4B" (effective 4B)
# convention is kept so the size is not misread as a 7B-class model.
MODEL_FAMILY = "Kitsune-Tales"
# One model per language, each with an explicit suffix: -jp (Japanese) and -en (English).
LANG_SUFFIX: dict[str, str] = {"ja": "jp", "en": "en"}


def model_slug(lang: str = "ja") -> str:
    return f"kitsune-tales-e4b-{LANG_SUFFIX[lang]}"


def model_name(lang: str = "ja") -> str:
    return f"Kitsune-Tales-E4B-{LANG_SUFFIX[lang].upper()}"


MODEL_NAME = model_name("ja")  # Kitsune-Tales-E4B-JP
MODEL_SLUG = model_slug("ja")  # kitsune-tales-e4b-jp
MODEL_NAME_EN = model_name("en")  # Kitsune-Tales-E4B-EN
MODEL_SLUG_EN = model_slug("en")  # kitsune-tales-e4b-en
WANDB_PROJECT = "kitsune-tales"
HF_NAMESPACE = "whoashish115"
HF_MODEL_REPO = f"{HF_NAMESPACE}/{MODEL_SLUG}"
HF_ADAPTER_REPO = f"{HF_NAMESPACE}/{MODEL_SLUG}-lora"
HF_GGUF_REPO = f"{HF_NAMESPACE}/{MODEL_SLUG}-gguf"
HF_DATASET_REPO = f"{HF_NAMESPACE}/kitsune-tales-jp-fantasy-sft"
HF_MODEL_REPO_EN = f"{HF_NAMESPACE}/{MODEL_SLUG_EN}"
HF_ADAPTER_REPO_EN = f"{HF_NAMESPACE}/{MODEL_SLUG_EN}-lora"
HF_GGUF_REPO_EN = f"{HF_NAMESPACE}/{MODEL_SLUG_EN}-gguf"
HF_DATASET_REPO_EN = f"{HF_NAMESPACE}/kitsune-tales-en-fantasy-sft"
HF_SPACE_REPO = f"{HF_NAMESPACE}/kitsune-tales"

# llama.cpp release v0.5.0 (2026-09-23); registers Qwen3_5ForCausalLM in conversion/qwen.py.
LLAMA_CPP_COMMIT = "7fe450e19305b828c199d602c23a8337aaa1f03b"

MODEL_LICENSES: dict[str, str] = {
    BASE_MODEL: "apache-2.0",
    ALT_BASE_MODEL: "apache-2.0",
    GENERATOR_MODEL: "apache-2.0",
    GENERATOR2_MODEL: "apache-2.0",
    JUDGE_MODEL: "apache-2.0",
}
