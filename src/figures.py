"""Paper figures for the report, the model cards and the project site.

    python -m kitsune.figures        # reports/figures/<name>.{svg,png} (light) and reports/figures/dark/<name>.svg
Inputs are the evaluation files in ``reports/`` and the trainer logs in ``reports/train_logs/`` (``trainer_state.json`` of
every run); the compute figure reads the phase ledger from the site export. Nothing is re-estimated here: every
interval is the 95 % percentile bootstrap that ``kitsune.eval.report`` computed over prompts.
Conventions. Each system keeps one color in every figure (a categorical set that passes the dataviz CVD checks in
both themes); figure numbers live in the captions, not inside the images, so the report and the site can order them
independently. Light SVGs have a transparent background for the site, PNGs a white one for README and model cards.
"""
from __future__ import annotations
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, PercentFormatter
R = Path("reports")
OUT = R / "figures"
THEMES: dict[str, dict[str, Any]] = {
    "light": {
        "ink": "#1a1a2e", "ink2": "#45465e", "muted": "#777995", "grid": "#e1e1f0",
        "good": "#0ca30c", "bad": "#d03b3b", "neutral": "#c3c4d6",
        "seq": ("#9ec5f4", "#5598e7", "#1c5cab"), "heat": ("#f3f6fc", "#1c5cab"),
        "sys": {
            "kitsune-sft": "#2a78d6", "kitsune-en": "#2a78d6", "base": "#eb6834", "base-en": "#eb6834",
            "kitsune": "#1baf7a", "kitsune-en-sft": "#1baf7a", "qwen3.5-4b": "#eda100", "qwen3.5-9b": "#e87ba4",
            "teacher": "#008300",
        },
    },
    "dark": {
        "ink": "#ecebfa", "ink2": "#c3c3dc", "muted": "#8b8cab", "grid": "#2a2b48",
        "good": "#3ccf7a", "bad": "#ff7b6b", "neutral": "#5a5c78",
        "seq": ("#1f4f8f", "#3987e5", "#9cc3f5"), "heat": ("#1b1d38", "#8fbaf2"),
        "sys": {
            "kitsune-sft": "#3987e5", "kitsune-en": "#3987e5", "base": "#d95926", "base-en": "#d95926",
            "kitsune": "#199e70", "kitsune-en-sft": "#199e70", "qwen3.5-4b": "#c98500", "qwen3.5-9b": "#d55181",
            "teacher": "#3f9f3f",
        },
    },
}  # fmt: skip
P: dict[str, Any] = dict(THEMES["light"])
def _save(fig: Any, name: str, theme: str) -> None:
    if theme == "light":
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", transparent=True)
        fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    else:
        (OUT / "dark").mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / "dark" / f"{name}.svg", bbox_inches="tight", transparent=True)
    plt.close(fig)

def _load(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))

def _train(run: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    rows = [(x["step"], x[key]) for x in _hist(run) if key in x and "eval_loss" not in x]
    return tuple(np.array(rows).T)

def _evals(run: str) -> list[dict]:
    return [x for x in _hist(run) if "eval_loss" in x]

def _ema(y: np.ndarray, a: float = 0.15) -> np.ndarray:
    out = np.empty_like(y, dtype=float)
    for i, v in enumerate(y):
        out[i] = v if i == 0 else a * v + (1 - a) * out[i - 1]
    return out

def _results(lang: str) -> dict:
    raise NotImplementedError

def _generations(lang: str, system: str) -> list[dict]:
    raise NotImplementedError

def _pct_axis(ax: Any, axis: str = "x") -> None:
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(PercentFormatter(1.0, decimals=0))

def data_rejections() -> Any:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
    for ax, (lang, path) in zip(
        axes,
        (("Japanese", R / "data" / "stats.json"), ("English", R / "data_en" / "stats.json")),
        strict=True,
    ):
        d = _load(path)
        top = sorted(d["drop_reasons"].items(), key=lambda kv: -kv[1])[:10]
        y = np.arange(len(top))
        share = [v / d["n_generations"] for _, v in top]
        cols = [P["sys"]["kitsune"] if k.startswith("llm_label") else P["muted"] for k, _ in top]
        ax.barh(y, share, color=cols, height=0.62, zorder=3)
        for yi, (_, v), s in zip(y, top, share, strict=True):
            ax.text(s + max(share) * 0.02, yi, f"{v:,}", va="center", fontsize=8, color=P["ink2"])
        ax.set_yticks(y, [REASON.get(k, k.replace("_", " ")) for k, _ in top], fontsize=8.2)
        ax.set_ylim(len(top) - 0.5, -0.5)
        ax.set_xlim(0, max(share) * 1.2)
        _pct_axis(ax)
        ax.set_xlabel("share of all generations (a story can fail several checks)")
        ax.set_title(f"{lang}: ten most frequent rejection reasons")
        _no_ygrid(ax)
    fig.legend(
        handles=[Patch(color=P["muted"], label="rule filter"), Patch(color=P["sys"]["kitsune"], label="cross-model LLM label")],
        loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06),
    )  # fmt: skip
    fig.tight_layout()
    return fig

def _run_color(key: str) -> str:
    raise NotImplementedError
