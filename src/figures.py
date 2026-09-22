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
LABEL = {
    "base": "Gemma 4 E4B (base)",
    "base-en": "Gemma 4 E4B (base)",
    "kitsune-sft": "kitsune-tales-e4b-jp (SFT, released)",
    "kitsune": "JP SFT + DPO v2 (not released)",
    "kitsune-en-sft": "EN SFT",
    "kitsune-en": "kitsune-tales-e4b-en (SFT + DPO, released)",
    "qwen3.5-4b": "Qwen3.5-4B",
    "qwen3.5-9b": "Qwen3.5-9B",
    "teacher": "Teacher: Qwen3.6-35B-A3B",
}
SHORT = {
    **LABEL,
    "kitsune-sft": "JP released (SFT)",
    "kitsune-en": "EN released (SFT + DPO)",
    "kitsune": "JP SFT + DPO v2",
    "teacher": "Teacher (35B)",
}
SYSTEMS_JA = ["base", "qwen3.5-4b", "qwen3.5-9b", "teacher", "kitsune", "kitsune-sft"]
SYSTEMS_EN = ["base-en", "kitsune-en-sft", "kitsune-en"]
FMT_EN = {"あらすじ": "synopsis", "短編": "short story", "続き": "continuation"}

def C(system: str) -> str:
    return P["sys"][system]

def _style(theme: str) -> None:
    P.clear()
    P.update(THEMES[theme])
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlecolor": P["ink"],
            "axes.labelcolor": P["ink2"],
            "axes.edgecolor": P["muted"],
            "axes.facecolor": "none",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": P["grid"],
            "grid.linewidth": 0.7,
            "xtick.color": P["ink2"],
            "ytick.color": P["ink2"],
            "text.color": P["ink"],
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "legend.labelcolor": P["ink2"],
            "figure.facecolor": "none",
            "svg.fonttype": "none",
        }
    )

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

def _hist(run: str) -> list[dict]:
    return _load(R / "train_logs" / f"{run}.json")["log_history"]

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

def _band(fmt: str, lang: str) -> tuple[int, int]:
    from kitsune.en import FORMATS_EN
    from kitsune.taxonomy import FORMATS

    spec = FORMATS[fmt] if lang == "ja" else FORMATS_EN[fmt]
    return spec.target_min, spec.target_max

def _pct_axis(ax: Any, axis: str = "x") -> None:
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(PercentFormatter(1.0, decimals=0))

# =========================================================================== data

def data_funnel() -> Any:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 2.9))
    for ax, (lang, path) in zip(
        axes,
        (("Japanese", R / "data" / "stats.json"), ("English", R / "data_en" / "stats.json")),
        strict=True,
    ):
        d = _load(path)
        drops = d["drop_reasons"]
        llm = sum(v for k, v in drops.items() if k.startswith("llm_label"))
        rules = sum(drops.values()) - llm
        stages = [
            ("generated", d["n_generations"]),
            ("pass rule filters", d["n_generations"] - rules),
            ("pass cross-model labels", d["n_after_filters"]),
            ("after MinHash dedup", d["n_kept_synthetic"]),
            ("train split (synthetic)", d["n_train"] - d.get("n_refusal_templates", 0)),
        ]
        y = np.arange(len(stages))
        ax.barh(y, [s[1] for s in stages], color=P["sys"]["kitsune-sft"], height=0.62, zorder=3)
        for yi, (_, v) in zip(y, stages, strict=True):
            ax.text(
                v + d["n_generations"] * 0.012,
                yi,
                f"{v:,}  ({v / d['n_generations']:.0%})",
                va="center",
                fontsize=8.5,
                color=P["ink"],
            )
        ax.set_yticks(y, [s[0] for s in stages])
        ax.set_ylim(len(stages) - 0.5, -0.5)
        ax.set_xlim(0, d["n_generations"] * 1.3)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
        ax.set_title(f"{lang}: {d['n_generations']:,} generations from two models")
        _no_ygrid(ax)
    fig.tight_layout()
    return fig

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

# =========================================================================== SFT

def sft_loss() -> Any:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2), sharey=True)
    for ax, (run, title, sid) in zip(
        axes, (("sft-main", "Japanese SFT, 10,090 examples", "kitsune-sft"), ("sft-en-main", "English SFT, 6,647 examples", "kitsune-en-sft")), strict=True
    ):  # fmt: skip
        s, y = _train(run, "loss")
        ax.plot(s, y, color=C(sid), alpha=0.3, lw=1, label="train loss, every 10 steps")
        ax.plot(s, _ema(y), color=C(sid), lw=2, label="train loss, EMA (α = 0.15)")
        ev = _evals(run)
        es, ey = [x["step"] for x in ev], [x["eval_loss"] for x in ev]
        ax.plot(es, ey, "o-", color=P["ink"], ms=4.5, lw=1.2, mfc=P["ink"], mew=0, label="validation loss")
        ax.annotate(
            f"{ey[-1]:.3f}",
            (es[-1], ey[-1]),
            xytext=(2, 8),
            textcoords="offset points",
            fontsize=8.5,
            color=P["ink"],
        )
        ax.set_title(title)
        ax.set_xlabel("optimizer step (batch 16)")
    axes[0].set_ylabel("cross-entropy on assistant tokens")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    return fig

def sft_dynamics() -> Any:
    runs = (("sft-main", "Japanese", "kitsune-sft"), ("sft-en-main", "English", "kitsune-en-sft"))
    panels = (
        ("learning_rate", None, "(a) learning rate", False),
        ("grad_norm", None, "(b) gradient norm (before clipping at 1.0)", True),
        ("mean_token_accuracy", "eval_mean_token_accuracy", "(c) next-token accuracy", True),
        ("entropy", "eval_entropy", "(d) predictive entropy (nats)", True),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.4))
    for ax, (key, ekey, title, smooth) in zip(axes.flat, panels, strict=True):
        for run, name, sid in runs:
            h = _hist(run)
            rows = [(x["epoch"], x[key]) for x in h if key in x and "eval_loss" not in x]
            ep, y = np.array(rows).T
            if smooth:
                ax.plot(ep, y, color=C(sid), alpha=0.25, lw=1)
                ax.plot(ep, _ema(y, 0.2), color=C(sid), lw=1.8, label=f"{name} train")
            else:
                ax.plot(ep, y, color=C(sid), lw=1.8, label=name)
            if ekey:
                ev = [(x["epoch"], x[ekey]) for x in h if ekey in x]
                ee, eyv = np.array(ev).T
                ax.plot(ee, eyv, "D", color=C(sid), ms=5, mec=P["ink"], mew=0.6, label=f"{name} validation")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        if key == "mean_token_accuracy":
            _pct_axis(ax, "y")
    axes[0, 0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 1e4:.1f}e-4" if v else "0"))
    axes[0, 0].legend(loc="upper right")
    axes[1, 0].legend(loc="lower right", ncol=2, fontsize=8)
    fig.tight_layout()
    return fig

def _run_color(key: str) -> str:
    if key == "muted":
        return P["muted"]
    if key.startswith("seq"):
        return P["seq"][int(key[3:])]
    return C(key)

def sft_runs() -> Any:
    raise NotImplementedError

def ablations() -> Any:
    raise NotImplementedError

DPO_RUNS = (
    ("dpo-main-v2", "JP DPO v2, 1,709 pairs (85 held out)", "kitsune"),
    ("dpo-en-main", "EN DPO, 1,596 pairs incl. 135 safety (79 held out)", "kitsune-en"),
)

def dpo_rewards() -> Any:
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.4), sharex="row")
    for row, (run, _name, sid) in enumerate(DPO_RUNS):
        for col, (a, b, title) in enumerate(
            (
                ("rewards/chosen", "rewards/rejected", "implicit reward β·log(π/π_ref)"),
                ("logps/chosen", "logps/rejected", "sequence log-probability"),
                ("rewards/margins", None, "reward margin, train"),
            )
        ):
            ax = axes[row, col]
            s, y = _train(run, a)
            ax.plot(s, y, color=C(sid), alpha=0.25, lw=1)
            ax.plot(s, _ema(y, 0.3), color=C(sid), lw=2, label="chosen" if b else "margin")
            if b:
                s2, y2 = _train(run, b)
                ax.plot(s2, y2, color=P["muted"], alpha=0.25, lw=1)
                ax.plot(s2, _ema(y2, 0.3), color=P["muted"], lw=2, ls="--", label="rejected")
            ax.axhline(0, color=P["ink2"], lw=0.7) if col != 1 else None
            ax.set_title(f"{'JP v2' if row == 0 else 'EN'}: {title}")
            if row == 1:
                ax.set_xlabel("optimizer step")
        axes[row, 0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return fig

# =========================================================================== automatic metrics
def _metric_panel(ax: Any, res: dict, systems: list[str], title: str) -> None:
    raise NotImplementedError
