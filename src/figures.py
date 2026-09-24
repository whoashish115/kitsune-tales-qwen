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
    return _load(R / ("results.json" if lang == "ja" else "results_en.json"))


def _generations(lang: str, system: str) -> list[dict]:
    path = R / ("generations" if lang == "ja" else "generations_en") / f"{system}.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [r for r in map(json.loads, f) if r["suite"] == "test"]


def _length(text: str, lang: str) -> int:
    if lang == "ja":
        from kitsune.taxonomy import count_chars

        return count_chars(text)
    from kitsune.en import count_words

    return count_words(text)


def _band(fmt: str, lang: str) -> tuple[int, int]:
    from kitsune.en import FORMATS_EN
    from kitsune.taxonomy import FORMATS

    spec = FORMATS[fmt] if lang == "ja" else FORMATS_EN[fmt]
    return spec.target_min, spec.target_max


def _no_ygrid(ax: Any) -> None:
    ax.grid(axis="y", visible=False)


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


REASON = {
    "length:too_long": "too long",
    "length:too_short": "too short",
    "tag_consistency:no_cue": "no cue for a requested genre",
    "tag_consistency:tags": "genre tags inconsistent",
    "tag_consistency:title_not_reflected": "title not reflected",
    "japanese_purity:simplified_chinese": "simplified Chinese characters",
    "japanese_purity:non_jis_kanji": "non-JIS kanji",
    "fantasy_rule:few_fantasy_terms": "too few fantasy terms",
    "title_clean:simplified_chinese": "Chinese characters in title",
    "generation:truncated": "truncated generation",
    "no_artifacts:artifact": "markdown / meta artifacts",
    "llm_label:not_fantasy": "LLM label: not fantasy",
    "llm_label:not_general_audience": "LLM label: not general audience",
    "safety_rule:unsafe": "safety lexicon",
    "self_correction:self_correction": "self-correction in the text",
}


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


SFT_RUNS = [
    ("sft-pilot", "pilot, 5 % data", "muted", "-"),
    ("abl-data10", "10 % data", "seq0", "-"),
    ("abl-data30", "30 % data", "seq1", "-"),
    ("sft-main", "100 % data (released JP)", "seq2", "-"),
    ("abl-r16", "25 % data, r = 16", "qwen3.5-4b", "--"),
    ("abl-r64", "25 % data, r = 64", "qwen3.5-9b", "--"),
    ("sft-en-main", "English, 100 % data", "kitsune-en-sft", ":"),
]


def _run_color(key: str) -> str:
    if key == "muted":
        return P["muted"]
    if key.startswith("seq"):
        return P["seq"][int(key[3:])]
    return C(key)


def sft_runs() -> Any:
    n_train = _load(Path("reports/site/kitsune.json"))["training"]["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5), sharey=True)
    for run, name, ck, ls in SFT_RUNS:
        n = n_train[run]["train_examples"]
        h = _hist(run)
        tr = np.array([(x["epoch"] * n, x["loss"]) for x in h if "loss" in x and "eval_loss" not in x]).T
        axes[0].plot(tr[0], _ema(tr[1], 0.25), color=_run_color(ck), lw=1.8, ls=ls, label=name)
        ev = np.array([(x["epoch"] * n, x["eval_loss"]) for x in h if "eval_loss" in x]).T
        axes[1].plot(ev[0], ev[1], "o", color=_run_color(ck), ls=ls, lw=1.4, ms=4, label=name)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("training examples seen (log scale)")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axes[0].set_ylabel("cross-entropy on assistant tokens")
    axes[0].set_title("(a) training loss (EMA)")
    axes[1].set_title("(b) validation loss (JP runs share one split; the pilot used half of it)")
    axes[1].legend(loc="upper right", fontsize=7.8)
    fig.tight_layout()
    return fig


def ablations() -> Any:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2))
    for run, name, ck in (
        ("abl-data10", "10 % data", "seq0"),
        ("abl-data30", "30 % data", "seq1"),
        ("sft-main", "100 % data", "seq2"),
    ):
        ev = _evals(run)
        axes[0].plot(
            [x["epoch"] for x in ev],
            [x["eval_loss"] for x in ev],
            "o-",
            color=_run_color(ck),
            ms=3.8,
            lw=1.6,
            label=name,
        )
    axes[0].set_title("(a) validation loss during one epoch, r = 32")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("validation loss")
    axes[0].legend()
    runs = _results("ja")["training"]
    fr = [0.1, 0.3, 1.0]
    ls = [runs["abl-data10"]["eval_loss"], runs["abl-data30"]["eval_loss"], runs["sft-main"]["eval_loss"]]
    axes[1].plot(fr, ls, "o-", color=P["seq"][2], lw=2, ms=6, label="r = 32, data scaling")
    for f, v in zip(fr, ls, strict=True):
        axes[1].annotate(f"{v:.3f}", (f, v), xytext=(6, 4), textcoords="offset points", fontsize=8.5)
    for run, m, ck in (("abl-r16", "s", "qwen3.5-4b"), ("abl-r64", "D", "qwen3.5-9b")):
        v = runs[run]["eval_loss"]
        axes[1].plot([0.25], [v], m, color=C(ck), ms=7, label=f"r = {run.split('-')[1][1:]}, 25 % data")
        axes[1].annotate(
            f"{v:.3f}", (0.25, v), xytext=(-8, -3), textcoords="offset points", fontsize=8.5, ha="right"
        )
    axes[1].set_xscale("log")
    axes[1].set_xticks([0.1, 0.25, 0.3, 1.0], ["10 %", "", "30 %", "100 %"])
    axes[1].minorticks_off()
    axes[1].set_xlim(0.07, 1.4)
    axes[1].set_ylim(1.17, 1.39)
    axes[1].set_title("(b) final validation loss vs data share and rank")
    axes[1].set_xlabel("share of the Japanese training set (log scale; ranks at 25 %)")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


# =========================================================================== DPO

DPO_RUNS = (
    ("dpo-main-v2", "JP DPO v2, 1,709 pairs (85 held out)", "kitsune"),
    ("dpo-en-main", "EN DPO, 1,596 pairs incl. 135 safety (79 held out)", "kitsune-en"),
)


def dpo_training() -> Any:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for run, name, sid in DPO_RUNS:
        s, y = _train(run, "loss")
        axes[0].plot(s, y, color=C(sid), alpha=0.3, lw=1)
        axes[0].plot(s, _ema(y, 0.3), color=C(sid), lw=2, label=name)
        ev = _evals(run)
        es = [x["step"] for x in ev]
        for ax, key, fmt in (
            (axes[1], "eval_rewards/accuracies", "{:.0%}"),
            (axes[2], "eval_rewards/margins", "{:.2f}"),
        ):
            vals = [x[key] for x in ev]
            ax.plot(es, vals, "o-", color=C(sid), ms=5, lw=1.4)
            ax.annotate(
                fmt.format(vals[-1]),
                (es[-1], vals[-1]),
                xytext=(5, -3),
                textcoords="offset points",
                fontsize=8,
            )
    axes[0].axhline(np.log(2), color=P["muted"], lw=1, ls="--")
    axes[0].text(
        0.99,
        np.log(2) + 0.012,
        "ln 2 = no preference",
        color=P["muted"],
        fontsize=8,
        ha="right",
        transform=axes[0].get_yaxis_transform(),
    )
    axes[1].axhline(0.5, color=P["muted"], lw=1, ls="--")
    axes[1].text(
        0.99,
        0.505,
        "chance",
        color=P["muted"],
        fontsize=8,
        ha="right",
        transform=axes[1].get_yaxis_transform(),
    )
    axes[0].set_title("(a) DPO loss, train")
    axes[1].set_title("(b) held-out preference accuracy")
    axes[2].set_title("(c) held-out reward margin")
    for ax in axes:
        ax.set_xlabel("optimizer step (effective batch 16)")
    _pct_axis(axes[1], "y")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    return fig


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

ROWS = [
    ("test", "length_ok", "Requested length ↑"),
    ("test", "markdown", "Markdown artifacts ↓"),
    ("test", "degenerate", "Degenerate outputs ↓"),
    ("test", "repetitive", "Repetitive ↓"),
    ("policy", "refusal_rate_disallowed", "Refuses disallowed ↑"),
    ("policy", "violation_rate_disallowed", "Policy violations ↓"),
    ("policy", "fantasy_rate_offgenre", "Off-genre → fantasy ↑"),
    ("policy", "fantasy_rate_adversarial", "Adversarial → fantasy ↑"),
]


def _metric_panel(ax: Any, res: dict, systems: list[str], title: str) -> None:
    off = np.linspace(-0.3, 0.3, len(systems))
    for i, (part, key, _) in enumerate(ROWS):
        for j, s in enumerate(systems):
            c = res["systems"][s][part][key]
            lo, hi = max(0.0, c["mean"] - c["low"]), max(0.0, c["high"] - c["mean"])
            ax.errorbar(
                c["mean"], i + off[j], xerr=[[lo], [hi]], fmt="o", color=C(s), ms=5, elinewidth=1.4, capsize=0
            )
    ax.set_yticks(range(len(ROWS)), [r[2] for r in ROWS])
    ax.set_ylim(len(ROWS) - 0.5, -0.5)
    ax.set_xlim(-0.03, 1.03)
    _pct_axis(ax)
    ax.set_xlabel("share of outputs, mean and 95 % bootstrap CI over prompts")
    ax.set_title(title)
    _no_ygrid(ax)
    for i in range(len(ROWS) - 1):
        ax.axhline(i + 0.5, color=P["grid"], lw=0.7)


def eval_metrics() -> Any:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharey=True, gridspec_kw={"width_ratios": [1.25, 1]})
    _metric_panel(
        axes[0],
        _results("ja"),
        SYSTEMS_JA,
        "Japanese: 270 test prompts × 3 seeds; 72 policy prompts × 3 seeds",
    )
    _metric_panel(axes[1], _results("en"), SYSTEMS_EN, "English: same design")
    for ax, systems in ((axes[0], SYSTEMS_JA), (axes[1], SYSTEMS_EN)):
        handles = [
            plt.Line2D([], [], marker="o", ls="", color=C(s), label=LABEL[s]) for s in reversed(systems)
        ]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


def eval_lengths() -> Any:
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.4))
    for row, (lang, systems, unit) in enumerate(
        (
            ("ja", ["base", "teacher", "kitsune-sft"], "characters"),
            ("en", ["base-en", "kitsune-en-sft", "kitsune-en"], "words"),
        )
    ):
        gens = {s: _generations(lang, s) for s in systems}
        for col, fmt in enumerate(("あらすじ", "短編", "続き")):
            ax = axes[row, col]
            lo, hi = _band(fmt, lang)
            ax.axvspan(lo, hi, color=P["sys"]["kitsune-sft"], alpha=0.1, lw=0)
            data = {s: [_length(r["text"], lang) for r in gens[s] if r["format"] == fmt] for s in systems}
            top = np.percentile(np.concatenate(list(data.values())), 99)
            bins = np.linspace(0, top, 42)
            for s, d in data.items():
                inside = np.mean([(lo <= v <= hi) for v in d])
                ax.hist(
                    d,
                    bins=bins,
                    histtype="step",
                    lw=1.8,
                    color=C(s),
                    label=f"{SHORT[s]}: {inside:.0%} in range",
                    density=True,
                )
            ax.set_yticks([])
            _no_ygrid(ax)
            ax.set_title(
                f"{'JP' if lang == 'ja' else 'EN'} {FMT_EN[fmt]}, target {lo:,}–{hi:,} {unit}", fontsize=9
            )
            ax.set_xlabel(unit)
            ax.legend(loc="upper right", fontsize=7.2)
    fig.tight_layout()
    return fig


def _length_grid(lang: str, system: str) -> tuple[list[str], np.ndarray]:
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in _generations(lang, system):
        lo, hi = _band(r["format"], lang)
        cells[(r["genres"][0], r["format"])].append(float(lo <= _length(r["text"], lang) <= hi))
    genres = list(dict.fromkeys(g for g, _ in sorted(cells)))
    from kitsune.taxonomy import GENRES

    genres = [g for g in GENRES if any(k[0] == g for k in cells)]
    grid = np.array([[np.mean(cells[(g, f)]) if cells[(g, f)] else np.nan for f in FMT_EN] for g in genres])
    return genres, grid


def eval_length_grid() -> Any:
    from kitsune.en import GENRE_NAME_EN

    cmap = LinearSegmentedColormap.from_list("k", P["heat"])
    panels = (
        ("ja", "base", "JP base"),
        ("ja", "kitsune-sft", "JP released"),
        ("en", "base-en", "EN base"),
        ("en", "kitsune-en", "EN released"),
    )
    fig, axes = plt.subplots(1, 4, figsize=(12, 4.2), sharey=True)
    for ax, (lang, s, title) in zip(axes, panels, strict=True):
        genres, grid = _length_grid(lang, s)
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid[i, j]
                ax.text(
                    j,
                    i,
                    f"{v:.0%}",
                    ha="center",
                    va="center",
                    fontsize=7.8,
                    color=P["heat"][0] if v > 0.55 else P["ink"],
                )
        ax.set_xticks(range(3), [FMT_EN[f] for f in FMT_EN], fontsize=8)
        ax.set_yticks(range(len(genres)), [GENRE_NAME_EN[g] for g in genres])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        m = float(np.nanmean(grid))
        ax.set_title(f"{title}: {m:.1%} in range" if m < 0.1 else f"{title}: {m:.0%} in range")
    fig.tight_layout()
    return fig


def eval_diversity() -> Any:
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.4), gridspec_kw={"width_ratios": [1.6, 1]})
    for row, (lang, systems, unit) in enumerate(
        (("ja", SYSTEMS_JA, "character"), ("en", SYSTEMS_EN, "word"))
    ):
        res = _results(lang)["systems"]
        x = np.arange(len(systems))
        ax = axes[row, 0]
        for k, n in enumerate((2, 3, 4)):
            vals = [res[s]["diversity"][f"corpus_distinct_{n}"] for s in systems]
            ax.bar(x + (k - 1) * 0.26, vals, 0.24, color=P["seq"][k], zorder=3, label=f"distinct-{n}")
        ax.set_xticks(x, [SHORT[s] for s in systems], fontsize=7.8)
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"unique {unit} n-grams / all")
        ax.set_title(
            f"({'a' if row == 0 else 'c'}) {'Japanese' if lang == 'ja' else 'English'}: corpus distinct-n ↑"
        )
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left", ncol=3, fontsize=8)
        ax2 = axes[row, 1]
        for i, s in enumerate(systems):
            c = res[s]["test"]["self_bleu"]
            ax2.errorbar(
                c["mean"],
                i,
                xerr=[[c["mean"] - c["low"]], [c["high"] - c["mean"]]],
                fmt="o",
                color=C(s),
                ms=5,
                capsize=0,
            )
        ax2.set_yticks(range(len(systems)), [SHORT[s] for s in systems], fontsize=7.8)
        ax2.set_ylim(len(systems) - 0.5, -0.5)
        ax2.set_title(f"({'b' if row == 0 else 'd'}) self-BLEU across seeds ↓")
        ax2.set_xlabel("mean, 95 % CI")
        _no_ygrid(ax2)
    fig.tight_layout()
    return fig


# =========================================================================== judge

JUDGE_ROWS = {
    "ja": [
        ("kitsune-sft_vs_base", "released vs base, full outputs"),
        ("excerpt-kitsune-sft_vs_excerpt-base", "released vs base, equal-length openings"),
        ("kitsune-sft_vs_teacher", "released vs teacher (35B)"),
        ("kitsune_vs_kitsune-sft", "DPO v2 vs released SFT"),
        ("kitsune_vs_base", "DPO v2 vs base"),
        ("kitsune_vs_teacher", "DPO v2 vs teacher (35B)"),
    ],
    "en": [
        ("kitsune-en_vs_base-en", "released vs base, full outputs"),
        ("excerpt-kitsune-en_vs_excerpt-base-en", "released vs base, equal-length openings"),
        ("kitsune-en_vs_kitsune-en-sft", "released (SFT + DPO) vs SFT"),
    ],
}


def _judge(lang: str) -> list[tuple[dict, str]]:
    res = _results(lang)["judge"]
    return [(res[f"judge:{k}"], name) for k, name in JUDGE_ROWS[lang] if f"judge:{k}" in res]


def judge_preference() -> Any:
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True, gridspec_kw={"height_ratios": [6, 3]})
    for ax, lang in zip(axes, ("ja", "en"), strict=True):
        rows = _judge(lang)
        for i, (v, name) in enumerate(rows):
            c = v["net_preference"]
            lm = "equal-length" in name
            col = P["sys"]["kitsune-sft"] if lm else P["ink2"]
            ax.errorbar(
                c["mean"],
                i,
                xerr=[[c["mean"] - c["low"]], [c["high"] - c["mean"]]],
                fmt="D" if lm else "o",
                color=col,
                ms=6,
                elinewidth=1.6,
                capsize=0,
            )
            ax.annotate(
                f"{c['mean']:+.2f}  [{c['low']:+.2f}, {c['high']:+.2f}]   n = {v['n_pairs']}",
                (c["high"], i),
                xytext=(6, -3),
                textcoords="offset points",
                fontsize=8,
                color=P["ink2"],
            )
        ax.set_yticks(range(len(rows)), [r[1] for r in rows])
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.axvline(0, color=P["ink"], lw=0.9)
        ax.set_xlim(-1, 1)
        _no_ygrid(ax)
        val = _results(lang)["judge"]["judge:validation"]["accuracy_all"]["mean"]
        ax.set_title(
            f"{'Japanese' if lang == 'ja' else 'English'} (judge accuracy on known-answer pairs: {val:.1%})"
        )
    axes[1].set_xlabel(
        "← prefers the second system      net preference = P(win) − P(loss)      prefers the first system →"
    )
    fig.tight_layout()
    return fig


def judge_outcomes() -> Any:
    rows = [(f"JP  {n}", v) for v, n in _judge("ja")] + [(f"EN  {n}", v) for v, n in _judge("en")]
    fig, ax = plt.subplots(figsize=(9.5, 4))
    for i, (_, v) in enumerate(rows):
        w, t, lo = v["win_rate"], v["tie_rate"], v["loss_rate"]
        ax.barh(i, w, color=P["sys"]["kitsune-sft"], height=0.64, zorder=3)
        ax.barh(i, t, left=w, color=P["neutral"], height=0.64, zorder=3)
        ax.barh(i, lo, left=w + t, color=P["sys"]["base"], height=0.64, zorder=3)
        for x0, val in ((w / 2, w), (w + t / 2, t), (w + t + lo / 2, lo)):
            if val >= 0.07:
                ax.text(
                    x0,
                    i,
                    f"{val:.0%}",
                    ha="center",
                    va="center",
                    fontsize=7.8,
                    color="white" if val is not t else P["ink"],
                )
        ax.text(
            1.01,
            i,
            f"order-consistent {v['position_consistency']:.0%}",
            va="center",
            fontsize=7.8,
            color=P["muted"],
        )
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8.2)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0, 1)
    _pct_axis(ax)
    _no_ygrid(ax)
    ax.axhline(5.5, color=P["muted"], lw=0.8)
    ax.legend(
        handles=[Patch(color=P["sys"]["kitsune-sft"], label="first system wins (both orders)"), Patch(color=P["neutral"], label="tie or order-inconsistent"), Patch(color=P["sys"]["base"], label="second system wins")],
        loc="upper center", bbox_to_anchor=(0.45, -0.08), ncol=3, fontsize=8,
    )  # fmt: skip
    fig.tight_layout()
    return fig


CORRUPT = {
    "chinese": "Chinese mixed in (JP)",
    "script_leak": "other script mixed in (EN)",
    "loop": "paragraph loop",
    "shuffle": "sentences shuffled",
    "truncate": "cut off mid-story",
    "wrong_story": "story for another request",
}


def judge_validation() -> Any:
    fig, ax = plt.subplots(figsize=(8.4, 3))
    keys = list(CORRUPT)
    x = np.arange(len(keys) + 1)
    for k, (lang, col) in enumerate((("ja", "kitsune-sft"), ("en", "kitsune-en-sft"))):
        v = _results(lang)["judge"]["judge:validation"]
        by = v["accuracy_by_corruption"]
        vals = [by.get(c, np.nan) for c in keys] + [v["accuracy_all"]["mean"]]
        xs = x + (k - 0.5) * 0.36
        ax.bar(
            xs,
            vals,
            0.34,
            color=C(col),
            zorder=3,
            label=f"{'Japanese' if lang == 'ja' else 'English'} judge prompts (n = {v['n']})",
        )
        for xi, vi in zip(xs, vals, strict=True):
            if not np.isnan(vi):
                ax.text(xi, vi + 0.02, f"{vi:.0%}", ha="center", fontsize=7.5, color=P["ink2"])
    ax.axhline(0.5, color=P["muted"], lw=1, ls="--")
    ax.set_xticks(x, [CORRUPT[c].replace(" (", "\n(") for c in keys] + ["all pairs"], fontsize=7.8)
    ax.set_ylim(0, 1.12)
    _pct_axis(ax, "y")
    ax.set_ylabel("picks the intact story")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


# =========================================================================== safety, side effects, compute


def safety() -> Any:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.3))
    for ax, lang, systems, title in (
        (axes[0], "ja", SYSTEMS_JA, "Japanese, 45 disallowed prompts × 3 seeds"),
        (axes[1], "en", SYSTEMS_EN, "English, 45 × 3"),
    ):
        res = _results(lang)
        for i, s in enumerate(systems):
            p = res["systems"][s]["policy"]
            ref, vio = p["refusal_rate_disallowed"]["mean"], p["violation_rate_disallowed"]["mean"]
            red = max(0.0, 1 - ref - vio)
            ax.barh(i, ref, color=P["good"], height=0.62, zorder=3)
            ax.barh(i, red, left=ref, color=P["neutral"], height=0.62, zorder=3)
            ax.barh(i, vio, left=ref + red, color=P["bad"], height=0.62, zorder=3)
            ax.text(1.01, i, f"{vio:.0%} violations", va="center", fontsize=8, color=P["ink2"])
        ax.set_yticks(range(len(systems)), [SHORT[s] for s in systems], fontsize=8)
        ax.set_ylim(5.5, -0.5)
        ax.set_xlim(0, 1)
        _pct_axis(ax)
        ax.set_title(title)
        _no_ygrid(ax)
    fig.legend(
        handles=[Patch(color=P["good"], label="refused"), Patch(color=P["neutral"], label="safe fantasy redirect"), Patch(color=P["bad"], label="violation: uses the real person / IP, or unsafe")],
        loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.09),
    )  # fmt: skip
    fig.tight_layout()
    return fig


def lmeval() -> Any:
    lm = _load(R / "lm_eval_summary.json")
    tasks = {
        "ja_leaderboard_jcommonsenseqa": "JCommonsenseQA",
        "ja_leaderboard_jnli": "JNLI",
        "ja_leaderboard_marc_ja": "MARC-ja",
        "ja_leaderboard_xwinograd": "XWinograd-ja",
    }
    fig, ax = plt.subplots(figsize=(6.8, 3))
    x = np.arange(len(tasks))
    for s, dx in (("base", -0.18), ("kitsune-sft", 0.18)):
        m = [lm[s][t]["acc,none"] for t in tasks]
        e = [lm[s][t]["acc_stderr,none"] for t in tasks]
        ax.bar(
            x + dx,
            m,
            0.34,
            yerr=e,
            color=C(s),
            label=LABEL[s],
            error_kw={"elinewidth": 1, "ecolor": P["ink2"]},
            zorder=3,
        )
        for xi, mi in zip(x + dx, m, strict=True):
            ax.text(xi, mi + 0.035, f"{mi:.1%}", ha="center", fontsize=7.5, color=P["ink2"])
    ax.set_xticks(x, list(tasks.values()))
    ax.set_ylim(0, 1.08)
    _pct_axis(ax, "y")
    ax.set_ylabel("accuracy (± 1 SE, 500 items)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=8, ncol=2, bbox_to_anchor=(0, 1.13))
    fig.tight_layout()
    return fig


def ppl_leakage() -> Any:
    site = _load(Path("reports/site/kitsune.json"))["eval"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.1))
    groups = (("ja", ["base", "kitsune-sft", "kitsune"]), ("en", ["base-en", "kitsune-en-sft", "kitsune-en"]))
    ax = axes[0]
    xpos = 0.0
    ticks, labels = [], []
    for lang, systems in groups:
        for s in systems:
            v = site[lang]["ppl"][s]["ppl"]
            ax.bar(xpos, v, 0.7, color=C(s), zorder=3)
            ax.text(xpos, v + 0.12, f"{v:.2f}", ha="center", fontsize=8, color=P["ink2"])
            ticks.append(xpos)
            labels.append(SHORT[s].replace(" (", "\n("))
            xpos += 1
        xpos += 0.6
    ax.set_xticks(ticks, labels, fontsize=7.2)
    ax.set_ylabel("perplexity on the validation split")
    ax.set_title(
        f"(a) validation perplexity (JP {site['ja']['ppl']['base']['n_tokens']:,} tokens, EN {site['en']['ppl']['base-en']['n_tokens']:,})"
    )
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    xpos = 0.0
    ticks, labels = [], []
    for lang, systems in groups:
        for s in systems:
            lk = site[lang]["leakage"].get(s)
            if not lk:
                continue
            ax.bar(xpos, lk["overlap_rate_mean"], 0.7, color=C(s), zorder=3)
            ax.text(
                xpos,
                lk["overlap_rate_mean"] + 0.0006,
                f"{lk['overlap_rate_mean']:.1%}\nmax {lk['max_span_max']:.0f}",
                ha="center",
                fontsize=7.4,
                color=P["ink2"],
            )
            ticks.append(xpos)
            labels.append(SHORT[s].replace(" (", "\n("))
            xpos += 1
        xpos += 0.6
    ax.set_xticks(ticks, labels, fontsize=7.2)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax.set_ylim(0, 0.024)
    ax.set_ylabel("32-char windows found in training data")
    ax.set_title("(b) verbatim overlap with training stories (max = longest span, chars)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def compute() -> Any:
    site = _load(Path("reports/site/kitsune.json"))["budget"]
    ph = sorted(site["phases"], key=lambda p: p["usd"])
    fig, ax = plt.subplots(figsize=(6.8, 3))
    ax.barh(
        [p["phase"] for p in ph], [p["usd"] for p in ph], color=P["sys"]["kitsune-sft"], height=0.62, zorder=3
    )
    total = sum(p["usd"] for p in ph)
    for i, p in enumerate(ph):
        ax.text(
            p["usd"] + 0.12,
            i,
            f"${p['usd']:.2f}  ({p['usd'] / total:.0%})",
            va="center",
            fontsize=8.2,
            color=P["ink2"],
        )
    ax.set_xlabel(f"USD, per-job ledger (sum ${total:.2f}; Modal billed ${site['total_billed']:.2f})")
    ax.set_xlim(0, max(p["usd"] for p in ph) * 1.35)
    _no_ygrid(ax)
    fig.tight_layout()
    return fig


FIGURES = [
    data_funnel, data_rejections, sft_loss, sft_dynamics, sft_runs, ablations, dpo_training, dpo_rewards,
    eval_metrics, eval_lengths, eval_length_grid, eval_diversity, judge_preference, judge_outcomes, judge_validation,
    safety, lmeval, ppl_leakage, compute,
]  # fmt: skip


def main() -> None:
    for theme in ("light", "dark"):
        _style(theme)
        for f in FIGURES:
            _save(f(), f.__name__, theme)
    print(f"wrote {len(FIGURES)} figures × 2 themes to {OUT}")


if __name__ == "__main__":
    main()
