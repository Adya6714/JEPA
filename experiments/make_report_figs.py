"""Publication-style figures F1–F4 → figs/final/."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "figs" / "final"

# Colorblind-safe (Okabe–Ito)
PALETTE = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
    "gray": "#999999",
}
METHOD_COLORS = {
    "procrustes": PALETTE["blue"],
    "ridge": PALETTE["orange"],
    "mlp_cos": PALETTE["sky"],
    "mlp_vicreg": PALETTE["green"],
    "infonce_t0.07": PALETTE["vermillion"],
    "infonce_t0.2": PALETTE["purple"],
}
FEATURE_COLORS = {
    "I": PALETTE["blue"],
    "T": PALETTE["orange"],
    "g_proc": PALETTE["sky"],
    "g_ridge": PALETTE["green"],
    "I_g_ridge": PALETTE["vermillion"],
    "R": PALETTE["purple"],
    "IR": PALETTE["black"],
    "R_shuffled_labels": PALETTE["gray"],
}


def setup_style():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": False,
        }
    )


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Wrote {path}")


def fig_f1_leakage():
    data = load("01_leakage.json")
    rows = data["summary"]
    labels = [f"{r['feature']}\n{r['model'].replace('Classifier','').replace('Regression','')}" for r in rows]
    x = np.arange(len(rows))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(
        x - w / 2,
        [r["random_mean"] for r in rows],
        w,
        yerr=[r["random_std"] for r in rows],
        capsize=2,
        color=PALETTE["sky"],
        label="random split",
        edgecolor="none",
    )
    ax.bar(
        x + w / 2,
        [r["group_mean"] for r in rows],
        w,
        yerr=[r["group_std"] for r in rows],
        capsize=2,
        color=PALETTE["orange"],
        label="group split",
        edgecolor="none",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.7, 1.02)
    ax.set_title("F1  Leakage audit: random vs group-aware splits")
    ax.legend(frameon=False, loc="lower right")
    leak = data["leak_fraction_random"]
    ax.text(
        0.02,
        0.98,
        f"Exact-T leak (random): {leak['mean']:.1%}±{leak['std']:.1%}",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        color=PALETTE["gray"],
    )
    save(fig, "F1_leakage.png")


def fig_f2_hubness():
    data = load("02b.json")
    methods = data["methods"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for m in methods:
        raw_x = data["summary"][m]["raw"]["hub_skew"]["mean"]
        raw_y = data["summary"][m]["raw"]["class_R@1"]["mean"]
        cen_x = data["summary"][m]["centered"]["hub_skew"]["mean"]
        cen_y = data["summary"][m]["centered"]["class_R@1"]["mean"]
        color = METHOD_COLORS.get(m, PALETTE["black"])
        ax.annotate(
            "",
            xy=(cen_x, cen_y),
            xytext=(raw_x, raw_y),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2, alpha=0.85),
        )
        ax.scatter([raw_x], [raw_y], s=55, color=color, zorder=3, edgecolors="white", linewidths=0.6)
        ax.scatter(
            [cen_x],
            [cen_y],
            s=45,
            facecolors="white",
            edgecolors=color,
            linewidths=1.4,
            zorder=3,
        )
        # label near raw point
        ax.annotate(
            m.replace("infonce_", "inf."),
            (raw_x, raw_y),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=7.5,
            color=color,
        )
    ax.set_xlabel("Hub skew (top-10 gallery counts)")
    ax.set_ylabel("Class R@1")
    ax.set_title("F2  Hubness vs class retrieval (raw → centered)")
    # Legend proxies
    ax.scatter([], [], s=55, color=PALETTE["gray"], label="raw")
    ax.scatter(
        [],
        [],
        s=45,
        facecolors="white",
        edgecolors=PALETTE["gray"],
        linewidths=1.4,
        label="centered",
    )
    ax.legend(frameon=False, loc="upper right")
    ax.set_xlim(left=-0.2)
    ax.set_ylim(0, 1.02)
    save(fig, "F2_hubness.png")


def fig_f3_complementarity():
    data = load("03_probes.json")["3b"]
    keys = ["I", "T", "R", "IR", "R_shuffled_labels"]
    labels = ["I", "T", "R=T−g(I)", "concat[I,R]", "R (shuf. y)"]
    means = [data[k]["mean"] for k in keys]
    stds = [data[k]["std"] for k in keys]
    colors = [FEATURE_COLORS[k] for k in keys]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    x = np.arange(len(keys))
    ax.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="none")
    ax.axhline(1 / 31, color=PALETTE["gray"], ls="--", lw=1.2, label="chance 1/31")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Test LR accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("F3  Complementarity: what remains in T − g(I)?")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "F3_complementarity.png")


def fig_f4_fewshot():
    data = load("04_fewshot.json")
    ks = data["k_shots"]
    features = data["features"]
    label_map = {
        "I": "I",
        "T": "T (centered)",
        "g_proc": r"$g_{\mathrm{proc}}(I)$",
        "g_ridge": r"$g_{\mathrm{ridge}}(I)$",
        "I_g_ridge": r"$\mathrm{concat}[I,\,g_{\mathrm{ridge}}(I)]$",
    }
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for fname in features:
        means = [data["summary"][fname][str(k)]["mean"] for k in ks]
        stds = [data["summary"][fname][str(k)]["std"] for k in ks]
        ax.errorbar(
            ks,
            means,
            yerr=stds,
            marker="o",
            ms=5,
            capsize=3,
            lw=1.6,
            color=FEATURE_COLORS.get(fname, PALETTE["black"]),
            label=label_map[fname],
        )
    ax.axhline(data.get("chance", 1 / 31), color=PALETTE["gray"], ls="--", lw=1.2, label="chance 1/31")
    ax.set_xlabel("Labeled examples per class ($k$)")
    ax.set_ylabel("Test accuracy")
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.05)
    ax.set_title("F4  Few-shot probes (50 runs: 5 seeds × 10 draws)")
    ax.legend(frameon=False, loc="lower right", fontsize=7.5)
    save(fig, "F4_fewshot.png")


def main():
    setup_style()
    fig_f1_leakage()
    fig_f2_hubness()
    fig_f3_complementarity()
    if (RESULTS / "04_fewshot.json").exists():
        fig_f4_fewshot()
    else:
        print("Skipping F4: results/04_fewshot.json not found yet")


if __name__ == "__main__":
    main()
