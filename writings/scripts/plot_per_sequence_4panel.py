"""Render the 4-panel per-sequence OOD figure for Materials chapter.

Reads the raw JSON outputs of the benchmark
(``general/results/raw/<model>__<scenario>.json``) for the four OOD families
referenced in Fig.~4.1 of the thesis and produces a single, publication-ready
2x2 panel with a coherent palette (no model-by-model PDF stitching).

Output:
    writings/02_Images/per_sequence/per_sequence_4panel.pdf
    writings/02_Images/per_sequence/per_sequence_4panel.png
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy


HERE = os.path.dirname(os.path.abspath(__file__))
WRITINGS = os.path.dirname(HERE)
WORKSPACE = os.path.dirname(WRITINGS)
RAW = os.path.join(WORKSPACE, "general", "results", "raw")
OUT_DIR = os.path.join(WRITINGS, "02_Images", "per_sequence")


PANELS = [
    ("sev_empirical",        "Empirico (in-distribution)"),
    ("sev_extrapolate_high", "Extrapolacion de severidades altas"),
    ("struct_more_total",    "Estructural: mayor numero de trials"),
    ("joint_extrap_both",    "Extrapolacion conjunta (sev. y estructura)"),
]

MODELS = ["pes_ql", "pes_dql", "pes_dqn", "pes_rdqn", "pes_a2c", "pes_trf", "pes_ens"]

# Coherent palette: cool→warm progression highlights the ranking
# (tabular → recurrent → policy-gradient → attention → ensemble).
PALETTE = {
    "pes_ql":   "#5B6C8A",   # slate blue
    "pes_dql":  "#3F8EFC",   # bright blue
    "pes_dqn":  "#1FB57A",   # teal-green
    "pes_rdqn": "#7B61FF",   # violet (recurrent)
    "pes_a2c":  "#F2A93B",   # amber
    "pes_trf":  "#E04141",   # red (best individual)
    "pes_ens":  "#111111",   # black, thicker (winner)
}


def _load(model: str, scenario: str) -> numpy.ndarray:
    path = os.path.join(RAW, f"{model}__{scenario}.json")
    if not os.path.isfile(path):
        return numpy.empty(0)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    arr = numpy.asarray(data.get("per_sequence_perf", []), dtype=float)
    return arr


def main() -> None:
    """Render the 2x2 per-sequence histograms panel as PDF and PNG."""
    os.makedirs(OUT_DIR, exist_ok=True)

    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "font.size":        9.5,
        "axes.titlesize":   10.5,
        "axes.labelsize":   9.5,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.25,
        "grid.linestyle":   "--",
        "legend.frameon":    False,
        "legend.fontsize":   8.5,
        "figure.dpi":        150,
    })

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.94, bottom=0.10,
                        wspace=0.10, hspace=0.30)

    bins = numpy.linspace(0.0, 1.0, 26)
    handles, labels = None, None

    for ax, (scenario, title) in zip(axes.flat, PANELS):
        ax.set_title(title, loc="left", fontweight="bold", pad=4)
        for model in MODELS:
            arr = _load(model, scenario)
            if arr.size == 0:
                continue
            color = PALETTE[model]
            is_winner = model == "pes_ens"
            ax.hist(
                arr,
                bins=bins,
                histtype="step",
                linewidth=2.0 if is_winner else 1.2,
                alpha=0.95 if is_winner else 0.80,
                color=color,
                label=model,
                zorder=3 if is_winner else 2,
            )
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks(numpy.arange(0.0, 1.01, 0.2))
        if ax in (axes[1, 0], axes[1, 1]):
            ax.set_xlabel("Retorno normalizado por secuencia ($r$)")
        if ax in (axes[0, 0], axes[1, 0]):
            ax.set_ylabel("Frecuencia (n=64)")

        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=len(MODELS),
        bbox_to_anchor=(0.5, -0.005),
        handlelength=2.0,
        columnspacing=1.6,
    )
    fig.subplots_adjust(bottom=0.14)

    base = os.path.join(OUT_DIR, "per_sequence_4panel")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {base}.pdf and .png")


if __name__ == "__main__":
    main()
