"""Render combined publication-ready figures for the thesis.

Produces four single-PDF panels that replace ad-hoc subfigure grids in the
LaTeX sources, with a coherent palette and minimal visual chrome:

* ``per_sequence_4panel.{pdf,png}``       -- Fig. 4.1 (4 OOD families)
* ``per_sequence_extra_6panel.{pdf,png}`` -- Fig. 5.3 (histogramas adicionales)
* ``ood_curves_3panel.{pdf,png}``         -- Fig. 5.4 (estresores universales)
* ``heatmaps_4panel.{pdf,png}``           -- Fig. 5.1 (matrices agregadas)

All artefacts are written under ``writings/02_Images/`` only.
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm
import numpy


HERE = os.path.dirname(os.path.abspath(__file__))
WRITINGS = os.path.dirname(HERE)
WORKSPACE = os.path.dirname(WRITINGS)
RAW = os.path.join(WORKSPACE, "general", "results", "raw")
RESULTS = os.path.join(WORKSPACE, "general", "results")

OUT_PS = os.path.join(WRITINGS, "02_Images", "per_sequence")
OUT_OOD = os.path.join(WRITINGS, "02_Images", "ood_curves")
OUT_HM = os.path.join(WRITINGS, "02_Images", "heatmaps")

MODELS = ["pes_ql", "pes_dql", "pes_dqn", "pes_rdqn", "pes_a2c", "pes_trf", "pes_ens"]

PALETTE = {
    "pes_ql":   "#8C97A8",
    "pes_dql":  "#3F8EFC",
    "pes_dqn":  "#1FB57A",
    "pes_rdqn": "#7B61FF",
    "pes_a2c":  "#F2A93B",
    "pes_trf":  "#E04141",
    "pes_ens":  "#111111",
}

RC = {
    "font.family":      "DejaVu Sans",
    "font.size":        10.0,
    "axes.titlesize":   11.0,
    "axes.labelsize":   10.0,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.grid.which":  "both",
    "grid.alpha":        0.45,
    "grid.linestyle":   "--",
    "grid.linewidth":    0.6,
    "legend.frameon":    False,
    "legend.fontsize":   9.0,
    "figure.dpi":        150,
}

# Crop range for per-sequence plots (data lives in [0.7, 1.0]).
YLIM_PS = (0.55, 1.02)


def _load_perf(model: str, scenario: str) -> numpy.ndarray:
    """Return the per-sequence performance vector for ``model``/``scenario``."""
    path = os.path.join(RAW, f"{model}__{scenario}.json")
    if not os.path.isfile(path):
        return numpy.empty(0)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return numpy.asarray(data.get("per_sequence_perf", []), dtype=float)


def _plot_sorted_curves(ax, scenario: str) -> None:
    """Draw one ascending sorted-performance curve per model on ``ax``."""
    for model in MODELS:
        arr = numpy.sort(_load_perf(model, scenario))
        if arr.size == 0:
            continue
        is_winner = model == "pes_ens"
        x = numpy.arange(1, arr.size + 1)
        ax.plot(
            x, arr,
            color=PALETTE[model], label=model,
            lw=2.4 if is_winner else 1.4,
            alpha=0.95 if is_winner else 0.85,
            zorder=3 if is_winner else 2,
        )


# --------------------------------------------------------------------------
# Fig 4.1 -- per_sequence_4panel
# --------------------------------------------------------------------------

PANELS_4 = [
    ("sev_empirical",        "Empírico (in-distribution)"),
    ("sev_extrapolate_high", "Extrapolación de severidades altas"),
    ("struct_more_total",    "Estructural: mayor número de trials"),
    ("joint_extrap_both",    "Extrapolación conjunta (sev. + estructura)"),
]


def render_per_sequence_4panel() -> None:
    """Render Fig. 4.1: sorted per-sequence curves for four OOD families."""
    os.makedirs(OUT_PS, exist_ok=True)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=0.14,
                            wspace=0.10, hspace=0.32)

        for ax, (scenario, title) in zip(axes.flat, PANELS_4):
            ax.set_title(title, loc="left", fontweight="bold", pad=4)
            _plot_sorted_curves(ax, scenario)
            ax.set_ylim(*YLIM_PS)
            ax.set_xlim(1, 64)
            ax.minorticks_on()
            ax.grid(which="major", alpha=0.45, linestyle="--", linewidth=0.6)
            ax.grid(which="minor", alpha=0.18, linestyle=":",  linewidth=0.4)

        for ax in axes[1]:
            ax.set_xlabel("Secuencia (ordenada por desempeño)")
        for ax in axes[:, 0]:
            ax.set_ylabel(r"Retorno por secuencia ($r$)")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(MODELS),
                   bbox_to_anchor=(0.5, 0.01), handlelength=2.4,
                   columnspacing=1.8)
        base = os.path.join(OUT_PS, "per_sequence_4panel")
        fig.savefig(base + ".pdf", bbox_inches="tight")
        fig.savefig(base + ".png", bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] {base}.pdf")


# --------------------------------------------------------------------------
# Fig 5.3 -- per_sequence_extra_6panel
# --------------------------------------------------------------------------

PANELS_EXTRA = [
    ("sev_bimodal",          "Severidad: bimodal"),
    ("sev_gauss_high",       "Severidad: gauss-high"),
    ("sev_beta_highskew",    "Severidad: beta high-skew"),
    ("len_poisson",          "Longitud: Poisson"),
    ("len_extrapolate_long", "Longitud: extrapolación larga"),
    ("joint_high_long",      "Conjunta: severo + largo"),
]


def render_per_sequence_extra() -> None:
    """Render Fig. 5.3: sorted curves for six additional scenarios."""
    os.makedirs(OUT_PS, exist_ok=True)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.2), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.06, right=0.99, top=0.94, bottom=0.15,
                            wspace=0.10, hspace=0.34)

        for ax, (scenario, title) in zip(axes.flat, PANELS_EXTRA):
            ax.set_title(title, loc="left", fontweight="bold", pad=4)
            _plot_sorted_curves(ax, scenario)
            ax.set_ylim(*YLIM_PS)
            ax.set_xlim(1, 64)
            ax.minorticks_on()
            ax.grid(which="major", alpha=0.45, linestyle="--", linewidth=0.6)
            ax.grid(which="minor", alpha=0.18, linestyle=":",  linewidth=0.4)

        for ax in axes[1]:
            ax.set_xlabel("Secuencia (ordenada)")
        for ax in axes[:, 0]:
            ax.set_ylabel(r"Retorno ($r$)")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(MODELS),
                   bbox_to_anchor=(0.5, 0.01), handlelength=2.4,
                   columnspacing=1.8)
        base = os.path.join(OUT_PS, "per_sequence_extra_6panel")
        fig.savefig(base + ".pdf", bbox_inches="tight")
        fig.savefig(base + ".png", bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] {base}.pdf")


# --------------------------------------------------------------------------
# Fig 5.4 -- ood_curves_3panel
# --------------------------------------------------------------------------

OOD_PANELS = [
    ("sev_extrapolate_high", "pes_trf", "pes_dql", "Severidad extrapolada"),
    ("joint_extrap_both",    "pes_trf", "pes_a2c", "Extrapolación conjunta"),
    ("len_extrapolate_long", "pes_trf", "pes_ens", "Longitudes extrapoladas"),
]


def render_ood_curves() -> None:
    """Render Fig. 5.4: three universal-stressor head-to-head comparisons."""
    os.makedirs(OUT_OOD, exist_ok=True)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=True)
        fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.18,
                            wspace=0.10)

        for ax, (scenario, m1, m2, title) in zip(axes, OOD_PANELS):
            arr1 = numpy.sort(_load_perf(m1, scenario))
            arr2 = numpy.sort(_load_perf(m2, scenario))
            x1 = numpy.arange(1, arr1.size + 1)
            x2 = numpy.arange(1, arr2.size + 1)

            if arr2.size:
                ax.plot(x2, arr2, color=PALETTE[m2], lw=1.6, label=m2,
                        alpha=0.9)
                ax.axhline(arr2.mean(), color=PALETTE[m2], ls=":", lw=1.0,
                           alpha=0.7)
            if arr1.size:
                ax.plot(x1, arr1, color=PALETTE[m1], lw=2.4, label=m1,
                        zorder=3)
                ax.axhline(arr1.mean(), color=PALETTE[m1], ls=":", lw=1.0,
                           alpha=0.7)

            ax.set_title(title, loc="left", fontweight="bold", pad=4)
            ax.set_xlabel("Secuencia (ordenada por desempeño)")
            ax.set_ylim(0.55, 1.02)
            ax.set_xlim(1, 64)
            ax.minorticks_on()
            ax.grid(which="major", alpha=0.45, linestyle="--", linewidth=0.6)
            ax.grid(which="minor", alpha=0.18, linestyle=":",  linewidth=0.4)
            ax.legend(loc="lower right")

        axes[0].set_ylabel(r"Retorno normalizado ($r$)")
        fig.suptitle("Desempeño bajo los tres estresores universales",
                     y=0.99, fontsize=11.5, fontweight="bold")

        base = os.path.join(OUT_OOD, "ood_curves_3panel")
        fig.savefig(base + ".pdf", bbox_inches="tight")
        fig.savefig(base + ".png", bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] {base}.pdf")


# --------------------------------------------------------------------------
# Fig 5.1 -- heatmaps_4panel
# --------------------------------------------------------------------------

HM_PANELS = [
    ("matrix_global_mean.csv",     r"Retorno medio  $\bar r$",                "viridis",   "linear",    None),
    ("matrix_welch_logp.csv",      r"Significancia  $-\log_{10} p$ (Welch)",  "magma",     "linear",    None),
    ("matrix_ood_degradation.csv", r"Degradación OOD  $\Delta r$",            "diverging", "diverging", 0.0),
    ("matrix_action_kl.csv",       r"Divergencia KL de acciones",             "cividis",   "log",       None),
]


def _read_matrix(path):
    """Read a benchmark matrix CSV. Returns (row labels, col labels, ndarray)."""
    rows, data = [], []
    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
        cols = header[1:]
        for line in fh:
            parts = line.strip().split(",")
            rows.append(parts[0])
            data.append([float(x) if x else numpy.nan for x in parts[1:]])
    return rows, cols, numpy.asarray(data, dtype=float)


def _short_scen(name: str) -> str:
    """Compress long scenario names so heatmap xticks stay readable."""
    return (name
            .replace("sev_", "sev·")
            .replace("len_", "len·")
            .replace("joint_", "joint·")
            .replace("struct_", "struct·")
            .replace("extrapolate_", "extr·")
            .replace("_", " "))


def render_heatmaps() -> None:
    """Render Fig. 5.1: four aggregate benchmark matrices on a 2x2 grid."""
    os.makedirs(OUT_HM, exist_ok=True)

    diverging = LinearSegmentedColormap.from_list(
        "ood_div", ["#9b1d20", "#f7f7f7", "#1a7f37"], N=256)

    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.8))
        fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.13,
                            wspace=0.30, hspace=0.60)

        for ax, (fname, title, cmap_name, scale, center) in zip(axes.flat, HM_PANELS):
            rows, cols, M = _read_matrix(os.path.join(RESULTS, fname))

            base_cmap = diverging if cmap_name == "diverging" else plt.get_cmap(cmap_name)
            cmap = base_cmap.copy()
            cmap.set_bad("#e9e9e9")
            kwargs = {"cmap": cmap, "aspect": "auto"}

            if scale == "log":
                M_pos = numpy.where(M > 0, M, numpy.nan)
                if numpy.any(numpy.isfinite(M_pos)):
                    vmin = max(numpy.nanmin(M_pos), 1e-3)
                    vmax = numpy.nanmax(M_pos)
                else:
                    vmin, vmax = 1e-3, 1.0
                kwargs["norm"] = LogNorm(vmin=vmin, vmax=vmax)
                M_show = M_pos
            elif scale == "diverging":
                vabs = numpy.nanmax(numpy.abs(M)) or 1.0
                kwargs["norm"] = TwoSlopeNorm(
                    vcenter=center if center is not None else 0.0,
                    vmin=-vabs, vmax=vabs,
                )
                M_show = M
            else:
                M_show = M

            im = ax.imshow(M_show, **kwargs)
            ax.set_title(title, loc="left", fontweight="bold", pad=6)
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels([_short_scen(c) for c in cols], rotation=55,
                               ha="right", fontsize=6.8)
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels(rows, fontsize=8.5)
            ax.tick_params(axis="both", length=0)
            ax.grid(False)

            if scale != "log":
                for i in range(M.shape[0]):
                    for j in range(M.shape[1]):
                        v = M[i, j]
                        if not numpy.isfinite(v):
                            continue
                        rgba = im.cmap(im.norm(v))
                        lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                        txt = "#ffffff" if lum < 0.5 else "#111111"
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                fontsize=5.6, color=txt)

            cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
            cb.ax.tick_params(labelsize=8)

        fig.suptitle("Matrices agregadas sobre las 22 condiciones del benchmark",
                     y=0.99, fontsize=12.0, fontweight="bold")

        base = os.path.join(OUT_HM, "heatmaps_4panel")
        fig.savefig(base + ".pdf", bbox_inches="tight")
        fig.savefig(base + ".png", bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] {base}.pdf")


def main() -> None:
    """Render all four publication figures."""
    render_per_sequence_4panel()
    render_per_sequence_extra()
    render_ood_curves()
    render_heatmaps()


if __name__ == "__main__":
    main()
