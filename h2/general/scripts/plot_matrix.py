"""Publication-quality visualisation of benchmark matrices.

Generates, under ``general/results/``:

* ``heatmap_global_mean.[png|pdf]``     -- model x scenario performance.
* ``heatmap_ood_degradation.[png|pdf]`` -- baseline_mean - cell_mean.
* ``heatmap_welch_logp.[png|pdf]``      -- log10(p) clipped to [-10, 0].
* ``heatmap_action_kl.[png|pdf]``       -- KL action drift (log scale).
* ``per_sequence_histograms/<scenario>.[png|pdf]`` -- one panel per scenario
  with overlaid model histograms.

Design choices for publication readability:

* Each colour map is paired with an explicit ``vmin``/``vmax`` so plots
  across runs are directly comparable; extreme values are clipped and
  annotated (e.g. ``"\u2264-10"``).
* Per-cell text colour is derived from the cell's normalised luminance so
  contrast is correct everywhere (the old "global max" heuristic failed
  on non-zero-centred data).
* NaN cells are rendered as a neutral grey with no annotation.
* Both ``.png`` (raster) and ``.pdf`` (vector) outputs are written; the
  PDF is the format intended for inclusion in publications.

Run after :mod:`general.scripts.aggregate`.
"""
##########################
##  Imports externos    ##
##########################
import csv as _csv
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy

##########################
##  Imports internos    ##
##########################
from .runner import GENERAL_ROOT

RESULTS_DIR = os.path.join(GENERAL_ROOT, 'results')
HIST_DIR = os.path.join(RESULTS_DIR, 'per_sequence_histograms')

# ---------------------------------------------------------------------------
# Publication style defaults.
# ---------------------------------------------------------------------------
_PUB_RC = {
    'font.family':       'DejaVu Sans',
    'font.size':         10,
    'axes.titlesize':    12,
    'axes.labelsize':    10,
    'xtick.labelsize':   8.5,
    'ytick.labelsize':   9.5,
    'legend.fontsize':   8,
    'figure.dpi':        140,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'pdf.fonttype':      42,   # TrueType for editability in vector editors.
    'ps.fonttype':       42,
}

# Significance thresholds for the Welch heatmap colourbar (alpha=0.05/0.01/0.001).
_LOG10_ALPHA_LINES = (math.log10(0.05), math.log10(0.01), math.log10(0.001))


# ---------------------------------------------------------------------------
# Custom publication palettes.
# ---------------------------------------------------------------------------
# Hand-tuned colour ramps (loosely inspired by seaborn's `crest`, `mako`,
# `flare` and `vlag`) so the four heatmaps share a coherent aesthetic
# without the harshness of the default `RdBu_r` / `viridis` schemes.
_PALETTE_ANCHORS = {
    # Sequential blue-green for "performance" (low -> high).
    'mpes_perf': ['#f3f7f4', '#cfe7e2', '#7fc6c0', '#3b9aa1', '#1f6e83',
                  '#1b455f', '#16263f'],
    # Diverging cool/warm for OOD degradation (centered at zero).
    # Cold side = "better OOD than baseline", warm = "worse OOD".
    'mpes_div':  ['#2a6489', '#5994b8', '#a7c8db', '#f1f1ee',
                  '#f1c5a3', '#d77c5d', '#9e3a26'],
    # Sequential warm-purple for log10(p): deeper = stronger evidence.
    # Anchors run from dark (low log_p, strong evidence) to light (log_p~0).
    'mpes_pval': ['#1f0033', '#5a1450', '#9c3060', '#d96a52', '#f6a36b',
                  '#ffd9a3', '#fff4e6'],
    # Sequential blue-teal-cream for KL drift (low = close to in-dist).
    'mpes_kl':   ['#fdf8ec', '#d9ecdb', '#7ec5b6', '#3a8a9a', '#1d5c80',
                  '#162d52', '#0a0f29'],
}


def _register_palettes() -> None:
    """Register custom colormaps the first time the module is used."""
    cmaps = matplotlib.colormaps
    for name, anchors in _PALETTE_ANCHORS.items():
        if name in cmaps:
            continue
        cmap = mcolors.LinearSegmentedColormap.from_list(name, anchors, N=256)
        cmaps.register(cmap=cmap, name=name)


_register_palettes()


##########################
##  IO helpers          ##
##########################
def _load_summary() -> dict:
    """Load the consolidated ``matrix_summary.json``."""
    path = os.path.join(RESULTS_DIR, 'matrix_summary.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _read_matrix_csv(path: str) -> "tuple[list, list, numpy.ndarray]":
    """Read a ``model x scenario`` CSV produced by ``aggregate.py``.

    Empty cells become ``NaN``.
    """
    with open(path, 'r', encoding='utf-8') as f:
        rows = list(_csv.reader(f))
    header = rows[0][1:]
    body = numpy.full((len(rows) - 1, len(header)), numpy.nan)
    models = []
    for i, row in enumerate(rows[1:]):
        models.append(row[0])
        for j, val in enumerate(row[1:]):
            if val:
                try:
                    body[i, j] = float(val)
                except ValueError:
                    pass
    return models, header, body


##########################
##  Heatmap primitive   ##
##########################
def _save(fig, base_path: str) -> None:
    """Save ``fig`` as both ``.png`` and ``.pdf``."""
    fig.savefig(base_path + '.png')
    fig.savefig(base_path + '.pdf')
    plt.close(fig)


def _heatmap(matrix: numpy.ndarray,
             models: list,
             scenarios: list,
             title: str,
             out_base: str,
             cmap: str = 'viridis',
             vmin: "float | None" = None,
             vmax: "float | None" = None,
             norm: "mcolors.Normalize | None" = None,
             fmt: str = '{:.3f}',
             clip_low_label: "str | None" = None,
             clip_high_label: "str | None" = None,
             cbar_label: str = '',
             cbar_ticks: "list | None" = None) -> None:
    """Draw a single ``model x scenario`` heatmap.

    Parameters
    ----------
    matrix : ``ndarray`` of shape ``(n_models, n_scenarios)``
        Values to visualise; ``NaN`` cells are rendered as light grey.
    models, scenarios : list[str]
        Row and column labels.
    title : str
        Figure title.
    out_base : str
        Output path *without* extension; both ``.png`` and ``.pdf`` are
        written.
    cmap : str
        Matplotlib colormap name.
    vmin, vmax : float, optional
        Manual colour-scale limits (overridden by ``norm`` when given).
    norm : ``matplotlib.colors.Normalize``, optional
        Custom normaliser (e.g. :class:`~matplotlib.colors.LogNorm`).
    fmt : str
        ``str.format`` template for in-cell numeric annotations.
    clip_low_label, clip_high_label : str, optional
        Text to draw in cells whose original value is below ``vmin`` /
        above ``vmax`` (used to flag clipped cells).
    cbar_label : str
        Label shown next to the colour bar.
    cbar_ticks : list, optional
        Manual tick locations on the colour bar.
    """
    with plt.rc_context(_PUB_RC):
        n_rows, n_cols = matrix.shape
        fig_w = max(10.0, n_cols * 0.55)
        fig_h = max(3.6, n_rows * 0.55)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor('white')

        # Mask NaN -> soft grey background; values get clipped to [vmin, vmax].
        masked = numpy.ma.masked_invalid(matrix)
        cmap_obj = matplotlib.colormaps.get_cmap(cmap).copy()
        cmap_obj.set_bad(color='#ececec')

        if norm is None:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        im = ax.imshow(masked, cmap=cmap_obj, norm=norm, aspect='auto',
                       interpolation='nearest')

        # Axis decoration.
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(scenarios, rotation=55, ha='right')
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(models)
        ax.set_xticks(numpy.arange(-0.5, n_cols), minor=True)
        ax.set_yticks(numpy.arange(-0.5, n_rows), minor=True)
        ax.grid(which='minor', color='white', linewidth=1.1)
        ax.tick_params(which='minor', length=0)
        ax.tick_params(which='major', length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, pad=12, fontweight='semibold')

        # Per-cell annotations with luminance-aware text colour.
        for i in range(n_rows):
            for j in range(n_cols):
                v = matrix[i, j]
                if not numpy.isfinite(v):
                    continue
                # Clipping markers.
                if clip_low_label is not None and vmin is not None and v < vmin:
                    text = clip_low_label
                    txt_v = vmin
                elif clip_high_label is not None and vmax is not None and v > vmax:
                    text = clip_high_label
                    txt_v = vmax
                else:
                    text = fmt.format(v)
                    txt_v = float(numpy.clip(v, norm.vmin, norm.vmax))
                # Luminance of the cell colour drives text colour.
                rgba = cmap_obj(norm(txt_v))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                colour = 'white' if lum < 0.55 else '#1a1a1a'
                ax.text(j, i, text, ha='center', va='center',
                        color=colour, fontsize=7)

        cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.012)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(length=0)
        if cbar_label:
            cbar.set_label(cbar_label)
        if cbar_ticks is not None:
            cbar.set_ticks(cbar_ticks)

        fig.tight_layout()
        _save(fig, out_base)


##########################
##  Matrix builders     ##
##########################
def _matrix_from(summary: dict, getter) -> numpy.ndarray:
    """Build a ``(n_models, n_scenarios)`` matrix from the summary JSON."""
    models = summary['models']
    scenarios = summary['scenarios']
    out = numpy.full((len(models), len(scenarios)), numpy.nan)
    for i, m in enumerate(models):
        for j, s in enumerate(scenarios):
            cell = summary['cells'].get(m, {}).get(s, {})
            v = getter(cell)
            if v is not None:
                out[i, j] = float(v)
    return out


##########################
##  Public API          ##
##########################
def plot_all() -> None:
    """Render every benchmark figure into ``general/results/``."""
    summary = _load_summary()
    models = summary['models']
    scenarios = summary['scenarios']
    baseline_id = summary['baseline_scenario']
    base_idx = scenarios.index(baseline_id) if baseline_id in scenarios else None

    # ----- Global mean performance --------------------------------------
    M_mean = _matrix_from(summary, lambda c: c.get('global_mean_perf'))
    # Auto vmin/vmax with a small margin for stable colour mapping.
    finite = M_mean[numpy.isfinite(M_mean)]
    if finite.size:
        vmin_m = float(numpy.floor(finite.min() * 100) / 100)
        vmax_m = float(numpy.ceil(finite.max() * 100) / 100)
    else:
        vmin_m, vmax_m = 0.0, 1.0
    _heatmap(M_mean, models, scenarios,
             'Global mean performance per (model, scenario)',
             os.path.join(RESULTS_DIR, 'heatmap_global_mean'),
             cmap='mpes_perf', vmin=vmin_m, vmax=vmax_m,
             cbar_label='mean performance')

    # ----- OOD degradation ----------------------------------------------
    def _degr(c):
        m = c.get('model')
        if not m:
            return None
        b = summary['cells'].get(m, {}).get(baseline_id, {}).get('global_mean_perf')
        v = c.get('global_mean_perf')
        if b is None or v is None:
            return None
        return b - v
    M_degr = _matrix_from(summary, _degr)
    # Hide the baseline column (always 0 by construction).
    if base_idx is not None:
        M_degr[:, base_idx] = numpy.nan
    finite = M_degr[numpy.isfinite(M_degr)]
    bound = float(numpy.nanmax(numpy.abs(finite))) if finite.size else 0.1
    bound = max(round(bound + 0.005, 2), 0.05)
    _heatmap(M_degr, models, scenarios,
             'OOD degradation: baseline_mean \u2212 cell_mean',
             os.path.join(RESULTS_DIR, 'heatmap_ood_degradation'),
             cmap='mpes_div', vmin=-bound, vmax=bound, fmt='{:+.3f}',
             cbar_label='\u0394 perf. (+ = worse OOD)')

    # ----- log10(Welch p) -----------------------------------------------
    welch_logp_csv = os.path.join(RESULTS_DIR, 'matrix_welch_logp.csv')
    if os.path.isfile(welch_logp_csv):
        wmodels, wscen, wbody = _read_matrix_csv(welch_logp_csv)
    else:
        # Backwards compatibility: derive log10(p) from raw p with a floor.
        welch_csv = os.path.join(RESULTS_DIR, 'matrix_welch_p.csv')
        wmodels, wscen, body_p = _read_matrix_csv(welch_csv)
        with numpy.errstate(divide='ignore'):
            wbody = numpy.where(numpy.isfinite(body_p) & (body_p > 0),
                                numpy.log10(numpy.maximum(body_p, 1e-300)),
                                numpy.nan)
    # Hide the baseline column (no test by construction).
    if baseline_id in wscen:
        wbody[:, wscen.index(baseline_id)] = numpy.nan

    vmin_p, vmax_p = -10.0, 0.0
    cticks = [vmin_p] + list(_LOG10_ALPHA_LINES) + [vmax_p]
    _heatmap(wbody, wmodels, wscen,
             'Welch t-test vs in-distribution baseline '
             '(log\u2081\u2080 p, clipped to [\u221210, 0])',
             os.path.join(RESULTS_DIR, 'heatmap_welch_logp'),
             cmap='mpes_pval', vmin=vmin_p, vmax=vmax_p, fmt='{:.2f}',
             clip_low_label='\u2264-10',
             cbar_label='log\u2081\u2080(p) -- lower = stronger evidence',
             cbar_ticks=cticks)

    # ----- KL action drift (log-norm scale) -----------------------------
    kl_csv = os.path.join(RESULTS_DIR, 'matrix_action_kl.csv')
    if os.path.isfile(kl_csv):
        kmodels, kscen, kbody = _read_matrix_csv(kl_csv)
        if baseline_id in kscen:
            kbody[:, kscen.index(baseline_id)] = numpy.nan
        finite = kbody[numpy.isfinite(kbody) & (kbody > 0)]
        if finite.size:
            kl_floor = max(1e-3, float(numpy.percentile(finite, 5)))
            kl_top = float(numpy.nanmax(finite))
        else:
            kl_floor, kl_top = 1e-3, 1.0
        # LogNorm ignores values <= 0; clip them up to the floor for display.
        kbody_disp = kbody.copy()
        mask_zero = numpy.isfinite(kbody_disp) & (kbody_disp <= 0)
        kbody_disp[mask_zero] = kl_floor
        norm = mcolors.LogNorm(vmin=kl_floor, vmax=kl_top)
        _heatmap(kbody_disp, kmodels, kscen,
                 'KL(action distribution \u2225 in-distribution policy)',
                 os.path.join(RESULTS_DIR, 'heatmap_action_kl'),
                 cmap='mpes_kl', norm=norm, fmt='{:.2f}',
                 cbar_label='KL divergence (log scale, nats)')

    # ----- Per-scenario histograms --------------------------------------
    os.makedirs(HIST_DIR, exist_ok=True)
    with plt.rc_context(_PUB_RC):
        for s in scenarios:
            fig, ax = plt.subplots(figsize=(7, 4))
            any_data = False
            for m in models:
                cell = summary['cells'].get(m, {}).get(s, {})
                arr = cell.get('per_sequence_perf', [])
                if not arr:
                    continue
                ax.hist(arr, bins=20, range=(0, 1), histtype='step',
                        label=m, linewidth=1.5)
                any_data = True
            if not any_data:
                plt.close(fig)
                continue
            ax.set_title(f'Per-sequence performance distribution -- {s}')
            ax.set_xlabel('performance')
            ax.set_ylabel('count')
            ax.legend(loc='best')
            ax.set_xlim(0, 1)
            fig.tight_layout()
            fig.savefig(os.path.join(HIST_DIR, f'{s}.png'))
            fig.savefig(os.path.join(HIST_DIR, f'{s}.pdf'))
            plt.close(fig)


def _main() -> None:
    plot_all()
    print(f'[plot_matrix] figures written to {RESULTS_DIR}')


if __name__ == '__main__':
    _main()
