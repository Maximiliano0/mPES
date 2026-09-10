"""Shared plotting primitives and statistics for the mPES benchmark.

Centralises the publication style, the colour palettes, the annotated heatmap
renderer and the statistical helpers (Welch, Cohen's d, Kullback-Leibler) that
were previously duplicated across several figure modules.

Figure conventions exposed to :mod:`figures`:

* ``MODEL_COLOURS`` / :func:`model_colour` — one fixed colour per model; the
  best model of each suite (``pes_trf``, ``pes_ens``) uses the warm accent.
* ``BEST_LINEWIDTH`` / ``BASE_LINEWIDTH`` — stroke widths that emphasise the
  best model whenever several models are overlaid.
"""
##########################
##  Imports externos    ##
##########################
import csv
import math
import os
from dataclasses import dataclass, field

import matplotlib
matplotlib.use('Agg')
from matplotlib import colors as mcolors
from matplotlib import pyplot
import numpy


###############
##  Style
###############
PUB_RC = {
    'font.family':     'DejaVu Sans',
    'font.size':       10,
    'axes.titlesize':  12,
    'axes.labelsize':  10,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 8,
    'figure.dpi':      140,
    'savefig.dpi':     300,
    'savefig.bbox':    'tight',
    'pdf.fonttype':    42,
    'ps.fonttype':     42,
}

# Soft sequential/diverging ramps shared by every heatmap in the benchmark.
_PALETTE_ANCHORS = {
    'mpes_perf': ['#f3f7f4', '#cfe7e2', '#7fc6c0', '#3b9aa1', '#1f6e83',
                  '#1b455f', '#16263f'],
    'mpes_div':  ['#2a6489', '#5994b8', '#a7c8db', '#f1f1ee',
                  '#f1c5a3', '#d77c5d', '#9e3a26'],
    'mpes_pval': ['#1f0033', '#5a1450', '#9c3060', '#d96a52', '#f6a36b',
                  '#ffd9a3', '#fff4e6'],
    'mpes_kl':   ['#fdf8ec', '#d9ecdb', '#7ec5b6', '#3a8a9a', '#1d5c80',
                  '#162d52', '#0a0f29'],
}

# Stable, colour-blind friendly line colours for per-model curves.
MODEL_LINE_COLOURS = ('#8d99ae', '#457b9d', '#2a9d8f', '#c9a227',
                      '#f4a261', '#e76f51', '#6d597a', '#264653')

# Fixed model -> colour mapping so a model keeps its hue across every figure.
# The best model of each suite (pes_trf / pes_ens) gets the warm accent.
MODEL_COLOURS = {
    'pes_base': '#9aa5b1', 'pes_ql': '#5b8fb9', 'pes_dql': '#2f6690',
    'pes_dqn': '#2a9d8f', 'pes_rdqn': '#7fb069', 'pes_a2c': '#e9a13b',
    'pes_trf': '#c44e52',
    'pes_ens': '#c44e52', 'pes_ens_sprb': '#5b8fb9', 'pes_ens_accq': '#2f6690',
    'pes_ens_consensus': '#2a9d8f', 'pes_ens_consensus_prior': '#7fb069',
    'pes_ens_trf_guard': '#e9a13b',
}

#: Line widths for the best model of a panel versus the remaining ones.
BEST_LINEWIDTH = 3.0
BASE_LINEWIDTH = 1.5

ALPHA_LEVELS = (math.log10(0.05), math.log10(0.01), math.log10(0.001))


def model_colour(model: str, index: int = 0) -> str:
    """Return the colour assigned to one model.

    Parameters
    ----------
    model : str
        Package name (``pes_dqn``, ``pes_ens_sprb`` …).
    index : int, optional
        Position of the model in the plotted list; used to pick a fallback
        colour from ``MODEL_LINE_COLOURS`` when ``model`` is not listed in
        ``MODEL_COLOURS`` (default 0).

    Returns
    -------
    str
        Hex colour string.
    """
    return MODEL_COLOURS.get(model, MODEL_LINE_COLOURS[index % len(MODEL_LINE_COLOURS)])


def _register_palettes() -> None:
    """Register the custom colormaps once per interpreter."""
    for name, anchors in _PALETTE_ANCHORS.items():
        if name in matplotlib.colormaps:
            continue
        matplotlib.colormaps.register(
            cmap=mcolors.LinearSegmentedColormap.from_list(name, anchors, N=256),
            name=name)


_register_palettes()


###############
##  IO helpers
###############
def save_figure(figure, base_path: str) -> None:
    """Save ``figure`` as a raster PNG and close it.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure to write.
    base_path : str
        Output path without extension; ``.png`` is appended and parent
        directories are created on demand.
    """
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    figure.savefig(base_path + '.png')
    pyplot.close(figure)


def read_matrix_csv(path: str) -> "tuple[list[str], list[str], numpy.ndarray]":
    """Read a ``model x scenario`` CSV, mapping empty cells to ``NaN``.

    Parameters
    ----------
    path : str
        CSV file whose first column holds model names and whose header row
        holds scenario identifiers.

    Returns
    -------
    models : list of str
        Row labels in file order.
    scenarios : list of str
        Column labels in file order.
    matrix : ndarray, shape ``(len(models), len(scenarios))``
        Float values; empty or unparsable cells are ``NaN``.
    """
    with open(path, 'r', encoding='utf-8', newline='') as handle:
        rows = list(csv.reader(handle))
    scenarios = rows[0][1:]
    models: list[str] = []
    matrix = numpy.full((len(rows) - 1, len(scenarios)), numpy.nan)
    for row_index, row in enumerate(rows[1:]):
        models.append(row[0])
        for column_index, value in enumerate(row[1:]):
            if value:
                try:
                    matrix[row_index, column_index] = float(value)
                except ValueError:
                    continue
    return models, scenarios, matrix


def style_axes(axis) -> None:
    """Apply the restrained axis style shared by the line figures.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axes to style in place (light grid below the data, no top/right spines).
    """
    axis.grid(alpha=0.22, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)


###############
##  Heatmap
###############
@dataclass
class HeatmapSpec:
    """Rendering options for :func:`heatmap`.

    Parameters
    ----------
    title : str
        Figure title.
    cbar_label : str
        Label drawn next to the colour bar.
    cmap : str
        Registered colormap name.
    vmin, vmax : float, optional
        Colour-scale limits; ignored when ``norm`` is given.
    norm : matplotlib.colors.Normalize, optional
        Custom normaliser, e.g. :class:`~matplotlib.colors.LogNorm`.
    fmt : str
        ``str.format`` template for the in-cell annotation.
    clip_low_label, clip_high_label : str, optional
        Text drawn in cells whose value falls outside ``[vmin, vmax]``.
    cbar_ticks : list, optional
        Manual colour-bar tick locations.
    """

    title: str
    cbar_label: str = ''
    cmap: str = 'mpes_perf'
    vmin: "float | None" = None
    vmax: "float | None" = None
    norm: "mcolors.Normalize | None" = None
    fmt: str = '{:.3f}'
    clip_low_label: "str | None" = None
    clip_high_label: "str | None" = None
    cbar_ticks: "list | None" = field(default=None)
    xlabel: str = 'Escenario'
    ylabel: str = 'Modelo'


def heatmap(matrix: numpy.ndarray, models: "list[str]", scenarios: "list[str]",
            out_base: str, spec: HeatmapSpec) -> None:
    """Draw and save one annotated ``model x scenario`` heatmap.

    ``NaN`` cells are rendered as neutral grey and left unannotated so that
    missing measurements are never confused with a real value.

    Parameters
    ----------
    matrix : ndarray, shape ``(len(models), len(scenarios))``
        Values to colour and annotate.
    models : list of str
        Row labels.
    scenarios : list of str
        Column labels (scenario identifiers).
    out_base : str
        Output path without extension, forwarded to :func:`save_figure`.
    spec : HeatmapSpec
        Rendering options (title, colormap, limits, annotation format).
    """
    with pyplot.rc_context(PUB_RC):
        n_rows, n_cols = matrix.shape
        figure, axis = pyplot.subplots(
            figsize=(max(10.0, n_cols * 0.62), max(3.6, n_rows * 0.58)))
        figure.patch.set_facecolor('white')

        colour_map = matplotlib.colormaps[spec.cmap].copy()
        colour_map.set_bad(color='#ececec')
        norm = spec.norm or mcolors.Normalize(vmin=spec.vmin, vmax=spec.vmax)
        image = axis.imshow(numpy.ma.masked_invalid(matrix), cmap=colour_map,
                            norm=norm, aspect='auto', interpolation='nearest')

        axis.set_xticks(range(n_cols), scenarios, rotation=55, ha='right')
        axis.set_yticks(range(n_rows), models)
        axis.set_xticks(numpy.arange(-0.5, n_cols), minor=True)
        axis.set_yticks(numpy.arange(-0.5, n_rows), minor=True)
        axis.grid(which='minor', color='white', linewidth=1.1)
        axis.tick_params(which='both', length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_xlabel(spec.xlabel)
        axis.set_ylabel(spec.ylabel)
        axis.set_title(spec.title, pad=12, fontweight='semibold')

        for row in range(n_rows):
            for column in range(n_cols):
                value = matrix[row, column]
                if not numpy.isfinite(value):
                    continue
                text, reference = _cell_text(value, spec)
                rgba = colour_map(norm(reference))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                axis.text(column, row, text, ha='center', va='center',
                          fontsize=7, color='white' if luminance < 0.55 else '#1a1a1a')

        colour_bar = figure.colorbar(image, ax=axis, shrink=0.85, pad=0.012)
        colour_bar.ax.tick_params(length=0)
        if spec.cbar_label:
            colour_bar.set_label(spec.cbar_label)
        if spec.cbar_ticks is not None:
            colour_bar.set_ticks(spec.cbar_ticks)

        figure.tight_layout()
        save_figure(figure, out_base)


def _cell_text(value: float, spec: HeatmapSpec) -> "tuple[str, float]":
    """Return the annotation and the colour-reference value for one cell."""
    if spec.clip_low_label is not None and spec.vmin is not None and value < spec.vmin:
        return spec.clip_low_label, spec.vmin
    if spec.clip_high_label is not None and spec.vmax is not None and value > spec.vmax:
        return spec.clip_high_label, spec.vmax
    low = spec.vmin if spec.vmin is not None else float(value)
    high = spec.vmax if spec.vmax is not None else float(value)
    return spec.fmt.format(value), float(numpy.clip(value, low, high))


###############
##  Statistics
###############
def welch_test(first: numpy.ndarray, second: numpy.ndarray) -> "tuple[float, float, float]":
    """Two-sample Welch t-test.

    Returns
    -------
    t : float
        Welch statistic.
    p : float
        Two-sided p-value.
    log10_p : float
        Base-10 logarithm of ``p``, evaluated in log-space so that extreme
        tails do not underflow to zero.
    """
    if first.size < 2 or second.size < 2:
        return float('nan'), float('nan'), float('nan')
    first_var = float(numpy.var(first, ddof=1))
    second_var = float(numpy.var(second, ddof=1))
    error = math.sqrt(first_var / first.size + second_var / second.size)
    if error == 0.0:
        return float('nan'), float('nan'), float('nan')
    statistic = (float(numpy.mean(first)) - float(numpy.mean(second))) / error
    numerator = (first_var / first.size + second_var / second.size) ** 2
    denominator = ((first_var / first.size) ** 2 / (first.size - 1)
                   + (second_var / second.size) ** 2 / (second.size - 1))
    degrees = numerator / denominator if denominator > 0 else (first.size + second.size - 2)
    try:
        from scipy.stats import t as student_t
        log10_p = (float(student_t.logsf(abs(statistic), df=degrees))
                   + math.log(2.0)) / math.log(10.0)
        p_value = 2.0 * float(student_t.sf(abs(statistic), df=degrees))
    except ImportError:
        p_value = math.erfc(abs(statistic) / math.sqrt(2.0))
        log10_p = math.log10(p_value) if p_value > 0 else float('-inf')
    return statistic, p_value, log10_p


def cohen_d(first: numpy.ndarray, second: numpy.ndarray) -> float:
    """Standardised mean difference using the pooled standard deviation.

    Parameters
    ----------
    first, second : ndarray
        Samples to compare; the sign is positive when ``first`` has the
        larger mean.

    Returns
    -------
    float
        Cohen's ``d``; ``NaN`` when either sample has fewer than two values or
        the pooled standard deviation is zero.
    """
    if first.size < 2 or second.size < 2:
        return float('nan')
    pooled = math.sqrt(((first.size - 1) * numpy.var(first, ddof=1)
                        + (second.size - 1) * numpy.var(second, ddof=1))
                       / (first.size + second.size - 2))
    if pooled == 0.0:
        return float('nan')
    return float((numpy.mean(first) - numpy.mean(second)) / pooled)


def kl_divergence(first: numpy.ndarray, second: numpy.ndarray,
                  epsilon: float = 1e-9) -> float:
    """Kullback-Leibler divergence ``KL(first || second)`` between two PMFs.

    Parameters
    ----------
    first, second : ndarray
        Non-negative vectors of equal length; both are smoothed by ``epsilon``
        and renormalised before the divergence is computed.
    epsilon : float, optional
        Additive smoothing that avoids ``log(0)`` (default ``1e-9``).

    Returns
    -------
    float
        Divergence in nats; ``NaN`` when the inputs are empty or of
        different length.
    """
    if first.size == 0 or second.size == 0 or first.size != second.size:
        return float('nan')
    left = numpy.asarray(first, dtype=float) + epsilon
    right = numpy.asarray(second, dtype=float) + epsilon
    left = left / left.sum()
    right = right / right.sum()
    return float(numpy.sum(left * numpy.log(left / right)))


def symmetric_kl(first: numpy.ndarray, second: numpy.ndarray) -> float:
    """Order-independent Kullback-Leibler divergence between two PMFs.

    Parameters
    ----------
    first, second : ndarray
        PMFs forwarded to :func:`kl_divergence` in both directions.

    Returns
    -------
    float
        ``0.5 * (KL(first || second) + KL(second || first))``; ``NaN`` if
        either direction is undefined.
    """
    forward = kl_divergence(first, second)
    reverse = kl_divergence(second, first)
    if math.isnan(forward) or math.isnan(reverse):
        return float('nan')
    return 0.5 * (forward + reverse)


def histogram_pmf(values: numpy.ndarray, bins: int = 20) -> numpy.ndarray:
    """Bin performance values on the common ``[0, 1]`` scale.

    Parameters
    ----------
    values : ndarray
        Per-sequence performance values.
    bins : int, optional
        Number of equal-width bins over ``[0, 1]`` (default 20).

    Returns
    -------
    ndarray, shape ``(bins,)``
        Unnormalised bin counts as floats, suitable for :func:`kl_divergence`.
    """
    counts, _ = numpy.histogram(values, bins=numpy.linspace(0.0, 1.0, bins + 1))
    return counts.astype(float)
