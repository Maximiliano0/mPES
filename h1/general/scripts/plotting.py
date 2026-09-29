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

#: Printable area of the thesis page (A4 with 2.54 cm margins), in inches.
#: Figures rendered at this width are included at ``\linewidth`` (portrait),
#: so their font sizes are the printed sizes.
PAGE_WIDTH_IN = 6.27

#: Style for figures rendered at their printed size (see ``PAGE_WIDTH_IN``).
PRINT_RC = {
    **PUB_RC,
    'font.size':       8,
    'axes.titlesize':  8.5,
    'axes.labelsize':  8,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
}
#: Line widths of the per-model curves at printed size.
PRINT_BEST_LINEWIDTH = 1.8
PRINT_BASE_LINEWIDTH = 0.9

#: Base colours of every figure: a blue family in harmony with the ITBA logo,
#: a muted coral as the single warm accent and soft neutrals. Large filled
#: areas (heatmap cells, bars, histograms) use the pastel steps; lines and
#: markers use the mid steps, which stay legible on white.
PALETTE = {
    'blue_dark':  '#2f6399',  # emphasis, table headers, dark end of the ramps
    'blue':       '#6c9cca',  # standard marks and lines
    'blue_light': '#bcd3e8',  # pastel fills
    'blue_pale':  '#e6eef7',  # backgrounds, box fills, striped rows
    'coral':      '#d87069',  # warm accent: mean lines, the highlighted model
    'coral_light': '#f2cfc1',  # pastel warm fill (dispersion bands)
    'teal':       '#1ba6ae',  # secondary line (medians, cumulative mean)
    'grey':       '#a0a8b3',  # references and de-emphasised marks
    'ink':        '#2b3440',  # text and outlines
}

# Soft sequential/diverging ramps shared by every heatmap in the benchmark.
# Sequential maps use one hue from pastel to a medium-dark step (never black);
# the diverging map uses the blue family and the coral accent around a
# neutral light grey.
_PALETTE_ANCHORS = {
    # Positioned anchors: most of the lightness range is spent on the upper
    # part of the scale, where the model means concentrate.
    'mpes_perf': [(0.0, '#f5f8fb'), (0.3, '#dde8f3'), (0.5, '#bcd3e8'),
                  (0.65, '#94b8da'), (0.78, '#6c9cca'), (0.9, '#4677ad'),
                  (1.0, '#285384')],
    'mpes_div':  ['#3f73a8', '#7ea3cc', '#bed2e7', '#f3f3f1',
                  '#f2cfc1', '#e09a84', '#c0624c'],
    'mpes_pval': ['#3b4a86', '#5a68a3', '#7f8bbf', '#a6afd6', '#c9cfe7',
                  '#e2e6f3', '#f8f9fc'],
    'mpes_kl':   ['#f5f8fb', '#d3e2f0', '#b0cae4', '#8db3d8', '#6c9cca',
                  '#4a80b6', '#2f6399'],
}

# Muted categorical line colours, CVD-checked on all pairs (Machado 2009,
# OKLab dE >= 8 under protan/deutan, >= 15 under normal vision).
MODEL_LINE_COLOURS = ('#8aa4fa', '#0c739f', '#1ba6ae', '#387a23',
                      '#c0a033', '#d87069', '#a0a8b3')

# Fixed model -> colour mapping so a model keeps its hue across every figure.
# The best model of each suite (pes_trf / pes_ens) gets the warm accent and
# the untuned reference (pes_base) the neutral grey.
MODEL_COLOURS = {
    'pes_base': '#a0a8b3', 'pes_ql': '#8aa4fa', 'pes_dql': '#0c739f',
    'pes_dqn': '#1ba6ae', 'pes_rdqn': '#387a23', 'pes_a2c': '#c0a033',
    'pes_trf': '#d87069',
    'pes_ens': '#d87069', 'pes_ens_sprb': '#8aa4fa', 'pes_ens_accq': '#0c739f',
    'pes_ens_consensus': '#1ba6ae', 'pes_ens_consensus_prior': '#387a23',
    'pes_ens_trf_guard': '#c0a033',
}

#: Line widths for the best model of a panel versus the remaining ones.
BEST_LINEWIDTH = 3.0
BASE_LINEWIDTH = 1.5

#: Colours of the non-model references drawn next to the model curves:
#: the random decision maker and the per-sequence bounds S_peor / S_mejor.
REFERENCE_COLOURS = {'random': '#7d8896', 'worst': '#c0624c', 'best': '#2f6399'}

#: Style of the horizontal per-model mean line shared by every per-sequence figure.
MEAN_LINESTYLE = ':'
MEAN_LINEWIDTH = 1.1

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
    separators : list of int, optional
        Column indices before which a vertical rule is drawn (e.g. the first
        held-out replica, to set it apart from the stress scenarios).
    figsize : tuple of float, optional
        Figure size in inches. When given, the figure is drawn at its printed
        size with :data:`PRINT_RC`; otherwise it grows with the matrix shape.
    annot_fontsize : float
        Font size of the in-cell annotations.
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
    separators: "list[int]" = field(default_factory=list)
    figsize: "tuple[float, float] | None" = None
    annot_fontsize: float = 7.0


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
    n_rows, n_cols = matrix.shape
    printed = spec.figsize is not None
    with pyplot.rc_context(PRINT_RC if printed else PUB_RC):
        figure, axis = pyplot.subplots(
            figsize=spec.figsize or (max(10.0, n_cols * 0.62), max(3.6, n_rows * 0.58)))
        figure.patch.set_facecolor('white')

        colour_map = matplotlib.colormaps[spec.cmap].copy()
        colour_map.set_bad(color='#ececec')
        norm = spec.norm or mcolors.Normalize(vmin=spec.vmin, vmax=spec.vmax)
        image = _draw_cells(axis, matrix, models, scenarios, spec, colour_map, norm,
                            spec.separators)
        axis.set_xlabel(spec.xlabel)
        axis.set_ylabel(spec.ylabel)
        axis.set_title(spec.title, pad=12, fontweight='semibold')

        colour_bar = figure.colorbar(image, ax=axis, shrink=0.85,
                                     pad=0.012, fraction=0.03 if printed else 0.15)
        colour_bar.ax.tick_params(length=0)
        if spec.cbar_label:
            colour_bar.set_label(spec.cbar_label)
        if spec.cbar_ticks is not None:
            colour_bar.set_ticks(spec.cbar_ticks)

        figure.tight_layout()
        save_figure(figure, out_base)


def heatmap_split(matrix: numpy.ndarray, models: "list[str]", scenarios: "list[str]",
                  out_base: str, spec: HeatmapSpec, split: int,
                  panel_titles: "tuple[str, str]" = ('', '')) -> None:
    """Draw a wide ``model x scenario`` heatmap as two stacked portrait panels.

    The columns ``[0, split)`` go to the upper panel and ``[split, n)`` to the
    lower one. Both panels share the colour scale and the cell size, so the
    figure fits the portrait text width (:data:`PAGE_WIDTH_IN`) at its printed
    size with :data:`PRINT_RC`, and the colour bar is drawn horizontally below.

    Parameters
    ----------
    matrix, models, scenarios, out_base, spec
        As in :func:`heatmap`; ``spec.separators`` uses the full column indices.
    split : int
        First column of the lower panel.
    panel_titles : tuple of str
        Titles of the upper and lower panels.
    """
    n_rows, n_cols = matrix.shape
    parts = ((0, split), (split, n_cols))
    widest = max(stop - start for start, stop in parts)
    label_w, right_w = 1.55, 0.08
    cell_w = (PAGE_WIDTH_IN - label_w - right_w) / widest
    cell_h = min(cell_w * 0.8, 0.30)
    # Room below each panel for its rotated scenario labels, from the longest one.
    ticks_h = [0.25 + 0.047 * max(len(name) for name in scenarios[start:stop])
               for start, stop in parts]
    panel_h, gap_h, title_h, cbar_h = n_rows * cell_h, 0.30, 0.55, 0.55
    height = title_h + 2 * panel_h + sum(ticks_h) + gap_h + cbar_h

    with pyplot.rc_context(PRINT_RC):
        figure = pyplot.figure(figsize=(PAGE_WIDTH_IN, height))
        figure.patch.set_facecolor('white')
        colour_map = matplotlib.colormaps[spec.cmap].copy()
        colour_map.set_bad(color='#ececec')
        norm = spec.norm or mcolors.Normalize(vmin=spec.vmin, vmax=spec.vmax)
        figure.suptitle(spec.title, y=1.0 - 0.08 / height, va='top', fontweight='semibold',
                        fontsize=PRINT_RC['axes.titlesize'])

        top = height - title_h
        images = []
        for index, (start, stop) in enumerate(parts):
            bottom = top - panel_h
            axis = figure.add_axes((label_w / PAGE_WIDTH_IN, bottom / height,
                                    (stop - start) * cell_w / PAGE_WIDTH_IN, panel_h / height))
            images.append(_draw_cells(axis, matrix[:, start:stop], models, scenarios[start:stop],
                                      spec, colour_map, norm,
                                      [column - start for column in spec.separators
                                       if start < column < stop]))
            axis.set_ylabel(spec.ylabel)
            if panel_titles[index]:
                axis.set_title(panel_titles[index], fontsize=PRINT_RC['axes.labelsize'], pad=4)
            top = bottom - ticks_h[index] - gap_h

        bar_axis = figure.add_axes((label_w / PAGE_WIDTH_IN, (top + gap_h - 0.14) / height,
                                    widest * cell_w / PAGE_WIDTH_IN, 0.12 / height))
        colour_bar = figure.colorbar(images[-1], cax=bar_axis, orientation='horizontal')
        colour_bar.ax.tick_params(length=0)
        if spec.cbar_label:
            colour_bar.set_label(spec.cbar_label)
        if spec.cbar_ticks is not None:
            colour_bar.set_ticks(spec.cbar_ticks)
        save_figure(figure, out_base)


def _draw_cells(axis, matrix: numpy.ndarray, models: "list[str]", scenarios: "list[str]",
                spec: HeatmapSpec, colour_map, norm, separators: "list[int]"):
    """Draw one annotated heatmap panel on ``axis`` and return its image."""
    n_rows, n_cols = matrix.shape
    image = axis.imshow(numpy.ma.masked_invalid(matrix), cmap=colour_map,
                        norm=norm, aspect='auto', interpolation='nearest')
    axis.set_xticks(range(n_cols), scenarios, rotation=55, ha='right')
    axis.set_yticks(range(n_rows), models)
    axis.set_xticks(numpy.arange(-0.5, n_cols), minor=True)
    axis.set_yticks(numpy.arange(-0.5, n_rows), minor=True)
    axis.grid(which='minor', color='white', linewidth=1.1)
    axis.tick_params(which='both', length=0)
    for column in separators:
        axis.axvline(column - 0.5, color=PALETTE['ink'], linewidth=2.0, zorder=4)
    for spine in axis.spines.values():
        spine.set_visible(False)
    for row in range(n_rows):
        for column in range(n_cols):
            value = matrix[row, column]
            if not numpy.isfinite(value):
                continue
            text, reference = _cell_text(value, spec)
            rgba = colour_map(norm(reference))
            axis.text(column, row, text, ha='center', va='center',
                      fontsize=spec.annot_fontsize, color=_annotation_colour(rgba))
    return image


def _annotation_colour(rgba) -> str:
    """Return white or dark ink, whichever contrasts more with the cell (WCAG)."""
    def linear(channel: float) -> float:
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
    luminance = (0.2126 * linear(rgba[0]) + 0.7152 * linear(rgba[1])
                 + 0.0722 * linear(rgba[2]))
    ink = PALETTE['ink']
    ink_luminance = 0.0223  # relative luminance of PALETTE['ink']
    on_white = 1.05 / (luminance + 0.05)
    on_ink = (luminance + 0.05) / (ink_luminance + 0.05)
    return 'white' if on_white > on_ink else ink


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
