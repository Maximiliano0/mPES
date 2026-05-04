"""Visualisation of benchmark matrices.

Generates:

* ``heatmap_global_mean.png``     -- model x scenario performance heatmap.
* ``heatmap_ood_degradation.png`` -- baseline_mean - cell_mean.
* ``heatmap_welch_p.png``         -- log10(p) significance map.
* ``heatmap_action_kl.png``       -- action distribution drift vs baseline.
* ``per_sequence_histograms/<scenario>.png`` -- 7 KDE-like histograms
  overlayed per scenario.

Run after :mod:`general.aggregate`.
"""
##########################
##  Imports externos    ##
##########################
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy

##########################
##  Imports internos    ##
##########################
from .runner import GENERAL_ROOT

RESULTS_DIR = os.path.join(GENERAL_ROOT, 'results')
HIST_DIR = os.path.join(RESULTS_DIR, 'per_sequence_histograms')


def _load_summary() -> dict:
    path = os.path.join(RESULTS_DIR, 'matrix_summary.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _heatmap(matrix: numpy.ndarray, models: list, scenarios: list,
             title: str, out_path: str, cmap: str = 'viridis',
             fmt: str = '{:.3f}', center_zero: bool = False):
    fig, ax = plt.subplots(figsize=(max(10, len(scenarios) * 0.55),
                                    max(4, len(models) * 0.55)))
    if center_zero:
        vmax = float(numpy.nanmax(numpy.abs(matrix))) if matrix.size else 1.0
        im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')
    else:
        im = ax.imshow(matrix, cmap=cmap, aspect='auto')
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=60, ha='right', fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if not math.isnan(v):
                ax.text(j, i, fmt.format(v), ha='center', va='center',
                        color='white' if abs(v) > (numpy.nanmax(numpy.abs(matrix)) * 0.5) else 'black',
                        fontsize=6)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def _matrix_from(summary: dict, getter) -> numpy.ndarray:
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


def plot_all():
    """Render every benchmark figure into ``general/results/figures/``."""
    summary = _load_summary()
    models = summary['models']
    scenarios = summary['scenarios']
    baseline_id = summary['baseline_scenario']

    # --- Heatmaps ---
    M_mean = _matrix_from(summary, lambda c: c.get('global_mean_perf'))
    _heatmap(M_mean, models, scenarios,
             'Global mean performance (raw)',
             os.path.join(RESULTS_DIR, 'heatmap_global_mean.png'),
             cmap='viridis')

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
    _heatmap(M_degr, models, scenarios,
             'OOD degradation (baseline_mean - cell_mean)',
             os.path.join(RESULTS_DIR, 'heatmap_ood_degradation.png'),
             cmap='RdBu_r', fmt='{:+.3f}', center_zero=True)

    # log10(p) heatmap from precomputed CSV.
    welch_csv = os.path.join(RESULTS_DIR, 'matrix_welch_p.csv')
    if os.path.isfile(welch_csv):
        import csv as _csv
        with open(welch_csv, 'r', encoding='utf-8') as f:
            rows = list(_csv.reader(f))
        header = rows[0][1:]
        body = numpy.full((len(rows) - 1, len(header)), numpy.nan)
        for i, row in enumerate(rows[1:]):
            for j, val in enumerate(row[1:]):
                if val:
                    p = max(float(val), 1e-300)
                    body[i, j] = math.log10(p)
        _heatmap(body, [r[0] for r in rows[1:]], header,
                 'log10(Welch p) vs in-distribution baseline',
                 os.path.join(RESULTS_DIR, 'heatmap_welch_logp.png'),
                 cmap='magma', fmt='{:.1f}')

    # KL action drift from precomputed CSV.
    kl_csv = os.path.join(RESULTS_DIR, 'matrix_action_kl.csv')
    if os.path.isfile(kl_csv):
        import csv as _csv
        with open(kl_csv, 'r', encoding='utf-8') as f:
            rows = list(_csv.reader(f))
        header = rows[0][1:]
        body = numpy.full((len(rows) - 1, len(header)), numpy.nan)
        for i, row in enumerate(rows[1:]):
            for j, val in enumerate(row[1:]):
                if val:
                    body[i, j] = float(val)
        _heatmap(body, [r[0] for r in rows[1:]], header,
                 'KL(action_dist || in-distribution policy)',
                 os.path.join(RESULTS_DIR, 'heatmap_action_kl.png'),
                 cmap='cividis', fmt='{:.2f}')

    # --- Per-sequence histograms ---
    os.makedirs(HIST_DIR, exist_ok=True)
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
        ax.legend(fontsize=7, loc='best')
        ax.set_xlim(0, 1)
        fig.tight_layout()
        fig.savefig(os.path.join(HIST_DIR, f'{s}.png'), dpi=120,
                    bbox_inches='tight')
        plt.close(fig)


def _main():
    plot_all()
    print(f'[plot_matrix] figures written to {RESULTS_DIR}')


if __name__ == '__main__':
    _main()
