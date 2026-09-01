"""Benchmark-wide normalised reward plots.

Reads the per-sequence performance traces aggregated under
``general/results/raw/<model>__<scenario>.json`` and produces, under
``general/results/normalized_reward/``:

* ``cumulative_<scenario>.png``  -- one line per model, cumulative
  normalised reward (running sum of ``perf_i``) across the 64
  evaluation sequences for that scenario.
* ``running_mean_<scenario>.png`` -- one line per model, normalised
  average reward (running mean of ``perf_i``) for that scenario.
* ``overview_cumulative_by_family.png`` -- multi-panel figure, one panel
  per scenario family, lines = models, x = sequence index, y =
  cumulative normalised reward (in-distribution baseline only).
* ``overview_running_mean_by_family.png`` -- as above but running mean.

Normalisation (1)
-----------------
``per_sequence_perf`` is already the normalised performance metric
defined as

.. math::
    \\text{perf}_i = \\frac{S_{\\text{worst},i} - S_{\\text{final},i}}
                          {S_{\\text{worst},i} - S_{\\text{best},i}}

i.e. the terminal reward of sequence *i* mapped into ``[0, 1]`` using the
per-sequence worst-case (no resources) and best feasible severity bounds.

Therefore:

* normalised **cumulative** reward after *n* sequences = ``cumsum(perf)``
* normalised **average** reward after *n* sequences = ``cumsum(perf) / n``

This script does NOT modify any other code, nor any of the input JSON
files; it only writes new PNG plots.

Usage
-----
    python -m general.scripts.plot_normalized_reward
"""

##########################
##  Imports externos    ##
##########################
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy


##########################
##  Constantes          ##
##########################
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(REPO_ROOT, 'general', 'results', 'raw')
OUT_DIR = os.path.join(REPO_ROOT, 'general', 'results', 'normalized_reward')

# Stable colour mapping for models (matches the project's narrative order)
MODEL_ORDER = ['pes_ql', 'pes_dql', 'pes_dqn', 'pes_rdqn', 'pes_a2c', 'pes_trf', 'pes_ens']
MODEL_COLORS = {
    'pes_ql':   '#1f77b4',
    'pes_dql':  '#ff7f0e',
    'pes_dqn':  '#2ca02c',
    'pes_rdqn': '#d62728',
    'pes_a2c':  '#9467bd',
    'pes_trf':  '#8c564b',
    'pes_ens':  '#000000',
}


##########################
##  Helpers             ##
##########################
def _load_all_raw() -> Dict[Tuple[str, str], dict]:
    """Return mapping ``(model, scenario) -> raw JSON dict``."""
    data: Dict[Tuple[str, str], dict] = {}
    for path in glob.glob(os.path.join(RAW_DIR, '*.json')):
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                d = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        model = d.get('model')
        scenario = d.get('scenario')
        if model and scenario and d.get('per_sequence_perf'):
            data[(model, scenario)] = d
    return data


def _models_present(data: Dict[Tuple[str, str], dict]) -> List[str]:
    """Return ordered list of models present in raw data."""
    present = set(m for (m, _s) in data.keys())
    return [m for m in MODEL_ORDER if m in present]


def _scenarios_present(data: Dict[Tuple[str, str], dict]) -> List[str]:
    """Return sorted list of scenarios present in raw data."""
    return sorted(set(s for (_m, s) in data.keys()))


def _family_of(data: Dict[Tuple[str, str], dict], scenario: str) -> Optional[str]:
    """Return the family label for a scenario (peek at any model entry)."""
    for (_m, s), d in data.items():
        if s == scenario:
            return d.get('family')
    return None


##########################
##  Single-scenario plots
##########################
def _plot_one_scenario(scenario: str,
                       data: Dict[Tuple[str, str], dict],
                       models: List[str]) -> None:
    """Plot cumulative and running-mean curves for one scenario."""
    # Cumulative
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for model in models:
        entry = data.get((model, scenario))
        if entry is None:
            continue
        perf = numpy.asarray(entry['per_sequence_perf'], dtype=numpy.float64)
        x = numpy.arange(1, len(perf) + 1)
        cum = numpy.cumsum(perf)
        ax.plot(x, cum, '-', color=MODEL_COLORS[model], linewidth=1.8,
                label=f'{model} (mean={perf.mean():.3f})')
    ax.set_title(f'Cumulative normalised reward — scenario: {scenario}')
    ax.set_xlabel('Sequence index')
    ax.set_ylabel('$\\Sigma_{i=1}^{n}$ perf$_i$')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'cumulative_{scenario}.png'),
                dpi=140, bbox_inches='tight')
    plt.close(fig)

    # Running mean
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for model in models:
        entry = data.get((model, scenario))
        if entry is None:
            continue
        perf = numpy.asarray(entry['per_sequence_perf'], dtype=numpy.float64)
        x = numpy.arange(1, len(perf) + 1)
        running = numpy.cumsum(perf) / x
        ax.plot(x, running, '-', color=MODEL_COLORS[model], linewidth=1.8,
                label=f'{model} (final={running[-1]:.3f})')
    ax.set_title(f'Normalised average reward (running mean) — scenario: {scenario}')
    ax.set_xlabel('Sequence index')
    ax.set_ylabel('Running mean of perf$_i$ $\\in [0, 1]$')
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'running_mean_{scenario}.png'),
                dpi=140, bbox_inches='tight')
    plt.close(fig)


##########################
##  Overview by family   ##
##########################
def _plot_overview_by_family(data: Dict[Tuple[str, str], dict],
                             models: List[str]) -> None:
    """Generate two grid overviews (cumulative + running mean) by family."""
    scenarios = _scenarios_present(data)
    families: Dict[str, List[str]] = {}
    for s in scenarios:
        fam = _family_of(data, s) or 'unknown'
        families.setdefault(fam, []).append(s)

    fam_order = [f for f in ['severity', 'length', 'structural', 'joint', 'unknown']
                 if f in families]

    for mode, ylabel, fname in [
        ('cum', '$\\Sigma$ perf$_i$', 'overview_cumulative_by_family.png'),
        ('run', 'Running mean perf$_i$', 'overview_running_mean_by_family.png'),
    ]:
        ncols = max(len(families[f]) for f in fam_order)
        nrows = len(fam_order)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(3.2 * ncols, 2.6 * nrows),
                                 squeeze=False, sharex=True)
        for r, fam in enumerate(fam_order):
            scen_list = families[fam]
            for c in range(ncols):
                ax = axes[r, c]
                if c >= len(scen_list):
                    ax.axis('off')
                    continue
                scenario = scen_list[c]
                for model in models:
                    entry = data.get((model, scenario))
                    if entry is None:
                        continue
                    perf = numpy.asarray(entry['per_sequence_perf'],
                                         dtype=numpy.float64)
                    x = numpy.arange(1, len(perf) + 1)
                    if mode == 'cum':
                        y = numpy.cumsum(perf)
                    else:
                        y = numpy.cumsum(perf) / x
                    ax.plot(x, y, '-', color=MODEL_COLORS[model], linewidth=1.2,
                            label=model)
                ax.set_title(f'{fam}/{scenario}', fontsize=8)
                ax.grid(True, alpha=0.25)
                if mode == 'run':
                    ax.set_ylim(-0.02, 1.05)
                if r == nrows - 1:
                    ax.set_xlabel('Sequence', fontsize=8)
                if c == 0:
                    ax.set_ylabel(ylabel, fontsize=8)
                ax.tick_params(labelsize=7)
        # single shared legend
        handles = [plt.Line2D([0], [0], color=MODEL_COLORS[m], linewidth=2.0, label=m)
                   for m in models]
        fig.legend(handles=handles, loc='lower center', ncol=len(models),
                   frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
        title = ('Cumulative normalised reward by scenario family'
                 if mode == 'cum'
                 else 'Normalised average reward (running mean) by scenario family')
        fig.suptitle(title + ' — normalisation (1): per-sequence worst/best feasible severity',
                     fontsize=11, y=1.0)
        fig.tight_layout(rect=(0, 0.03, 1, 0.98))
        fig.savefig(os.path.join(OUT_DIR, fname), dpi=140, bbox_inches='tight')
        plt.close(fig)


##########################
##  Main                ##
##########################
def main() -> None:
    """Generate benchmark-wide normalised reward plots."""
    os.makedirs(OUT_DIR, exist_ok=True)
    data = _load_all_raw()
    if not data:
        print('No raw benchmark JSONs with per_sequence_perf found.')
        return
    models = _models_present(data)
    scenarios = _scenarios_present(data)

    print('=' * 72)
    print('Benchmark-wide normalised reward plots (normalisation 1)')
    print('=' * 72)
    print(f'Models:     {models}')
    print(f'Scenarios:  {len(scenarios)}  '
          f'(first 5: {scenarios[:5]} ...)')
    print(f'Output dir: {os.path.relpath(OUT_DIR, REPO_ROOT)}')
    print('-' * 72)

    for scenario in scenarios:
        _plot_one_scenario(scenario, data, models)
    _plot_overview_by_family(data, models)

    n_per_scen = 2 * len(scenarios)
    print(f'Wrote {n_per_scen} per-scenario plots + 2 family overviews.')


if __name__ == '__main__':
    main()
