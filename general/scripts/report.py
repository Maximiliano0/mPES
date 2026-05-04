"""Compose ``general/results/benchmark_report.md`` from the aggregated
matrices and per-cell raw JSONs.

The report is the executive summary the user reads to compare models:

* Per-model "best/worst scenario" callouts.
* Family-wise mean degradation (severity / length / joint / structural).
* Top 5 most-degraded cells overall.
* References to all heatmaps and per-scenario histograms.

Run after :mod:`general.scripts.aggregate` and :mod:`general.scripts.plot_matrix`.
"""
##########################
##  Imports externos    ##
##########################
import datetime as _dt
import json
import os

import numpy

##########################
##  Imports internos    ##
##########################
from .runner import GENERAL_ROOT

RESULTS_DIR = os.path.join(GENERAL_ROOT, 'results')


def _load_summary() -> dict:
    with open(os.path.join(RESULTS_DIR, 'matrix_summary.json'),
              'r', encoding='utf-8') as f:
        return json.load(f)


def write_report() -> str:
    """Render ``benchmark_report.md`` from the aggregated summary JSON."""
    summary = _load_summary()
    models = summary['models']
    scenarios = summary['scenarios']
    baseline_id = summary['baseline_scenario']
    cells = summary['cells']

    lines = []
    lines.append('# mPES OOD Benchmark Report')
    lines.append('')
    lines.append(f'_Generated: {_dt.datetime.utcnow().isoformat()}Z_')
    lines.append('')
    lines.append(f'**Reference baseline scenario:** `{baseline_id}`')
    lines.append('')
    lines.append(f'**Models evaluated:** {len(models)} -- {", ".join(models)}')
    lines.append(f'**Scenarios evaluated:** {len(scenarios)}')
    lines.append('')

    # --- Per-model summary ---
    lines.append('## 1. Per-model best / worst')
    lines.append('')
    lines.append('| Model | Baseline mean | Best scenario | Best mean | Worst scenario | Worst mean | Mean degradation |')
    lines.append('|---|---:|---|---:|---|---:|---:|')
    for m in models:
        baseline = cells.get(m, {}).get(baseline_id, {}).get('global_mean_perf')
        if baseline is None:
            lines.append(f'| {m} | - | - | - | - | - | - |')
            continue
        means = {s: cells.get(m, {}).get(s, {}).get('global_mean_perf')
                 for s in scenarios if s != baseline_id}
        means = {k: v for k, v in means.items() if v is not None}
        if not means:
            lines.append(f'| {m} | {baseline:.4f} | - | - | - | - | - |')
            continue
        best_s = max(means, key=means.get)
        worst_s = min(means, key=means.get)
        avg_degr = float(numpy.mean([baseline - v for v in means.values()]))
        lines.append(
            f'| {m} | {baseline:.4f} | `{best_s}` | {means[best_s]:.4f} | '
            f'`{worst_s}` | {means[worst_s]:.4f} | {avg_degr:+.4f} |')
    lines.append('')

    # --- Family-wise degradation ---
    lines.append('## 2. Mean degradation by scenario family')
    lines.append('')
    families: dict = {}
    for s in scenarios:
        family = None
        for m in models:
            f = cells.get(m, {}).get(s, {}).get('family')
            if f:
                family = f
                break
        if family:
            families.setdefault(family, []).append(s)
    lines.append('| Family | # scenarios | Mean degradation across models |')
    lines.append('|---|---:|---:|')
    for fam, ss in families.items():
        if fam == 'baseline':
            continue
        degrs = []
        for m in models:
            base = cells.get(m, {}).get(baseline_id, {}).get('global_mean_perf')
            if base is None:
                continue
            for s in ss:
                v = cells.get(m, {}).get(s, {}).get('global_mean_perf')
                if v is not None:
                    degrs.append(base - v)
        mean_d = float(numpy.mean(degrs)) if degrs else float('nan')
        lines.append(f'| {fam} | {len(ss)} | {mean_d:+.4f} |')
    lines.append('')

    # --- Top 5 most-degraded cells ---
    lines.append('## 3. Top 5 most-degraded cells')
    lines.append('')
    rows = []
    for m in models:
        base = cells.get(m, {}).get(baseline_id, {}).get('global_mean_perf')
        if base is None:
            continue
        for s in scenarios:
            if s == baseline_id:
                continue
            v = cells.get(m, {}).get(s, {}).get('global_mean_perf')
            if v is None:
                continue
            rows.append((base - v, m, s, base, v))
    rows.sort(reverse=True)
    lines.append('| Rank | Model | Scenario | Baseline | Cell | Degradation |')
    lines.append('|---:|---|---|---:|---:|---:|')
    for i, (d, m, s, b, v) in enumerate(rows[:5], 1):
        lines.append(f'| {i} | {m} | `{s}` | {b:.4f} | {v:.4f} | {d:+.4f} |')
    lines.append('')

    # --- Figure references ---
    lines.append('## 4. Figures')
    lines.append('')
    for fig in ('heatmap_global_mean.png', 'heatmap_ood_degradation.png',
                'heatmap_welch_logp.png', 'heatmap_action_kl.png'):
        if os.path.isfile(os.path.join(RESULTS_DIR, fig)):
            lines.append(f'* ![{fig}]({fig})')
    lines.append('')
    lines.append('Per-scenario histograms in `per_sequence_histograms/`.')
    lines.append('')

    out = os.path.join(RESULTS_DIR, 'benchmark_report.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return out


def _main():
    out = write_report()
    print(f'[report] wrote {out}')


if __name__ == '__main__':
    _main()
