"""Aggregation, statistics and reporting for the mPES benchmark.

Reads the per-cell payloads written by :mod:`general.scripts.benchmark` and
produces, for each suite:

* ``matrices/<metric>.csv`` -- ``model x scenario`` matrices.
* ``summary.json``          -- machine-readable consolidation.
* ``report.md``             -- executive Markdown summary.

Every stress metric compares a cell against the same model's
``sev_empirical`` reference condition.

Usage
-----
.. code-block:: powershell

    python -m general.scripts.analysis --suite individual
    python -m general.scripts.analysis --suite ensemble
"""
##########################
##  Imports externos    ##
##########################
import argparse
import csv
import datetime as _dt
import json
import math
import os

import numpy

##########################
##  Imports internos    ##
##########################
from .benchmark import (REFERENCE_SCENARIO, SUITES, SUITE_PACKAGES, load_cells,
                        matrices_dir, report_path, scenario_catalogue,
                        summary_path, suite_dir)
from .plotting import cohen_d, kl_divergence, welch_test


MATRIX_METRICS = ('global_mean', 'std', 'min', 'max', 'stress_degradation',
                  'welch_p', 'welch_logp', 'cohen_d', 'action_kl')
FAMILY_LABELS = {
    'severity': 'Severidad',
    'length': 'Longitud',
    'joint': 'Conjunta',
    'structural': 'Estructura',
}


###############
##  Matrix IO
###############
def _write_matrix(path: str, models: "list[str]", scenarios: "list[str]", getter) -> None:
    """Write one ``model x scenario`` CSV using ``getter(model, scenario)``."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['model'] + scenarios)
        for model in models:
            writer.writerow([model] + [getter(model, scenario) for scenario in scenarios])


def _format(value, template: str = '{:.6f}') -> str:
    """Format a metric, mapping missing or non-finite values to an empty cell."""
    if value is None:
        return ''
    number = float(value)
    return '' if math.isnan(number) or math.isinf(number) else template.format(number)


###############
##  Aggregation
###############
def aggregate(suite: str = 'individual', reference_pkg: str = 'pes_dqn') -> str:
    """Build the matrices and the summary JSON of one suite."""
    if suite not in SUITE_PACKAGES:
        raise ValueError(f'Unknown benchmark suite: {suite}')
    data = load_cells(suite)
    if not data:
        raise RuntimeError(f'No cells found for suite {suite!r}.')

    catalogue = scenario_catalogue(reference_pkg)
    scenarios = [scenario.scenario_id for scenario in catalogue]
    models = [model for model in SUITE_PACKAGES[suite] if model in data]

    def cell(model, scenario) -> dict:
        return data.get(model, {}).get(scenario, {})

    def performance(model, scenario) -> numpy.ndarray:
        return numpy.asarray(cell(model, scenario).get('per_sequence_perf', []), dtype=float)

    def actions(model, scenario) -> numpy.ndarray:
        distribution = cell(model, scenario).get('action_distribution')
        return numpy.asarray(distribution, dtype=float) if distribution else numpy.array([])

    reference_perf = {model: performance(model, REFERENCE_SCENARIO) for model in models}
    reference_actions = {model: actions(model, REFERENCE_SCENARIO) for model in models}
    welch_cache: dict = {}

    def welch(model, scenario):
        key = (model, scenario)
        if key not in welch_cache:
            current, baseline = performance(model, scenario), reference_perf.get(model)
            if scenario == REFERENCE_SCENARIO or current.size == 0 or baseline is None or baseline.size == 0:
                welch_cache[key] = (float('nan'),) * 3
            else:
                welch_cache[key] = welch_test(current, baseline)
        return welch_cache[key]

    def degradation(model, scenario):
        baseline = cell(model, REFERENCE_SCENARIO).get('global_mean_perf')
        current = cell(model, scenario).get('global_mean_perf')
        return None if baseline is None or current is None else baseline - current

    getters = {
        'global_mean': lambda m, s: _format(cell(m, s).get('global_mean_perf')),
        'std': lambda m, s: _format(cell(m, s).get('std_perf')),
        'min': lambda m, s: _format(cell(m, s).get('min_perf')),
        'max': lambda m, s: _format(cell(m, s).get('max_perf')),
        'stress_degradation': lambda m, s: _format(degradation(m, s)),
        'welch_p': lambda m, s: _format(welch(m, s)[1], '{:.6e}'),
        'welch_logp': lambda m, s: _format(welch(m, s)[2]),
        'cohen_d': lambda m, s: _format(
            cohen_d(performance(m, s), reference_perf.get(m, numpy.array([])))
            if s != REFERENCE_SCENARIO else None),
        'action_kl': lambda m, s: _format(
            kl_divergence(actions(m, s), reference_actions.get(m, numpy.array([])))
            if s != REFERENCE_SCENARIO else None),
    }
    for metric in MATRIX_METRICS:
        _write_matrix(os.path.join(matrices_dir(suite), f'{metric}.csv'),
                      models, scenarios, getters[metric])

    summary = {
        'suite': suite,
        'reference_scenario': REFERENCE_SCENARIO,
        'models': models,
        'scenarios': scenarios,
        'cells': data,
    }
    os.makedirs(suite_dir(suite), exist_ok=True)
    with open(summary_path(suite), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    return summary_path(suite)


###############
##  Report
###############
def write_report(suite: str = 'individual') -> str:
    """Render the executive Markdown report of one suite."""
    with open(summary_path(suite), 'r', encoding='utf-8') as handle:
        summary = json.load(handle)
    models, scenarios = summary['models'], summary['scenarios']
    reference, cells = summary['reference_scenario'], summary['cells']

    lines = ['# mPES Under Stress Experiments — ' + suite, '',
             f'_Generated: {_dt.datetime.now(_dt.timezone.utc).isoformat()}_', '',
             f'**Reference condition:** `{reference}`', '',
             f'**Models:** {len(models)} — {", ".join(models)}',
             f'**Scenarios:** {len(scenarios)}', '',
             '## 1. Per-model best / worst', '',
             '| Model | Reference | Best scenario | Best | Worst scenario | Worst | Mean degradation |',
             '|---|---:|---|---:|---|---:|---:|']
    for model in models:
        baseline = cells.get(model, {}).get(reference, {}).get('global_mean_perf')
        means = {scenario: value for scenario, value in (
            (s, cells.get(model, {}).get(s, {}).get('global_mean_perf'))
            for s in scenarios if s != reference) if value is not None}
        if baseline is None or not means:
            lines.append(f'| {model} | - | - | - | - | - | - |')
            continue
        best = max(means, key=means.__getitem__)
        worst = min(means, key=means.__getitem__)
        mean_degradation = float(numpy.mean([baseline - value for value in means.values()]))
        lines.append(f'| {model} | {baseline:.4f} | `{best}` | {means[best]:.4f} | '
                     f'`{worst}` | {means[worst]:.4f} | {mean_degradation:+.4f} |')

    lines += ['', '## 2. Mean degradation by scenario family', '',
              '| Family | # scenarios | Mean degradation |', '|---|---:|---:|']
    families: dict = {}
    for scenario in scenarios:
        for model in models:
            family = cells.get(model, {}).get(scenario, {}).get('family')
            if family:
                families.setdefault(family, []).append(scenario)
                break
    for family, family_scenarios in families.items():
        if family == 'baseline':
            continue
        degradations = []
        for model in models:
            baseline = cells.get(model, {}).get(reference, {}).get('global_mean_perf')
            if baseline is None:
                continue
            for scenario in family_scenarios:
                value = cells.get(model, {}).get(scenario, {}).get('global_mean_perf')
                if value is not None:
                    degradations.append(baseline - value)
        mean = float(numpy.mean(degradations)) if degradations else float('nan')
        label = FAMILY_LABELS.get(family, family)
        lines.append(f'| {label} | {len(family_scenarios)} | {mean:+.4f} |')

    lines += ['', '## 3. Most degraded cells', '',
              '| Rank | Model | Scenario | Reference | Cell | Degradation |',
              '|---:|---|---|---:|---:|---:|']
    rows = []
    for model in models:
        baseline = cells.get(model, {}).get(reference, {}).get('global_mean_perf')
        if baseline is None:
            continue
        for scenario in scenarios:
            value = cells.get(model, {}).get(scenario, {}).get('global_mean_perf')
            if scenario != reference and value is not None:
                rows.append((baseline - value, model, scenario, baseline, value))
    rows.sort(reverse=True)
    for rank, (delta, model, scenario, baseline, value) in enumerate(rows[:5], 1):
        lines.append(f'| {rank} | {model} | `{scenario}` | {baseline:.4f} | '
                     f'{value:.4f} | {delta:+.4f} |')

    lines += ['', '## 4. Artefacts', '',
              '* Matrices: `matrices/*.csv`',
              '* Figures: `figures/*.png`',
              '* Cells: `cells/<model>__<scenario>.json`', '']
    with open(report_path(suite), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))
    return report_path(suite)


def main() -> None:
    """Aggregate the requested suites and render their reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=SUITES, action='append')
    parser.add_argument('--reference-pkg', default='pes_dqn')
    parser.add_argument('--no-report', action='store_true')
    args = parser.parse_args()
    for suite in args.suite or list(SUITES):
        print(f'[analysis] {suite} summary -> {aggregate(suite, args.reference_pkg)}')
        if not args.no_report:
            print(f'[analysis] {suite} report  -> {write_report(suite)}')


if __name__ == '__main__':
    main()
