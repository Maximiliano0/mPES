"""Figure generation for the mPES benchmark.

Replaces the previous five plotting modules with a single generator that
renders, per suite, a coherently numbered figure set under
``results/<suite>/figures``:

===============================================  ==========================================
``01_desempeno_por_escenario``                   mean normalised performance
``02_degradacion_por_escenario``                 degradation vs ``sev_empirical``
``03_welch_logp_por_escenario``                  ``log10(p)`` of the Welch t-test
``04_kl_acciones_por_escenario``                 KL of the action distributions
``05_curvas_por_familia``                        per-sequence curves by stress family
``06_curvas_estresores_universales``             per-sequence curves under extrapolation
``07_cohen_d_por_escenario``                     standardised effect size
``08_ranking_desempeno``                         reference vs stress ranking
``09_degradacion_por_familia``                   sensitivity per stress family
``10_desempeno_vs_estabilidad``                  mean performance vs dispersion
``11_perfiles_generalizacion``                   response profile per family
``12_pares_welch_logp`` / ``13_pares_cohen_d`` / ``14_pares_kl``  pairwise model contrasts
``histogramas/<scenario>``                       per-sequence distribution per scenario
``recompensa/<scenario>``                        cumulative and running-mean reward
===============================================  ==========================================

Usage
-----
.. code-block:: powershell

    python -m general.scripts.figures --suite individual
    python -m general.scripts.figures --only matrices
"""
##########################
##  Imports externos    ##
##########################
import argparse
import json
import os

from matplotlib import colors as mcolors
from matplotlib import pyplot
import numpy

##########################
##  Imports internos    ##
##########################
from .benchmark import (REFERENCE_MODEL, REFERENCE_SCENARIO, SUITES,
                        SUITE_PACKAGES, comparison_metrics_path, figures_dir,
                        load_cells, matrices_dir, summary_path)
from .plotting import (ALPHA_LEVELS, MODEL_LINE_COLOURS, HeatmapSpec, cohen_d,
                       heatmap, histogram_pmf, read_matrix_csv, save_figure,
                       style_axes, symmetric_kl, welch_test)


ENSEMBLE_REFERENCE = 'pes_ens'
FAMILY_ORDER = ('severity', 'length', 'joint', 'structural')
FAMILY_LABELS = {'severity': 'Severidad', 'length': 'Longitud',
                 'joint': 'Conjunta', 'structural': 'Estructura'}
CURVE_SCENARIOS = ('sev_bimodal', 'sev_gauss_high', 'sev_beta_highskew',
                   'len_poisson', 'len_extrapolate_long', 'joint_high_long')
UNIVERSAL_SCENARIOS = ('sev_extrapolate_high', 'joint_extrap_both',
                       'len_extrapolate_long')
PANEL_COLOURS = {'contrast': '#4c72b0', 'joint': '#dd8452',
                 'best': '#c44e52', 'ens': '#111111'}


###############
##  Data access
###############
def load_model_cells() -> dict:
    """Return ``{model: {scenario: cell}}`` merged across every suite."""
    merged: dict = {}
    for suite in SUITES:
        for model, scenarios in load_cells(suite).items():
            merged.setdefault(model, {}).update(scenarios)
    return merged


def suite_models(suite: str) -> "list[str]":
    """Models plotted for ``suite``, always led by the reference model."""
    packages = [model for model in SUITE_PACKAGES[suite] if model != REFERENCE_MODEL]
    return [REFERENCE_MODEL] + packages


def _performance(cells: dict, model: str, scenario: str) -> numpy.ndarray:
    """Per-sequence performance vector of one cell."""
    return numpy.asarray(
        cells.get(model, {}).get(scenario, {}).get('per_sequence_perf', []), dtype=float)


def _scenarios_of(summary: dict) -> "list[str]":
    """Scenario order as recorded in the suite summary."""
    return list(summary['scenarios'])


def _load_summary(suite: str) -> dict:
    """Load the aggregated summary of one suite."""
    with open(summary_path(suite), 'r', encoding='utf-8') as handle:
        return json.load(handle)


###############
##  Matrix figures
###############
def render_matrix_figures(suite: str) -> None:
    """Render the metric heatmaps from the aggregated CSV matrices."""
    output = figures_dir(suite)
    specs = {
        'global_mean': ('01_desempeno_por_escenario', HeatmapSpec(
            title=f'{suite.capitalize()}: desempeño normalizado medio por modelo y escenario',
            cbar_label='Rendimiento normalizado medio', cmap='mpes_perf', fmt='{:.2f}')),
        'stress_degradation': ('02_degradacion_por_escenario', HeatmapSpec(
            title=f'{suite.capitalize()}: degradación respecto a {REFERENCE_SCENARIO}',
            cbar_label='baseline_mean − cell_mean; positivo = pérdida',
            cmap='mpes_div', fmt='{:+.3f}')),
        'welch_logp': ('03_welch_logp_por_escenario', HeatmapSpec(
            title=f'{suite.capitalize()}: log10(p) del test t de Welch',
            cbar_label='log10(p); más bajo = evidencia más fuerte',
            cmap='mpes_pval', vmin=-10.0, vmax=0.0, fmt='{:.2f}',
            clip_low_label='≤-10', cbar_ticks=[-10.0, *ALPHA_LEVELS, 0.0])),
        'cohen_d': ('07_cohen_d_por_escenario', HeatmapSpec(
            title=f'{suite.capitalize()}: tamaño de efecto (d de Cohen)',
            cbar_label='d de Cohen; positivo = mejor que la referencia',
            cmap='mpes_div', fmt='{:+.2f}')),
    }
    for metric, (name, spec) in specs.items():
        path = os.path.join(matrices_dir(suite), f'{metric}.csv')
        if not os.path.isfile(path):
            continue
        models, scenarios, matrix = read_matrix_csv(path)
        _autoscale(spec, matrix)
        heatmap(matrix, models, scenarios, os.path.join(output, name), spec)
    _render_action_kl(suite, output)


def _autoscale(spec: HeatmapSpec, matrix: numpy.ndarray) -> None:
    """Fill missing colour limits from the data, keeping diverging maps centred."""
    if spec.vmin is not None and spec.vmax is not None:
        return
    finite = matrix[numpy.isfinite(matrix)]
    if not finite.size:
        spec.vmin, spec.vmax = 0.0, 1.0
        return
    if spec.cmap == 'mpes_div':
        bound = max(float(numpy.max(numpy.abs(finite))), 1e-3)
        spec.vmin, spec.vmax = -bound, bound
    else:
        spec.vmin = float(numpy.floor(finite.min() * 100) / 100)
        spec.vmax = float(numpy.ceil(finite.max() * 100) / 100)


def _render_action_kl(suite: str, output: str) -> None:
    """Render the action-KL heatmap on a logarithmic colour scale."""
    path = os.path.join(matrices_dir(suite), 'action_kl.csv')
    if not os.path.isfile(path):
        return
    models, scenarios, matrix = read_matrix_csv(path)
    positive = matrix[numpy.isfinite(matrix) & (matrix > 0)]
    if not positive.size:
        return
    floor = max(1e-4, float(numpy.percentile(positive, 5)))
    ceiling = float(numpy.max(positive))
    display = numpy.where(numpy.isfinite(matrix) & (matrix <= 0), floor, matrix)
    heatmap(display, models, scenarios,
            os.path.join(output, '04_kl_acciones_por_escenario'),
            HeatmapSpec(
                title=f'{suite.capitalize()}: divergencia KL de acciones vs {REFERENCE_SCENARIO}',
                cbar_label='KL(escenario ‖ referencia), escala logarítmica',
                cmap='mpes_kl', fmt='{:.2f}',
                norm=mcolors.LogNorm(vmin=floor, vmax=ceiling)))


###############
##  Curve figures
###############
def render_curve_figures(suite: str, cells: dict) -> None:
    """Render the per-sequence curves by family and under universal stressors."""
    output = figures_dir(suite)
    models = suite_models(suite)
    curve_models = models

    figure, axes = pyplot.subplots(2, 3, figsize=(16.5, 8.5), sharey=True)
    for axis, scenario in zip(axes.flat, CURVE_SCENARIOS):
        for index, model in enumerate(curve_models):
            values = numpy.sort(_performance(cells, model, scenario))
            if not values.size:
                continue
            is_ens = model == ENSEMBLE_REFERENCE
            axis.plot(numpy.arange(1, values.size + 1), values,
                      color=PANEL_COLOURS['ens'] if is_ens
                      else MODEL_LINE_COLOURS[index % len(MODEL_LINE_COLOURS)],
                      linewidth=2.8 if is_ens else 1.6, label=model)
        axis.set_title(scenario, fontsize=11)
        axis.set_ylim(0, 1.02)
        style_axes(axis)
    for axis in axes[1]:
        axis.set_xlabel('Secuencia ordenada')
    for axis in axes[:, 0]:
        axis.set_ylabel('Rendimiento normalizado')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=4, frameon=False,
                  bbox_to_anchor=(0.5, -0.02))
    figure.suptitle(f'{suite.capitalize()}: desempeño por secuencia y familia de perturbación',
                    fontweight='semibold')
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    save_figure(figure, os.path.join(output, '05_curvas_por_familia'))

    _render_universal(suite, cells, models, output)


def _stress_mean(cells: dict, model: str, scenarios: "list[str]") -> float:
    """Mean performance of one model over the non-reference scenarios."""
    values = [cells.get(model, {}).get(scenario, {}).get('global_mean_perf')
              for scenario in scenarios if scenario != REFERENCE_SCENARIO]
    values = [value for value in values if value is not None]
    return float(numpy.mean(values)) if values else float('-inf')


def _render_universal(suite: str, cells: dict, models: "list[str]", output: str) -> None:
    """Render the three universal-stressor panels with per-model means."""
    scenarios = sorted({scenario for model in models
                        for scenario in cells.get(model, {})})
    if suite == 'individual':
        best = 'pes_trf'
        partners = {'sev_extrapolate_high': ('pes_dql', 'contrast'),
                    'joint_extrap_both': ('pes_a2c', 'joint'),
                    'len_extrapolate_long': ('pes_dqn', 'contrast')}
    else:
        candidates = [m for m in models if m not in (REFERENCE_MODEL, ENSEMBLE_REFERENCE)]
        best = max(candidates, key=lambda model: _stress_mean(cells, model, scenarios))
        others = [m for m in candidates if m != best]

        def weakest(scenario: str) -> str:
            return min(others, key=lambda model: cells.get(model, {}).get(
                scenario, {}).get('global_mean_perf', float('inf')))

        partners = {'sev_extrapolate_high': (weakest('sev_extrapolate_high'), 'contrast'),
                    'joint_extrap_both': (weakest('joint_extrap_both'), 'joint'),
                    'len_extrapolate_long': (ENSEMBLE_REFERENCE, 'ens')}

    figure, axes = pyplot.subplots(1, 3, figsize=(15.5, 5.0), sharey=True)
    for axis, scenario in zip(axes, UNIVERSAL_SCENARIOS):
        partner, role = partners[scenario]
        for model, colour in ((partner, PANEL_COLOURS[role]), (best, PANEL_COLOURS['best'])):
            values = numpy.sort(_performance(cells, model, scenario))
            if not values.size:
                continue
            axis.plot(numpy.arange(1, values.size + 1), values, color=colour,
                      linewidth=2.8 if model == ENSEMBLE_REFERENCE else 1.9, label=model)
            axis.axhline(values.mean(), color=colour, linestyle=':', linewidth=1.2)
        axis.set_title(scenario, fontsize=11)
        axis.set_xlabel('Secuencia ordenada')
        axis.set_ylim(0, 1.02)
        style_axes(axis)
        axis.legend(frameon=False, loc='upper left', fontsize=9)
    axes[0].set_ylabel('Rendimiento normalizado')
    figure.suptitle(f'{suite.capitalize()}: desempeño bajo los estresores universales '
                    '(línea punteada = media)', fontweight='semibold')
    figure.tight_layout(rect=(0, 0.02, 1, 0.93))
    save_figure(figure, os.path.join(output, '06_curvas_estresores_universales'))


###############
##  Comparison figures
###############
def _model_metrics(summary: dict, model: str) -> dict:
    """Reference, stress and per-family statistics of one model."""
    reference = summary['reference_scenario']
    cells = summary['cells'].get(model, {})
    means = {scenario: cell['global_mean_perf'] for scenario, cell in cells.items()
             if cell.get('global_mean_perf') is not None}
    baseline = means.get(reference, float('nan'))
    stress = {scenario: value for scenario, value in means.items() if scenario != reference}
    families: dict = {}
    for scenario, value in stress.items():
        family = cells.get(scenario, {}).get('family')
        if family in FAMILY_ORDER:
            families.setdefault(family, []).append(baseline - value)
    worst = max(stress, key=lambda s: baseline - stress[s]) if stress else None
    return {
        'reference_mean': baseline,
        'stress_mean': float(numpy.mean(list(stress.values()))) if stress else float('nan'),
        'mean_degradation': float(numpy.mean([baseline - v for v in stress.values()]))
        if stress else float('nan'),
        'worst_scenario': worst,
        'worst_degradation': (baseline - stress[worst]) if worst else float('nan'),
        'family_degradation': {family: float(numpy.mean(values))
                               for family, values in families.items()},
        'stress_std': float(numpy.mean([cells[s]['std_perf'] for s in stress
                                        if cells[s].get('std_perf') is not None]))
        if stress else float('nan'),
    }


def render_comparison_figures(suite: str) -> dict:
    """Render the ranking, sensitivity, stability and pairwise figures."""
    summary = _load_summary(suite)
    models, scenarios = summary['models'], _scenarios_of(summary)
    metrics = {model: _model_metrics(summary, model) for model in models}
    output = figures_dir(suite)

    _plot_ranking(suite, models, metrics, output)
    _plot_family_sensitivity(suite, models, metrics, output)
    _plot_stability(suite, models, metrics, output)
    _plot_generalisation(suite, summary, models, scenarios, output)
    pairwise = _plot_pairwise(suite, summary, models, scenarios, output)

    payload = {'suite': suite, 'reference_scenario': summary['reference_scenario'],
               'models': metrics, 'pairwise': pairwise}
    with open(comparison_metrics_path(suite), 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return payload


def _plot_ranking(suite: str, models: "list[str]", metrics: dict, output: str) -> None:
    """Horizontal ranking of reference versus stress performance."""
    order = sorted(models, key=lambda model: metrics[model]['stress_mean'], reverse=True)
    positions = numpy.arange(len(order))
    width = 0.36
    figure, axis = pyplot.subplots(figsize=(10.5, 5.8))
    axis.barh(positions + width / 2, [metrics[m]['reference_mean'] for m in order],
              width, label=f'Condición {REFERENCE_SCENARIO}', color='#a7c8db')
    axis.barh(positions - width / 2, [metrics[m]['stress_mean'] for m in order],
              width, label='Media bajo estrés', color='#1f6e83')
    axis.set_yticks(positions, order)
    axis.invert_yaxis()
    axis.set_xlim(0, 1.02)
    axis.set_xlabel('Rendimiento normalizado')
    axis.set_title(f'{suite.capitalize()}: ranking de desempeño')
    axis.legend(frameon=False, ncol=2, loc='lower right')
    style_axes(axis)
    for row, model in enumerate(order):
        axis.text(metrics[model]['stress_mean'] + 0.012, row - width / 2,
                  f"{metrics[model]['stress_mean']:.3f}", va='center', fontsize=9)
    save_figure(figure, os.path.join(output, '08_ranking_desempeno'))


def _plot_family_sensitivity(suite: str, models: "list[str]", metrics: dict,
                             output: str) -> None:
    """Grouped bars of the mean degradation per stress family."""
    families = [family for family in FAMILY_ORDER
                if any(family in metrics[model]['family_degradation'] for model in models)]
    positions = numpy.arange(len(families))
    width = 0.82 / max(len(models), 1)
    figure, axis = pyplot.subplots(figsize=(11, 5.8))
    for index, model in enumerate(models):
        values = [metrics[model]['family_degradation'].get(family, numpy.nan)
                  for family in families]
        axis.bar(positions + (index - (len(models) - 1) / 2) * width, values, width,
                 label=model, color=MODEL_LINE_COLOURS[index % len(MODEL_LINE_COLOURS)])
    axis.axhline(0, color='#252525', linewidth=0.8)
    axis.set_xticks(positions, [FAMILY_LABELS[family] for family in families])
    axis.set_ylabel(f'Degradación frente a {REFERENCE_SCENARIO}')
    axis.set_title(f'{suite.capitalize()}: sensibilidad media al estrés')
    axis.legend(frameon=False, ncol=2, loc='upper left')
    style_axes(axis)
    save_figure(figure, os.path.join(output, '09_degradacion_por_familia'))


def _plot_stability(suite: str, models: "list[str]", metrics: dict, output: str) -> None:
    """Scatter of mean stress performance against sequence-level dispersion."""
    figure, axis = pyplot.subplots(figsize=(8.2, 6.2))
    x_values = [metrics[model]['stress_std'] for model in models]
    y_values = [metrics[model]['stress_mean'] for model in models]
    axis.scatter(x_values, y_values, s=105, color='#e76f51',
                 edgecolor='white', linewidth=1.2)
    for model, x_value, y_value in zip(models, x_values, y_values):
        axis.annotate(model, (x_value, y_value), xytext=(7, 5),
                      textcoords='offset points', fontsize=9)
    axis.set_xlabel('Desviación estándar media por escenario')
    axis.set_ylabel('Rendimiento medio bajo estrés')
    axis.set_title(f'{suite.capitalize()}: desempeño y estabilidad')
    style_axes(axis)
    save_figure(figure, os.path.join(output, '10_desempeno_vs_estabilidad'))


def _plot_generalisation(suite: str, summary: dict, models: "list[str]",
                         scenarios: "list[str]", output: str) -> None:
    """Per-family response profiles across the stress catalogue."""
    grouped = {family: [scenario for scenario in scenarios
                        if scenario != summary['reference_scenario']
                        and summary['cells'].get(models[0], {}).get(
                            scenario, {}).get('family') == family]
               for family in FAMILY_ORDER}
    figure, axes = pyplot.subplots(2, 2, figsize=(14, 9), squeeze=False)
    for axis, family in zip(axes.flatten(), FAMILY_ORDER):
        family_scenarios = grouped[family]
        if not family_scenarios:
            axis.axis('off')
            continue
        positions = numpy.arange(len(family_scenarios))
        for index, model in enumerate(models):
            values = [summary['cells'].get(model, {}).get(scenario, {}).get(
                'global_mean_perf', numpy.nan) for scenario in family_scenarios]
            axis.plot(positions, values, marker='o', linewidth=2.0, markersize=4.5,
                      color=MODEL_LINE_COLOURS[index % len(MODEL_LINE_COLOURS)], label=model)
        axis.set_xticks(positions, family_scenarios, rotation=42, ha='right', fontsize=8)
        axis.set_ylim(0.6, 1.02)
        axis.set_ylabel('Rendimiento normalizado')
        axis.set_title(FAMILY_LABELS[family])
        style_axes(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=4, frameon=False,
                  bbox_to_anchor=(0.5, 0.005))
    figure.suptitle(f'{suite.capitalize()}: perfiles de generalización',
                    fontsize=14, fontweight='semibold')
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    save_figure(figure, os.path.join(output, '11_perfiles_generalizacion'))


def _plot_pairwise(suite: str, summary: dict, models: "list[str]",
                   scenarios: "list[str]", output: str) -> dict:
    """Pairwise Welch, Cohen and KL contrasts between the models of a suite."""
    reference = summary['reference_scenario']
    common = [scenario for scenario in scenarios if scenario != reference
              and all(scenario in summary['cells'].get(model, {}) for model in models)]
    vectors = {model: numpy.asarray(
        [value for scenario in common
         for value in summary['cells'][model][scenario].get('per_sequence_perf', [])],
        dtype=float) for model in models}
    distributions = {model: histogram_pmf(vectors[model]) for model in models}

    size = len(models)
    log_p = numpy.full((size, size), numpy.nan)
    effect = numpy.full((size, size), numpy.nan)
    divergence = numpy.full((size, size), numpy.nan)
    for row, first in enumerate(models):
        for column, second in enumerate(models):
            if row == column:
                continue
            _, _, logp = welch_test(vectors[first], vectors[second])
            log_p[row, column] = logp
            effect[row, column] = cohen_d(vectors[first], vectors[second])
            divergence[row, column] = symmetric_kl(distributions[first],
                                                   distributions[second])

    heatmap(log_p, models, models, os.path.join(output, '12_pares_welch_logp'),
            HeatmapSpec(title=f'{suite.capitalize()}: evidencia estadística entre modelos',
                        cbar_label='log10(p); más bajo = evidencia más fuerte',
                        cmap='mpes_pval', vmin=-10.0, vmax=0.0, fmt='{:.1f}',
                        clip_low_label='≤-10', xlabel='Modelo de referencia',
                        ylabel='Modelo comparado'))
    bound = max(float(numpy.nanmax(numpy.abs(effect))), 1e-3)
    heatmap(effect, models, models, os.path.join(output, '13_pares_cohen_d'),
            HeatmapSpec(title=f'{suite.capitalize()}: tamaño de efecto entre modelos',
                        cbar_label='d de Cohen', cmap='mpes_div',
                        vmin=-bound, vmax=bound, fmt='{:+.2f}',
                        xlabel='Modelo de referencia', ylabel='Modelo comparado'))
    heatmap(divergence, models, models, os.path.join(output, '14_pares_kl'),
            HeatmapSpec(title=f'{suite.capitalize()}: divergencia entre distribuciones',
                        cbar_label='KL simétrica del rendimiento', cmap='mpes_kl',
                        vmin=0.0, vmax=float(numpy.nanmax(divergence)) or 1.0,
                        fmt='{:.2f}', xlabel='Modelo de referencia',
                        ylabel='Modelo comparado'))
    return {'scenarios_used': common, 'models': models,
            'welch_log10_p': log_p.tolist(), 'cohen_d': effect.tolist(),
            'symmetric_kl': divergence.tolist()}


###############
##  Distribution figures
###############
def render_distribution_figures(suite: str, cells: dict) -> None:
    """Render per-scenario histograms and normalised reward curves."""
    summary = _load_summary(suite)
    models, scenarios = summary['models'], _scenarios_of(summary)
    histograms = os.path.join(figures_dir(suite), 'histogramas')
    rewards = os.path.join(figures_dir(suite), 'recompensa')

    for scenario in scenarios:
        series = {model: _performance(cells, model, scenario) for model in models}
        series = {model: values for model, values in series.items() if values.size}
        if not series:
            continue
        figure, axis = pyplot.subplots(figsize=(7.5, 4.4))
        for index, (model, values) in enumerate(series.items()):
            axis.hist(values, bins=20, range=(0, 1), histtype='step', linewidth=1.6,
                      color=MODEL_LINE_COLOURS[index % len(MODEL_LINE_COLOURS)], label=model)
        axis.set_title(f'Distribución del desempeño por secuencia — {scenario}')
        axis.set_xlabel('Rendimiento normalizado')
        axis.set_ylabel('Frecuencia')
        axis.set_xlim(0, 1)
        axis.legend(frameon=False, fontsize=8)
        style_axes(axis)
        save_figure(figure, os.path.join(histograms, scenario))

        figure, axes = pyplot.subplots(1, 2, figsize=(13, 4.8))
        for index, (model, values) in enumerate(series.items()):
            steps = numpy.arange(1, values.size + 1)
            colour = MODEL_LINE_COLOURS[index % len(MODEL_LINE_COLOURS)]
            axes[0].plot(steps, numpy.cumsum(values), color=colour, linewidth=1.7, label=model)
            axes[1].plot(steps, numpy.cumsum(values) / steps, color=colour,
                         linewidth=1.7, label=model)
        axes[0].set_title('Recompensa normalizada acumulada')
        axes[0].set_ylabel('Suma de rendimiento')
        axes[1].set_title('Recompensa normalizada media móvil')
        axes[1].set_ylabel('Media acumulada')
        axes[1].set_ylim(0, 1.02)
        for axis in axes:
            axis.set_xlabel('Secuencia')
            style_axes(axis)
        axes[1].legend(frameon=False, fontsize=8, loc='lower right')
        figure.suptitle(f'{suite.capitalize()} — {scenario}', fontweight='semibold')
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        save_figure(figure, os.path.join(rewards, scenario))


###############
##  Entry point
###############
def render_suite(suite: str, only: "str | None" = None) -> None:
    """Render the requested figure groups for one suite."""
    cells = load_model_cells()
    if only in (None, 'matrices'):
        render_matrix_figures(suite)
    if only in (None, 'curves'):
        render_curve_figures(suite, cells)
    if only in (None, 'comparison'):
        render_comparison_figures(suite)
    if only in (None, 'distributions'):
        render_distribution_figures(suite, cells)
    print(f'[figures] {suite} -> {figures_dir(suite)}')


def main() -> None:
    """Render figures for the requested suites."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=SUITES, action='append')
    parser.add_argument('--only', choices=('matrices', 'curves', 'comparison',
                                           'distributions'))
    args = parser.parse_args()
    for suite in args.suite or list(SUITES):
        render_suite(suite, args.only)


if __name__ == '__main__':
    main()
