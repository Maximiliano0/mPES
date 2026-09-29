"""Figure generation for the mPES benchmark.

Replaces the previous five plotting modules with a single generator that
renders, per suite, a coherently numbered figure set under
``results/<suite>/figures``:

===============================================  ==========================================
``01_desempeno_por_escenario``                   mean normalised performance
``02_degradacion_por_escenario``                 degradation vs the reference condition
``03_welch_logp_por_escenario``                  ``log10(p)`` of the Welch t-test
``04_kl_acciones_por_escenario``                 KL of the action distributions
``05_curvas_por_familia``                        per-sequence curves by perturbation family
``06_curvas_estresores_universales``             per-sequence curves under extrapolation
``07_cohen_d_por_escenario``                     standardised effect size
``08_ranking_desempeno``                         reference vs perturbed ranking
``09_degradacion_por_familia``                   sensitivity per perturbation family
``10_desempeno_vs_estabilidad``                  mean performance vs dispersion
``11_perfiles_generalizacion``                   response profile per family
``12_pares_welch_logp`` / ``13_pares_cohen_d`` / ``14_pares_kl``  pairwise model contrasts
``15_referencia_vs_heldout``                     reference vs held-out replicas (overfitting)
``histogramas/<scenario>``                       per-sequence distribution per scenario
``recompensa/<scenario>``                        cumulative and running-mean reward
``modelos/<PKG>_results``                        six-panel reference figure per model
===============================================  ==========================================

Shared conventions (see :mod:`plotting`): each model keeps a fixed colour
(``MODEL_COLOURS``) and, whenever several models are overlaid, the one with
the highest mean over the perturbation scenarios is drawn with
``BEST_LINEWIDTH`` and named in the super-title. The reference condition is
``sev_base`` (``REFERENCE_SCENARIO`` in :mod:`benchmark`). The held-out
replicas (``heldout_s*``) appear as separate columns in the heatmaps but are
excluded from every stress aggregate (ranking, families, profiles, pairs).

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
import glob
import json
import os

from matplotlib import colors as mcolors
from matplotlib import pyplot
from matplotlib.layout_engine import ConstrainedLayoutEngine
import numpy

##########################
##  Imports internos    ##
##########################
from .benchmark import (REFERENCE_MODEL, REFERENCE_SCENARIO, SUITES,
                        SUITE_PACKAGES, WORK_ROOT, comparison_metrics_path, figures_dir,
                        generalisation_scenarios, heldout_scenarios, load_cells,
                        matrices_dir, summary_path)
from .plotting import (ALPHA_LEVELS, BASE_LINEWIDTH, BEST_LINEWIDTH,
                       MEAN_LINESTYLE, MEAN_LINEWIDTH, PAGE_WIDTH_IN, PALETTE, PRINT_BASE_LINEWIDTH,
                       PRINT_BEST_LINEWIDTH, PRINT_RC, PUB_RC, HeatmapSpec, cohen_d,
                       heatmap, heatmap_split, histogram_pmf, model_colour, read_matrix_csv,
                       save_figure, style_axes, symmetric_kl, welch_test)


#: Suite names shown in figure titles.
SUITE_TITLES = {'individual': 'Modelos individuales', 'ensemble': 'Ensambles'}
ENSEMBLE_REFERENCE = 'pes_ens'
FAMILY_ORDER = ('severity', 'length', 'joint', 'structural')
FAMILY_LABELS = {'severity': 'Severidad', 'length': 'Longitud',
                 'joint': 'Conjunta', 'structural': 'Estructura'}
#: Panels of the model x scenario heatmaps, drawn in portrait (``heatmap_split``).
SCENARIO_PANEL_TITLES = ('Referencia, severidad y longitud',
                         'Conjunta, estructura y réplicas held-out')
#: Printed size of the model x model heatmaps (full text width).
PAIRWISE_HEATMAP_SIZE = (PAGE_WIDTH_IN, 4.3)
CURVE_SCENARIOS = ('sev_bimodal', 'sev_gauss_high', 'sev_beta_highskew',
                   'len_poisson', 'len_extrapolate_long', 'joint_high_long')
UNIVERSAL_SCENARIOS = ('sev_extrapolate_high', 'joint_extrap_both',
                       'len_extrapolate_long')


###############
##  Data access
###############
def load_model_cells() -> dict:
    """Return the benchmark cells of every suite merged into one mapping.

    Returns
    -------
    dict
        ``{model: {scenario: cell}}`` where ``cell`` is the JSON payload
        written by ``benchmark.py`` for that ``(model, scenario)`` pair.
    """
    merged: dict = {}
    for suite in SUITES:
        for model, scenarios in load_cells(suite).items():
            merged.setdefault(model, {}).update(scenarios)
    return merged


def suite_models(suite: str) -> "list[str]":
    """Return the models plotted for one suite, led by the reference model.

    Parameters
    ----------
    suite : str
        Suite identifier (``'individual'`` or ``'ensemble'``).

    Returns
    -------
    list of str
        ``REFERENCE_MODEL`` followed by the suite packages in catalogue order.
    """
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


def _linewidth(model: str, best: "str | None", printed: bool = False) -> float:
    """Thick stroke for the best model of a panel, regular stroke otherwise."""
    if printed:
        return PRINT_BEST_LINEWIDTH if model == best else PRINT_BASE_LINEWIDTH
    return BEST_LINEWIDTH if model == best else BASE_LINEWIDTH


def _best_model(cells: dict, models: "list[str]", scenarios: "list[str]") -> "str | None":
    """Model with the highest mean performance over the non-reference scenarios."""
    candidates = [model for model in models if model != REFERENCE_MODEL] or list(models)
    if not candidates:
        return None
    return max(candidates, key=lambda model: _stress_mean(cells, model, scenarios))


###############
##  Matrix figures
###############
def render_matrix_figures(suite: str) -> None:
    """Render the metric heatmaps (01-04, 07) from the aggregated CSV matrices.

    Parameters
    ----------
    suite : str
        Suite whose ``matrices/*.csv`` are read and whose ``figures/`` folder
        receives the PNG files.
    """
    output = figures_dir(suite)
    specs = {
        'global_mean': ('01_desempeno_por_escenario', HeatmapSpec(
            title=f'{SUITE_TITLES.get(suite, suite)}: desempeño normalizado medio por modelo y escenario',
            cbar_label='Desempeño normalizado medio', cmap='mpes_perf', fmt='{:.2f}')),
        'stress_degradation': ('02_degradacion_por_escenario', HeatmapSpec(
            title=f'{SUITE_TITLES.get(suite, suite)}: degradación respecto a {REFERENCE_SCENARIO}',
            cbar_label='Degradación (positivo = pérdida)',
            cmap='mpes_div', fmt='{:+.3f}')),
        'welch_logp': ('03_welch_logp_por_escenario', HeatmapSpec(
            title=f'{SUITE_TITLES.get(suite, suite)}: log10(p) de la prueba t de Welch',
            cbar_label='log10(p); más bajo = evidencia más fuerte',
            cmap='mpes_pval', vmin=-10.0, vmax=0.0, fmt='{:.2f}',
            clip_low_label='≤-10', cbar_ticks=[-10.0, *ALPHA_LEVELS, 0.0])),
        'cohen_d': ('07_cohen_d_por_escenario', HeatmapSpec(
            title=f'{SUITE_TITLES.get(suite, suite)}: tamaño de efecto (d de Cohen)',
            cbar_label='d de Cohen; positivo = mejor que la referencia',
            cmap='mpes_div', fmt='{:+.2f}')),
    }
    for metric, (name, spec) in specs.items():
        path = os.path.join(matrices_dir(suite), f'{metric}.csv')
        if not os.path.isfile(path):
            continue
        models, scenarios, matrix = read_matrix_csv(path)
        _autoscale(spec, matrix)
        spec.separators = _heldout_separator(scenarios)
        heatmap_split(matrix, models, scenarios, os.path.join(output, name), spec,
                      split=_joint_start(scenarios), panel_titles=SCENARIO_PANEL_TITLES)
    _render_action_kl(suite, output)


def _joint_start(scenarios: "list[str]") -> int:
    """First column of the lower panel: the first joint scenario (or the middle)."""
    joint = [index for index, name in enumerate(scenarios) if name.startswith('joint_')]
    return joint[0] if joint else len(scenarios) // 2


def _heldout_separator(scenarios: "list[str]") -> "list[int]":
    """Column index of the first held-out replica, if any."""
    replicas = heldout_scenarios(scenarios)
    return [scenarios.index(replicas[0])] if replicas else []


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
                title=f'{SUITE_TITLES.get(suite, suite)}: divergencia KL de acciones vs {REFERENCE_SCENARIO}',
                cbar_label='KL(escenario ‖ referencia), escala logarítmica',
                cmap='mpes_kl', fmt='{:.2f}',
                norm=mcolors.LogNorm(vmin=floor, vmax=ceiling),
                separators=_heldout_separator(scenarios)))


###############
##  Curve figures
###############
def _fit_ylim(axis, values: "list[numpy.ndarray]") -> None:
    """Fit the y-range of ``axis`` to the lowest and highest plotted value."""
    plotted = numpy.concatenate(values)
    low, high = float(plotted.min()), float(plotted.max())
    pad = 0.04 * (high - low or 1.0)
    axis.set_ylim(low - pad, high + pad)


def render_curve_figures(suite: str, cells: dict) -> None:
    """Render the per-sequence curves by family (05) and under extrapolation (06).

    The model with the highest mean over the perturbation scenarios is drawn
    with the thick stroke in every panel.

    Parameters
    ----------
    suite : str
        Suite identifier.
    cells : dict
        Merged cells as returned by :func:`load_model_cells`.
    """
    output = figures_dir(suite)
    models = suite_models(suite)
    scenarios = sorted({scenario for model in models for scenario in cells.get(model, {})})
    best = _best_model(cells, models, scenarios)

    with pyplot.rc_context(PRINT_RC):
        figure, axes = pyplot.subplots(2, 3, figsize=(PAGE_WIDTH_IN, 5.0))
        for axis, scenario in zip(axes.flat, CURVE_SCENARIOS):
            plotted = []
            for index, model in enumerate(models):
                values = numpy.sort(_performance(cells, model, scenario))
                if not values.size:
                    continue
                plotted.append(values)
                axis.plot(numpy.arange(1, values.size + 1), values,
                          color=model_colour(model, index),
                          linewidth=_linewidth(model, best, printed=True),
                          zorder=3 if model == best else 2, label=model)
            axis.set_title(scenario)
            if plotted:
                _fit_ylim(axis, plotted)
            style_axes(axis)
        for axis in axes[1]:
            axis.set_xlabel('Secuencia ordenada')
        for axis in axes[:, 0]:
            axis.set_ylabel('Desempeño normalizado')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='lower center', ncol=4,
                      frameon=False, bbox_to_anchor=(0.5, 0.0))
        figure.suptitle(f'{SUITE_TITLES.get(suite, suite)}: desempeño por secuencia y familia de '
                        f'perturbación\n(trazo grueso = {best})', fontweight='semibold')
        figure.tight_layout(rect=(0, 0.08, 1, 0.96))
        save_figure(figure, os.path.join(output, '05_curvas_por_familia'))

        _render_universal(suite, cells, models, best, output)


def _stress_mean(cells: dict, model: str, scenarios: "list[str]") -> float:
    """Mean performance of one model over the stress scenarios."""
    values = [cells.get(model, {}).get(scenario, {}).get('global_mean_perf')
              for scenario in generalisation_scenarios(scenarios)]
    values = [value for value in values if value is not None]
    return float(numpy.mean(values)) if values else float('-inf')


def _render_universal(suite: str, cells: dict, models: "list[str]",
                      best: "str | None", output: str) -> None:
    """Render the three extrapolation panels with per-model means.

    The individual suite contrasts ``pes_trf`` with one partner per panel;
    the ensemble suite draws every variant. The best model uses the thick
    stroke in both cases.
    """
    if suite == 'individual':
        best = 'pes_trf'
        partners = {'sev_extrapolate_high': 'pes_dql',
                    'joint_extrap_both': 'pes_a2c',
                    'len_extrapolate_long': 'pes_dqn'}
        panel_models = {scenario: [partners[scenario], best]
                        for scenario in UNIVERSAL_SCENARIOS}
    else:
        ordered = [model for model in models if model != REFERENCE_MODEL]
        panel_models = {scenario: ordered for scenario in UNIVERSAL_SCENARIOS}

    figure, axes = pyplot.subplots(1, 3, figsize=(PAGE_WIDTH_IN, 3.4))
    for axis, scenario in zip(axes, UNIVERSAL_SCENARIOS):
        plotted = []
        for index, model in enumerate(panel_models[scenario]):
            values = numpy.sort(_performance(cells, model, scenario))
            if not values.size:
                continue
            plotted.append(values)
            colour = model_colour(model, index)
            axis.plot(numpy.arange(1, values.size + 1), values, color=colour,
                      linewidth=_linewidth(model, best, printed=True),
                      zorder=3 if model == best else 2, label=model)
            axis.axhline(values.mean(), color=colour, linestyle=MEAN_LINESTYLE,
                         linewidth=MEAN_LINEWIDTH * 0.7)
        axis.set_title(scenario)
        axis.set_xlabel('Secuencia ordenada')
        if plotted:
            _fit_ylim(axis, plotted)
        style_axes(axis)
    axes[0].set_ylabel('Desempeño normalizado')
    same_models = len({tuple(models_) for models_ in panel_models.values()}) == 1
    if same_models:
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='lower center', ncol=3,
                      frameon=False, bbox_to_anchor=(0.5, 0.0))
        bottom = 0.14
    else:
        # Sorted curves climb towards the top-right corner, leaving the bottom-right free.
        for axis in axes:
            axis.legend(frameon=False, loc='lower right')
        bottom = 0.02
    figure.suptitle(f'{SUITE_TITLES.get(suite, suite)}: desempeño en escenarios de extrapolación\n'
                    f'(trazo grueso = {best}; línea punteada = media)',
                    fontweight='semibold')
    figure.tight_layout(rect=(0, bottom, 1, 0.93))
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
    stress = {scenario: means[scenario] for scenario in generalisation_scenarios(list(means))}
    replicas = heldout_scenarios(list(means))
    pooled = numpy.concatenate([numpy.asarray(cells[s].get('per_sequence_perf', []), dtype=float)
                                for s in replicas]) if replicas else numpy.array([])
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
        'heldout_replica_means': {s: means[s] for s in replicas},
        'heldout_mean': float(pooled.mean()) if pooled.size else float('nan'),
        'heldout_gap': float(baseline - pooled.mean()) if pooled.size else float('nan'),
    }


def render_comparison_figures(suite: str) -> dict:
    """Render the ranking, sensitivity, stability, profile and pairwise figures.

    Also writes ``comparison_metrics.json`` for the suite.

    Parameters
    ----------
    suite : str
        Suite identifier.

    Returns
    -------
    dict
        Payload written to ``comparison_metrics.json``: per-model metrics under
        ``'models'`` and the pairwise Welch / Cohen / KL matrices under
        ``'pairwise'``.
    """
    summary = _load_summary(suite)
    models, scenarios = summary['models'], _scenarios_of(summary)
    metrics = {model: _model_metrics(summary, model) for model in models}
    output = figures_dir(suite)

    with pyplot.rc_context(PUB_RC):
        _plot_ranking(suite, models, metrics, output)
        _plot_family_sensitivity(suite, models, metrics, output)
        _plot_stability(suite, models, metrics, output)
        _plot_generalisation(suite, summary, models, scenarios, output)
        pairwise = _plot_pairwise(suite, summary, models, scenarios, output)
        _plot_heldout(suite, models, metrics, output)

    payload = {'suite': suite, 'reference_scenario': summary['reference_scenario'],
               'models': metrics, 'pairwise': pairwise}
    with open(comparison_metrics_path(suite), 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return payload


def _plot_ranking(suite: str, models: "list[str]", metrics: dict, output: str) -> None:
    """Horizontal ranking of reference versus perturbed performance."""
    order = sorted(models, key=lambda model: metrics[model]['stress_mean'], reverse=True)
    positions = numpy.arange(len(order))
    width = 0.36
    figure, axis = pyplot.subplots(figsize=(10.0, 5.6))
    edge = [PALETTE['ink'] if row == 0 else 'none' for row in range(len(order))]
    axis.barh(positions + width / 2, [metrics[m]['reference_mean'] for m in order],
              width, label=f'Condición de referencia ({REFERENCE_SCENARIO})',
              color=PALETTE['blue_light'], edgecolor=edge, linewidth=1.2)
    axis.barh(positions - width / 2, [metrics[m]['stress_mean'] for m in order],
              width, label='Media en los escenarios de perturbación',
              color=PALETTE['blue_dark'], edgecolor=edge, linewidth=1.2)
    axis.set_yticks(positions, order)
    axis.get_yticklabels()[0].set_fontweight('bold')
    axis.invert_yaxis()
    axis.set_xlim(0, 1.08)
    axis.set_xlabel('Desempeño normalizado')
    axis.set_title(f'{SUITE_TITLES.get(suite, suite)}: ranking de desempeño (borde = mejor modelo)')
    axis.legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.14))
    style_axes(axis)
    for row, model in enumerate(order):
        axis.text(metrics[model]['stress_mean'] + 0.01, row - width / 2,
                  f"{metrics[model]['stress_mean']:.3f}", va='center', fontsize=9)
        axis.text(metrics[model]['reference_mean'] + 0.01, row + width / 2,
                  f"{metrics[model]['reference_mean']:.3f}", va='center', fontsize=9,
                  color=PALETTE['ink'])
    figure.tight_layout()
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
                 label=model, color=model_colour(model, index))
    axis.axhline(0, color=PALETTE['ink'], linewidth=0.8)
    axis.set_xticks(positions, [FAMILY_LABELS[family] for family in families])
    axis.set_ylabel(f'Degradación frente a {REFERENCE_SCENARIO}')
    axis.set_title(f'{SUITE_TITLES.get(suite, suite)}: degradación media por familia de perturbación')
    axis.legend(frameon=False, ncol=min(len(models), 4), loc='upper center',
                bbox_to_anchor=(0.5, -0.10))
    style_axes(axis)
    figure.tight_layout()
    save_figure(figure, os.path.join(output, '09_degradacion_por_familia'))


def _plot_stability(suite: str, models: "list[str]", metrics: dict, output: str) -> None:
    """Scatter of mean stress performance against sequence-level dispersion."""
    figure, axis = pyplot.subplots(figsize=(8.2, 6.2))
    x_values = [metrics[model]['stress_std'] for model in models]
    y_values = [metrics[model]['stress_mean'] for model in models]
    axis.scatter(x_values, y_values, s=110,
                 color=[model_colour(model, index) for index, model in enumerate(models)],
                 edgecolor='white', linewidth=1.2, zorder=3)
    for model, x_value, y_value in zip(models, x_values, y_values):
        axis.annotate(model, (x_value, y_value), xytext=(7, 5),
                      textcoords='offset points', fontsize=9)
    axis.set_xlabel('Desviación estándar media por escenario')
    axis.set_ylabel('Desempeño medio en los escenarios de perturbación')
    axis.set_title(f'{SUITE_TITLES.get(suite, suite)}: desempeño y estabilidad')
    style_axes(axis)
    figure.tight_layout()
    save_figure(figure, os.path.join(output, '10_desempeno_vs_estabilidad'))


def _plot_generalisation(suite: str, summary: dict, models: "list[str]",
                         scenarios: "list[str]", output: str) -> None:
    """Per-family response profiles across the stress catalogue."""
    grouped = {family: [scenario for scenario in scenarios
                        if scenario != summary['reference_scenario']
                        and summary['cells'].get(models[0], {}).get(
                            scenario, {}).get('family') == family]
               for family in FAMILY_ORDER}
    stress_means: "dict[str, float]" = {
        model: float(numpy.nanmean([summary['cells'].get(model, {}).get(s, {}).get(
            'global_mean_perf', numpy.nan) for s in generalisation_scenarios(scenarios)]))
        for model in models}
    candidates = [m for m in models if m != REFERENCE_MODEL]
    best = max(candidates, key=lambda m: stress_means[m]) if candidates else None
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
            axis.plot(positions, values, marker='o', markersize=4.5,
                      linewidth=_linewidth(model, best), zorder=3 if model == best else 2,
                      color=model_colour(model, index), label=model)
        axis.set_xticks(positions, family_scenarios,
                        rotation=42, ha='right', fontsize=8)
        axis.set_ylim(0.6, 1.02)
        axis.set_ylabel('Desempeño normalizado')
        axis.set_title(FAMILY_LABELS[family])
        style_axes(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False,
                  fontsize=9.5, bbox_to_anchor=(0.5, 0.005))
    figure.suptitle(f'{SUITE_TITLES.get(suite, suite)}: perfiles de generalización '
                    f'(trazo grueso = {best})', fontsize=14, fontweight='semibold')
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    save_figure(figure, os.path.join(output, '11_perfiles_generalizacion'))


def _plot_pairwise(suite: str, summary: dict, models: "list[str]",
                   scenarios: "list[str]", output: str) -> dict:
    """Pairwise Welch, Cohen and KL contrasts between the models of a suite."""
    common = [scenario for scenario in generalisation_scenarios(scenarios)
              if all(scenario in summary['cells'].get(model, {}) for model in models)]
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
            HeatmapSpec(title=f'{SUITE_TITLES.get(suite, suite)}: evidencia estadística entre modelos',
                        cbar_label='log10(p); más bajo = evidencia más fuerte',
                        cmap='mpes_pval', vmin=-10.0, vmax=0.0, fmt='{:.1f}',
                        clip_low_label='≤-10', xlabel='Modelo de referencia',
                        ylabel='Modelo comparado', figsize=PAIRWISE_HEATMAP_SIZE,
                        annot_fontsize=8))
    bound = max(float(numpy.nanmax(numpy.abs(effect))), 1e-3)
    heatmap(effect, models, models, os.path.join(output, '13_pares_cohen_d'),
            HeatmapSpec(title=f'{SUITE_TITLES.get(suite, suite)}: tamaño de efecto entre modelos',
                        cbar_label='d de Cohen', cmap='mpes_div',
                        vmin=-bound, vmax=bound, fmt='{:+.2f}',
                        xlabel='Modelo de referencia', ylabel='Modelo comparado',
                        figsize=PAIRWISE_HEATMAP_SIZE, annot_fontsize=8))
    heatmap(divergence, models, models, os.path.join(output, '14_pares_kl'),
            HeatmapSpec(title=f'{SUITE_TITLES.get(suite, suite)}: divergencia entre distribuciones',
                        cbar_label='KL simétrica del desempeño', cmap='mpes_kl',
                        vmin=0.0, vmax=float(numpy.nanmax(divergence)) or 1.0,
                        fmt='{:.2f}', xlabel='Modelo de referencia',
                        ylabel='Modelo comparado', figsize=PAIRWISE_HEATMAP_SIZE,
                        annot_fontsize=8))
    return {'scenarios_used': common, 'models': models,
            'welch_log10_p': log_p.tolist(), 'cohen_d': effect.tolist(),
            'symmetric_kl': divergence.tolist()}


def _plot_heldout(suite: str, models: "list[str]", metrics: dict, output: str) -> None:
    """Reference score against the held-out replicas of the same distribution."""
    shown = [model for model in models if metrics[model]['heldout_replica_means']]
    if not shown:
        return
    order = sorted(shown, key=lambda model: metrics[model]['reference_mean'], reverse=True)
    positions = numpy.arange(len(order))
    figure, axis = pyplot.subplots(figsize=(9.5, 0.55 * len(order) + 1.8))
    for row, model in enumerate(order):
        reference, pooled = metrics[model]['reference_mean'], metrics[model]['heldout_mean']
        replica_means = list(metrics[model]['heldout_replica_means'].values())
        axis.plot([pooled, reference], [row, row], color=PALETTE['grey'], linewidth=2, zorder=1)
        axis.scatter(replica_means, [row] * len(replica_means), s=16, color=PALETTE['blue_dark'],
                     alpha=0.35, linewidth=0, zorder=2)
        axis.text(max([reference, pooled, *replica_means]) + 0.004, row,
                  f"{metrics[model]['heldout_gap']:+.3f}", va='center', fontsize=9,
                  color=PALETTE['ink'])
    axis.scatter([metrics[m]['reference_mean'] for m in order], positions, s=70,
                 color=PALETTE['blue_light'], edgecolor=PALETTE['ink'], linewidth=0.8, zorder=3,
                 label=f'Referencia ({REFERENCE_SCENARIO}, 64 secuencias)')
    axis.scatter([metrics[m]['heldout_mean'] for m in order], positions, s=70,
                 color=PALETTE['blue_dark'], edgecolor='white', linewidth=1.2, zorder=3,
                 label='Réplicas held-out (media agrupada; puntos claros = cada réplica)')
    axis.set_yticks(positions, order)
    axis.invert_yaxis()
    axis.set_xlabel('Desempeño normalizado')
    axis.set_title(f'{SUITE_TITLES.get(suite, suite)}: referencia frente a réplicas held-out '
                   '(número = referencia - held-out)')
    axis.legend(frameon=False, ncol=1, loc='upper center', bbox_to_anchor=(0.5, -0.12))
    style_axes(axis)
    figure.tight_layout()
    save_figure(figure, os.path.join(output, '15_referencia_vs_heldout'))


###############
##  Distribution figures
###############
def render_distribution_figures(suite: str, cells: dict) -> None:
    """Render per-scenario histograms and normalised reward curves.

    Parameters
    ----------
    suite : str
        Suite identifier.
    cells : dict
        Merged cells as returned by :func:`load_model_cells`.
    """
    summary = _load_summary(suite)
    models, scenarios = summary['models'], _scenarios_of(summary)
    histograms = os.path.join(figures_dir(suite), 'histogramas')
    rewards = os.path.join(figures_dir(suite), 'recompensa')
    best = _best_model(cells, models, scenarios)

    for scenario in scenarios:
        series = {model: _performance(cells, model, scenario) for model in models}
        series = {model: values for model, values in series.items() if values.size}
        if not series:
            continue

        with pyplot.rc_context(PUB_RC):
            figure, axis = pyplot.subplots(figsize=(7.5, 4.4))
            for index, (model, values) in enumerate(series.items()):
                axis.hist(values, bins=20, range=(0, 1), histtype='step',
                          linewidth=_linewidth(model, best),
                          color=model_colour(model, index), label=model)
            axis.set_title(f'Distribución del desempeño por secuencia — {scenario}')
            axis.set_xlabel('Desempeño normalizado')
            axis.set_ylabel('Frecuencia')
            axis.set_xlim(0, 1)
            axis.legend(frameon=False, fontsize=8, loc='upper left')
            style_axes(axis)
            figure.tight_layout()
            save_figure(figure, os.path.join(histograms, scenario))

            figure, axes = pyplot.subplots(1, 2, figsize=(13, 4.8))
            for index, (model, values) in enumerate(series.items()):
                steps = numpy.arange(1, values.size + 1)
                colour = model_colour(model, index)
                width = _linewidth(model, best)
                axes[0].plot(steps, numpy.cumsum(values), color=colour,
                             linewidth=width, label=model)
                axes[1].plot(steps, numpy.cumsum(values) / steps, color=colour,
                             linewidth=width, label=model)
            axes[0].set_title('Recompensa normalizada acumulada')
            axes[0].set_ylabel('Suma de desempeño')
            axes[1].set_title('Recompensa normalizada media móvil')
            axes[1].set_ylabel('Media acumulada')
            axes[1].set_ylim(0, 1.02)
            for axis in axes:
                axis.set_xlabel('Secuencia')
                style_axes(axis)
            handles, labels = axes[0].get_legend_handles_labels()
            figure.legend(handles, labels, loc='lower center', ncol=len(labels),
                          frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
            figure.suptitle(f'{SUITE_TITLES.get(suite, suite)} — {scenario} (trazo grueso = {best})',
                            fontweight='semibold')
            figure.tight_layout(rect=(0, 0.07, 1, 0.94))
            save_figure(figure, os.path.join(rewards, scenario))


###############
##  Per-model panels
###############
#: Printed size of the per-model reference figure (full text width, two per page).
MODEL_PANEL_SIZE = (PAGE_WIDTH_IN, 3.4)
#: Names used in the thesis text, shown in the title of each per-model figure.
MODEL_DISPLAY_NAMES = {
    'pes_base': 'Q-Learning base', 'pes_ql': 'Q-Learning', 'pes_dql': 'Double Q-Learning',
    'pes_dqn': 'DQN', 'pes_rdqn': 'DQN recurrente', 'pes_a2c': 'A2C', 'pes_trf': 'Transformer',
    'pes_ens': 'Ensamble ponderado', 'pes_ens_sprb': 'Voto suave', 'pes_ens_accq': 'Voto por acción',
    'pes_ens_consensus': 'Consenso', 'pes_ens_consensus_prior': 'Consenso con prior',
    'pes_ens_trf_guard': 'Compuerta del Transformer',
}
MODEL_PANEL_RC = {**PRINT_RC, 'axes.titlesize': 8, 'axes.labelsize': 7.5,
                  'axes.titleweight': 'bold', 'axes.labelweight': 'bold',
                  'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5}


def _package_report(model: str) -> "dict | None":
    """Latest reference-run report JSON written by the package itself."""
    pattern = os.path.join(WORK_ROOT, model, 'outputs', REFERENCE_SCENARIO, '**',
                           f'{model.upper()}_results_*.json')
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        return None
    with open(max(matches, key=os.path.getmtime), 'r', encoding='utf-8') as handle:
        return json.load(handle)


def _panel_statistics(performances: numpy.ndarray, blocks: "list[numpy.ndarray]",
                      report: "dict | None") -> dict:
    """Summary-table statistics, taken from the package report when available.

    The benchmark cells store the per-sequence values rounded to four
    decimals, so the report of the package run is preferred for the table;
    both must agree on the mean.
    """
    computed = {
        'overall_mean': float(numpy.mean(performances)),
        'overall_median': float(numpy.median(performances)),
        'overall_std': float(numpy.std(performances)),
        'overall_min': float(numpy.min(performances)),
        'overall_max': float(numpy.max(performances)),
        'percentile_25': float(numpy.percentile(performances, 25)),
        'percentile_75': float(numpy.percentile(performances, 75)),
        'total_sequences': int(performances.size),
        'first_block_mean': float(numpy.mean(blocks[0])),
        'last_block_mean': float(numpy.mean(blocks[-1])),
    }
    computed['improvement'] = computed['last_block_mean'] - computed['first_block_mean']
    reported = (report or {}).get('performance_statistics')
    if not reported:
        return computed
    if abs(reported['overall_mean'] - computed['overall_mean']) > 1e-3:
        raise ValueError('Package report and benchmark cell disagree on the reference mean')
    return {key: reported.get(key, value) for key, value in computed.items()}


def render_model_panels(suite: str, cells: dict) -> None:
    """Render each model's six-panel reference figure at its printed size.

    Reproduces the layout of the packages' ``result_formatter`` (trend,
    distribution, per-block box plot, cumulative mean, block means and a
    summary table) from the benchmark cell of the reference condition, so the
    thesis can include it at ``\\linewidth`` with legible fonts. Files go to
    ``results/<suite>/figures/modelos/<PKG>_results.png``.

    Parameters
    ----------
    suite : str
        Suite identifier.
    cells : dict
        Merged cells as returned by :func:`load_model_cells`.
    """
    output = os.path.join(figures_dir(suite), 'modelos')
    for model in suite_models(suite):
        cell = cells.get(model, {}).get(REFERENCE_SCENARIO)
        if not cell or not cell.get('per_sequence_perf'):
            continue
        performances = numpy.asarray(cell['per_sequence_perf'], dtype=float)
        per_block = int(cell.get('num_sequences_per_block') or performances.size)
        blocks = [performances[start:start + per_block]
                  for start in range(0, performances.size, per_block)]
        report = _package_report(model)
        stats = _panel_statistics(performances, blocks, report)
        name = MODEL_DISPLAY_NAMES.get(model, model)
        with pyplot.rc_context(MODEL_PANEL_RC):
            _draw_model_panel(performances, blocks, stats,
                              f'{name} ({model}): desempeño por secuencia en la referencia',
                              os.path.join(output, f'{model.upper()}_results'))


def _draw_model_panel(performances: numpy.ndarray, blocks: "list[numpy.ndarray]",
                      stats: dict, title: str, out_base: str) -> None:
    """Draw one six-panel reference figure (layout of ``result_formatter``).

    The figure is drawn at full text width and at a height that lets two of
    them share an A4 page, so its fonts are the printed sizes.
    """
    figure = pyplot.figure(figsize=MODEL_PANEL_SIZE,
                           layout=ConstrainedLayoutEngine(h_pad=0.02, w_pad=0.03,
                                                          hspace=0.04, wspace=0.03))
    grid = figure.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.72])
    steps = numpy.arange(1, performances.size + 1)
    mean, std = stats['overall_mean'], stats['overall_std']

    trend = figure.add_subplot(grid[0, :2])
    trend.plot(steps, performances, '-o', color=PALETTE['blue'], linewidth=1.0, markersize=2)
    trend.axhline(y=mean, color=PALETTE['coral'], linestyle='--', linewidth=1.0, label=f'Media: {mean:.3f}')
    trend.fill_between(steps, mean - std, mean + std, alpha=0.6, color=PALETTE['coral_light'],
                       linewidth=0)
    trend.set(xlabel='Secuencia', ylabel='Desempeño',
              title='Desempeño por secuencia', ylim=(0, 1.05))
    trend.legend(loc='lower left')

    histogram = figure.add_subplot(grid[0, 2])
    histogram.hist(performances, bins=15, color=PALETTE['blue_light'], edgecolor='white',
                   linewidth=0.6)
    histogram.axvline(x=mean, color=PALETTE['coral'], linestyle='--', linewidth=1.0, label='Media')
    histogram.axvline(x=stats['overall_median'], color=PALETTE['blue_dark'], linestyle=':',
                      linewidth=1.0, label='Mediana')
    histogram.set(xlabel='Desempeño', ylabel='Frecuencia', title='Distribución')
    histogram.legend(loc='upper left')

    boxes = figure.add_subplot(grid[1, 0])
    drawn = boxes.boxplot(blocks, tick_labels=[f'B{i + 1}' for i in range(len(blocks))],
                          patch_artist=True, flierprops={'markersize': 3},
                          boxprops={'linewidth': 0.7}, whiskerprops={'linewidth': 0.7},
                          medianprops={'linewidth': 1.0, 'color': PALETTE['coral']})
    for patch in drawn['boxes']:
        patch.set_facecolor(PALETTE['blue_pale'])
        patch.set_edgecolor(PALETTE['blue_dark'])
    boxes.set(xlabel='Bloque', ylabel='Desempeño', title='Desempeño por bloque',
              ylim=(0, 1.05))

    cumulative = figure.add_subplot(grid[1, 1])
    cumulative.plot(steps, numpy.cumsum(performances) / steps, '-o', color=PALETTE['teal'],
                    linewidth=1.0, markersize=2)
    cumulative.set(xlabel='Secuencia', ylabel='Media acumulada',
                   title='Media acumulada', ylim=(0, 1.05))

    block_means = figure.add_subplot(grid[1, 2])
    block_means.bar(range(1, len(blocks) + 1), [float(numpy.mean(b)) for b in blocks],
                    color=PALETTE['blue_light'], edgecolor=PALETTE['blue'], linewidth=0.6)
    block_means.set(xlabel='Bloque', ylabel='Media',
                    title='Media por bloque', ylim=(0, 1.05))
    block_means.set_xticks(range(1, len(blocks) + 1))
    for axis in (trend, cumulative):
        axis.grid(True, alpha=0.3)
    for axis in (boxes, block_means):
        axis.grid(True, alpha=0.3, axis='y')

    summary = figure.add_subplot(grid[2, :])
    summary.axis('off')
    header = ['Estadístico', 'Valor'] * 3
    rows = [header,
            ['Media', f'{mean:.4f}', 'Mediana', f"{stats['overall_median']:.4f}",
             'Secuencias', f"{stats['total_sequences']}"],
            ['Desv. estándar', f'{std:.4f}', 'Mínimo', f"{stats['overall_min']:.4f}",
             'Máximo', f"{stats['overall_max']:.4f}"],
            ['Q1', f"{stats['percentile_25']:.4f}", 'Q3', f"{stats['percentile_75']:.4f}",
             'Último − primero', f"{stats['improvement']:.4f}"],
            ['Primer bloque', f"{stats['first_block_mean']:.4f}",
             'Último bloque', f"{stats['last_block_mean']:.4f}", '', '']]
    table = summary.table(cellText=rows, cellLoc='center', loc='center',
                          colWidths=[0.16, 0.11] * 3)
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    for cell in table.get_celld().values():
        cell.set_edgecolor(PALETTE['grey'])
        cell.set_linewidth(0.6)
    for column in range(len(header)):
        table[(0, column)].set_facecolor(PALETTE['blue_dark'])
        table[(0, column)].set_text_props(weight='bold', color='white')
    for row in range(2, len(rows), 2):
        for column in range(len(header)):
            table[(row, column)].set_facecolor(PALETTE['blue_pale'])

    figure.suptitle(title, fontsize=8.5, fontweight='bold')
    save_figure(figure, out_base)


###############
##  Entry point
###############
def render_suite(suite: str, only: "str | None" = None) -> None:
    """Render the requested figure groups for one suite.

    Parameters
    ----------
    suite : str
        Suite identifier.
    only : str, optional
        Restrict rendering to one group: ``'matrices'``, ``'curves'``,
        ``'comparison'``, ``'distributions'`` or ``'models'``. ``None`` renders all.
    """
    cells = load_model_cells()
    if only in (None, 'matrices'):
        render_matrix_figures(suite)
    if only in (None, 'curves'):
        render_curve_figures(suite, cells)
    if only in (None, 'comparison'):
        render_comparison_figures(suite)
    if only in (None, 'distributions'):
        render_distribution_figures(suite, cells)
    if only in (None, 'models'):
        render_model_panels(suite, cells)
    print(f'[figures] {suite} -> {figures_dir(suite)}')


def main() -> None:
    """Parse ``--suite`` / ``--only`` and render figures for the requested suites."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=SUITES, action='append')
    parser.add_argument('--only', choices=('matrices', 'curves', 'comparison',
                                           'distributions', 'models'))
    args = parser.parse_args()
    for suite in args.suite or list(SUITES):
        render_suite(suite, args.only)


if __name__ == '__main__':
    main()
