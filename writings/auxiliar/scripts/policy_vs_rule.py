"""Política aprendida frente a la regla fija ``a = min(S + k, R)`` y al óptimo.

Para la defensa (Bernardez, «Recomendaciones adicionales»): compara, decisión
por decisión, cuánto asigna cada política según la severidad inicial ``S`` de
la ciudad, en las 64 secuencias de la referencia y en las 320 de las cinco
réplicas *held-out*.

* **Modelos**: lee las respuestas que guardó el benchmark
  (``*_responses_*.txt``, columnas ``InitialSeverity, Response``) en la carpeta
  del informe de cada celda (``h1/general/results/<suite>/cells``). No ejecuta
  ningún modelo.
* **Regla fija**: ``a = min(clip(S, 9) + k, R, 10)`` sobre las mismas secuencias.
* **Óptimo**: la asignación canónica de ``weighted_ens_oracle.json``.

Una decisión es *con recursos* si quedaba presupuesto (``R > 0``) al tomarla; el
presupuesto se reconstruye en cada secuencia a partir de las respuestas
(``R`` inicial = 30, asignación efectiva ``min(a, R)``). Las medias de
``r̄`` se toman de ``fixed_rule.json`` (mismas celdas del benchmark).

Salidas: ``h1/general/results/ensemble/policy_vs_rule.json`` y las figuras
``figures/politica_vs_regla.png`` (ancho de página) y
``figures/politica_vs_regla_slide.png`` (16:9, para la presentación).

Uso (desde la raíz del repositorio, con ``win_mpes_env`` activado)::

    python writings/auxiliar/scripts/policy_vs_rule.py
"""

##########################
##  Imports externos    ##
##########################
import argparse
import glob
import importlib
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy
from matplotlib import pyplot

##########################
##  Configuración       ##
##########################
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
H1 = os.path.join(ROOT, 'h1')
RESULTS = os.path.join(H1, 'general', 'results')
SCENARIO_ROOT = os.path.join(H1, 'general', 'work', 'pes_ens', 'scenarios')
DEFAULT_OUTPUT = os.path.join(RESULTS, 'ensemble', 'policy_vs_rule.json')
FIGURE_BASE = os.path.join(RESULTS, 'ensemble', 'figures', 'politica_vs_regla')
BUDGET = 30
MAX_ALLOC = 10
MAX_SEVERITY = 9
RULE_K = 2
GROUPS = {'referencia': ['sev_base'], 'heldout': [f'heldout_s{i}' for i in range(1, 6)]}
INDIVIDUAL = ('pes_trf', 'pes_dqn', 'pes_a2c', 'pes_rdqn', 'pes_dql', 'pes_ql', 'pes_base')
ENSEMBLES = ('pes_ens', 'pes_ens_trf_guard', 'pes_ens_consensus_prior', 'pes_ens_consensus',
             'pes_ens_sprb', 'pes_ens_accq')
NAMES = {'pes_trf': 'Transformer', 'pes_dqn': 'DQN', 'pes_a2c': 'A2C', 'pes_rdqn': 'DQN recurrente',
         'pes_dql': 'Double Q-Learning', 'pes_ql': 'Q-Learning', 'pes_base': 'Q-Learning base',
         'pes_ens': 'Ensamble ponderado', 'pes_ens_trf_guard': 'Compuerta del Transformer',
         'pes_ens_consensus_prior': 'Consenso con prior', 'pes_ens_consensus': 'Consenso',
         'pes_ens_sprb': 'Voto suave', 'pes_ens_accq': 'Voto por acción'}
# Políticas dibujadas en la figura, además del óptimo y la regla.
PLOTTED = ('pes_trf', 'pes_ens', 'pes_dqn')
sys.path.insert(0, H1)

##########################
##  Imports internos    ##
##########################
# Paleta y estilo del benchmark (h1/general/scripts/plotting.py), cargados por nombre
# porque h1/ no es un paquete instalado.
plotting = importlib.import_module('general.scripts.plotting')
MODEL_COLOURS = plotting.MODEL_COLOURS
PAGE_WIDTH_IN = plotting.PAGE_WIDTH_IN
PALETTE = plotting.PALETTE
PRINT_RC = plotting.PRINT_RC
PUB_RC = plotting.PUB_RC

Decision = Tuple[int, int, bool]  # (S, asignación efectiva, con recursos)


###############
##  Helpers
###############
def read_column(path: str) -> List[float]:
    """Read a one-column CSV of numbers."""
    with open(path, 'r', encoding='utf-8') as handle:
        return [float(line) for line in handle if line.strip()]


def load_sequences(scenario: str) -> List[List[int]]:
    """Initial severities of every sequence of one scenario, in benchmark order."""
    folder = os.path.join(SCENARIO_ROOT, scenario)
    severities = read_column(os.path.join(folder, 'initial_severity.csv'))
    lengths = [int(n) for n in read_column(os.path.join(folder, 'sequence_lengths.csv'))]
    sequences, start = [], 0
    for length in lengths:
        sequences.append([int(s) for s in severities[start:start + length]])
        start += length
    return sequences


def responses_path(suite: str, model: str, scenario: str) -> str:
    """Locate the ``*_responses_*.txt`` written next to the report of one benchmark cell."""
    with open(os.path.join(RESULTS, suite, 'cells', f'{model}__{scenario}.json'), 'r', encoding='utf-8') as handle:
        report = json.load(handle)['report_json'].replace('\\', '/')
    rest = report.split('/', 1)[1] if report.startswith('general/') else report
    candidates = [os.path.join(H1, *report.split('/')),
                  os.path.join(H1, 'general', 'work', *rest.split('/'))]
    for candidate in candidates:
        folder = os.path.dirname(candidate)
        found = glob.glob(os.path.join(folder, '*_responses_*.txt')) + glob.glob(
            os.path.join(folder, '*_responses_*.csv'))
        if found:
            return found[0]
    raise FileNotFoundError(f'sin respuestas para {model} / {scenario}: {candidates}')


def model_decisions(suite: str, model: str, scenario: str) -> List[Decision]:
    """Decisions of one model, with the budget reconstructed per sequence.

    ``.txt`` files (``pes_*`` individuales y ``pes_ens``) carry ``InitialSeverity, Response``;
    ``.csv`` files (ensambles optimizables) carry ``trial_index, allocated_resources`` and the
    severity comes from the scenario sequences.
    """
    path = responses_path(suite, model, scenario)
    has_severity = path.endswith('.txt')
    rows = numpy.atleast_2d(numpy.genfromtxt(path, delimiter=',', comments='#', usecols=(0, 1)))
    decisions: List[Decision] = []
    index = 0
    for sequence in load_sequences(scenario):
        remaining = BUDGET
        for severity in sequence:
            logged_s = int(rows[index, 0]) if has_severity else severity
            action = int(rows[index, 1])
            if logged_s != severity:
                raise ValueError(f'{model}/{scenario}: severidad {logged_s} != {severity} en la fila {index}')
            effective = min(action, remaining)
            decisions.append((severity, effective, remaining > 0))
            remaining -= effective
            index += 1
    if index != rows.shape[0]:
        raise ValueError(f'{model}/{scenario}: {rows.shape[0]} respuestas para {index} decisiones')
    return decisions


def rule_decisions(scenario: str, offset: int) -> List[Decision]:
    """Decisions of the fixed rule ``a = min(clip(S, 9) + k, R, 10)``."""
    decisions: List[Decision] = []
    for sequence in load_sequences(scenario):
        remaining = BUDGET
        for severity in sequence:
            action = min(min(severity, MAX_SEVERITY) + offset, remaining, MAX_ALLOC)
            decisions.append((severity, action, remaining > 0))
            remaining -= action
    return decisions


def summarise(decisions: List[Decision]) -> dict:
    """Mean allocation by severity and mean excess ``a - S`` over the decisions with resources."""
    live = [(s, a) for s, a, has_budget in decisions if has_budget]
    by_severity = {}
    for severity in sorted({s for s, _ in live}):
        actions = [a for s, a in live if s == severity]
        by_severity[str(severity)] = {'n': len(actions), 'media': float(numpy.mean(actions))}
    return {'decisiones': len(decisions), 'con_recursos': len(live),
            'exceso_medio': float(numpy.mean([a - s for s, a in live])),
            'por_severidad': by_severity}


def optimum_summary(oracle: dict, group: str) -> dict:
    """Optimal allocation by severity (decisions with resources) from ``weighted_ens_oracle.json``."""
    key = 'sev_base' if group == 'referencia' else group
    rows = [r for r in oracle['per_severity'][key] if r.get('count_with_resources')]
    total = sum(r['count_with_resources'] for r in rows)
    excess = sum(r['mean_minus_S_with_resources'] * r['count_with_resources'] for r in rows) / total
    return {'con_recursos': total, 'exceso_medio': excess,
            'por_severidad': {str(r['S']): {'n': r['count_with_resources'], 'media': r['mean_with_resources']}
                              for r in rows}}


def plot(summary: Dict[str, dict], base_path: str, slide: bool) -> None:
    """Mean allocation versus initial severity for the optimum, the rule and the plotted models."""
    rc = PUB_RC if slide else PRINT_RC
    size = (10.0, 5.6) if slide else (PAGE_WIDTH_IN, 3.2)
    with pyplot.rc_context(rc):
        figure, axis = pyplot.subplots(figsize=size)
        observed = sorted({int(s) for key in ('optimo', 'regla', *PLOTTED)
                           for s in summary[key]['por_severidad']})
        grid = numpy.arange(observed[0], observed[-1] + 1)
        axis.plot(grid, grid, linestyle=':', color=PALETTE['grey'], linewidth=1.0, label='$a = S$')
        # pes_trf y pes_ens comparten el acento cálido de la paleta (mejor modelo de cada grupo):
        # aquí el ensamble usa el azul oscuro para distinguirlos.
        colours = {'pes_trf': MODEL_COLOURS['pes_trf'], 'pes_ens': PALETTE['blue_dark'],
                   'pes_dqn': MODEL_COLOURS['pes_dqn']}
        curves = [('optimo', 'Óptimo', PALETTE['ink'], '-', 2.6),
                  ('regla', f'Regla $a = S + {RULE_K}$', PALETTE['ink'], '--', 1.3)]
        curves += [(m, NAMES[m], colours[m], '-', 1.7) for m in PLOTTED]
        for key, label, colour, style, width in curves:
            points = summary[key]['por_severidad']
            xs = sorted(int(s) for s in points)
            excess = f"{summary[key]['exceso_medio']:+.1f}".replace('.', ',')
            axis.plot(xs, [points[str(s)]['media'] for s in xs], linestyle=style, color=colour,
                      linewidth=width, marker='o', markersize=5 if slide else 3.2,
                      label=f'{label} (S {excess[0]} {excess[1:]})')
        axis.set_xlabel('Severidad inicial de la ciudad, $S$')
        axis.set_ylabel('Asignación media con recursos')
        axis.set_xticks(grid)
        axis.set_xlim(grid[0] - 0.3, grid[-1] + 0.3)
        axis.set_ylim(0, MAX_ALLOC + 0.5)
        plotting.style_axes(axis)
        axis.legend(loc='lower right', frameon=False)
        plotting.save_figure(figure, base_path)


###############
##  Main
###############
def main() -> None:
    """Compare every policy with the fixed rule and the optimum and write the JSON and the figures."""
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n', 1)[0])
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='results JSON')
    args = parser.parse_args()

    with open(os.path.join(RESULTS, 'ensemble', 'weighted_ens_oracle.json'), 'r', encoding='utf-8') as handle:
        oracle = json.load(handle)
    with open(os.path.join(RESULTS, 'ensemble', 'fixed_rule.json'), 'r', encoding='utf-8') as handle:
        fixed = json.load(handle)['resumen']

    models = {**{m: 'individual' for m in INDIVIDUAL}, **{m: 'ensemble' for m in ENSEMBLES}}
    result: dict = {'meta': {'presupuesto': BUDGET, 'k_regla': RULE_K, 'grupos': GROUPS}}
    for group, scenarios in {**GROUPS, 'todas': GROUPS['referencia'] + GROUPS['heldout']}.items():
        summary: Dict[str, dict] = {}
        for model, suite in models.items():
            try:
                summary[model] = summarise([d for sc in scenarios for d in model_decisions(suite, model, sc)])
            except FileNotFoundError:
                # pes_ens_sprb, pes_ens_accq y pes_ens_consensus no guardan sus decisiones.
                summary[model] = {'sin_datos': True}
        summary['regla'] = summarise([d for sc in scenarios for d in rule_decisions(sc, RULE_K)])
        summary['optimo'] = optimum_summary(oracle, 'all' if group == 'todas' else group)
        if group in fixed:
            for model in models:
                summary[model]['rbar'] = fixed[group]['modelos'][model]['media']
            summary['regla']['rbar'] = fixed[group][f'S+{RULE_K}']
            summary['optimo']['rbar'] = 1.0
        result[group] = summary

    pooled = result['todas']
    plot(pooled, FIGURE_BASE, slide=False)
    plot(pooled, FIGURE_BASE + '_slide', slide=True)

    print(f"{'política':28s}{'a-S ref':>9s}{'a-S held':>10s}{'r ref':>8s}{'r held':>8s}")
    order = ['optimo', 'regla'] + sorted(models, key=lambda m: -result['referencia'][m]['rbar'])
    for key in order:
        ref, held = result['referencia'][key], result['heldout'][key]
        nan = float('nan')
        print(f"{NAMES.get(key, key):28s}{ref.get('exceso_medio', nan):+9.2f}{held.get('exceso_medio', nan):+10.2f}"
              f"{ref.get('rbar', nan):8.3f}{held.get('rbar', nan):8.3f}")
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=1, ensure_ascii=False)
    print(f'-> {os.path.relpath(args.output, ROOT)}')
    print(f'-> {os.path.relpath(FIGURE_BASE, ROOT)}{{,_slide}}.png')


if __name__ == '__main__':
    main()
