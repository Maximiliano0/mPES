"""Regla fija ``a = min(S + k, R)`` frente a los modelos, en todos los escenarios.

La asignación óptima con información completa entrega, mientras queda
presupuesto, alrededor de ``S + 1,5`` unidades por ciudad
(``weighted_ens_oracle.py``). Este script evalúa como política la regla fija
que asigna ``min(clip(S, 9) + k, R, 10)`` sin usar ningún modelo, para
``k = 0 … 3``, en las mismas secuencias de los ``27`` escenarios del
benchmark y con la métrica de los paquetes, y la compara con los ``13``
modelos del benchmark (siete individuales y seis ensambles).

Con una asignación constante ``a = S + k``, la severidad de una ciudad tras
``n`` evoluciones es ``S + k - k * 1,4**n`` (mientras sea positiva): ``k = 0``
la deja constante y cualquier ``k > 0`` la extingue. La dinámica fija así el
punto de equilibrio ``a = S``; el desplazamiento ``k`` se calibra con los datos.

Uso (desde la raíz del repositorio, con ``win_mpes_env`` activado)::

    python writings/auxiliar/scripts/fixed_rule.py
"""

##########################
##  Imports externos    ##
##########################
import argparse
import json
import os
import sys

import numpy

##########################
##  Configuración       ##
##########################
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
RESULTS = os.path.join(ROOT, 'h1', 'general', 'results')
SCENARIO_ROOT = os.path.join(ROOT, 'h1', 'general', 'work', 'pes_ens', 'scenarios')
DEFAULT_OUTPUT = os.path.join(RESULTS, 'ensemble', 'fixed_rule.json')
OFFSETS = (0, 1, 2, 3)
INDIVIDUAL = ('pes_trf', 'pes_dqn', 'pes_a2c', 'pes_rdqn', 'pes_dql', 'pes_ql', 'pes_base')
ENSEMBLES = ('pes_ens', 'pes_ens_trf_guard', 'pes_ens_consensus_prior', 'pes_ens_consensus',
             'pes_ens_sprb', 'pes_ens_accq')
MODELS = {**{m: ('individual', m) for m in INDIVIDUAL}, **{m: ('ensemble', m) for m in ENSEMBLES}}
RULE = 'S+2'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

##########################
##  Imports internos    ##
##########################
# pylint: disable=wrong-import-position
import weighted_ens_oracle as oracle


###############
##  Helpers
###############
def benchmark_mean(suite: str, model: str, scenario: str) -> float:
    """Mean ``r̄`` of one benchmark cell."""
    path = os.path.join(RESULTS, suite, 'cells', f'{model}__{scenario}.json')
    with open(path, 'r', encoding='utf-8') as handle:
        return float(json.load(handle)['global_mean_perf'])


def group_of(scenario: str) -> str:
    """Reference, held-out replica or generalisation scenario."""
    if scenario == 'sev_base':
        return 'referencia'
    return 'heldout' if scenario.startswith('heldout') else 'generalizacion'


def summarise(members: list, columns: list) -> dict:
    """Means of one scenario group, rule wins and each model against the rule and the optimum."""
    summary: dict = {column: float(numpy.mean([row[column] for row in members]))
                     for column in columns}
    summary['escenarios'] = len(members)
    for rule in columns[:len(OFFSETS)]:
        for key, model in (('trf', 'pes_trf'), ('ens', 'pes_ens')):
            summary[f'{rule}_supera_{key}'] = sum(row[rule] > row[model] for row in members)
    # Each model against the rule S+2 and against the optimum (r̄ = 1).
    summary['modelos'] = {
        model: {'media': summary[model],
                'dif_regla': summary[model] - summary[RULE],
                'brecha_optimo': 1.0 - summary[model],
                'supera_regla': sum(row[model] > row[RULE] for row in members)}
        for model in MODELS}
    return summary


###############
##  Main
###############
def main() -> None:
    """Evaluate the fixed rule on every scenario and summarise it by group."""
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n', 1)[0])
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='results JSON')
    args = parser.parse_args()

    rows = {}
    for scenario in sorted(os.listdir(SCENARIO_ROOT)):
        row: dict = {'grupo': group_of(scenario)}
        row.update({f'S+{k}': oracle.prior_policy(scenario, k) for k in OFFSETS})
        row.update({key: benchmark_mean(suite, model, scenario)
                    for key, (suite, model) in MODELS.items()})
        rows[scenario] = row

    columns = [f'S+{k}' for k in OFFSETS] + list(MODELS)
    summary = {group: summarise([row for row in rows.values() if row['grupo'] == group], columns)
               for group in ('referencia', 'generalizacion', 'heldout')}

    for group, values in summary.items():
        print(f"{group} ({values['escenarios']} escenarios)")
        for rule in columns[:len(OFFSETS)]:
            print(f'  {rule:26s}{values[rule]:8.3f}')
        for model, info in sorted(values['modelos'].items(), key=lambda item: -item[1]['media']):
            print(f"  {model:26s}{info['media']:8.3f}  vs {RULE} {info['dif_regla']:+.3f}"
                  f"  supera a {RULE} en {info['supera_regla']}/{values['escenarios']}")
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump({'escenarios': rows, 'resumen': summary}, handle, indent=1, ensure_ascii=False)
    print(f'-> {os.path.relpath(args.output, ROOT)}')


if __name__ == '__main__':
    main()
