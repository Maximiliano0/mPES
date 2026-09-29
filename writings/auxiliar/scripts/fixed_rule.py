"""Regla fija ``a = min(S + k, R)`` frente a los modelos, en todos los escenarios.

La asignación óptima con información completa entrega, mientras queda
presupuesto, alrededor de ``S + 1,5`` unidades por ciudad
(``weighted_ens_oracle.py``). Este script evalúa como política la regla fija
que asigna ``min(clip(S, 9) + k, R, 10)`` sin usar ningún modelo, para
``k = 0 … 3``, en las mismas secuencias de los ``27`` escenarios del
benchmark y con la métrica de los paquetes, y la compara con el Transformer
(``pes_trf``) y el ensamble ponderado (``pes_ens``).

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
MODELS = {'trf': ('individual', 'pes_trf'), 'ens': ('ensemble', 'pes_ens')}
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
        row.update({key: benchmark_mean(suite, model, scenario) for key, (suite, model) in MODELS.items()})
        rows[scenario] = row

    columns = [f'S+{k}' for k in OFFSETS] + list(MODELS)
    summary = {}
    for group in ('referencia', 'generalizacion', 'heldout'):
        members = [row for row in rows.values() if row['grupo'] == group]
        summary[group] = {column: float(numpy.mean([row[column] for row in members])) for column in columns}
        summary[group]['escenarios'] = len(members)
        for rule in columns[:len(OFFSETS)]:
            summary[group][f'{rule}_supera_trf'] = sum(row[rule] > row['trf'] for row in members)
            summary[group][f'{rule}_supera_ens'] = sum(row[rule] > row['ens'] for row in members)

    print(f"{'escenario':26s}" + ''.join(f'{c:>8s}' for c in columns))
    for scenario, row in rows.items():
        print(f'{scenario:26s}' + ''.join(f'{row[c]:8.3f}' for c in columns))
    for group, values in summary.items():
        print(f'{group:26s}' + ''.join(f'{values[c]:8.3f}' for c in columns)
              + f"   S+2 > trf en {values['S+2_supera_trf']}/{values['escenarios']}")
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump({'escenarios': rows, 'resumen': summary}, handle, indent=1, ensure_ascii=False)
    print(f'-> {os.path.relpath(args.output, ROOT)}')


if __name__ == '__main__':
    main()
