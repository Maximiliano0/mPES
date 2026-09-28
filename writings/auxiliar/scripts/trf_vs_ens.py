"""Comparación de cada ensamble con el Transformer individual en generalización.

Reproduce la Tabla ``tab:trf-vs-ens`` de Resultados a partir de los
``summary.json`` de ``h1/general/results/{individual,ensemble}/``:

* ``diferencia``: media de generalización del ensamble menos la de
  ``pes_trf`` (promedio de las medias de los escenarios distintos de la
  referencia).
* ``d`` y ``p``: d de Cohen y test de Welch sobre todas las secuencias de
  esos escenarios, con las mismas funciones que los mapas por pares
  (``general.scripts.plotting``).
* ``mayor media en``: escenarios en que la media del ensamble, redondeada a
  los 6 decimales de ``matrices/global_mean.csv``, supera a la del
  Transformer.

Uso (desde la raíz del repositorio)::

    python writings/auxiliar/scripts/trf_vs_ens.py
    python writings/auxiliar/scripts/trf_vs_ens.py --output trf_vs_ens.json
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
H1 = os.path.join(ROOT, 'h1')
RESULTS = os.path.join(H1, 'general', 'results')
sys.path.insert(0, H1)

##########################
##  Imports internos    ##
##########################
# pylint: disable=wrong-import-position
from general.scripts.plotting import cohen_d, welch_test

TRANSFORMER = 'pes_trf'
# Decimales de ``matrices/global_mean.csv``; con esa precisión un empate no cuenta como mayor media.
MEAN_DECIMALS = 6


###############
##  Helpers
###############
def load_summary(suite: str) -> dict:
    """Load the ``summary.json`` of one benchmark suite.

    Parameters
    ----------
    suite : str
        ``'individual'`` or ``'ensemble'``.

    Returns
    -------
    dict
        Parsed summary.
    """
    with open(os.path.join(RESULTS, suite, 'summary.json'), encoding='utf-8') as handle:
        return json.load(handle)


def compare(ensemble_cells: dict, transformer_cells: dict, scenarios: 'list[str]') -> dict:
    """Compare one ensemble with the Transformer over the given scenarios.

    Parameters
    ----------
    ensemble_cells : dict
        ``{scenario: cell}`` of the ensemble.
    transformer_cells : dict
        ``{scenario: cell}`` of ``pes_trf``.
    scenarios : list of str
        Generalisation scenarios (reference excluded).

    Returns
    -------
    dict
        Difference of means, Cohen's d, Welch p and log10 p, and number of
        scenarios in which the ensemble has the larger mean.
    """
    ensemble_means = numpy.asarray([ensemble_cells[s]['global_mean_perf'] for s in scenarios], dtype=float)
    transformer_means = numpy.asarray([transformer_cells[s]['global_mean_perf'] for s in scenarios], dtype=float)
    ensemble_pool = numpy.asarray([v for s in scenarios for v in ensemble_cells[s]['per_sequence_perf']],
                                  dtype=float)
    transformer_pool = numpy.asarray([v for s in scenarios for v in transformer_cells[s]['per_sequence_perf']],
                                     dtype=float)
    _, p_value, log10_p = welch_test(ensemble_pool, transformer_pool)
    return {'gen_mean': float(numpy.mean(ensemble_means)),
            'difference': float(numpy.mean(ensemble_means) - numpy.mean(transformer_means)),
            'cohen_d': cohen_d(ensemble_pool, transformer_pool),
            'welch_p': p_value, 'welch_log10_p': log10_p,
            'wins': int(numpy.sum(numpy.round(ensemble_means, MEAN_DECIMALS)
                                  > numpy.round(transformer_means, MEAN_DECIMALS))),
            'n_scenarios': len(scenarios)}


###############
##  Main
###############
def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=(__doc__ or '').splitlines()[0])
    parser.add_argument('--output', default=None, help='optional JSON file for the comparison')
    args = parser.parse_args()

    individual = load_summary('individual')
    ensemble = load_summary('ensemble')
    reference = ensemble['reference_scenario']
    transformer_cells = individual['cells'][TRANSFORMER]
    results = {}
    for model, cells in ensemble['cells'].items():
        scenarios = [s for s in cells if s != reference and s in transformer_cells]
        results[model] = compare(cells, transformer_cells, scenarios)
    for model, row in sorted(results.items(), key=lambda item: -item[1]['difference']):
        print(f'{model:26s} gen={row["gen_mean"]:.4f} diff={row["difference"]:+.4f} d={row["cohen_d"]:+.3f} '
              f'p={row["welch_p"]:.3g} log10p={row["welch_log10_p"]:.2f} '
              f'mayor media en {row["wins"]} de {row["n_scenarios"]}')
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as handle:
            json.dump(results, handle, indent=2)


if __name__ == '__main__':
    main()
