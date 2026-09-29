"""Caída de cada modelo entre la referencia y sus réplicas fuera de muestra.

Compara, para los 13 modelos, la media en ``sev_base`` (las 64 secuencias
fijas sobre las que se eligieron hiperparámetros y parámetros de ensamble)
con la de las réplicas ``heldout_s1`` … ``heldout_s5``: sorteos i.i.d. de
las mismas frecuencias de severidad y longitud que nunca intervinieron en
el entrenamiento ni en Optuna. Lee los ``summary.json`` de
``h1/general/results/{individual,ensemble}/``:

* ``caída``: referencia menos media agrupada de las réplicas (positivo =
  la referencia es optimista), con su error estándar.
* ``d`` y ``p``: d de Cohen y test de Welch entre las secuencias agrupadas
  de las réplicas y las de la referencia (``general.scripts.plotting``).
* ``DE entre réplicas``: desvío de las cinco medias por réplica.
* Rangos de los 13 modelos en la referencia y en las réplicas, su
  correlación de Spearman y la comparación pareada (mismas secuencias)
  del ensamble ponderado con el Transformer en las réplicas.

Uso (desde la raíz del repositorio)::

    python writings/auxiliar/scripts/heldout_gap.py
    python writings/auxiliar/scripts/heldout_gap.py --output h1/general/results/heldout/heldout_gap.json
"""

##########################
##  Imports externos    ##
##########################
import argparse
import json
import os
import sys

import numpy
from scipy import stats

##########################
##  Configuración       ##
##########################
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
H1 = os.path.join(ROOT, 'h1')
RESULTS = os.path.join(H1, 'general', 'results')
DEFAULT_OUTPUT = os.path.join(RESULTS, 'heldout', 'heldout_gap.json')
sys.path.insert(0, H1)

##########################
##  Imports internos    ##
##########################
# pylint: disable=wrong-import-position
from general.scripts.benchmark import REFERENCE_SCENARIO, heldout_scenarios
from general.scripts.plotting import cohen_d, welch_test

# Cómo se eligió la configuración evaluada en la referencia.
SELECTION = {'pes_base': 'sin optimizar', 'pes_ens': 'manual'}
WEIGHTED, TRANSFORMER = 'pes_ens', 'pes_trf'


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


def _vector(cell: dict) -> numpy.ndarray:
    return numpy.asarray(cell.get('per_sequence_perf', []), dtype=float)


def gap(cells: dict, replicas: 'list[str]') -> dict:
    """Reference versus pooled held-out replicas for one model.

    Parameters
    ----------
    cells : dict
        ``{scenario: cell}`` of the model.
    replicas : list of str
        Held-out scenario ids present for the model.

    Returns
    -------
    dict
        Means, standard deviations, gap and its standard error, Cohen's d,
        Welch p and log10 p, and the per-replica means.
    """
    reference = _vector(cells[REFERENCE_SCENARIO])
    pooled = numpy.concatenate([_vector(cells[s]) for s in replicas])
    replica_means = [float(cells[s]['global_mean_perf']) for s in replicas]
    _, p_value, log10_p = welch_test(pooled, reference)
    return {'reference_mean': float(reference.mean()), 'reference_std': float(reference.std(ddof=1)),
            'n_reference': int(reference.size),
            'heldout_mean': float(pooled.mean()), 'heldout_std': float(pooled.std(ddof=1)),
            'n_heldout': int(pooled.size),
            'replica_means': dict(zip(replicas, replica_means)),
            'sd_between_replicas': float(numpy.std(replica_means, ddof=1)),
            'gap': float(reference.mean() - pooled.mean()),
            'gap_se': float(numpy.sqrt(reference.var(ddof=1) / reference.size
                                       + pooled.var(ddof=1) / pooled.size)),
            'cohen_d': cohen_d(pooled, reference), 'welch_p': p_value, 'welch_log10_p': log10_p}


def _field(result, name: str) -> float:
    """Read ``statistic`` / ``pvalue`` from a scipy result object."""
    return float(getattr(result, name))


def _ranks(values: 'dict[str, float]') -> 'dict[str, int]':
    order = sorted(values, key=lambda model: -values[model])
    return {model: rank for rank, model in enumerate(order, 1)}


def paired(first: dict, second: dict, replicas: 'list[str]') -> dict:
    """Paired comparison of two models on the same held-out sequences.

    Returns
    -------
    dict
        Mean per-sequence difference (first - second), paired t-test and
        Wilcoxon signed-rank p-values, and replicas where ``first`` has the larger mean.
    """
    diff = numpy.concatenate([_vector(first[s]) - _vector(second[s]) for s in replicas])
    wins = sum(first[s]['global_mean_perf'] > second[s]['global_mean_perf'] for s in replicas)
    return {'mean_difference': float(diff.mean()),
            'paired_t_p': _field(stats.ttest_rel(numpy.concatenate([_vector(first[s]) for s in replicas]),
                                                 numpy.concatenate([_vector(second[s]) for s in replicas])),
                                 'pvalue'),
            'wilcoxon_p': _field(stats.wilcoxon(diff[diff != 0]), 'pvalue') if numpy.any(diff != 0) else 1.0,
            'replicas_won': int(wins), 'n_replicas': len(replicas), 'n_sequences': int(diff.size)}


###############
##  Main
###############
def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=(__doc__ or '').splitlines()[0])
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='JSON file for the results')
    args = parser.parse_args()

    models: dict = {}
    for suite in ('individual', 'ensemble'):
        summary = load_summary(suite)
        replicas = heldout_scenarios(summary['scenarios'])
        for model in summary['models']:
            cells = summary['cells'][model]
            present = [s for s in replicas if _vector(cells.get(s, {})).size]
            if REFERENCE_SCENARIO not in cells or len(present) != len(replicas) or not present:
                print(f'{model}: faltan réplicas held-out ({len(present)} de {len(replicas)})')
                continue
            models[model] = {'suite': suite, 'selection': SELECTION.get(model, 'optuna'),
                             **gap(cells, present), '_cells': cells, '_replicas': present}

    reference_rank = _ranks({m: row['reference_mean'] for m, row in models.items()})
    heldout_rank = _ranks({m: row['heldout_mean'] for m, row in models.items()})
    for model, row in models.items():
        row['rank_reference'], row['rank_heldout'] = reference_rank[model], heldout_rank[model]
    names = list(models)
    spearman = stats.spearmanr([models[m]['reference_mean'] for m in names],
                               [models[m]['heldout_mean'] for m in names])
    rho, rho_p = _field(spearman, 'statistic'), _field(spearman, 'pvalue')
    comparison = (paired(models[WEIGHTED]['_cells'], models[TRANSFORMER]['_cells'],
                         models[WEIGHTED]['_replicas'])
                  if WEIGHTED in models and TRANSFORMER in models else None)

    for model, row in sorted(models.items(), key=lambda item: item[1]['rank_reference']):
        print(f'{model:24s} {row["selection"]:13s} ref={row["reference_mean"]:.4f} '
              f'heldout={row["heldout_mean"]:.4f} (DE réplicas {row["sd_between_replicas"]:.4f}) '
              f'caída={row["gap"]:+.4f}±{row["gap_se"]:.4f} d={row["cohen_d"]:+.2f} '
              f'p={row["welch_p"]:.3g} rango {row["rank_reference"]}->{row["rank_heldout"]}')
    print(f'Spearman referencia vs held-out: rho={rho:.3f} p={rho_p:.3g}')
    if comparison:
        print(f'{WEIGHTED} - {TRANSFORMER} en held-out (pareado): {comparison}')

    payload = {'reference_scenario': REFERENCE_SCENARIO,
               'models': {m: {k: v for k, v in row.items() if not k.startswith('_')}
                          for m, row in models.items()},
               'spearman_reference_vs_heldout': {'rho': rho, 'p': rho_p, 'n_models': len(names)},
               'weighted_vs_transformer_heldout': comparison}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f'-> {os.path.relpath(args.output, ROOT)}')


if __name__ == '__main__':
    main()
