"""Seed variability of the individual models and the ensembles in the 21 generalisation scenarios.

Reads the per-sequence results that ``seed_variability eval`` (individual models, ``work/seeds/<pkg>/<run>/
generalizacion.json``) and ``ensemble_seeds run --scenarios`` (ensembles, ``work/ens_seeds/generalizacion/<pkg>/
<combination>.json``) leave on disk, and repeats for every training seed the generalisation analysis of the
benchmark:

* **Per model and ensemble, across seeds or combinations**: generalisation mean (mean of the 21 scenario means),
  sample standard deviation, range and 95 % ``t`` interval, change from its own reference and worst scenario of
  every run.
* **Between seeds** (``n = 10`` per model, the right test for training variability): Welch test and Cohen's ``d``
  on the per-seed generalisation means of every pair of models.
* **Per seed, the benchmark analysis**: Cohen's ``d`` and Welch's ``p`` between two models with the sequences of
  the 21 scenarios pooled, pairing the ``i``-th new seed of each model; summarised by median and range of ``d`` and
  the number of seeds with ``p < 0.05``.
* **Spearman**: ordering of the models by their per-seed means against the delivered ordering, per seed index and
  overall, and stability of the ordering of the 21 scenarios of every model.
* **Ensembles**: each combination against the Transformer of the same combination (difference in percent,
  sequence-level ``d`` and ``p``, paired test across combinations).

The delivered runs (seed ``0`` and the ``entregado`` combination) are checked against the benchmark cells; the
check is stored under ``control``. Differences are in percent of the comparison mean, as in the benchmark report.

Usage (from ``h1/``, with ``win_mpes_env`` activated, after the evaluations)::

    python -m general.scripts.seed_generalisation
"""

##########################
##  Imports externos    ##
##########################
import argparse
import json
import math
import os
import sys
from collections import Counter
from itertools import combinations
from typing import Dict, List, Optional

import numpy
from scipy import stats

##########################
##  Imports internos    ##
##########################
from .ensemble_seeds import DELIVERED, ENSEMBLES, GEN_WORK, PLAN_FILE, WORK as ENS_WORK
from .plotting import cohen_d, welch_test
from .seed_variability import (GEN_FILE, MODELS, NEW_SEEDS, ORDER, SCENARIOS, WORK as SEEDS_WORK,
                               generalisation_scenarios)

##########################
##  Configuración       ##
##########################
GENERAL = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS = os.path.join(GENERAL, 'results')
OUTPUTS = {'individual': os.path.join(RESULTS, 'individual', 'seed_generalisation.json'),
           'ensemble': os.path.join(RESULTS, 'ensemble', 'seed_generalisation.json')}
REFERENCE = SCENARIOS[0]
HELDOUT = SCENARIOS[1:]
ALPHA = 0.05
TOLERANCE = 1e-4            # the benchmark cells keep four decimals


###############
##  Helpers
###############
def read_json(path: str) -> dict:
    """Read a UTF-8 JSON file."""
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def write_json(path: str, payload: dict) -> None:
    """Write ``payload`` as indented UTF-8 JSON, creating the folder."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False)


def finite(value: float) -> Optional[float]:
    """``value`` as a float, or ``None`` when it is not finite (JSON has no NaN)."""
    return float(value) if math.isfinite(value) else None


def spread(values: List[float]) -> dict:
    """Mean, sample standard deviation, range and 95 % ``t`` interval of ``values``.

    Parameters
    ----------
    values : list of float
        One value per seed or combination.

    Returns
    -------
    dict
        ``n``, ``media``, ``desviacion`` (``ddof=1``), ``minimo``, ``maximo`` and ``ic95`` (``[low, high]``);
        statistics that need two values are ``None`` with fewer.
    """
    array = numpy.asarray(values, dtype=float)
    if array.size == 0:
        return {'n': 0, 'media': None, 'desviacion': None, 'minimo': None, 'maximo': None, 'ic95': None}
    mean = float(array.mean())
    if array.size < 2:
        return {'n': 1, 'media': mean, 'desviacion': None, 'minimo': mean, 'maximo': mean, 'ic95': None}
    deviation = float(array.std(ddof=1))
    half = float(stats.t.ppf(1 - ALPHA / 2, array.size - 1)) * deviation / math.sqrt(array.size)
    return {'n': int(array.size), 'media': mean, 'desviacion': deviation, 'minimo': float(array.min()),
            'maximo': float(array.max()), 'ic95': [mean - half, mean + half]}


def percent(first: float, second: float) -> float:
    """Difference ``first - second`` in percent of ``second``."""
    return 100.0 * (first - second) / second


def spearman(first: List[float], second: List[float]) -> Optional[float]:
    """Spearman rank correlation, or ``None`` with fewer than three pairs or a constant input."""
    if len(first) < 3 or numpy.ptp(first) == 0 or numpy.ptp(second) == 0:
        return None
    ranks = numpy.vstack([stats.rankdata(first), stats.rankdata(second)])   # average ranks for ties
    return finite(float(numpy.corrcoef(ranks)[0, 1]))


def compare_sequences(first: numpy.ndarray, second: numpy.ndarray) -> dict:
    """Cohen's ``d``, Welch's ``p`` and the difference in percent between two pooled sequence samples."""
    _t, p_value, log10_p = welch_test(first, second)
    return {'d': finite(cohen_d(first, second)), 'p': finite(p_value), 'log10_p': finite(log10_p),
            'diferencia_pct': percent(float(first.mean()), float(second.mean()))}


def summarise_comparisons(rows: List[dict]) -> dict:
    """Median and range of ``d``, seeds with ``p < ALPHA`` and seeds where the first sample is higher."""
    d_values = [row['d'] for row in rows if row['d'] is not None]
    p_values = [row['p'] for row in rows if row['p'] is not None]
    return {'n': len(rows),
            'd_mediana': float(numpy.median(d_values)) if d_values else None,
            'd_minimo': min(d_values) if d_values else None, 'd_maximo': max(d_values) if d_values else None,
            'p_menor_005': int(sum(p < ALPHA for p in p_values)),
            'primero_mayor': int(sum(row['diferencia_pct'] > 0 for row in rows))}


###############
##  Runs
###############
class Run:
    """Per-sequence results of one model or ensemble run in every evaluated scenario.

    Parameters
    ----------
    per_sequence : dict
        ``{scenario: [r per sequence]}``.
    generalisation : list of str
        The 21 generalisation scenarios.
    """

    def __init__(self, per_sequence: Dict[str, List[float]], generalisation: List[str]) -> None:
        self.per_sequence = {s: numpy.asarray(v, dtype=float) for s, v in per_sequence.items()}
        self.generalisation = generalisation

    def complete(self) -> bool:
        """Whether the reference, the replicas and the 21 generalisation scenarios are all present."""
        return all(s in self.per_sequence for s in SCENARIOS + self.generalisation)

    def mean(self, scenario: str) -> float:
        """Mean ``r`` of one scenario."""
        return float(self.per_sequence[scenario].mean())

    def generalisation_mean(self) -> float:
        """Mean of the 21 generalisation scenario means (as in the benchmark report)."""
        return float(numpy.mean([self.mean(s) for s in self.generalisation]))

    def heldout_mean(self) -> float:
        """Mean of the five held-out replica means."""
        return float(numpy.mean([self.mean(s) for s in HELDOUT]))

    def pooled(self) -> numpy.ndarray:
        """Sequences of the 21 generalisation scenarios pooled into one sample."""
        return numpy.concatenate([self.per_sequence[s] for s in self.generalisation])

    def profile(self) -> List[float]:
        """Mean of every generalisation scenario, in scenario order."""
        return [self.mean(s) for s in self.generalisation]

    def describe(self) -> dict:
        """Means, change from the reference and worst generalisation scenario of this run."""
        reference = self.mean(REFERENCE)
        worst = min(self.generalisation, key=self.mean)
        return {'referencia': reference, 'heldout': self.heldout_mean(), 'generalizacion': self.generalisation_mean(),
                'generalizacion_menos_referencia_pct': percent(self.generalisation_mean(), reference),
                'peor_escenario': worst, 'peor_media': self.mean(worst),
                'caida_peor_pct': 100.0 * (reference - self.mean(worst)) / reference}


def control(run: Run, suite: str, pkg: str) -> dict:
    """Largest per-sequence difference between a delivered run and the benchmark cells.

    Parameters
    ----------
    run : Run
        Delivered model or ensemble.
    suite : str
        ``individual`` or ``ensemble``.
    pkg : str
        Package name of the cells.

    Returns
    -------
    dict
        ``max_dif``, ``escenarios`` compared and ``ok`` (every difference below ``TOLERANCE``).
    """
    worst, compared = 0.0, 0
    for scenario, values in run.per_sequence.items():
        cell = os.path.join(RESULTS, suite, 'cells', f'{pkg}__{scenario}.json')
        if not os.path.isfile(cell):
            continue
        reference = numpy.asarray(read_json(cell)['per_sequence_perf'], dtype=float)
        if reference.shape != values.shape:
            return {'max_dif': None, 'escenarios': compared, 'ok': False, 'error': f'{scenario}: forma distinta'}
        worst = max(worst, float(numpy.max(numpy.abs(values - reference))))
        compared += 1
    return {'max_dif': worst, 'escenarios': compared, 'ok': bool(compared and worst < TOLERANCE)}


def load_individual(generalisation: List[str]) -> Dict[str, Dict[int, Run]]:
    """Complete runs of every individual model, keyed by model and seed (``0`` = delivered)."""
    runs: Dict[str, Dict[int, Run]] = {}
    for key in ORDER:
        folder = os.path.join(SEEDS_WORK, MODELS[key]['pkg'].split('.')[1])
        runs[key] = {}
        for seed in [0] + NEW_SEEDS:
            path = os.path.join(folder, 'entregado' if seed == 0 else f'seed_{seed}', GEN_FILE)
            if os.path.isfile(path):
                run = Run(read_json(path)['por_secuencia'], generalisation)
                if run.complete():
                    runs[key][seed] = run
    return runs


def load_ensembles(generalisation: List[str]) -> Dict[str, Dict[str, Run]]:
    """Complete runs of every ensemble, keyed by ensemble and combination id.

    The reference and the held-out replicas missing from a generalisation run are taken from the reference study
    (``work/ens_seeds/results/<pkg>/<combination>.json``), so ``run --scenarios gen`` is enough.
    """
    runs: Dict[str, Dict[str, Run]] = {}
    for pkg in ENSEMBLES:
        runs[pkg] = {}
        folder = os.path.join(GEN_WORK, pkg)
        for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
            if not name.endswith('.json'):
                continue
            result = read_json(os.path.join(folder, name))
            per_sequence = dict(result['por_secuencia'])
            reference_file = os.path.join(ENS_WORK, 'results', pkg, name)
            if os.path.isfile(reference_file):
                for scenario, values in read_json(reference_file)['por_secuencia'].items():
                    per_sequence.setdefault(scenario, values)
            run = Run(per_sequence, generalisation)
            if run.complete():
                runs[pkg][result['combinacion']] = run
    return runs


###############
##  Analysis
###############
def individual_analysis(runs: Dict[str, Dict[int, Run]]) -> dict:
    """Seed statistics, between-seed and per-seed comparisons and Spearman orderings of the individual models."""
    models: dict = {}
    for key, by_seed in runs.items():
        new = {seed: run for seed, run in by_seed.items() if seed != 0}
        described = {seed: run.describe() for seed, run in by_seed.items()}
        models[key] = {
            'paquete': MODELS[key]['pkg'],
            'entregado': described.get(0),
            'control': control(by_seed[0], 'individual', MODELS[key]['pkg'].split('.')[1]) if 0 in by_seed else None,
            'semillas': {str(seed): described[seed] for seed in sorted(new)},
            'semillas_nuevas': {field: spread([described[s][field] for s in new])
                                for field in ('referencia', 'heldout', 'generalizacion',
                                              'generalizacion_menos_referencia_pct', 'caida_peor_pct')},
            'peor_escenario_frecuencia': dict(Counter(described[s]['peor_escenario'] for s in new).most_common())}
    present = [key for key in ORDER if len([s for s in runs[key] if s != 0]) >= 2]
    delivered_order = [key for key in present if 0 in runs[key]]
    pairs: dict = {}
    for first, second in combinations(present, 2):
        means_a = numpy.asarray([runs[first][s].generalisation_mean() for s in runs[first] if s != 0])
        means_b = numpy.asarray([runs[second][s].generalisation_mean() for s in runs[second] if s != 0])
        _t, p_value, _log10 = welch_test(means_a, means_b)
        per_seed = [compare_sequences(runs[first][s].pooled(), runs[second][s].pooled())
                    for s in NEW_SEEDS if s in runs[first] and s in runs[second]]
        pairs[f'{first}_vs_{second}'] = {
            'entre_semillas': {'welch_p': finite(p_value), 'd': finite(cohen_d(means_a, means_b)),
                               'diferencia_pct': percent(float(means_a.mean()), float(means_b.mean())),
                               'rangos_superpuestos': bool(means_a.min() <= means_b.max()
                                                           and means_b.min() <= means_a.max())},
            'por_semilla': summarise_comparisons(per_seed),
            'entregados': (compare_sequences(runs[first][0].pooled(), runs[second][0].pooled())
                           if 0 in runs[first] and 0 in runs[second] else None)}
    return {'modelos': models, 'pares': pairs, 'spearman': orderings(runs, delivered_order)}


def orderings(runs: Dict[str, Dict[int, Run]], keys: List[str]) -> dict:
    """Spearman of the model ordering (per seed index and by seed means) and of each model's scenario profile."""
    if len(keys) < 3:
        return {}
    delivered = [runs[key][0].generalisation_mean() for key in keys]
    per_index = []
    for seed in NEW_SEEDS:
        if all(seed in runs[key] for key in keys):
            per_index.append(spearman(delivered, [runs[key][seed].generalisation_mean() for key in keys]))
    seed_means = [float(numpy.mean([run.generalisation_mean() for s, run in runs[key].items() if s != 0]))
                  for key in keys]
    profiles = {}
    for key in keys:
        values = [spearman(runs[key][0].profile(), run.profile()) for s, run in runs[key].items() if s != 0]
        values = [v for v in values if v is not None]
        profiles[key] = {'mediana': float(numpy.median(values)) if values else None,
                         'minimo': min(values) if values else None}
    valid = [v for v in per_index if v is not None]
    return {'modelos': keys,
            'orden_medias_semillas_vs_entregados': spearman(delivered, seed_means),
            'orden_por_semilla': {'n': len(valid), 'mediana': float(numpy.median(valid)) if valid else None,
                                  'minimo': min(valid) if valid else None},
            'perfil_de_escenarios_vs_entregado': profiles}


def ensemble_analysis(runs: Dict[str, Dict[str, Run]], individual: Dict[str, Dict[int, Run]]) -> dict:
    """Combination statistics of every ensemble and comparison with the Transformer of the same combination."""
    plan = {combo['id']: combo['semillas'] for combo in read_json(PLAN_FILE)['combinaciones']}
    transformers = individual.get('trf', {})
    ensembles: dict = {}
    for pkg, by_combo in runs.items():
        combos = {cid: run for cid, run in by_combo.items() if cid != DELIVERED}
        described = {cid: run.describe() for cid, run in by_combo.items()}
        versus, own, ours = [], [], []
        for cid, run in combos.items():
            seed = plan.get(cid, {}).get('trf')
            if seed in transformers:
                row = compare_sequences(run.pooled(), transformers[seed].pooled())
                row.update({'combinacion': cid, 'semilla_trf': seed})
                versus.append(row)
                ours.append(run.generalisation_mean())
                own.append(transformers[seed].generalisation_mean())
        differences = numpy.asarray(ours) - numpy.asarray(own)
        paired_p = (finite(float(stats.ttest_rel(ours, own)[1]))
                    if len(ours) >= 2 and float(numpy.std(differences)) > 0 else None)
        pct = [row['diferencia_pct'] for row in versus]
        ensembles[pkg] = {
            'entregado': described.get(DELIVERED),
            'control': control(by_combo[DELIVERED], 'ensemble', pkg) if DELIVERED in by_combo else None,
            'combinaciones': {cid: described[cid] for cid in sorted(combos)},
            'combinaciones_resumen': {field: spread([described[c][field] for c in combos])
                                      for field in ('referencia', 'heldout', 'generalizacion',
                                                    'generalizacion_menos_referencia_pct', 'caida_peor_pct')},
            'frente_a_su_transformer': {
                'por_combinacion': versus, 'secuencias': summarise_comparisons(versus),
                'supera': int(sum(p > 0 for p in pct)), 'n': len(pct),
                'diferencia_pct': spread(pct), 'pareada_p': paired_p}}
    return {'ensambles': ensembles, 'spearman': ensemble_orderings(runs, transformers, plan)}


def ensemble_orderings(runs: Dict[str, Dict[str, Run]], transformers: Dict[int, Run], plan: dict) -> dict:
    """Spearman of the ordering of the ensembles and their Transformer, per combination, against the delivered one."""
    keys = [pkg for pkg in ENSEMBLES if DELIVERED in runs.get(pkg, {})]
    if len(keys) < 2 or 0 not in transformers:
        return {}
    delivered = [runs[pkg][DELIVERED].generalisation_mean() for pkg in keys] + [transformers[0].generalisation_mean()]
    values = []
    for cid, seeds in plan.items():
        if seeds.get('trf') in transformers and all(cid in runs[pkg] for pkg in keys):
            values.append(spearman(delivered, [runs[pkg][cid].generalisation_mean() for pkg in keys]
                                   + [transformers[seeds['trf']].generalisation_mean()]))
    valid = [v for v in values if v is not None]
    return {'modelos': keys + ['pes_trf'], 'n': len(valid),
            'mediana': float(numpy.median(valid)) if valid else None, 'minimo': min(valid) if valid else None}


###############
##  Main
###############
def main() -> None:
    """Compute both analyses with the runs available and write their JSON files."""
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n', 1)[0])
    parser.parse_args()
    generalisation = generalisation_scenarios()
    individual = load_individual(generalisation)
    payload = {'escenarios_generalizacion': generalisation, **individual_analysis(individual)}
    write_json(OUTPUTS['individual'], payload)
    for key, info in payload['modelos'].items():
        new = info['semillas_nuevas']['generalizacion']
        check = info['control'] or {}
        print(f"{key:5s} semillas={new['n']:2d}  gen {new['media'] if new['media'] is not None else float('nan'):.4f}"
              f" ± {new['desviacion'] if new['desviacion'] is not None else float('nan'):.4f}  "
              f"control entregado: {check.get('ok')} ({check.get('escenarios', 0)} escenarios)")
    ensembles = ensemble_analysis(load_ensembles(generalisation), individual)
    write_json(OUTPUTS['ensemble'], {'escenarios_generalizacion': generalisation, **ensembles})
    for pkg, info in ensembles['ensambles'].items():
        versus = info['frente_a_su_transformer']
        check = info['control'] or {}
        print(f"{pkg:24s} combinaciones={info['combinaciones_resumen']['generalizacion']['n']:2d}  "
              f"supera a su TRF en {versus['supera']}/{versus['n']}  control entregado: {check.get('ok')}")
    print(f"-> {os.path.relpath(OUTPUTS['individual'], GENERAL)}, {os.path.relpath(OUTPUTS['ensemble'], GENERAL)}")


if __name__ == '__main__':
    sys.exit(main())
