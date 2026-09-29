"""Asignación óptima exacta frente a las reglas fijas del ensamble ponderado.

``pes_ens`` decide con reglas fijadas a mano: un *prior* gaussiano
``N(S, 3)`` con peso ``0.17``, la atenuación ``0.3`` de la acción 0 cuando
quedan recursos y la cota ``a >= floor(S/2)`` cuando ``S >= 6``. Este script
las contrasta con el óptimo del problema, que ``_best_feasible_sequence_severity``
(``h1/ens/pes_ens/src/exp_utils.py``) calcula por programación dinámica sólo
como valor:

* Reimplementa esa programación dinámica con reconstrucción de la solución y
  comprueba, secuencia por secuencia, que el mínimo coincide. Como el óptimo
  puede no ser único, calcula además para cada ciudad el conjunto de acciones
  compatibles con *alguna* asignación óptima. La asignación canónica es la de
  menor presupuesto total y, dentro de ella, la de menor acción en las
  primeras ciudades.
* Usa las 64 secuencias de ``sev_base`` y las de ``heldout_s1`` …
  ``heldout_s5`` (``h1/general/work/pes_ens/scenarios/<escenario>/``). El
  óptimo usa la severidad cruda; el agente la recorta a 9.
* Resume ``a*`` por severidad inicial ``S``, ajusta por máxima verosimilitud
  el ``sigma`` (y el centro) de una gaussiana discreta sobre las acciones
  factibles y evalúa como política el *prior* solo (``a = min(S, R)``).
* Auditoría de reglas (salvo ``--no-audit``): reproduce ``pes_ens`` con los
  valores fijos (``weighted_ens_sensitivity.py``) y, en cada estado que
  visita, compara la acción con la óptima clarividente desde ese estado
  (presupuesto restante y ciudades futuras conocidas). La pérdida de cada
  decisión se expresa en unidades de ``r̄`` y la suma a lo largo de una
  secuencia es exactamente ``1 - r̄``. Para cada regla se compara la acción
  final con la que saldría en el mismo estado sin esa regla.

Uso (desde la raíz del repositorio, con ``win_mpes_env`` activado)::

    python writings/auxiliar/scripts/weighted_ens_oracle.py
    python writings/auxiliar/scripts/weighted_ens_oracle.py --no-audit
"""

##########################
##  Imports externos    ##
##########################
import argparse
import contextlib
import dataclasses
import io
import json
import os
import sys
from typing import Dict, List

import numpy

##########################
##  Configuración       ##
##########################
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
H1 = os.path.join(ROOT, 'h1')
DEFAULT_OUTPUT = os.path.join(H1, 'general', 'results', 'ensemble', 'weighted_ens_oracle.json')
sys.path.insert(0, H1)

##########################
##  Imports internos    ##
##########################
# pylint: disable=wrong-import-position,protected-access
with contextlib.redirect_stdout(io.StringIO()):
    from ens.pes_ens import MAX_ALLOCATABLE_RESOURCES, MAX_SEVERITY
    from ens.pes_ens.config.CONFIG import ENS_SEVERITY_PRIOR_SIGMA
    from ens.pes_ens.ext.tools import convert_globalseq_to_seqs
    from ens.pes_ens.src import exp_utils

BUDGET = exp_utils._FEASIBLE_BUDGET_PER_SEQUENCE
MAX_ALLOC = MAX_ALLOCATABLE_RESOURCES
SCENARIOS = ['sev_base', 'heldout_s1', 'heldout_s2', 'heldout_s3', 'heldout_s4', 'heldout_s5']
GROUPS = {'sev_base': ['sev_base'], 'heldout': SCENARIOS[1:], 'all': SCENARIOS}
TIE_TOL = 1e-9
FLOOR_MIN_SEVERITY = 6
SIGMA_GRID = numpy.round(numpy.arange(0.25, 10.0001, 0.01), 2)
PRIOR_POLICY_OFFSETS = (0, 1, 2, 3)
OFFSET_GRID = numpy.round(numpy.arange(-4.0, 4.0001, 0.05), 2)


###############
##  Exact optimum with reconstruction
###############
class SequenceOracle:
    """Bounded-knapsack optimum of one sequence, with value tables for any sub-state.

    Parameters
    ----------
    severities : array-like
        Raw initial severity of each city, in arrival order.

    Attributes
    ----------
    cost : ndarray, shape ``(L, MAX_ALLOC + 1)``
        Final severity of city ``c`` under a constant allocation ``a``.
    suffix : ndarray, shape ``(L + 1, BUDGET + 1)``
        ``suffix[c, r]``: minimum total final severity of cities ``c..L-1``
        spending at most ``r``.
    prefix : ndarray, shape ``(L + 1, BUDGET + 1)``
        ``prefix[c, b]``: minimum total final severity of cities ``0..c-1``
        spending exactly ``b`` (``inf`` if impossible).
    """

    def __init__(self, severities) -> None:
        self.severities = [float(s) for s in severities]
        length = len(self.severities)
        self.cost = numpy.array([[exp_utils._evolve_single_city(s, a, length - c) for a in range(MAX_ALLOC + 1)]
                                 for c, s in enumerate(self.severities)])
        self.suffix = numpy.zeros((length + 1, BUDGET + 1))
        for c in reversed(range(length)):
            for r in range(BUDGET + 1):
                self.suffix[c, r] = min(self.cost[c, a] + self.suffix[c + 1, r - a]
                                        for a in range(min(MAX_ALLOC, r) + 1))
        self.prefix = numpy.full((length + 1, BUDGET + 1), numpy.inf)
        self.prefix[0, 0] = 0.0
        for c in range(length):
            for b in numpy.flatnonzero(numpy.isfinite(self.prefix[c])):
                for a in range(min(MAX_ALLOC, BUDGET - b) + 1):
                    self.prefix[c + 1, b + a] = min(self.prefix[c + 1, b + a], self.prefix[c, b] + self.cost[c, a])
        self.optimum = float(self.suffix[0, BUDGET])
        self.worst = float(numpy.sum(self.cost[:, 0]))

    def action_values(self, city: int, resources: int) -> numpy.ndarray:
        """Optimal total severity of cities ``city..`` after choosing each feasible action."""
        return numpy.array([self.cost[city, a] + self.suffix[city + 1, resources - a]
                            for a in range(min(MAX_ALLOC, resources) + 1)])

    def canonical(self) -> List[int]:
        """Optimal allocation using the least budget, then the smallest early actions."""
        resources = int(numpy.flatnonzero(self.suffix[0] <= self.optimum + TIE_TOL)[0])
        allocation = []
        for city in range(len(self.severities)):
            values = self.action_values(city, resources)
            action = int(numpy.flatnonzero(values <= self.suffix[city, resources] + TIE_TOL)[0])
            allocation.append(action)
            resources -= action
        return allocation

    def compatible(self) -> List[List[int]]:
        """Actions of each city that belong to at least one optimal allocation."""
        sets = []
        for city in range(len(self.severities)):
            actions = set()
            for b in numpy.flatnonzero(numpy.isfinite(self.prefix[city])):
                for a in range(min(MAX_ALLOC, BUDGET - b) + 1):
                    total = self.prefix[city, b] + self.cost[city, a] + self.suffix[city + 1, BUDGET - b - a]
                    if total <= self.optimum + TIE_TOL:
                        actions.add(a)
            sets.append(sorted(actions))
        return sets


def performance(allocation: List[int], severities: List[float]) -> float:
    """Normalised performance ``r̄`` of an allocation with the package helpers."""
    finals = exp_utils.get_array_of_sequence_severities_from_allocations(allocation, severities)
    return float(exp_utils.calculate_normalised_final_severity_performance_metric(finals, severities)[0])


def load_sequences(scenario: str) -> List[List[float]]:
    """Per-sequence raw initial severities of a benchmark scenario."""
    folder = os.path.join(H1, 'general', 'work', 'pes_ens', 'scenarios', scenario)
    lengths = numpy.atleast_1d(numpy.loadtxt(os.path.join(folder, 'sequence_lengths.csv'), delimiter=','))
    severities = numpy.loadtxt(os.path.join(folder, 'initial_severity.csv'), delimiter=',')
    return [[float(s) for s in seq] for seq in convert_globalseq_to_seqs(lengths, severities)]


###############
##  Statistics
###############
def solve_scenario(scenario: str) -> dict:
    """Solve every sequence of a scenario and collect one record per city."""
    records, oracles = [], []
    ties_sequences = 0
    for seq_id, severities in enumerate(load_sequences(scenario)):
        oracle = SequenceOracle(severities)
        reference = exp_utils._best_feasible_sequence_severity(severities)
        assert abs(oracle.optimum - reference) <= TIE_TOL, (scenario, seq_id, oracle.optimum, reference)
        allocation = oracle.canonical()
        assert abs(performance(allocation, severities) - 1.0) <= 1e-9, (scenario, seq_id)
        sets = oracle.compatible()
        ties_sequences += int(any(len(s) > 1 for s in sets))
        remaining = BUDGET
        for city, (severity, action, options) in enumerate(zip(severities, allocation, sets)):
            assert action in options
            records.append({'scenario': scenario, 'sequence': seq_id, 'city': city, 'S': severity,
                            'S_clip': min(severity, MAX_SEVERITY), 'evolutions': len(severities) - city,
                            'a': action, 'remaining': remaining, 'set': options})
            remaining -= action
        oracles.append(oracle)
    return {'records': records, 'oracles': oracles, 'ties_sequences': ties_sequences}


def per_severity(records: List[dict]) -> List[dict]:
    """Table of the canonical and tie-aware optimum by initial severity ``S = 0..9``."""
    rows = []
    for severity in range(MAX_SEVERITY + 1):
        subset = [r for r in records if int(r['S_clip']) == severity]
        row: dict = {'S': severity, 'count': len(subset)}
        if subset:
            actions = numpy.array([r['a'] for r in subset], dtype=float)
            lows = numpy.array([r['set'][0] for r in subset], dtype=float)
            highs = numpy.array([r['set'][-1] for r in subset], dtype=float)
            with_resources = [r for r in subset if r['remaining'] > 0]
            row.update({
                'mean': float(actions.mean()), 'sd': float(actions.std(ddof=1)) if len(subset) > 1 else 0.0,
                'median': float(numpy.median(actions)), 'mean_minus_S': float(actions.mean() - severity),
                'share_zero': float(numpy.mean(actions == 0)),
                'share_zero_with_resources': float(numpy.mean([r['a'] == 0 for r in with_resources]))
                if with_resources else None,
                'share_unique': float(numpy.mean([len(r['set']) == 1 for r in subset])),
                'mean_set_size': float(numpy.mean([len(r['set']) for r in subset])),
                'mean_min_optimal': float(lows.mean()), 'mean_max_optimal': float(highs.mean()),
                'share_zero_possible': float(numpy.mean([0 in r['set'] for r in subset])),
                'share_zero_forced': float(numpy.mean([r['set'] == [0] for r in subset])),
            })
            if severity >= FLOOR_MIN_SEVERITY:
                floor = severity // 2
                row.update({'floor': floor, 'share_ge_floor': float(numpy.mean(actions >= floor)),
                            'share_ge_floor_possible': float(numpy.mean(highs >= floor)),
                            'share_ge_floor_forced': float(numpy.mean(lows >= floor))})
            # Same statistics restricted to cities that still have budget (the
            # floor rule additionally requires the floor itself to be affordable).
            if with_resources:
                live = numpy.array([r['a'] for r in with_resources], dtype=float)
                row.update({'count_with_resources': len(with_resources),
                            'mean_with_resources': float(live.mean()),
                            'median_with_resources': float(numpy.median(live)),
                            'mean_minus_S_with_resources': float(live.mean() - severity)})
                affordable = [r for r in with_resources if r['remaining'] >= severity // 2]
                if severity >= FLOOR_MIN_SEVERITY and affordable:
                    row.update({'count_floor_affordable': len(affordable),
                                'share_ge_floor_affordable': float(numpy.mean(
                                    [r['a'] >= severity // 2 for r in affordable])),
                                'share_ge_floor_affordable_possible': float(numpy.mean(
                                    [r['set'][-1] >= severity // 2 for r in affordable]))})
        rows.append(row)
    return rows


def zero_summary(records: List[dict]) -> dict:
    """How often the optimum allocates nothing although budget remains."""
    live = [r for r in records if r['remaining'] > 0]
    return {'decisions_with_resources': len(live),
            'canonical_zero': int(sum(r['a'] == 0 for r in live)),
            'zero_possible': int(sum(0 in r['set'] for r in live)),
            'zero_forced': int(sum(r['set'] == [0] for r in live)),
            'share_canonical_zero': float(numpy.mean([r['a'] == 0 for r in live])),
            'share_zero_forced': float(numpy.mean([r['set'] == [0] for r in live]))}


def by_evolutions(records: List[dict]) -> List[dict]:
    """Mean canonical ``a*`` by number of remaining evolutions ``L - c``."""
    rows = []
    for evolutions in sorted({r['evolutions'] for r in records}):
        subset = [r for r in records if r['evolutions'] == evolutions]
        rows.append({'evolutions': evolutions, 'count': len(subset),
                     'mean_a': float(numpy.mean([r['a'] for r in subset])),
                     'mean_a_minus_S': float(numpy.mean([r['a'] - r['S'] for r in subset])),
                     'share_zero': float(numpy.mean([r['a'] == 0 for r in subset]))})
    return rows


def regression(records: List[dict]) -> dict:
    """Least squares ``a* ~ 1 + S + (L - c)`` and ``a* ~ 1 + S``, with their R²."""
    target = numpy.array([r['a'] for r in records], dtype=float)
    out = {}
    for name, columns in (('S', ['S']), ('S+evolutions', ['S', 'evolutions'])):
        design = numpy.column_stack([numpy.ones(len(records))] + [[r[c] for r in records] for c in columns])
        coef = numpy.linalg.lstsq(design, target, rcond=None)[0]
        residual = target - design @ coef
        out[name] = {'coef': dict(zip(['intercept'] + columns, map(float, coef))),
                     'r2': float(1.0 - residual.var() / target.var())}
    return out


def _likelihood_inputs(records: List[dict], tie_aware: bool) -> tuple:
    """Stack the centres, feasible masks and chosen-action masks of every city."""
    actions = numpy.arange(MAX_ALLOC + 1, dtype=float)
    feasible = numpy.array([actions <= min(MAX_ALLOC, r['remaining']) for r in records])
    chosen = numpy.zeros_like(feasible)
    for i, r in enumerate(records):
        chosen[i, r['set'] if tie_aware else [r['a']]] = True
    return numpy.array([r['S_clip'] for r in records], dtype=float), feasible, chosen & feasible


def _log_likelihood(inputs: tuple, sigma: float, offset: float) -> float:
    centres, feasible, chosen = inputs
    actions = numpy.arange(MAX_ALLOC + 1, dtype=float)
    logits = numpy.where(feasible, -((actions[None, :] - (centres[:, None] + offset)) ** 2) / (2.0 * sigma ** 2),
                         -numpy.inf)
    weights = numpy.exp(logits - numpy.max(logits, axis=1)[:, None])
    mass = numpy.maximum((weights * chosen).sum(axis=1), 1e-300)
    return float(numpy.sum(numpy.log(mass) - numpy.log(weights.sum(axis=1))))


def fit_prior(records: List[dict], tie_aware: bool) -> dict:
    """Maximum-likelihood ``sigma`` of a discretised Gaussian over the feasible actions.

    Parameters
    ----------
    records : list of dict
        City records with the canonical action, its tie set and the budget left.
    tie_aware : bool
        If True the likelihood of a city is the prior mass of its whole set of
        optimal actions; otherwise that of the canonical action.

    Returns
    -------
    dict
        Best ``sigma`` with the centre fixed at ``S``, best ``(sigma, offset)``
        and the mean log-likelihood at the fixed ``sigma = 3``.
    """
    inputs = _likelihood_inputs(records, tie_aware)
    fixed = [_log_likelihood(inputs, s, 0.0) for s in SIGMA_GRID]
    best_fixed = int(numpy.argmax(fixed))
    joint = numpy.array([[_log_likelihood(inputs, s, o) for s in SIGMA_GRID[::5]] for o in OFFSET_GRID])
    o_idx, s_idx = divmod(int(numpy.argmax(joint)), joint.shape[1])
    offset = float(OFFSET_GRID[o_idx])
    refine = [s for s in SIGMA_GRID if abs(s - SIGMA_GRID[::5][s_idx]) <= 0.05]
    refined = [_log_likelihood(inputs, s, offset) for s in refine]
    n = len(records)
    return {'n': n,
            'sigma_centre_S': float(SIGMA_GRID[best_fixed]), 'mean_loglik_centre_S': fixed[best_fixed] / n,
            'sigma_joint': float(refine[int(numpy.argmax(refined))]), 'offset_joint': offset,
            'mean_loglik_joint': max(refined) / n,
            'mean_loglik_sigma_fixed': _log_likelihood(inputs, float(ENS_SEVERITY_PRIOR_SIGMA), 0.0) / n}


def prior_policy_perfs(scenario: str, offset: int = 0) -> numpy.ndarray:
    """Per-sequence ``r̄`` of the prior's mode as a policy: ``a = min(clip(S, 9) + offset, R, 10)``."""
    perfs = []
    for severities in load_sequences(scenario):
        remaining, allocation = BUDGET, []
        for severity in severities:
            action = int(max(0, min(min(severity, MAX_SEVERITY) + offset, remaining, MAX_ALLOC)))
            allocation.append(action)
            remaining -= action
        perfs.append(performance(allocation, severities))
    return numpy.asarray(perfs, dtype=float)


def prior_policy(scenario: str, offset: int = 0) -> float:
    """Mean ``r̄`` of the prior's mode as a policy (see ``prior_policy_perfs``)."""
    return float(numpy.mean(prior_policy_perfs(scenario, offset)))


###############
##  Rule audit on the states visited by pes_ens
###############
def _audit_hook(bank, variants: dict, oracles: List[SequenceOracle], rows: List[dict]):
    """Build the ``on_decision`` hook that records the clairvoyant loss of each decision."""
    import weighted_ens_sensitivity as sens

    current = [-1]

    def hook(_env, state, outputs, state_norm, trace):
        if int(state[1]) == 0:
            current[0] += 1
        oracle = oracles[current[0]]
        city, resources = int(state[1]), int(state[0])
        values = oracle.action_values(city, resources)
        scale = oracle.worst - oracle.optimum

        def loss(action: int) -> float:
            return float(values[min(action, len(values) - 1)] - values.min()) / scale

        row = {'sequence': current[0], 'S': int(state[2]), 'resources': resources, 'a': trace['a_final'],
               'loss': loss(trace['a_final']), 'optimal': bool(loss(trace['a_final']) <= TIE_TOL),
               'a_trf': trace['a_trf'], 'loss_trf': loss(trace['a_trf'])}
        for name, params in variants.items():
            alternative, _ = sens.weighted_decision(bank.members, outputs, state_norm, resources, params)
            row[f'a_without_{name}'] = alternative
            row[f'loss_without_{name}'] = loss(alternative)
        rows.append(row)
    return hook


def audit_rules(solved: Dict[str, dict]) -> dict:
    """Clairvoyant loss of each pes_ens decision and of the same decision without each rule.

    Parameters
    ----------
    solved : dict
        ``solve_scenario`` output per scenario (its oracles are reused).

    Returns
    -------
    dict
        Per group: loss decomposition and, per rule, the number of decisions
        it changes and the mean loss difference (negative = the rule helps).
    """
    import weighted_ens_sensitivity as sens

    bank = sens.MemberBank()
    fixed = sens.WeightedParams()
    variants = {'eta_0.3': dataclasses.replace(fixed, zero_factor=1.0),
                'prior': dataclasses.replace(fixed, prior_weight=0.0),
                'floor': dataclasses.replace(fixed, safety_floor=False)}
    rows: Dict[str, List[dict]] = {s: [] for s in SCENARIOS}
    identity_gap = 0.0
    for scenario in SCENARIOS:
        hook = _audit_hook(bank, variants, solved[scenario]['oracles'], rows[scenario])
        counts = {'n': 0, 'prior': 0, 'floor': 0}
        perfs = sens.run_scenario(scenario, sens.make_decider(bank, fixed, counts, on_decision=hook))
        for seq_id, perf in enumerate(perfs):
            lost = sum(r['loss'] for r in rows[scenario] if r['sequence'] == seq_id)
            identity_gap = max(identity_gap, abs((1.0 - perf) - lost))
        print(f'  auditoría {scenario:11s} r̄={numpy.mean(perfs):.4f} decisiones={len(rows[scenario])}', flush=True)

    out: dict = {'identity_max_gap': identity_gap}
    for group, members in GROUPS.items():
        data = [r for s in members for r in rows[s]]
        live = [r for r in data if r['resources'] > 0]
        n_seq = sum(len(solved[s]['oracles']) for s in members)
        summary: dict = {'decisions': len(data), 'decisions_with_resources': len(live),
                         'share_optimal': float(numpy.mean([r['optimal'] for r in live])),
                         'mean_loss_per_sequence': sum(r['loss'] for r in data) / n_seq,
                         'share_trf_optimal': float(numpy.mean([r['loss_trf'] <= TIE_TOL for r in live])),
                         'mean_a_minus_S': {
                             'final': float(numpy.mean([r['a'] - r['S'] for r in live])),
                             'without_prior': float(numpy.mean([r['a_without_prior'] - r['S'] for r in live])),
                             'trf': float(numpy.mean([r['a_trf'] - r['S'] for r in live]))},
                         'rules': {}}
        for name in variants:
            changed = [r for r in live if r[f'a_without_{name}'] != r['a']]
            delta = numpy.array([r['loss'] - r[f'loss_without_{name}'] for r in changed])
            summary['rules'][name] = {
                'changed': len(changed),
                'share_changed': len(changed) / max(len(live), 1),
                'helps': int(numpy.sum(delta < -TIE_TOL)), 'hurts': int(numpy.sum(delta > TIE_TOL)),
                'neutral': int(numpy.sum(numpy.abs(delta) <= TIE_TOL)),
                'loss_change_per_sequence': float(delta.sum()) / n_seq,
                'share_rule_action_optimal': float(numpy.mean([r['loss'] <= TIE_TOL for r in changed]))
                if changed else None,
                'share_alternative_optimal': float(numpy.mean([r[f'loss_without_{name}'] <= TIE_TOL
                                                               for r in changed])) if changed else None,
            }
        out[group] = summary
    return out


###############
##  Main
###############
def _fmt(value, spec: str = '.2f') -> str:
    return '   -' if value is None else format(value, spec)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=(__doc__ or '').splitlines()[0])
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='results JSON')
    parser.add_argument('--no-audit', action='store_true', help='skip the replay of pes_ens (no TensorFlow)')
    args = parser.parse_args()

    solved = {s: solve_scenario(s) for s in SCENARIOS}
    records = {g: [r for s in members for r in solved[s]['records']] for g, members in GROUPS.items()}
    result: dict = {
        'meta': {'budget': BUDGET, 'max_alloc': MAX_ALLOC, 'tie_tolerance': TIE_TOL,
                 'canonical': 'least total budget, then smallest action for the earliest cities',
                 'prior_sigma_fixed': float(ENS_SEVERITY_PRIOR_SIGMA)},
        'scenarios': {s: {'sequences': len(solved[s]['oracles']), 'cities': len(solved[s]['records']),
                          'severity_min': min(r['S'] for r in solved[s]['records']),
                          'severity_max': max(r['S'] for r in solved[s]['records']),
                          'sequences_with_ties': solved[s]['ties_sequences'],
                          'cities_with_ties': sum(len(r['set']) > 1 for r in solved[s]['records']),
                          'budget_left_canonical': float(numpy.mean(
                              [r['remaining'] - r['a'] for r in solved[s]['records']
                               if r['evolutions'] == 1])),
                          'prior_policy_mean': {f'S{o:+d}': prior_policy(s, o) for o in PRIOR_POLICY_OFFSETS}}
                      for s in SCENARIOS},
        'per_severity': {g: per_severity(rec) for g, rec in records.items()},
        'zero_with_resources': {g: zero_summary(rec) for g, rec in records.items()},
        'by_evolutions': by_evolutions(records['all']),
        'regression': regression(records['all']),
        'regression_with_resources': regression([r for r in records['all'] if r['remaining'] > 0]),
        'prior_fit': {mode: fit_prior(records['all'], mode == 'tie_aware') for mode in ('canonical', 'tie_aware')},
    }
    result['prior_policy'] = {
        f'S{o:+d}': {'sev_base': result['scenarios']['sev_base']['prior_policy_mean'][f'S{o:+d}'],
                     'heldout_mean': float(numpy.mean([result['scenarios'][s]['prior_policy_mean'][f'S{o:+d}']
                                                       for s in GROUPS['heldout']]))}
        for o in PRIOR_POLICY_OFFSETS}

    print(f'Óptimo exacto verificado en {sum(len(v["oracles"]) for v in solved.values())} secuencias '
          f'(presupuesto {BUDGET}, tope {MAX_ALLOC}).')
    for s, info in result['scenarios'].items():
        print(f'  {s:11s} ciudades={info["cities"]:3d} S∈[{info["severity_min"]:.0f},{info["severity_max"]:.0f}] '
              f'secuencias con empates={info["sequences_with_ties"]:2d} ciudades con empates={info["cities_with_ties"]:3d} '
              'moda del prior como política r̄: ' + ' '.join(f'{k}={v:.4f}' for k, v in info['prior_policy_mean'].items()))
    print('\nTodas las secuencias:  S   n   media  DE   mediana a*-S  %a*=0  %único [min,max]opt  %a*>=S//2 (posible/forzado)')
    for row in result['per_severity']['all']:
        if not row['count']:
            continue
        floor = (f'{100 * row["share_ge_floor"]:5.1f} ({100 * row["share_ge_floor_possible"]:.1f}/'
                 f'{100 * row["share_ge_floor_forced"]:.1f})') if 'floor' in row else ''
        print(f'                       {row["S"]}  {row["count"]:3d}  {row["mean"]:5.2f} {row["sd"]:4.2f}  '
              f'{row["median"]:4.1f}  {row["mean_minus_S"]:+5.2f}  {100 * row["share_zero"]:5.1f}  '
              f'{100 * row["share_unique"]:5.1f}  [{row["mean_min_optimal"]:.2f},{row["mean_max_optimal"]:.2f}]  {floor}')
    print('\nCon recursos restantes: S   n   media  mediana a*-S  %a*>=S//2 si la cota es pagable (posible)')
    for row in result['per_severity']['all']:
        if row.get('count_with_resources'):
            floor = (f'{100 * row["share_ge_floor_affordable"]:5.1f} ({100 * row["share_ge_floor_affordable_possible"]:.1f})'
                     f' n={row["count_floor_affordable"]}') if 'share_ge_floor_affordable' in row else ''
            print(f'                        {row["S"]}  {row["count_with_resources"]:3d}  '
                  f'{row["mean_with_resources"]:5.2f}  {row["median_with_resources"]:4.1f}  '
                  f'{row["mean_minus_S_with_resources"]:+5.2f}  {floor}')
    zero = result['zero_with_resources']['all']
    print(f'\na*=0 con recursos: canónico {zero["canonical_zero"]}/{zero["decisions_with_resources"]} '
          f'({100 * zero["share_canonical_zero"]:.1f} %), posible {zero["zero_possible"]}, forzado {zero["zero_forced"]}')
    for mode, fit in result['prior_fit'].items():
        print(f'Ajuste del prior ({mode}): sigma(centro S)={fit["sigma_centre_S"]:.2f} '
              f'[logL/n={fit["mean_loglik_centre_S"]:.3f}; con sigma=3: {fit["mean_loglik_sigma_fixed"]:.3f}]  '
              f'conjunto: sigma={fit["sigma_joint"]:.2f}, centro S{fit["offset_joint"]:+.2f} '
              f'[logL/n={fit["mean_loglik_joint"]:.3f}]')
    for label in ('regression', 'regression_with_resources'):
        for name, reg in result[label].items():
            coef = ', '.join(f'{k}={v:+.3f}' for k, v in reg['coef'].items())
            print(f'{label}: a* ~ {name}: {coef}  R²={reg["r2"]:.3f}')

    if not args.no_audit:
        print('\nAuditoría de reglas sobre los estados de pes_ens:')
        result['audit'] = audit_rules(solved)
        print(f'  identidad Σ pérdidas = 1 - r̄: máx |Δ| = {result["audit"]["identity_max_gap"]:.1e}')
        for group in GROUPS:
            summary = result['audit'][group]
            print(f'  {group:8s} decisiones óptimas={100 * summary["share_optimal"]:.1f} % '
                  f'(TRF solo en los mismos estados {100 * summary["share_trf_optimal"]:.1f} %), '
                  f'pérdida media por secuencia={summary["mean_loss_per_sequence"]:.4f}')
            offset = summary['mean_a_minus_S']
            print(f'    a-S medio con recursos: final={offset["final"]:+.2f} '
                  f'sin prior={offset["without_prior"]:+.2f} TRF={offset["trf"]:+.2f}')
            for name, rule in summary['rules'].items():
                print(f'    {name:8s} cambia {rule["changed"]:4d} ({100 * rule["share_changed"]:4.1f} %) '
                      f'ayuda={rule["helps"]:3d} perjudica={rule["hurts"]:3d} neutra={rule["neutral"]:3d} '
                      f'Δpérdida/secuencia={rule["loss_change_per_sequence"]:+.4f} '
                      f'óptima con regla={_fmt(rule["share_rule_action_optimal"])} '
                      f'sin regla={_fmt(rule["share_alternative_optimal"])}')

    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=1)
    print(f'\nEscrito {os.path.relpath(args.output, ROOT)}')


if __name__ == '__main__':
    main()
