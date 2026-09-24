"""Frecuencia con que las reglas fijas de los ensambles cambian la decisión.

Reproduce, escenario por escenario, la evaluación de ``pes_ens``,
``pes_ens_consensus_prior`` y ``pes_ens_trf_guard`` sobre los CSV que el
benchmark materializa en ``h1/general/work/<pkg>/scenarios/<escenario>/`` y
cuenta, en las decisiones con recursos disponibles:

* ``pes_ens`` y ``pes_ens_consensus_prior``: cuántas veces el *prior* de
  severidad cambia la acción votada, cuántas veces la cota de seguridad cambia
  la acción resultante y cuántas veces la acción final difiere de la preferida
  por el Transformer.
* ``pes_ens_trf_guard``: cuántas veces la compuerta sigue al Transformer
  (``g >= 0.5``) y cuántas veces la acción final difiere de la suya.

Cada media se compara con ``h1/general/results/ensemble/matrices/global_mean.csv``
para comprobar que el replay coincide con el benchmark.

Uso (desde la raíz del repositorio, con ``win_mpes_env`` activado)::

    python writings/auxiliar/scripts/ensemble_decisions.py
    python writings/auxiliar/scripts/ensemble_decisions.py --scenarios sev_base
"""

##########################
##  Imports externos    ##
##########################
import argparse
import contextlib
import csv
import io
import json
import os
import sys

import numpy

##########################
##  Configuración       ##
##########################
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
H1 = os.path.join(ROOT, 'h1')
os.environ.setdefault('VIRTUAL_ENV', sys.prefix)
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
sys.path.insert(0, H1)

##########################
##  Imports internos    ##
##########################
# pylint: disable=wrong-import-position,protected-access
with contextlib.redirect_stdout(io.StringIO()):
    from ens.pes_ens.config.CONFIG import (ENS_MEMBER_MODELS, ENS_SEVERITY_PRIOR_SIGMA,
                                          ENS_SEVERITY_PRIOR_WEIGHT, ENS_SOFTMAX_TEMPERATURE)
    from ens.pes_ens.ext.ensemble_model import EnsembleAgent
    from ens.pes_ens_consensus_prior.ext import ensemble as consensus_prior
    from ens.pes_ens_trf_guard.ext.ensemble import TransformerGuardEnsemble
    from ens.pes_ens_trf_guard.src.pandemic_env import Pandemic, run_experiment
    from ens.pes_ens_trf_guard.src.tools_env import convert_globalseq_to_seqs

DEFAULT_WEIGHTS = {'dqn': 0.15, 'a2c': 0.10, 'rdqn': 0.25, 'trf': 0.50}
MAX_SEVERITY = 9
MAX_RESOURCES = 30
MAX_TRIALS = 10


###############
##  Helpers
###############
def load_params(pkg: str) -> dict:
    """Return the ``hyperparameters`` block of ``ens/<pkg>/inputs/best_params.json``.

    Parameters
    ----------
    pkg : str
        Ensemble package name.

    Returns
    -------
    dict
        Tuned parameters, exactly as ``evaluate_ens.py`` reads them.
    """
    with open(os.path.join(H1, 'ens', pkg, 'inputs', 'best_params.json'), encoding='utf-8') as handle:
        return json.load(handle)['hyperparameters']


def member_weights(params: dict) -> dict:
    """Build member weights with the same defaults as ``evaluate_ens.py``.

    Parameters
    ----------
    params : dict
        Tuned parameters.

    Returns
    -------
    dict
        Base weight per member name.
    """
    return {name: float(params.get(f'weight_{name}', default)) for name, default in DEFAULT_WEIGHTS.items()}


def normalise(state) -> numpy.ndarray:
    """Scale a raw ``[R, t, S]`` state as the optimisable ensembles do (no clipping).

    Parameters
    ----------
    state : sequence
        Raw environment state.

    Returns
    -------
    ndarray
        ``[R/30, t/10, S/9]`` as ``float32``.
    """
    return numpy.array([state[0] / MAX_RESOURCES, state[1] / MAX_TRIALS, state[2] / MAX_SEVERITY],
                       dtype=numpy.float32)


def run_scenario(pkg: str, scenario: str, decide) -> float:
    """Replay one benchmark cell and return its mean normalised performance.

    Parameters
    ----------
    pkg : str
        Package whose materialised scenario CSVs are used.
    scenario : str
        Scenario identifier.
    decide : callable
        ``(env, state, sequence_id) -> action``.

    Returns
    -------
    float
        Mean normalised performance over the scenario sequences.
    """
    folder = os.path.join(H1, 'general', 'work', pkg, 'scenarios', scenario)
    lengths = numpy.atleast_1d(numpy.loadtxt(os.path.join(folder, 'sequence_lengths.csv'), delimiter=','))
    severities = numpy.loadtxt(os.path.join(folder, 'initial_severity.csv'), delimiter=',')
    environment = Pandemic()
    environment.verbose = False
    with contextlib.redirect_stdout(io.StringIO()):
        _, performances, _ = run_experiment(environment, decide, False, lengths,
                                            convert_globalseq_to_seqs(lengths, severities),
                                            NumberOfIterations=len(lengths))
    return float(numpy.mean(performances))


def apply_prior_and_bound(scores: numpy.ndarray, severity: float, feasible: int,
                          weight: float, sigma: float) -> "tuple[int, int, int]":
    """Return the voted, prior-mixed and bounded actions of a prior-based ensemble.

    Parameters
    ----------
    scores : ndarray
        Normalised ensemble distribution over actions ``0..feasible``.
    severity : float
        Raw severity used by the prior and the safety bound.
    feasible : int
        Largest feasible action.
    weight, sigma : float
        Prior mixing weight and width.

    Returns
    -------
    tuple of int
        ``(a_vote, a_prior, a_final)``.
    """
    actions = numpy.arange(feasible + 1, dtype=numpy.float64)
    prior = numpy.exp(-((actions - severity) ** 2) / (2.0 * sigma ** 2))
    prior /= max(float(prior.sum()), 1e-12)
    a_vote = int(numpy.argmax(scores))
    a_prior = int(numpy.argmax((1.0 - weight) * scores + weight * prior))
    a_final = a_prior
    if severity >= 6.0:
        floor = int(severity // 2)
        if a_final < floor <= feasible:
            a_final = floor
    return a_vote, a_prior, a_final


###############
##  Instrumented decision rules
###############
def guard_rule(ensemble: TransformerGuardEnsemble, counts: dict):
    """Wrap ``TransformerGuardEnsemble.predict`` and count gate decisions."""
    current = [-1]
    ensemble._histories.clear()

    def decide(_env, state, sequence_id):
        if sequence_id != current[0]:
            if current[0] >= 0:
                ensemble.reset((0, current[0]))
            current[0] = sequence_id
        action, _confidence, diagnostics = ensemble.predict(normalise(state), int(state[0]), (0, sequence_id))
        if state[0] > 0:
            counts['n'] += 1
            counts['follow'] += int(diagnostics['gate'] >= 0.5)
            counts['trf_diff'] += int(action != diagnostics['trf']['action'])
        return action
    return decide


def consensus_prior_rule(ensemble, counts: dict):
    """Replicate ``ConfidenceConsensusEnsemble.predict`` with prior/bound counters."""
    current = [-1]
    ensemble._histories.clear()

    def decide(_env, state, sequence_id):
        if sequence_id != current[0]:
            if current[0] >= 0:
                ensemble.reset((0, current[0]))
            current[0] = sequence_id
        state_norm = normalise(state)
        feasible = min(max(int(state[0]), 0), 10)
        scores = numpy.zeros(feasible + 1)
        actions, confidences = {}, {}
        for name, model in ensemble._models.items():
            model_input = ensemble._model_input(name, state_norm, (0, sequence_id))
            values = numpy.asarray(model(model_input, training=False))[0].astype(numpy.float64)
            if consensus_prior.MODEL_ROLES[name] == 'actor':
                probabilities = numpy.clip(values, 0.0, None)
                probabilities /= max(float(probabilities.sum()), 1e-12)
            else:
                probabilities = consensus_prior._softmax(values)
            masked = probabilities[:feasible + 1]
            masked /= max(float(masked.sum()), 1e-12)
            confidences[name] = consensus_prior._confidence(masked)
            actions[name] = int(numpy.argmax(masked))
            weight = max(float(ensemble._weights.get(name, 1.0)), 0.0)
            scores += weight * confidences[name] ** ensemble._confidence_power * masked
        for index in range(feasible + 1):
            scores[index] += ensemble._agreement_bonus * sum(
                c for n, c in confidences.items() if actions[n] == index)
            scores[index] -= ensemble._disagreement_penalty * sum(
                c for n, c in confidences.items() if actions[n] != index)
        if feasible > 0:
            scores[0] *= 0.3
        scores = numpy.clip(scores, 0.0, None)
        scores /= max(float(scores.sum()), 1e-12)
        a_vote, a_prior, a_final = apply_prior_and_bound(
            scores, float(state_norm[2]) * MAX_SEVERITY, feasible,
            ensemble._prior_weight, ensemble._prior_sigma)
        if state[0] > 0:
            counts['n'] += 1
            counts['prior'] += int(a_prior != a_vote)
            counts['bound'] += int(a_final != a_prior)
            counts['trf_diff'] += int(a_final != actions['trf'])
        return a_final
    return decide


def weighted_rule(agent: EnsembleAgent, counts: dict):
    """Replicate ``EnsembleAgent.predict`` (``pes_ens``) with prior/bound counters."""
    agent._history_caches = {name: {} for name in agent._history_caches}

    def decide(_env, state, sequence_id):
        if int(state[1]) == 0:
            agent.reset_episode(0, sequence_id)
        resources = max(0, min(int(state[0]), MAX_RESOURCES))
        severity = max(0, min(int(state[2]), MAX_SEVERITY))  # pes_ens clips severity like its pygameMediator
        state_norm = numpy.array([resources / MAX_RESOURCES, min(int(state[1]), MAX_TRIALS) / MAX_TRIALS,
                                  severity / MAX_SEVERITY], dtype=numpy.float32)
        feasible = max(0, int(state[0]))
        ensemble = numpy.zeros(11)
        trf_action = 0
        for member in agent.members:
            raw = agent._member_distribution(member, state_norm, 0, sequence_id)
            masked = raw.astype(numpy.float64)
            if feasible < 10:
                masked[feasible + 1:] = 0.0
            masked = masked / masked.sum() if masked.sum() > 0 else numpy.eye(11)[0]
            clipped = numpy.clip(masked, 1e-9, 1.0)
            confidence = 1.0 - float(-numpy.sum(clipped * numpy.log2(clipped)) / numpy.log2(11))
            ensemble += member['weight_norm'] * (0.1 + confidence) * masked
            if member['name'] == 'trf':
                trf_action = int(numpy.argmax(masked))
        if feasible > 0:
            ensemble[0] *= 0.3
        ensemble /= ensemble.sum()
        upper = min(feasible, 10)
        a_vote, a_prior, a_final = apply_prior_and_bound(
            ensemble[:upper + 1] / ensemble[:upper + 1].sum(), float(severity), upper,
            agent._prior_weight, agent._prior_sigma)
        if state[0] > 0:
            counts['n'] += 1
            counts['prior'] += int(a_prior != a_vote)
            counts['bound'] += int(a_final != a_prior)
            counts['trf_diff'] += int(a_final != trf_action)
        return a_final
    return decide


###############
##  Main
###############
def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--scenarios', nargs='*', default=None, help='subset of scenario ids')
    parser.add_argument('--output', default=None, help='optional JSON file for the counts')
    args = parser.parse_args()

    with open(os.path.join(H1, 'general', 'results', 'ensemble', 'matrices', 'global_mean.csv'),
              encoding='utf-8') as handle:
        reference = {row['model']: row for row in csv.DictReader(handle)}
    scenarios = args.scenarios or [key for key in reference['pes_ens'] if key != 'model']

    guard_params = load_params('pes_ens_trf_guard')
    guard = TransformerGuardEnsemble(member_weights(guard_params), float(guard_params['confidence_power']),
                                     float(guard_params['trf_confidence_threshold']),
                                     float(guard_params['gate_slope']))
    prior_params = load_params('pes_ens_consensus_prior')
    prior_ensemble = consensus_prior.ConfidenceConsensusEnsemble(
        member_weights(prior_params), float(prior_params['confidence_power']),
        float(prior_params['agreement_bonus']), float(prior_params['disagreement_penalty']),
        float(prior_params['prior_weight']), float(prior_params['prior_sigma']))
    with contextlib.redirect_stdout(io.StringIO()):
        weighted = EnsembleAgent(ENS_MEMBER_MODELS, ENS_SOFTMAX_TEMPERATURE,
                                 ENS_SEVERITY_PRIOR_WEIGHT, ENS_SEVERITY_PRIOR_SIGMA)

    rules = (('pes_ens_trf_guard', lambda c: guard_rule(guard, c), ('n', 'follow', 'trf_diff')),
             ('pes_ens_consensus_prior', lambda c: consensus_prior_rule(prior_ensemble, c),
              ('n', 'prior', 'bound', 'trf_diff')),
             ('pes_ens', lambda c: weighted_rule(weighted, c), ('n', 'prior', 'bound', 'trf_diff')))
    results: dict = {pkg: {} for pkg, _, _ in rules}
    for scenario in scenarios:
        for pkg, factory, keys in rules:
            counts = dict.fromkeys(keys, 0)
            mean = run_scenario(pkg, scenario, factory(counts))
            results[pkg][scenario] = {'mean': mean, 'benchmark': float(reference[pkg][scenario]), **counts}
            print(f'{scenario:26s} {pkg:24s} mean={mean:.4f} bench={float(reference[pkg][scenario]):.4f} {counts}',
                  flush=True)

    for pkg, _, keys in rules:
        for label, subset in (('sev_base', ['sev_base']),
                              ('generalizacion', [s for s in scenarios if s != 'sev_base'])):
            totals = {k: sum(results[pkg][s][k] for s in subset if s in results[pkg]) for k in keys}
            if totals['n']:
                shares = {k: round(100.0 * v / totals['n'], 1) for k, v in totals.items() if k != 'n'}
                print(f'{pkg:24s} {label:15s} n={totals["n"]} {shares} (% de decisiones con recursos)')
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as handle:
            json.dump(results, handle, indent=2)


if __name__ == '__main__':
    main()
