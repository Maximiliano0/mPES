"""Sensibilidad uno-a-la-vez de los parámetros fijos del ensamble ponderado.

``pes_ens`` no tiene optimización: sus parámetros son constantes de
``h1/ens/pes_ens/config/CONFIG.py`` (temperatura, peso y ancho del *prior*,
pesos base) y literales de ``EnsembleAgent.predict`` en
``h1/ens/pes_ens/ext/ensemble_model.py`` (atenuación ``0.3`` de la acción 0,
desplazamiento ``0.1`` de la confianza y cota de seguridad ``S >= 6 ->
floor(S/2)``). Este script reproduce ``predict`` con esos valores como
argumentos y mueve uno por vez en una grilla, evaluando en ``sev_base`` y en
las réplicas ``heldout_s1`` … ``heldout_s5`` con los CSV del benchmark
(``h1/general/work/pes_ens/scenarios/<escenario>/``):

* Antes de barrer comprueba, en los seis escenarios, que la réplica con los
  valores fijos toma exactamente las mismas acciones que
  ``EnsembleAgent.predict`` y que sus medias coinciden con
  ``h1/general/results/ensemble/matrices/global_mean.csv`` (tolerancia
  ``1e-4``); si no, aborta.
* Nada se modifica en disco: los valores alternativos sólo existen en
  memoria. Los valores Q de cada miembro se guardan en caché por ventana de
  entrada, de modo que el costo lo fijan los estados nuevos que visita cada
  configuración.
* Por configuración guarda la media por escenario, la media de las cinco
  réplicas, la diferencia con ``pes_trf`` en los mismos escenarios
  (``h1/general/results/individual/cells/pes_trf__<escenario>.json``) y el
  error estándar pareado de esa diferencia en las 320 secuencias fuera de
  muestra. El JSON se escribe tras cada configuración y se reanuda
  saltando las ya presentes.

Uso (desde la raíz del repositorio, con ``win_mpes_env`` activado)::

    python writings/auxiliar/scripts/weighted_ens_sensitivity.py --estimate
    python writings/auxiliar/scripts/weighted_ens_sensitivity.py
"""

##########################
##  Imports externos    ##
##########################
import argparse
import contextlib
import csv
import dataclasses
import io
import json
import os
import sys
import time
from collections import deque
from typing import Callable, Dict, List, Optional, Tuple

import numpy

##########################
##  Configuración       ##
##########################
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
H1 = os.path.join(ROOT, 'h1')
RESULTS = os.path.join(H1, 'general', 'results')
DEFAULT_OUTPUT = os.path.join(RESULTS, 'ensemble', 'weighted_ens_sensitivity.json')
os.environ.setdefault('VIRTUAL_ENV', sys.prefix)
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
sys.path.insert(0, H1)

##########################
##  Imports internos    ##
##########################
# pylint: disable=wrong-import-position,protected-access
with contextlib.redirect_stdout(io.StringIO()):
    from ensemble_decisions import MAX_RESOURCES, MAX_SEVERITY, MAX_TRIALS
    from ens.pes_ens.config.CONFIG import (ENS_MEMBER_MODELS, ENS_SEVERITY_PRIOR_SIGMA,
                                          ENS_SEVERITY_PRIOR_WEIGHT, ENS_SOFTMAX_TEMPERATURE)
    from ens.pes_ens.ext.ensemble_model import (ACTION_DIM, EnsembleAgent, _softmax, make_history_window,
                                                normalize_state)
    from ens.pes_ens_trf_guard.src.pandemic_env import Pandemic, run_experiment
    from ens.pes_ens_trf_guard.src.tools_env import convert_globalseq_to_seqs
    from general.scripts.benchmark import REFERENCE_SCENARIO

HELDOUT = ['heldout_s1', 'heldout_s2', 'heldout_s3', 'heldout_s4', 'heldout_s5']
SCENARIOS = [REFERENCE_SCENARIO] + HELDOUT
TOLERANCE = 1e-4
# Literales de ``EnsembleAgent.predict`` (no son constantes del módulo).
FIXED_ZERO_FACTOR = 0.3
FIXED_CONFIDENCE_OFFSET = 0.1
FLOOR_MIN_SEVERITY = 6.0
BASE_WEIGHTS = {m['name']: float(m['weight']) for m in ENS_MEMBER_MODELS if m.get('enabled', True)}

# Grilla uno-a-la-vez: (campo de ``WeightedParams``, etiqueta, valores).
GRID = (('temperature', 'tau', (1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0)),
        ('prior_weight', 'w_pi', (0.0, 0.05, 0.10, 0.17, 0.25, 0.35, 0.50)),
        ('prior_sigma', 'sigma', (1.0, 2.0, 3.0, 4.0, 5.0)),
        ('zero_factor', 'eta', (0.1, 0.3, 0.5, 1.0)),
        ('trf_weight', 'w_trf', (1.0, 2.5, 5.0, 10.0)),
        ('confidence_offset', 'offset', (0.0, 0.1, 0.5, 1.0)),
        ('safety_floor', 'floor', (True, False)))


@dataclasses.dataclass(frozen=True)
class WeightedParams:
    """Tunable knobs of ``EnsembleAgent.predict``; defaults are the fixed values."""

    temperature: float = float(ENS_SOFTMAX_TEMPERATURE)
    prior_weight: float = float(ENS_SEVERITY_PRIOR_WEIGHT)
    prior_sigma: float = float(ENS_SEVERITY_PRIOR_SIGMA)
    zero_factor: float = FIXED_ZERO_FACTOR
    trf_weight: float = BASE_WEIGHTS['trf']
    confidence_offset: float = FIXED_CONFIDENCE_OFFSET
    safety_floor: bool = True

    def key(self) -> str:
        """Stable identifier used as the JSON key of a configuration."""
        return (f'tau={self.temperature:g}|w_pi={self.prior_weight:g}|sigma={self.prior_sigma:g}|'
                f'eta={self.zero_factor:g}|w_trf={self.trf_weight:g}|offset={self.confidence_offset:g}|'
                f'floor={"on" if self.safety_floor else "off"}')


###############
##  Member inference with cache
###############
class MemberBank:
    """Enabled pes_ens members with a cache of raw model outputs.

    The Q-values of a member depend only on its input window, never on the
    ensemble parameters, so every configuration can share them.
    """

    def __init__(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.agent = EnsembleAgent(ENS_MEMBER_MODELS, ENS_SOFTMAX_TEMPERATURE,
                                       ENS_SEVERITY_PRIOR_WEIGHT, ENS_SEVERITY_PRIOR_SIGMA)
        self.members = self.agent.members
        self._cache: Dict[Tuple[str, bytes], numpy.ndarray] = {}
        self.calls = 0

    def q_values(self, history: List[numpy.ndarray]) -> Dict[str, numpy.ndarray]:
        """Return each member's raw output for the current episode history.

        Parameters
        ----------
        history : list of ndarray
            Normalised states of the current sequence, current state last.

        Returns
        -------
        dict
            ``{member_name: raw output (11,)}``.
        """
        outputs = {}
        for member in self.members:
            if member['role'] == 'q_recurrent':
                window = make_history_window(deque(history[-member['history_len']:]), member['history_len'], 3)
                tensor = window[numpy.newaxis]
            else:
                tensor = history[-1][numpy.newaxis, :]
            key = (member['name'], tensor.tobytes())
            if key not in self._cache:
                self._cache[key] = member['model'](tensor, training=False).numpy().flatten()
                self.calls += 1
            outputs[member['name']] = self._cache[key]
        return outputs


###############
##  Decision rule
###############
def member_distribution(role: str, raw: numpy.ndarray, temperature: float) -> numpy.ndarray:
    """Convert a raw member output to probabilities as ``_member_distribution`` does."""
    if role == 'actor':
        probs = numpy.clip(raw, 0.0, None)
        total = float(numpy.sum(probs))
        if total <= 0.0:
            return numpy.full(ACTION_DIM, 1.0 / ACTION_DIM, dtype=numpy.float32)
        return (probs / total).astype(numpy.float32)
    return _softmax(raw, temperature)


def weighted_decision(members: List[dict], outputs: Dict[str, numpy.ndarray], state_norm: numpy.ndarray,
                      resources_left: int, params: WeightedParams) -> Tuple[int, dict]:
    """Replicate ``EnsembleAgent.predict`` plus the mediator's ``argmax`` with parameters.

    Parameters
    ----------
    members : list of dict
        ``EnsembleAgent.members`` (name, role, base weight).
    outputs : dict
        Raw output per member name (see :meth:`MemberBank.q_values`).
    state_norm : ndarray
        Normalised state with severity clipped to ``MAX_SEVERITY``.
    resources_left : int
        Remaining budget.
    params : WeightedParams
        Parameter values.

    Returns
    -------
    action : int
        Final allocation.
    trace : dict
        ``a_vote`` (after the action-0 factor), ``a_prior`` (after the prior)
        and ``a_final`` (after the safety floor), plus the Transformer's
        masked ``argmax`` ``a_trf``.
    """
    weights = {m['name']: (params.trf_weight if m['name'] == 'trf' else m['weight']) for m in members}
    weight_sum = sum(weights.values())
    max_feasible = max(0, int(resources_left))
    ensemble = numpy.zeros(ACTION_DIM, dtype=numpy.float64)
    log2_n = numpy.log2(ACTION_DIM)
    a_trf = 0
    for member in members:
        masked = member_distribution(member['role'], outputs[member['name']], params.temperature)
        masked = masked.astype(numpy.float64).copy()
        if max_feasible < ACTION_DIM - 1:
            masked[max_feasible + 1:] = 0.0
        total = float(numpy.sum(masked))
        if total <= 0.0:
            masked = numpy.zeros(ACTION_DIM, dtype=numpy.float64)
            masked[0] = 1.0
        else:
            masked = masked / total
        p = numpy.clip(masked, 1e-9, 1.0)
        confidence = 1.0 - float(-numpy.sum(p * numpy.log2(p)) / log2_n)
        ensemble += weights[member['name']] / weight_sum * (params.confidence_offset + confidence) * masked
        if member['name'] == 'trf':
            a_trf = int(numpy.argmax(masked))

    if max_feasible > 0:
        ensemble[0] *= params.zero_factor
    total = float(numpy.sum(ensemble))
    if total <= 0.0:
        ensemble = numpy.zeros(ACTION_DIM, dtype=numpy.float64)
        ensemble[0] = 1.0
    else:
        ensemble = ensemble / total
    a_vote = int(numpy.argmax(ensemble))

    severity_raw = float(state_norm[2]) * float(MAX_SEVERITY)
    prior_weight = float(numpy.clip(params.prior_weight, 0.0, 1.0))
    if prior_weight > 0.0:
        actions = numpy.arange(ACTION_DIM, dtype=numpy.float64)
        prior = numpy.exp(-((actions - severity_raw) ** 2) / (2.0 * max(params.prior_sigma, 1e-3) ** 2))
        if max_feasible < ACTION_DIM - 1:
            prior[max_feasible + 1:] = 0.0
        prior_sum = float(numpy.sum(prior))
        if prior_sum > 0.0:
            ensemble = (1.0 - prior_weight) * ensemble + prior_weight * prior / prior_sum
            ensemble = ensemble / float(numpy.sum(ensemble))
    a_prior = int(numpy.argmax(ensemble))

    if params.safety_floor and severity_raw >= FLOOR_MIN_SEVERITY:
        floor = int(severity_raw // 2)
        if a_prior < floor <= max_feasible:
            ensemble = numpy.zeros(ACTION_DIM, dtype=numpy.float64)
            ensemble[floor] = 1.0

    # ``predict`` returns float32; the mediator then marks infeasible actions
    # with 1e-5, takes the ``argmax`` and clips it to the budget.
    probs = ensemble.astype(numpy.float32).astype(numpy.float64)
    probs[numpy.arange(ACTION_DIM) > resources_left] = 0.00001
    action = int(numpy.clip(int(numpy.argmax(probs)), 0, int(resources_left)))
    return action, {'a_vote': a_vote, 'a_prior': a_prior, 'a_final': action, 'a_trf': a_trf}


def state_vector(state) -> numpy.ndarray:
    """Normalise a raw ``[R, t, S]`` state as ``provide_ens_agent_response`` does (severity clipped to 9)."""
    return normalize_state([max(0, min(int(state[0]), MAX_RESOURCES)), max(0, min(int(state[1]), MAX_TRIALS)),
                            max(0, min(int(state[2]), MAX_SEVERITY))], MAX_RESOURCES, MAX_TRIALS, MAX_SEVERITY)


def make_decider(bank: MemberBank, params: WeightedParams, counts: dict,
                 on_decision: Optional[Callable] = None, check_agent: bool = False):
    """Build a ``run_experiment`` action function for one configuration.

    Parameters
    ----------
    bank : MemberBank
        Members and output cache.
    params : WeightedParams
        Parameter values.
    counts : dict
        Updated in place with ``n`` (decisions with resources), ``prior``
        (prior changed the voted action) and ``floor`` (floor changed it).
    on_decision : callable, optional
        ``(env, state, outputs, state_norm, trace)`` hook called after each
        decision (used by the oracle audit).
    check_agent : bool, optional
        Also query the real ``EnsembleAgent.predict`` and assert that it picks
        the same action (only meaningful with the default parameters).
    """
    histories: Dict[int, List[numpy.ndarray]] = {}
    agent = bank.agent
    agent._history_caches = {name: {} for name in agent._history_caches}

    def decide(env, state, sequence_id):
        state_norm = state_vector(state)
        if int(state[1]) == 0:
            histories[sequence_id] = []
            agent.reset_episode(0, sequence_id)
        history = histories.setdefault(sequence_id, [])
        history.append(state_norm)
        outputs = bank.q_values(history)
        action, trace = weighted_decision(bank.members, outputs, state_norm, int(state[0]), params)
        if check_agent:
            probs, _ = agent.predict(state_norm, int(state[0]), 0, sequence_id)
            probs = probs.astype(numpy.float64)
            probs[numpy.arange(ACTION_DIM) > int(state[0])] = 0.00001
            expected = int(numpy.clip(int(numpy.argmax(probs)), 0, int(state[0])))
            if expected != action:
                raise AssertionError(f'replica {action} != EnsembleAgent {expected} at state {state}')
        if state[0] > 0:
            counts['n'] += 1
            counts['prior'] += int(trace['a_prior'] != trace['a_vote'])
            counts['floor'] += int(trace['a_final'] != trace['a_prior'])
        if on_decision is not None:
            on_decision(env, state, outputs, state_norm, trace)
        return action
    return decide


###############
##  Evaluation
###############
def load_scenario(scenario: str) -> Tuple[numpy.ndarray, list]:
    """Return the sequence lengths and per-sequence initial severities of a benchmark scenario."""
    folder = os.path.join(H1, 'general', 'work', 'pes_ens', 'scenarios', scenario)
    lengths = numpy.atleast_1d(numpy.loadtxt(os.path.join(folder, 'sequence_lengths.csv'), delimiter=','))
    severities = numpy.loadtxt(os.path.join(folder, 'initial_severity.csv'), delimiter=',')
    return lengths, convert_globalseq_to_seqs(lengths, severities)


def run_scenario(scenario: str, decide) -> List[float]:
    """Replay one scenario deterministically and return the per-sequence performances."""
    lengths, sequences = load_scenario(scenario)
    environment = Pandemic()
    environment.verbose = False
    with contextlib.redirect_stdout(io.StringIO()):
        _, performances, _ = run_experiment(environment, decide, False, lengths, sequences,
                                            NumberOfIterations=len(lengths))
    return [float(v) for v in performances]


def load_references() -> Tuple[dict, dict, dict]:
    """Return benchmark means of ``pes_ens``, and ``pes_trf`` means and per-sequence vectors."""
    with open(os.path.join(RESULTS, 'ensemble', 'matrices', 'global_mean.csv'), encoding='utf-8') as handle:
        bench = {row['model']: row for row in csv.DictReader(handle)}['pes_ens']
    trf_mean, trf_seq = {}, {}
    for scenario in SCENARIOS:
        with open(os.path.join(RESULTS, 'individual', 'cells', f'pes_trf__{scenario}.json'), encoding='utf-8') as fh:
            cell = json.load(fh)
        trf_seq[scenario] = numpy.asarray(cell['per_sequence_perf'], dtype=float)
        trf_mean[scenario] = float(cell['global_mean_perf'])
    return {s: float(bench[s]) for s in SCENARIOS}, trf_mean, trf_seq


def evaluate(bank: MemberBank, params: WeightedParams, trf_mean: dict, trf_seq: dict,
             check_agent: bool = False) -> dict:
    """Evaluate one configuration on the six scenarios.

    Returns
    -------
    dict
        Means, held-out aggregate, differences with ``pes_trf`` and rule counts.
    """
    start = time.perf_counter()
    means, per_sequence, counts = {}, {}, {'n': 0, 'prior': 0, 'floor': 0}
    for scenario in SCENARIOS:
        perfs = run_scenario(scenario, make_decider(bank, params, counts, check_agent=check_agent))
        per_sequence[scenario] = perfs
        means[scenario] = float(numpy.mean(perfs))
    heldout_diff = numpy.concatenate([numpy.asarray(per_sequence[s]) - trf_seq[s] for s in HELDOUT])
    return {
        'params': dataclasses.asdict(params),
        'means': means,
        'heldout_mean': float(numpy.mean([means[s] for s in HELDOUT])),
        'diff_vs_trf': {s: means[s] - trf_mean[s] for s in SCENARIOS},
        'heldout_diff_vs_trf': float(numpy.mean(heldout_diff)),
        'heldout_diff_se': float(numpy.std(heldout_diff, ddof=1) / numpy.sqrt(len(heldout_diff))),
        'counts': counts,
        'per_sequence': {s: [round(v, 6) for v in per_sequence[s]] for s in SCENARIOS},
        'elapsed_s': round(time.perf_counter() - start, 1),
    }


def verify_baseline(bank: MemberBank, bench: dict, trf_mean: dict, trf_seq: dict) -> dict:
    """Evaluate the fixed configuration, checking actions and benchmark means; abort on mismatch."""
    result = evaluate(bank, WeightedParams(), trf_mean, trf_seq, check_agent=True)
    for scenario in SCENARIOS:
        with open(os.path.join(RESULTS, 'ensemble', 'cells', f'pes_ens__{scenario}.json'), encoding='utf-8') as fh:
            cell_seq = numpy.asarray(json.load(fh)['per_sequence_perf'], dtype=float)
        gap = abs(result['means'][scenario] - bench[scenario])
        seq_gap = float(numpy.max(numpy.abs(numpy.asarray(result['per_sequence'][scenario]) - cell_seq)))
        print(f'  verificación {scenario:11s} réplica={result["means"][scenario]:.6f} '
              f'benchmark={bench[scenario]:.6f} |Δ|={gap:.1e} max|Δ| por secuencia={seq_gap:.1e}', flush=True)
        if gap > TOLERANCE or seq_gap > TOLERANCE:
            raise SystemExit(f'La réplica no reproduce el benchmark en {scenario}; se aborta.')
    return result


def grid_configs() -> List[Tuple[str, object, WeightedParams]]:
    """Expand :data:`GRID` into ``(label, value, params)`` triples (baseline duplicates included)."""
    return [(label, value, dataclasses.replace(WeightedParams(), **{field: value}))
            for field, label, values in GRID for value in values]


def save(path: str, data: dict) -> None:
    """Write the results JSON atomically."""
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as handle:
        json.dump(data, handle, indent=1)
    os.replace(tmp, path)


def print_curves(data: dict) -> None:
    """Print one table per parameter: reference, held-out mean and difference with the Transformer."""
    baseline = WeightedParams().key()
    for _field, label, _values in GRID:
        print(f'\n{label}:  valor   sev_base  heldout  Δ_TRF(ref)  Δ_TRF(heldout) ± EE')
        for row_label, value, params in grid_configs():
            if row_label != label or params.key() not in data['configs']:
                continue
            row = data['configs'][params.key()]
            mark = '*' if params.key() == baseline else ' '
            print(f'  {mark}{str(value):>6s}  {row["means"][REFERENCE_SCENARIO]:.4f}   {row["heldout_mean"]:.4f}  '
                  f'{row["diff_vs_trf"][REFERENCE_SCENARIO]:+.4f}      '
                  f'{row["heldout_diff_vs_trf"]:+.4f} ± {row["heldout_diff_se"]:.4f}')


###############
##  Main
###############
def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=(__doc__ or '').splitlines()[0])
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='results JSON (resumed if present)')
    parser.add_argument('--estimate', action='store_true',
                        help='verify the baseline, time one extra configuration and print the runtime estimate')
    args = parser.parse_args()

    bench, trf_mean, trf_seq = load_references()
    data: dict = {'configs': {}}
    if os.path.isfile(args.output):
        with open(args.output, encoding='utf-8') as handle:
            data = json.load(handle)
    data['meta'] = {
        'scenarios': SCENARIOS, 'benchmark_pes_ens': bench, 'pes_trf_mean': trf_mean,
        'pes_trf_heldout_mean': float(numpy.mean([trf_mean[s] for s in HELDOUT])),
        'fixed': dataclasses.asdict(WeightedParams()), 'base_weights': BASE_WEIGHTS,
        'grid': {label: list(values) for _f, label, values in GRID}, 'tolerance': TOLERANCE,
    }

    bank = MemberBank()
    start = time.perf_counter()
    print('Verificando la configuración fija contra EnsembleAgent.predict y el benchmark...', flush=True)
    baseline = verify_baseline(bank, bench, trf_mean, trf_seq)
    data['configs'][WeightedParams().key()] = baseline
    data['baseline_check'] = {'passed': True, 'elapsed_s': round(time.perf_counter() - start, 1)}
    save(args.output, data)
    print(f'  verificación superada en {time.perf_counter() - start:.0f} s ({bank.calls} llamadas a modelos)')

    pending = []
    for _label, _value, params in grid_configs():
        if params.key() not in data['configs'] and params.key() not in [p.key() for p in pending]:
            pending.append(params)
    if args.estimate:
        probe = WeightedParams(temperature=1.0)
        tic = time.perf_counter()
        data['configs'][probe.key()] = evaluate(bank, probe, trf_mean, trf_seq)
        elapsed = time.perf_counter() - tic
        save(args.output, data)
        print(f'Una configuración ({probe.key()}) en los 6 escenarios: {elapsed:.0f} s; '
              f'{len(pending)} configuraciones pendientes => estimado {len(pending) * elapsed / 60:.0f} min '
              f'(cota superior: la caché de Q compartida acelera las siguientes)')
        return

    for index, params in enumerate(pending, 1):
        data['configs'][params.key()] = evaluate(bank, params, trf_mean, trf_seq)
        save(args.output, data)
        row = data['configs'][params.key()]
        print(f'[{index}/{len(pending)}] {params.key()}  ref={row["means"][REFERENCE_SCENARIO]:.4f} '
              f'heldout={row["heldout_mean"]:.4f} Δ_TRF={row["heldout_diff_vs_trf"]:+.4f} '
              f'({row["elapsed_s"]:.0f} s)', flush=True)
    data['total_elapsed_s'] = round(time.perf_counter() - start, 1)
    save(args.output, data)
    print_curves(data)


if __name__ == '__main__':
    main()
