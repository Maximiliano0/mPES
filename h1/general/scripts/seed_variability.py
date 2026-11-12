"""Partial estimate of the seed-to-seed variability of the individual models.

Retrains Q-Learning (``pes_ql``), DQN (``pes_dqn``) and the Transformer
(``pes_trf``) with the best configuration of each package and several new
training seeds, and evaluates every model greedily on the reference
(``sev_base``) and the five held-out replicas. The delivered model of each
package is evaluated with the same code, so the delivered seed is one more
sample of the distribution.

The packages' ``train_*`` scripts are **never** called: they overwrite the
delivered model (``inputs/*.keras``, ``inputs/q.npy``). This harness calls the
training functions directly and writes everything under
``general/work/seeds/<pkg>/seed_<s>/``. Every (model, seed) is skipped when its
``result.json`` already exists, so an interrupted run resumes where it stopped.
While it runs, the process asks Windows not to sleep (``SetThreadExecutionState``).

Usage (from ``h1/``, with ``win_mpes_env`` activated)::

    python -m general.scripts.seed_variability run
    python -m general.scripts.seed_variability run --models ql dqn --seeds 1001 1002
    python -m general.scripts.seed_variability summary
"""

##########################
##  Imports externos    ##
##########################
import argparse
import ctypes
import importlib
import json
import os
import sys
import time
from datetime import datetime
from typing import Callable, Dict, List, Tuple

import numpy

##########################
##  Configuración       ##
##########################
H1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
GENERAL = os.path.join(H1, 'general')
WORK = os.path.join(GENERAL, 'work', 'seeds')
SCENARIO_ROOT = os.path.join(GENERAL, 'work', 'pes_ens', 'scenarios')
SUMMARY_FILE = os.path.join(GENERAL, 'results', 'individual', 'seed_variability.json')
LOG_FILE = os.path.join(WORK, 'seed_variability.log')
SCENARIOS = ['sev_base'] + [f'heldout_s{i}' for i in range(1, 6)]
MODELS = {
    'ql':  {'pkg': 'tabular.pes_ql', 'delivered': 'q.npy', 'seeds': [1001, 1002, 1003, 1004, 1005]},
    'dqn': {'pkg': 'ml.pes_dqn', 'delivered': 'dqn_model.keras', 'seeds': [1001, 1002, 1003, 1004, 1005]},
    'trf': {'pkg': 'ml.pes_trf', 'delivered': 'trf_model.keras', 'seeds': [1001, 1002, 1003]},
}
ORDER = ('ql', 'dqn', 'trf')
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


###############
##  Helpers
###############
def log(message: str) -> None:
    """Print ``message`` with a timestamp and append it to the run log."""
    line = f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}'
    print(line, flush=True)
    os.makedirs(WORK, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as handle:
        handle.write(line + '\n')


def keep_awake(enable: bool) -> None:
    """Ask Windows to keep the system awake while this process runs (no-op elsewhere)."""
    if os.name != 'nt':
        return
    flags = _ES_CONTINUOUS | (_ES_SYSTEM_REQUIRED if enable else 0)
    getattr(ctypes, 'windll').kernel32.SetThreadExecutionState(flags)


def package_inputs(pkg: str) -> str:
    """Return the ``inputs/`` directory of a package given its dotted name."""
    return os.path.join(H1, *pkg.split('.'), 'inputs')


def read_column(path: str) -> numpy.ndarray:
    """Read a one-column CSV of numbers."""
    return numpy.loadtxt(path, delimiter=',')


def best_params(pkg: str) -> dict:
    """Load ``inputs/best_params.json`` of a package."""
    with open(os.path.join(package_inputs(pkg), 'best_params.json'), 'r', encoding='utf-8') as handle:
        return json.load(handle)


def training_env(pandemic, pkg: str):
    """Build the training environment exactly as the package's ``train_*`` script does."""
    inputs = package_inputs(pkg)
    lengths = read_column(os.path.join(inputs, 'sequence_lengths.csv'))
    severities = read_column(os.path.join(inputs, 'initial_severity.csv'))
    cities, city_counts = numpy.unique(lengths, return_counts=True)
    values, value_counts = numpy.unique(severities, return_counts=True)
    env = pandemic.Pandemic()
    env.number_cities_prob = numpy.asarray((cities, city_counts / len(lengths))).T
    env.severity_prob = numpy.asarray((values, value_counts / len(severities))).T
    return env


def check_training_inputs(pkg: str) -> None:
    """Abort if the package CSVs are not the reference ones (the benchmark swaps them in place)."""
    for name in ('initial_severity.csv', 'sequence_lengths.csv'):
        ours = read_column(os.path.join(package_inputs(pkg), name))
        ref = read_column(os.path.join(SCENARIO_ROOT, 'sev_base', name))
        if ours.shape != ref.shape or not numpy.array_equal(ours, ref):
            raise RuntimeError(f'{pkg}/inputs/{name} no es la referencia: ¿hay un benchmark en curso?')


def evaluate(pandemic, tools, policy_factory: Callable[[], Callable]) -> Dict[str, float]:
    """Mean ``r̄`` of a greedy policy on the reference and each held-out replica."""
    means = {}
    for scenario in SCENARIOS:
        folder = os.path.join(SCENARIO_ROOT, scenario)
        lengths = read_column(os.path.join(folder, 'sequence_lengths.csv'))
        severities = read_column(os.path.join(folder, 'initial_severity.csv'))
        sevs = tools.convert_globalseq_to_seqs(lengths, severities)
        _seqs, perfs, _ = pandemic.run_experiment(pandemic.Pandemic(), policy_factory(), False, lengths, sevs)
        means[scenario] = float(numpy.mean(perfs))
    return means


###############
##  Policies
###############
def ql_policy(q_table: numpy.ndarray) -> Callable[[], Callable]:
    """Greedy Q-table policy, identical to ``eval_qf`` in ``pes_ql/ext/train_rl.py``."""
    def factory() -> Callable:
        def policy(_env, state, _seqid) -> int:
            s0 = min(int(state[0]), q_table.shape[0] - 1)
            s1 = min(int(state[1]), q_table.shape[1] - 1)
            s2 = min(int(state[2]), q_table.shape[2] - 1)
            options = q_table[s0, s1, s2].copy()
            actions = numpy.arange(len(options), dtype=numpy.float32)
            options[actions > state[0]] = -1e9
            return int(numpy.argmax(options))
        return policy
    return factory


def dqn_policy(pandemic, model_module, network) -> Callable[[], Callable]:
    """Greedy policy, identical to ``eval_qf`` in ``pes_dqn/ext/train_dqn.py``."""
    def factory() -> Callable:
        def policy(env, state, _seqid) -> int:
            norm = model_module.normalize_state(state, env.max_resources, env.max_seq_length, env.max_severity)
            q_values = network(norm[numpy.newaxis], training=False).numpy()[0].copy()
            return int(pandemic.dqn_agent_meta_cognitive(q_values, state[0], 10000)[0])
        return policy
    return factory


def trf_policy(pandemic, model_module, network, history_len: int) -> Callable[[], Callable]:
    """Greedy windowed policy, identical to ``eval_qf`` in ``pes_trf/ext/train_transformer.py``."""
    def factory() -> Callable:
        history = model_module.HistoryDeque(history_len, 3)
        current = {'seqid': -1}

        def policy(env, state, seqid) -> int:
            if seqid != current['seqid']:
                history.reset()
                current['seqid'] = seqid
            norm = model_module.normalize_state(state, env.max_resources, env.max_seq_length, env.max_severity)
            history.append_step(norm)
            q_values = network(history.current_window()[numpy.newaxis], training=False).numpy()[0].copy()
            return int(pandemic.trf_agent_meta_cognitive(q_values, state[0], 10000)[0])
        return policy
    return factory


###############
##  Training
###############
def deep_kwargs(params: dict, seed: int) -> dict:
    """Keyword arguments of ``DQNTraining`` / ``TRFTraining`` from ``best_params.json``."""
    hp = params['hyperparameters']
    starts = max(int(hp['learning_starts_frac'] * hp['buffer_size']), int(hp['batch_size']))
    pbrs = float(hp.get('penalty_coeff', 0.0)) if hp.get('use_pbrs') else 0.0
    return {'hidden_units': list(params['hidden_units']), 'batch_size': hp['batch_size'],
            'buffer_size': hp['buffer_size'], 'target_sync_freq': hp['target_sync_freq'],
            'max_grad_norm': hp['max_grad_norm'], 'seed': seed, 'penalty_coeff': pbrs,
            'compute_confidence': False, 'warmup_ratio': hp['warmup_ratio'],
            'target_ratio': hp['target_ratio'], 'learning_starts': starts}


def run_one(key: str, seed: int) -> Tuple[Dict[str, float], float]:
    """Train (or load, for the delivered model) one model and evaluate it.

    Parameters
    ----------
    key : str
        ``ql``, ``dqn`` or ``trf``.
    seed : int
        Training seed; ``0`` evaluates the delivered model without training.

    Returns
    -------
    tuple
        Mean ``r̄`` per scenario and training time in seconds.
    """
    spec = MODELS[key]
    pkg = spec['pkg']
    pandemic = importlib.import_module(f'{pkg}.ext.pandemic')
    tools = importlib.import_module(f'{pkg}.ext.tools')
    params = best_params(pkg)
    hp = params['hyperparameters']
    out_dir = os.path.join(WORK, pkg.split('.')[1], 'entregado' if seed == 0 else f'seed_{seed}')
    os.makedirs(out_dir, exist_ok=True)
    delivered = os.path.join(package_inputs(pkg), spec['delivered'])
    start = time.perf_counter()
    if key == 'ql':
        if seed == 0:
            q_table = numpy.load(delivered)
        else:
            _rewards, q_table, _conf = pandemic.QLearning(
                training_env(pandemic, pkg), hp['learning_rate'], hp['discount_factor'], hp['epsilon_initial'],
                hp['epsilon_min'], int(hp['num_episodes']), seed=seed, track_confidence=False)
            numpy.save(os.path.join(out_dir, 'q.npy'), q_table)
        factory = ql_policy(q_table)
    else:
        tf = importlib.import_module('tensorflow')
        model_module = importlib.import_module(f'{pkg}.ext.{"dqn_model" if key == "dqn" else "transformer_model"}')
        if seed == 0:
            network = tf.keras.models.load_model(delivered, safe_mode=False)
        else:
            kwargs = deep_kwargs(params, seed)
            if key == 'trf':
                kwargs.update({name: hp[name] for name in ('history_len', 'd_model', 'num_heads', 'key_dim',
                                                           'ff_dim', 'num_layers', 'dropout')})
                train = pandemic.TRFTraining
            else:
                train = pandemic.DQNTraining
            _rewards, network, _conf = train(training_env(pandemic, pkg), hp['learning_rate'],
                                             hp['discount_factor'], hp['epsilon_initial'], hp['epsilon_min'],
                                             int(hp['num_episodes']), **kwargs)
            network.save(os.path.join(out_dir, spec['delivered']))
        if key == 'dqn':
            factory = dqn_policy(pandemic, model_module, network)
        else:
            factory = trf_policy(pandemic, model_module, network, int(hp['history_len']))
    elapsed = time.perf_counter() - start
    return evaluate(pandemic, tools, factory), elapsed


def summarise() -> dict:
    """Collect every ``result.json`` into per-model statistics and write the summary JSON."""
    summary: dict = {'escenarios': SCENARIOS, 'modelos': {}}
    for key in ORDER:
        folder = os.path.join(WORK, MODELS[key]['pkg'].split('.')[1])
        runs: List[dict] = []
        for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
            path = os.path.join(folder, name, 'result.json')
            if os.path.isfile(path):
                with open(path, 'r', encoding='utf-8') as handle:
                    runs.append(json.load(handle))
        if not runs:
            continue
        ref = numpy.array([r['medias']['sev_base'] for r in runs])
        held = numpy.array([numpy.mean([r['medias'][s] for s in SCENARIOS[1:]]) for r in runs])
        summary['modelos'][key] = {
            'paquete': MODELS[key]['pkg'], 'n': len(runs),
            'corridas': [{'semilla': r['semilla'], 'referencia': r['medias']['sev_base'],
                          'heldout': float(numpy.mean([r['medias'][s] for s in SCENARIOS[1:]])),
                          'horas': r['segundos'] / 3600} for r in runs],
            'referencia': {'media': float(ref.mean()), 'desviacion': float(ref.std(ddof=1)) if len(runs) > 1 else 0.0,
                           'minimo': float(ref.min()), 'maximo': float(ref.max())},
            'heldout': {'media': float(held.mean()), 'desviacion': float(held.std(ddof=1)) if len(runs) > 1 else 0.0,
                        'minimo': float(held.min()), 'maximo': float(held.max())}}
    os.makedirs(os.path.dirname(SUMMARY_FILE), exist_ok=True)
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=1, ensure_ascii=False)
    return summary


###############
##  Main
###############
def main() -> None:
    """Run the pending (model, seed) pairs, or only rebuild the summary."""
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n', 1)[0])
    parser.add_argument('command', choices=('run', 'summary'))
    parser.add_argument('--models', nargs='+', choices=ORDER, default=list(ORDER))
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='training seeds (default: those of MODELS); 0 = delivered model')
    args = parser.parse_args()
    if args.command == 'run':
        keep_awake(True)
        try:
            for key in args.models:
                check_training_inputs(MODELS[key]['pkg'])
                for seed in [0] + (args.seeds or MODELS[key]['seeds']):
                    folder = os.path.join(WORK, MODELS[key]['pkg'].split('.')[1],
                                          'entregado' if seed == 0 else f'seed_{seed}')
                    result_file = os.path.join(folder, 'result.json')
                    if os.path.isfile(result_file):
                        log(f'{key} semilla {seed}: ya hecha, se omite')
                        continue
                    log(f'{key} semilla {seed}: inicio')
                    means, seconds = run_one(key, seed)
                    with open(result_file, 'w', encoding='utf-8') as handle:
                        json.dump({'modelo': key, 'semilla': seed, 'segundos': seconds, 'medias': means}, handle,
                                  indent=1)
                    log(f'{key} semilla {seed}: fin en {seconds / 3600:.2f} h; referencia '
                        f"{means['sev_base']:.4f}, held-out {numpy.mean([means[s] for s in SCENARIOS[1:]]):.4f}")
                    summarise()
        finally:
            keep_awake(False)
    summary = summarise()
    for key, info in summary['modelos'].items():
        log(f"{key}: n={info['n']}  ref {info['referencia']['media']:.4f} ± {info['referencia']['desviacion']:.4f}"
            f"  held-out {info['heldout']['media']:.4f} ± {info['heldout']['desviacion']:.4f}")


if __name__ == '__main__':
    sys.exit(main())
