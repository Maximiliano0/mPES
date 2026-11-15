"""Partial estimate of the seed-to-seed variability of the individual models.

Retrains Q-Learning (``pes_ql``), Double Q-Learning (``pes_dql``), DQN
(``pes_dqn``), the recurrent DQN (``pes_rdqn``), A2C (``pes_a2c``) and the
Transformer (``pes_trf``) with the best configuration of each package and
several new training seeds, and evaluates every model greedily on the
reference (``sev_base``) and the five held-out replicas. The delivered model of
each package is evaluated with the same code, so the delivered seed is one more
sample of the distribution. Each package is trained as its ``train_*`` script
does: ``pes_rdqn`` takes its architecture and episode count from
``config/CONFIG.py`` (``best_params.json`` keeps stale architecture keys).

The packages' ``train_*`` scripts are **never** called: they overwrite the
delivered model (``inputs/*.keras``, ``inputs/q.npy``). This harness calls the
training functions directly and writes everything under
``general/work/seeds/<pkg>/seed_<s>/``. Every (model, seed) is skipped when its
``result.json`` already exists, so an interrupted run resumes where it stopped.
While it runs, the process asks Windows not to sleep (``SetThreadExecutionState``).

Usage (from ``h1/``, with ``win_mpes_env`` activated)::

    python -m general.scripts.seed_variability run
    python -m general.scripts.seed_variability run --models ql dqn --seeds 1001 1002
    python -m general.scripts.seed_variability run --models a2c --seeds 1003
    python -m general.scripts.seed_variability summary

``run`` without ``--models`` covers the six models with seeds 1001-1010; A2C is then trained in a child process.
DQN, the recurrent DQN and the Transformer are single-threaded by design, so seeds can also run in parallel as
separate processes with disjoint ``--models``/``--seeds``.
"""

##########################
##  Imports externos    ##
##########################
import argparse
import ctypes
import importlib
import json
import os
import subprocess
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
NEW_SEEDS = list(range(1001, 1011))
MODELS = {
    'ql':   {'pkg': 'tabular.pes_ql', 'delivered': 'q.npy', 'seeds': NEW_SEEDS},
    'dql':  {'pkg': 'tabular.pes_dql', 'delivered': 'q.npy', 'seeds': NEW_SEEDS},
    'dqn':  {'pkg': 'ml.pes_dqn', 'delivered': 'dqn_model.keras', 'seeds': NEW_SEEDS},
    'rdqn': {'pkg': 'ml.pes_rdqn', 'delivered': 'rdqn_model.keras', 'seeds': NEW_SEEDS},
    'a2c':  {'pkg': 'ml.pes_a2c', 'delivered': 'ac_actor.keras', 'seeds': NEW_SEEDS},
    'trf':  {'pkg': 'ml.pes_trf', 'delivered': 'trf_model.keras', 'seeds': NEW_SEEDS},
}
ORDER = ('ql', 'dql', 'dqn', 'rdqn', 'a2c', 'trf')
TABULAR = ('ql', 'dql')
# ``pes_a2c`` sets TensorFlow intra-op threads to 0 (automatic) at import time and the other networks to 1; TF
# refuses to change it once initialised, so A2C runs in its own process when requested with other networks.
ISOLATED = ('a2c',)
MODEL_MODULES = {'dqn': 'dqn_model', 'rdqn': 'rdqn_model', 'a2c': 'ac_model', 'trf': 'transformer_model'}
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


def windowed_policy(agent: Callable, model_module, network, history_len: int) -> Callable[[], Callable]:
    """Greedy windowed policy, identical to ``eval_qf`` in ``train_transformer.py`` and ``train_rdqn.py``.

    Parameters
    ----------
    agent : callable
        ``trf_agent_meta_cognitive`` or ``rdqn_agent_meta_cognitive`` of the package.
    model_module : module
        ``transformer_model`` or ``rdqn_model`` (``normalize_state`` and ``HistoryDeque``).
    network : keras.Model
        Q-network fed with a window of ``history_len`` normalised states.
    history_len : int
        Window length.

    Returns
    -------
    callable
        Factory of a policy whose window is reset at the start of every sequence.
    """
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
            return int(agent(q_values, state[0], 10000)[0])
        return policy
    return factory


def a2c_policy(pandemic, model_module, network) -> Callable[[], Callable]:
    """Greedy actor policy, identical to ``eval_qf`` in ``pes_a2c/ext/train_a2c.py``."""
    def factory() -> Callable:
        def policy(env, state, _seqid) -> int:
            norm = model_module.normalize_state(state, env.available_resources_states - 1, env.trial_no_states - 1,
                                                env.severity_states - 1)
            probs = network(norm[numpy.newaxis, :], training=False)[0].numpy().copy()
            return int(pandemic.ac_agent_meta_cognitive(probs, state[0], 10000)[0])
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


def rdqn_config(pkg: str):
    """Return ``config/CONFIG.py`` of ``pes_rdqn``, the source of its architecture and episode count."""
    return importlib.import_module(f'{pkg}.config.CONFIG')


def train_tabular(key: str, pandemic, pkg: str, params: dict, seed: int) -> numpy.ndarray:
    """Train a Q-table as ``train_rl.py`` of ``pes_ql`` or ``pes_dql`` does.

    Parameters
    ----------
    key : str
        ``ql`` or ``dql``.
    pandemic : module
        ``ext.pandemic`` of the package.
    pkg : str
        Dotted package name.
    params : dict
        ``best_params.json`` of the package.
    seed : int
        Training seed.

    Returns
    -------
    numpy.ndarray
        Trained Q-table.
    """
    hp = params['hyperparameters']
    env = training_env(pandemic, pkg)
    episodes = int(hp['num_episodes'])
    if key == 'ql':
        _rewards, q_table, _conf = pandemic.QLearning(
            env, hp['learning_rate'], hp['discount_factor'], hp['epsilon_initial'], hp['epsilon_min'], episodes,
            seed=seed, track_confidence=False)
        return q_table
    env.verbose = False
    decay_rate = (hp['epsilon_min'] / hp['epsilon_initial']) ** (
        1.0 / ((hp['target_ratio'] - hp['warmup_ratio']) * episodes))
    _rewards, q_table, _conf = pandemic.QLearning(
        env, hp['learning_rate'], hp['discount_factor'], hp['epsilon_initial'], hp['epsilon_min'], episodes,
        decay_rate=decay_rate, warmup_ratio=hp['warmup_ratio'], target_ratio=hp['target_ratio'],
        double_q=bool(params.get('double_q', True)), penalty_coeff=float(hp['penalty_coeff']), seed=seed,
        track_confidence=False)
    return q_table


def train_network(key: str, pandemic, pkg: str, params: dict, seed: int):
    """Train a network as the ``train_*.py`` script of the package does.

    Parameters
    ----------
    key : str
        ``dqn``, ``rdqn``, ``a2c`` or ``trf``.
    pandemic : module
        ``ext.pandemic`` of the package.
    pkg : str
        Dotted package name.
    params : dict
        ``best_params.json`` of the package.
    seed : int
        Training seed.

    Returns
    -------
    keras.Model
        Q-network, or the actor for ``a2c``.
    """
    hp = params['hyperparameters']
    env = training_env(pandemic, pkg)
    if key == 'a2c':
        env.verbose = False
        layers = int(hp['n_hidden_layers'])
        _rewards, actor, _conf = pandemic.A2CTraining(
            env, hp['actor_lr'], hp['critic_lr'], hp['discount_factor'], hp['entropy_coeff'],
            hp.get('epsilon_initial', 0.0), hp.get('epsilon_min', 0.0), int(hp['num_episodes']),
            actor_hidden=[int(hp['actor_hidden_dim'])] * layers, critic_hidden=[int(hp['critic_hidden_dim'])] * layers,
            seed=seed, compute_confidence=False, warmup_ratio=hp.get('warmup_ratio', 0.0),
            target_ratio=hp.get('target_ratio', 1.0), penalty_coeff=hp['penalty_coeff'], gae_lambda=hp['gae_lambda'],
            max_grad_norm=hp['max_grad_norm'], lr_min_ratio=hp['lr_min_ratio'],
            spend_cost_coeff=hp['spend_cost_coeff'], last_action_bias=hp['last_action_bias'])
        return actor
    kwargs = deep_kwargs(params, seed)
    episodes = int(hp['num_episodes'])
    if key == 'trf':
        kwargs.update({name: hp[name] for name in ('history_len', 'd_model', 'num_heads', 'key_dim',
                                                   'ff_dim', 'num_layers', 'dropout')})
        train = pandemic.TRFTraining
    elif key == 'rdqn':
        cfg = rdqn_config(pkg)
        env.verbose = False
        kwargs.update({'hidden_units': list(cfg.RDQN_HIDDEN_UNITS), 'history_len': int(cfg.RDQN_HISTORY_LEN),
                       'lstm_units': int(cfg.RDQN_LSTM_UNITS)})
        episodes = int(cfg.RDQN_EPISODES)
        train = pandemic.RDQNTraining
    else:
        train = pandemic.DQNTraining
    _rewards, network, _conf = train(env, hp['learning_rate'], hp['discount_factor'], hp['epsilon_initial'],
                                     hp['epsilon_min'], episodes, **kwargs)
    return network


def network_policy(key: str, pandemic, pkg: str, network) -> Callable[[], Callable]:
    """Greedy evaluation policy of a trained network, as in the ``eval_qf`` of each package."""
    model_module = importlib.import_module(f'{pkg}.ext.{MODEL_MODULES[key]}')
    if key == 'dqn':
        return dqn_policy(pandemic, model_module, network)
    if key == 'a2c':
        return a2c_policy(pandemic, model_module, network)
    if key == 'rdqn':
        return windowed_policy(pandemic.rdqn_agent_meta_cognitive, model_module, network,
                               int(rdqn_config(pkg).RDQN_HISTORY_LEN))
    history_len = int(best_params(pkg)['hyperparameters']['history_len'])
    return windowed_policy(pandemic.trf_agent_meta_cognitive, model_module, network, history_len)


def run_one(key: str, seed: int) -> Tuple[Dict[str, float], float]:
    """Train (or load, for the delivered model) one model and evaluate it.

    Parameters
    ----------
    key : str
        One of ``ORDER``.
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
    out_dir = os.path.join(WORK, pkg.split('.')[1], 'entregado' if seed == 0 else f'seed_{seed}')
    os.makedirs(out_dir, exist_ok=True)
    delivered = os.path.join(package_inputs(pkg), spec['delivered'])
    start = time.perf_counter()
    if key in TABULAR:
        if seed == 0:
            q_table = numpy.load(delivered)
        else:
            q_table = train_tabular(key, pandemic, pkg, params, seed)
            numpy.save(os.path.join(out_dir, spec['delivered']), q_table)
        factory = ql_policy(q_table)
    else:
        if seed == 0:
            network = importlib.import_module('tensorflow').keras.models.load_model(delivered, safe_mode=False)
        else:
            network = train_network(key, pandemic, pkg, params, seed)
            network.save(os.path.join(out_dir, spec['delivered']))
        factory = network_policy(key, pandemic, pkg, network)
    elapsed = time.perf_counter() - start
    return evaluate(pandemic, tools, factory), elapsed


def spread(values: List[float]) -> Dict[str, float]:
    """Mean, sample standard deviation, minimum and maximum of ``values`` (``nan`` when empty).

    Parameters
    ----------
    values : list of float
        Mean ``r̄`` of each run.

    Returns
    -------
    dict
        ``media``, ``desviacion`` (``ddof=1``; ``0`` with a single value), ``minimo`` and ``maximo``.
    """
    if not values:
        nan = float('nan')
        return {'media': nan, 'desviacion': nan, 'minimo': nan, 'maximo': nan}
    array = numpy.asarray(values, dtype=float)
    return {'media': float(array.mean()), 'desviacion': float(array.std(ddof=1)) if array.size > 1 else 0.0,
            'minimo': float(array.min()), 'maximo': float(array.max())}


def summarise() -> dict:
    """Collect every ``result.json`` into per-model statistics and write the summary JSON.

    The delivered model (seed ``0``) is reported apart from the retrained seeds: its configuration and seed come
    from the best Optuna trial, so it is not an unbiased draw of the seed distribution.
    """
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
        corridas = [{'semilla': r['semilla'], 'referencia': r['medias']['sev_base'],
                     'heldout': float(numpy.mean([r['medias'][s] for s in SCENARIOS[1:]])),
                     'horas': r['segundos'] / 3600} for r in runs]
        delivered = [c for c in corridas if c['semilla'] == 0]
        retrained = [c for c in corridas if c['semilla'] != 0]
        summary['modelos'][key] = {
            'paquete': MODELS[key]['pkg'], 'corridas': corridas,
            'entregado': ({'referencia': delivered[0]['referencia'], 'heldout': delivered[0]['heldout']}
                          if delivered else None),
            'semillas_nuevas': {'n': len(retrained),
                                'referencia': spread([c['referencia'] for c in retrained]),
                                'heldout': spread([c['heldout'] for c in retrained])},
            'todas': {'n': len(corridas), 'referencia': spread([c['referencia'] for c in corridas]),
                      'heldout': spread([c['heldout'] for c in corridas])}}
    os.makedirs(os.path.dirname(SUMMARY_FILE), exist_ok=True)
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=1, ensure_ascii=False)
    return summary


def run_model(key: str, seeds: List[int]) -> None:
    """Train and evaluate the pending seeds of one model (the delivered model first), skipping finished ones.

    Parameters
    ----------
    key : str
        One of ``ORDER``.
    seeds : list of int
        New training seeds; the delivered model (seed ``0``) is always added in front.
    """
    check_training_inputs(MODELS[key]['pkg'])
    for seed in [0] + list(seeds):
        folder = os.path.join(WORK, MODELS[key]['pkg'].split('.')[1], 'entregado' if seed == 0 else f'seed_{seed}')
        result_file = os.path.join(folder, 'result.json')
        if os.path.isfile(result_file):
            log(f'{key} semilla {seed}: ya hecha, se omite')
            continue
        log(f'{key} semilla {seed}: inicio')
        means, seconds = run_one(key, seed)
        # Atomic write: parallel processes read every result.json in summarise().
        partial = result_file + '.tmp'
        with open(partial, 'w', encoding='utf-8') as handle:
            json.dump({'modelo': key, 'semilla': seed, 'segundos': seconds, 'medias': means}, handle, indent=1)
        os.replace(partial, result_file)
        log(f'{key} semilla {seed}: fin en {seconds / 3600:.2f} h; referencia '
            f"{means['sev_base']:.4f}, held-out {numpy.mean([means[s] for s in SCENARIOS[1:]]):.4f}")
        summarise()


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
            isolated = [key for key in args.models if key in ISOLATED
                        and any(other != key and other not in TABULAR for other in args.models)]
            for key in args.models:
                if key not in isolated:
                    run_model(key, args.seeds or MODELS[key]['seeds'])
            for key in isolated:
                seeds = [str(seed) for seed in args.seeds] if args.seeds else []
                log(f'{key}: en un proceso propio (configuración de hilos de TensorFlow incompatible)')
                subprocess.run([sys.executable, '-m', 'general.scripts.seed_variability', 'run', '--models', key]
                               + (['--seeds'] + seeds if seeds else []), cwd=H1, check=False)
        finally:
            keep_awake(False)
    summary = summarise()
    for key, info in summary['modelos'].items():
        new = info['semillas_nuevas']
        log(f"{key}: semillas nuevas n={new['n']}  ref {new['referencia']['media']:.4f} ± "
            f"{new['referencia']['desviacion']:.4f}  held-out {new['heldout']['media']:.4f} ± "
            f"{new['heldout']['desviacion']:.4f}")


if __name__ == '__main__':
    sys.exit(main())
