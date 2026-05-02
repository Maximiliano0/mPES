'''
pes_dqn - Pandemic Experiment Scenario: DQN-Agent Training Pipeline

Trains a Deep Q-Network agent on the Pandemic environment using
hyperparameters from config/CONFIG.py and evaluates it against a
random-player baseline.

Pipeline stages
---------------
1. Load training data (initial_severity.csv, sequence_lengths.csv)
2. Run random-player baseline and save performance plots
3. Train DQN agent (default 175 000 episodes, configurable via CLI)
4. Save trained model, rewards history, and training config to a dated directory
5. Evaluate trained agent on the same sequences and generate
   performance/confidence visualisations

Key differences from pes_ql/ext/train_rl.py
--------------------------------------------
- Uses a neural-network Q-function instead of a tabular Q-table
- Saves a .keras model file instead of a .npy Q-table
- DQN-specific hyperparameters: hidden_units, batch_size, buffer_size,
  target_sync_freq
- Default episodes: 175 000 (configurable via CLI; see ``DQN_EPISODES``)
- Uses SEED from CONFIG.py for reproducible training

Usage
-----
::

    # 1. Train with the hyperparameters baked into CONFIG.py
    python -m pes_dqn.ext.train_dqn [num_episodes]

    # 2. Reproduce the best trial of a Bayesian optimisation study
    #    (reads params + per-trial seed from the JSON sidecar
    #     ``best_params_<date>.json`` written by
    #     ``python -m pes_dqn.ext.optimize_dqn``; falls back to the
    #     SQLite study DB if the JSON file is absent).  Use this after
    #     copying the optimisation directory from Colab to your local
    #     machine.
    python -m pes_dqn.ext.train_dqn --from-best 2026-04-20

The ``--from-best`` flag is the supported way to make the local ``mean_perf``
match the Colab ``mean_perf`` of the best trial.  Bit-exact reproduction
requires identical TF version AND identical hardware (CPU vs GPU may differ
in the LSB).
'''

##########################
##  Imports internos    ##
##########################
from .pandemic import Pandemic, dqn_agent_meta_cognitive, run_experiment, DQNTraining
from .dqn_model import normalize_state
from ..src.terminal_utils import header, section, success, info, list_item
from .tools import plot_confidences, convert_globalseq_to_seqs
from ..config.CONFIG import (SEED, DQN_MODEL_FILE)
from .. import INPUTS_PATH

##########################
##  Imports externos    ##
##########################
import json
import os
import re
import sys
import glob
import numpy
import warnings
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime

# Device selection is centralised in pes_dqn/__init__.py via MPES_USE_GPU.
# Re-asserting it here only as a defensive fallback when this module is
# launched in isolation (e.g. ``python ext/train_dqn.py``) without the
# package having been imported first.
if os.environ.get('MPES_USE_GPU', '0') != '1':
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')


# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*Box.*precision lowered.*')
warnings.filterwarnings('ignore', message='.*A NumPy version.*SciPy.*')
warnings.filterwarnings('ignore', category=SyntaxWarning, message='.*invalid escape sequence.*')


###################################
##   Best-trial loader helpers  ##
###################################
_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')


def _parse_cli_args():
    """
    Parse ``train_dqn`` CLI arguments.

    Returns
    -------
    num_episodes : int or None
        Number of training episodes (``None`` → use default / best-trial value).
    from_best_date : str or None
        ``YYYY-MM-DD`` of an Optuna run to load best params from, or ``None``.
    """
    args = sys.argv[1:]
    num_episodes = None
    from_best_date = None
    i = 0
    while i < len(args):
        a = args[i]
        if a == '--from-best':
            if i + 1 >= len(args):
                raise SystemExit("--from-best requires a YYYY-MM-DD argument")
            cand = args[i + 1]
            if not _DATE_RE.match(cand):
                # Reject path-injection attempts like '../../etc'.
                raise SystemExit(
                    f"--from-best expects YYYY-MM-DD, got: {cand!r}"
                )
            from_best_date = cand
            i += 2
        elif a.isdigit():
            num_episodes = int(a)
            i += 1
        else:
            raise SystemExit(f"Unknown argument: {a}")
    return num_episodes, from_best_date


def _load_canonical_params():
    """Load best-trial params from the canonical ``pes_dqn/inputs/best_params.json``.

    This file is written automatically by ``optimize_dqn.main`` as a mirror of
    the dated sidecar.  It is the recommended source when the user just wants
    to retrain with the latest optimised hyperparameters without remembering a
    run date.

    Returns
    -------
    dict or None
        Same structure as :func:`_load_best_trial`, or ``None`` if the file
        does not exist.
    """
    canonical = os.path.join(INPUTS_PATH, 'best_params.json')
    if not os.path.isfile(canonical):
        return None
    with open(canonical, 'r', encoding='utf-8') as _f:
        payload = json.load(_f)
    opt_date = payload.get('opt_date', 'unknown')
    # Look for the matching .keras in the dated subdirectory (may not exist
    # if the run was cut short, which is fine — train_dqn retrains from scratch).
    model_path = os.path.join(
        INPUTS_PATH, f'{opt_date}_BAYESIAN_OPT', f'dqn_best_{opt_date}.keras'
    )
    if not os.path.isfile(model_path):
        model_path = None
    return {
        'params':       dict(payload['hyperparameters']),
        'trial_seed':   int(payload['trial_seed']),
        'trial_number': int(payload['best_trial_number']),
        'mean_perf':    float(payload['mean_perf']),
        'hidden_units': payload.get('hidden_units'),
        'model_path':   model_path,
        'source':       canonical,
    }


def _find_latest_opt_date():
    """Return the newest ``YYYY-MM-DD`` for which ``best_params_<date>.json``
    exists under ``INPUTS_PATH/<date>_BAYESIAN_OPT/``.

    Returns
    -------
    str or None
        The lexicographically highest valid date, or ``None`` if no such
        directory + sidecar exists.  Used by ``main()`` to auto-resolve
        ``from_best_date`` so the user does not need to type
        ``--from-best YYYY-MM-DD`` for the most recent run (matches
        pes_ql / pes_dql autoload behavior).
    """
    pattern = os.path.join(INPUTS_PATH, '*_BAYESIAN_OPT', 'best_params_*.json')
    candidates = []
    for path in glob.glob(pattern):
        date = os.path.basename(os.path.dirname(path)).split('_', 1)[0]
        if _DATE_RE.match(date):
            candidates.append(date)
    if not candidates:
        return None
    return max(candidates)


def _load_best_trial(opt_date):
    """
    Load the best Optuna trial's hyperparameters and seed from disk.

    Resolution order (mirrors pes_ql/pes_dql for cross-package consistency):

      1. ``pes_dqn/inputs/<opt_date>_BAYESIAN_OPT/best_params_<opt_date>.json``
         (small JSON sidecar written by ``optimize_dqn.main``).
      2. ``pes_dqn/inputs/<opt_date>_BAYESIAN_OPT/optuna_study_<opt_date>.db``
         (full SQLite study DB; required if the JSON is absent).

    Parameters
    ----------
    opt_date : str
        Run date in ``YYYY-MM-DD`` form (validated by ``_parse_cli_args``).

    Returns
    -------
    dict
        Mapping with keys: ``params`` (dict of hyperparameters),
        ``trial_seed`` (int), ``trial_number`` (int), ``mean_perf`` (float),
        ``hidden_units`` (list[int] or None), ``model_path`` (str or None).
    """
    opt_dir = os.path.join(INPUTS_PATH, f'{opt_date}_BAYESIAN_OPT')
    params_file = os.path.join(opt_dir, f'best_params_{opt_date}.json')
    model_path = os.path.join(opt_dir, f'dqn_best_{opt_date}.keras')
    if not os.path.isfile(model_path):
        model_path = None

    # 1. Prefer the JSON sidecar — no optuna dependency, copy-friendly.
    if os.path.isfile(params_file):
        with open(params_file, 'r', encoding='utf-8') as _f:
            payload = json.load(_f)
        return {
            'params':       dict(payload['hyperparameters']),
            'trial_seed':   int(payload['trial_seed']),
            'trial_number': int(payload['best_trial_number']),
            'mean_perf':    float(payload['mean_perf']),
            'hidden_units': payload.get('hidden_units'),
            'model_path':   model_path,
        }

    # 2. Fall back to the SQLite study DB (legacy / partial copies).
    import optuna  # local import — only needed for --from-best
    db_file = os.path.join(opt_dir, f'optuna_study_{opt_date}.db')
    if not os.path.isfile(db_file):
        raise FileNotFoundError(
            f"Neither best_params_{opt_date}.json nor optuna_study_{opt_date}.db "
            f"found in {opt_dir}.\nCopy the {opt_date}_BAYESIAN_OPT directory "
            f"(or at least best_params_{opt_date}.json) from Colab into "
            f"pes_dqn/inputs/ first."
        )
    study = optuna.load_study(
        study_name=f'dqn_opt_{opt_date}',
        storage=f'sqlite:///{db_file}',
    )
    best = study.best_trial
    trial_seed = best.user_attrs.get('trial_seed')
    if trial_seed is None:
        # Pre-fix studies didn't persist trial_seed; reconstruct it using
        # the per-trial seed formula in optimize_dqn.objective.
        trial_seed = SEED + int(best.number) + 1
    hidden_units = best.user_attrs.get('hidden_units')
    if hidden_units is None and 'hidden_layer_size' in best.params:
        hidden_units = ([best.params['hidden_layer_size']]
                        * best.params.get('num_hidden_layers', 1))
    return {
        'params': dict(best.params),
        'trial_seed': int(trial_seed),
        'trial_number': int(best.number),
        'mean_perf': float(best.user_attrs.get('mean_perf', best.value)),
        'hidden_units': hidden_units,
        'model_path': model_path,
    }

###################################
##             Main             ###
###################################


def main():
    """Run the full DQN-Agent training and evaluation pipeline.

    Trains the DQN agent from scratch using the hyperparameters defined in
    ``config/CONFIG.py``.  Usage: ``python -m pes_dqn.ext.train_dqn [episodes]``.
    """

    # ---- Parse CLI arguments ------------------------------------------------
    cli_num_episodes, from_best_date = _parse_cli_args()

    header("DQN-AGENT TRAINING PIPELINE", width=80)

    # Configure matplotlib for better aesthetics
    try:
        plt.style.use('ggplot')
    except BaseException:
        pass  # Use default if style is not available

    matplotlib_config = {
        'figure.figsize': (12, 6),
        'figure.dpi': 100,
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'lines.linewidth': 2,
        'lines.markersize': 6
    }
    plt.rcParams.update(matplotlib_config)

    # Create training output directory with date stamp
    train_date = datetime.now().strftime("%Y-%m-%d")
    train_dir = os.path.join(INPUTS_PATH, f'{train_date}_DQN_TRAIN')
    os.makedirs(train_dir, exist_ok=True)
    info(f"Output directory: {train_dir}")

    # Load initial severity and sequence lengths data
    section("Loading Training Data", width=80)
    trials_per_sequence = numpy.loadtxt(os.path.join(INPUTS_PATH, 'sequence_lengths.csv'), delimiter=',')
    all_severities = numpy.loadtxt(os.path.join(INPUTS_PATH, 'initial_severity.csv'), delimiter=',')

    list_item(f"Sequence lengths shape: {trials_per_sequence.shape}")
    list_item(f"Initial severities shape: {all_severities.shape}")
    list_item(f"Total trials: {int(sum(trials_per_sequence))}")
    print()

    # Convert global sequences to per-sequence format
    sevs = convert_globalseq_to_seqs(trials_per_sequence, all_severities)

    # Calculate probability distributions for number of cities (trials per sequence)
    val_cities, count_cities = numpy.unique(trials_per_sequence, return_counts=True)
    number_cities_prob = numpy.asarray((val_cities, count_cities / len(trials_per_sequence))).T

    # Calculate probability distributions for initial severities
    val_severity, count_severity = numpy.unique(all_severities, return_counts=True)
    severity_prob = numpy.asarray((val_severity, count_severity / len(all_severities))).T

    env = Pandemic()

    def random_qf(_env, _state, _seqid):
        """Return a random action from the environment's action space."""
        return env.sample()

    section("Random Player Baseline", width=80)
    info("Training random agent for comparison...")
    seqs1, perfs1, _ = run_experiment(env, random_qf, False, trials_per_sequence, sevs)
    success("Random player experiment completed")

    __fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(seqs1, color='#1f77b4', linewidth=2.5, marker='o', markersize=5, label='Random Player')
    ax.set_xlabel('Trial', fontsize=12, fontweight='bold')
    ax.set_ylabel('Final Severity Achieved', fontsize=12, fontweight='bold')
    ax.set_title('Random Player Baseline: Severity per Sequence', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            train_dir,
            f'random_player_sequence_performance_{train_date}.png'),
        dpi=150,
        bbox_inches='tight')
    plt.close()
    list_item("Saved: random_player_sequence_performance.png")

    __fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(perfs1, color='#ff7f0e', linewidth=2.5, marker='s', markersize=5, label='Random Player')
    ax.set_ylabel('Normalised Performance (0-1)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Trial', fontsize=12, fontweight='bold')
    ax.set_title('Random Player Baseline: Normalised Performance per Sequence', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            train_dir,
            f'random_player_normalised_performance_{train_date}.png'),
        dpi=150,
        bbox_inches='tight')
    plt.close()
    list_item("Saved: random_player_normalised_performance.png")
    print()

    # Initialize pandemic environment with the calculated probability distributions
    env = Pandemic()

    env.number_cities_prob = number_cities_prob
    env.severity_prob = severity_prob
    env.verbose = False

    section("DQN Training", width=80)

    # DQN hyperparameters — from CONFIG.py (Bayesian optimisation best trial #8, 2026-04-18)
    from ..config.CONFIG import (DQN_LEARNING_RATE, DQN_DISCOUNT,
                                 DQN_EPSILON_INITIAL, DQN_EPSILON_MIN,
                                 DQN_HIDDEN_UNITS, DQN_BATCH_SIZE,
                                 DQN_REPLAY_BUFFER_SIZE, DQN_TARGET_SYNC_FREQ,
                                 DQN_MAX_GRAD_NORM, DQN_PENALTY_COEFF,
                                 DQN_WARMUP_RATIO, DQN_TARGET_RATIO,
                                 DQN_LEARNING_STARTS_FRAC, DQN_EPISODES)

    best_info = None
    if from_best_date is None:
        # Resolution order (highest priority first):
        #   1. pes_dqn/inputs/best_params.json  (canonical mirror, always up-to-date)
        #   2. latest dated  *_BAYESIAN_OPT/best_params_<date>.json
        _canonical = _load_canonical_params()
        if _canonical is not None:
            best_info = _canonical
            info(f"Auto-loading best trial from inputs/best_params.json "
                 f"(trial #{best_info['trial_number'] + 1}, "
                 f"mean_perf={best_info['mean_perf']:.6f})")
            info("  override with --from-best YYYY-MM-DD to use a specific run")
        else:
            _auto = _find_latest_opt_date()
            if _auto is not None:
                info(f"Auto-loading best trial from {_auto} (latest BAYESIAN_OPT)")
                info("  override with --from-best YYYY-MM-DD or delete the dir to disable")
                from_best_date = _auto

    if best_info is None and from_best_date is not None:
        info(f"Loading best trial from Bayesian optimisation run {from_best_date}...")
        best_info = _load_best_trial(from_best_date)

    if best_info is not None:
        bp = best_info['params']
        train_seed       = best_info['trial_seed']
        learning_rate    = bp['learning_rate']
        discount_factor  = bp['discount_factor']
        epsilon_initial  = bp['epsilon_initial']
        epsilon_min      = bp['epsilon_min']
        hidden_units     = list(best_info['hidden_units']) if best_info['hidden_units'] \
                                                          else list(DQN_HIDDEN_UNITS)
        batch_size       = bp['batch_size']
        buffer_size      = bp['buffer_size']
        target_sync_freq = bp['target_sync_freq']
        max_grad_norm    = bp.get('max_grad_norm', DQN_MAX_GRAD_NORM)
        # PBRS: respect the explicit use_pbrs flag if present, else fall back to coeff>0
        if bp.get('use_pbrs', bp.get('penalty_coeff', 0.0) > 0):
            penalty_coeff = float(bp.get('penalty_coeff', DQN_PENALTY_COEFF))
        else:
            penalty_coeff = 0.0
        warmup_ratio        = bp.get('warmup_ratio',  DQN_WARMUP_RATIO)
        target_ratio        = bp.get('target_ratio',  DQN_TARGET_RATIO)
        learning_starts_frac = bp.get('learning_starts_frac', DQN_LEARNING_STARTS_FRAC)
        learning_starts     = max(int(learning_starts_frac * buffer_size), int(batch_size))
        # Episode-count resolution:
        #   * CLI-supplied value wins (e.g. ``train_dqn 40000`` for parity vs
        #     Optuna mean_perf).
        #   * Otherwise use the full training budget DQN_EPISODES, NOT the
        #     low ``bp['num_episodes']`` that Optuna used for fast trials.
        opt_episodes        = int(bp['num_episodes'])
        num_episodes        = cli_num_episodes if cli_num_episodes is not None \
                                                else int(DQN_EPISODES)
        info(f"  Reproducing trial #{best_info['trial_number'] + 1}  "
             f"(reported mean_perf = {best_info['mean_perf']:.6f} "
             f"with {opt_episodes:,} episodes)")
        info(f"  Trial seed = {train_seed}")
        if num_episodes != opt_episodes:
            info(f"  Training with {num_episodes:,} episodes (CONFIG.DQN_EPISODES) "
                 f"— pass `{opt_episodes}` as CLI arg for parity with Optuna mean_perf")
    else:
        train_seed       = SEED
        learning_rate    = DQN_LEARNING_RATE
        discount_factor  = DQN_DISCOUNT
        epsilon_initial  = DQN_EPSILON_INITIAL
        epsilon_min      = DQN_EPSILON_MIN
        hidden_units     = list(DQN_HIDDEN_UNITS)
        batch_size       = DQN_BATCH_SIZE
        buffer_size      = DQN_REPLAY_BUFFER_SIZE
        target_sync_freq = DQN_TARGET_SYNC_FREQ
        max_grad_norm    = DQN_MAX_GRAD_NORM
        penalty_coeff    = DQN_PENALTY_COEFF
        warmup_ratio     = DQN_WARMUP_RATIO
        target_ratio     = DQN_TARGET_RATIO
        learning_starts_frac = DQN_LEARNING_STARTS_FRAC
        # Match the formula used in optimize_dqn.objective() exactly.
        learning_starts  = max(int(learning_starts_frac * buffer_size), int(batch_size))
        num_episodes     = cli_num_episodes if cli_num_episodes is not None else DQN_EPISODES

    info(f"Starting DQN training ({num_episodes:,} episodes)...")
    info(f"Network: Input(3) → {' → '.join(f'Dense({u}, ReLU)' for u in hidden_units)} → Dense(11, linear)")
    info(f"Replay buffer: {buffer_size:,}  |  Batch size: {batch_size}  |  Target sync: {target_sync_freq:,}")
    info(f"Seed:          {train_seed}")
    info("(This may take several minutes)")
    print()

    rewards, dqn_model, confsrl = DQNTraining(
        env, learning_rate, discount_factor,
        epsilon_initial, epsilon_min, num_episodes,
        hidden_units=hidden_units, batch_size=batch_size,
        buffer_size=buffer_size, target_sync_freq=target_sync_freq,
        max_grad_norm=max_grad_norm, seed=train_seed,
        penalty_coeff=penalty_coeff,
        compute_confidence=False,
        warmup_ratio=warmup_ratio, target_ratio=target_ratio,
        learning_starts=learning_starts)
    print()
    success("Training completed")
    list_item(f"Network parameters: {dqn_model.count_params():,}")
    list_item(f"Rewards history length: {len(rewards)}")

    info("Saving trained model...")

    # Save DQN model and rewards with date stamp
    model_file = os.path.join(train_dir, f'dqn_model_{train_date}.keras')
    rewards_file = os.path.join(train_dir, f'rewards_{train_date}.npy')
    config_file = os.path.join(train_dir, f'training_config_{train_date}.txt')

    dqn_model.save(model_file)
    numpy.save(rewards_file, numpy.array(rewards))

    # Also save to the canonical location for __main__.py
    dqn_model.save(os.path.join(INPUTS_PATH, DQN_MODEL_FILE))
    numpy.save(os.path.join(INPUTS_PATH, 'rewards.npy'), numpy.array(rewards))

    # Create configuration file
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("DQN-AGENT TRAINING CONFIGURATION\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Training Date: {train_date}\n")
        f.write(f"Training Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("DQN HYPERPARAMETERS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Learning Rate (Adam):        {learning_rate}\n")
        f.write(f"Discount Factor (γ):         {discount_factor}\n")
        f.write(f"Initial Epsilon (ε):         {epsilon_initial}\n")
        f.write(f"Minimum Epsilon (ε_min):     {epsilon_min}\n")
        f.write(f"Number of Episodes:          {num_episodes:,}\n")
        f.write(f"Epsilon Decay:               Exponential with warm-up "
                f"({epsilon_initial} → {epsilon_min}, warmup_ratio={warmup_ratio}, "
                f"target_ratio={target_ratio})\n")
        f.write(f"Hidden Units:                {hidden_units}\n")
        f.write(f"Batch Size:                  {batch_size}\n")
        f.write(f"Replay Buffer Size:          {buffer_size:,}\n")
        f.write(f"Target Sync Frequency:       {target_sync_freq:,} steps\n")
        f.write(f"Max Gradient Norm:           {max_grad_norm}\n")
        f.write(f"PBRS β (penalty_coeff):      {penalty_coeff}\n")
        f.write(f"Learning starts (frac):      {learning_starts_frac} "
                f"→ {learning_starts:,} transitions\n\n")

        f.write("TRAINING RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Network Parameters:          {dqn_model.count_params():,}\n")
        f.write(f"State Space Dimension:       3 (resources, trial, severity)\n")
        f.write(f"Action Space:                11 (0-10)\n")
        f.write(f"Rewards History Length:       {len(rewards)}\n\n")

        f.write("OUTPUT FILES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Model File:                  dqn_model_{train_date}.keras\n")
        f.write(f"Rewards File:                rewards_{train_date}.npy\n")
        f.write(f"Configuration File:          training_config_{train_date}.txt\n\n")

        f.write("DESCRIPTION\n")
        f.write("-" * 80 + "\n")
        f.write("Files saved from DQN training on the Pandemic Scenario.\n")
        f.write("The model maps normalised (resources, trial, severity) states to Q-values.\n")
        f.write("The rewards file contains average reward progression every 10,000 episodes.\n")

    success(f"✓ Model saved to dqn_model_{train_date}.keras")
    success(f"✓ Rewards saved to rewards_{train_date}.npy")
    success(f"✓ Configuration saved to training_config_{train_date}.txt")
    list_item(f"Training Directory: {train_dir}")
    print()

    section("Training Performance Analysis", width=80)

    if rewards:
        info("Generating reward history visualization...")

        __fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(100 * (numpy.arange(len(rewards)) + 1), rewards, color='#2ca02c', linewidth=2.5,
                label='Average Reward')
        ax.fill_between(100 * (numpy.arange(len(rewards)) + 1), rewards, alpha=0.2, color='#2ca02c')
        ax.set_xlabel('Episodes', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
        ax.set_title('DQN Training: Average Reward Progression', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(train_dir, f'dqn_agent_rewards_vs_episodes_{train_date}.png'), dpi=150,
                    bbox_inches='tight')
        plt.close()
        list_item(f"Saved: dqn_agent_rewards_vs_episodes_{train_date}.png")
    else:
        info("No rewards history available — skipping reward plot")
    print()

    if True:  # pylint: disable=using-constant-test
        section("DQN-Agent Evaluation", width=80)

        # Re-seed all RNGs so the evaluation is isolated from the
        # random-baseline experiment that consumed RNG state above.
        tf.keras.utils.set_random_seed(SEED)
        env.action_space.seed(SEED)

        # Dimension limits for normalisation (must match training)
        _max_res = env.max_resources
        _max_seq = env.max_seq_length
        _max_sev = env.max_severity

        # ── Parity evaluation: matches optimize_dqn.objective.qf exactly ──
        # Pure deterministic argmax over masked Q-values.  This is the
        # function whose mean_perf is reported as the Optuna trial value, so
        # it must be used here to verify Colab → local reproducibility.
        info("Running parity evaluation (matches Optuna trial mean_perf)...")

        def parity_qf(_env, state, _seqid):
            norm_s = normalize_state(state, _max_res, _max_seq, _max_sev)
            q_vals = dqn_model(norm_s[numpy.newaxis], training=False).numpy()[0].copy()
            response, _conf, _rt_h, _rt_r = dqn_agent_meta_cognitive(
                q_vals, state[0], 10000
            )
            return response

        _, parity_perfs, _ = run_experiment(
            env, parity_qf, False, trials_per_sequence, sevs
        )
        parity_mean = float(numpy.mean(parity_perfs))
        list_item(f"Parity mean_perf (matches Optuna): {parity_mean:.6f}")
        if best_info is not None:
            _expected = best_info['mean_perf']
            _delta = abs(parity_mean - _expected)
            list_item(f"Optuna reported mean_perf:         {_expected:.6f}")
            # Tolerance bands account for GPU ↔ CPU kernel drift (Colab
            # cuDNN vs Windows generic kernels), which empirically reaches
            # ~3% mean-perf delta even with identical weights / seeds.
            # Anything above 5% likely indicates a real bug.
            if _delta < 1e-6:
                _verdict = 'OK (bit-exact)'
            elif _delta < 1e-3:
                _verdict = 'within float tolerance'
            elif _delta < 5e-2:
                _verdict = 'GPU↔CPU kernel drift (expected)'
            else:
                _verdict = 'MISMATCH — check TF version / hardware / weights'
            list_item(f"|Δ| = {_delta:.6f}  ({_verdict})")
        print()

        info("Running evaluation experiment with trained agent...")
        confsrl = []

        def eval_qf(_env, state, _seqid):
            """Select the best action using the trained Q-network with meta-cognitive confidence."""
            norm_s = normalize_state(state, _env.max_resources,
                                     _env.max_seq_length, _env.max_severity)
            q_vals = dqn_model(norm_s[numpy.newaxis], training=False).numpy()[0].copy()

            _response, confidence, _rt_hold, _rt_release = dqn_agent_meta_cognitive(
                q_vals, state[0], 10000
            )

            if (state[0] == 0):
                confidence = -1.0

            confsrl.append(confidence)
            return _response

        seqs, perfs, _ = run_experiment(env, eval_qf, False, trials_per_sequence, sevs)
        success("Evaluation experiment completed")

        info("Generating performance visualizations...")

        _fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(seqs, color='#d62728', linewidth=2.5, marker='o', markersize=5, label='DQN-Agent')
        ax.set_xlabel('Trial', fontsize=12, fontweight='bold')
        ax.set_ylabel('Final Severity Achieved', fontsize=12, fontweight='bold')
        ax.set_title('DQN-Agent Evaluation: Severity per Sequence', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=11)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                train_dir,
                f'dqn_agent_sequence_performance_{train_date}.png'),
            dpi=150,
            bbox_inches='tight')
        plt.close()
        list_item(f"Saved: dqn_agent_sequence_performance_{train_date}.png")

        _fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(perfs, color='#9467bd', linewidth=2.5, marker='s', markersize=5, label='DQN-Agent')
        ax.set_ylabel('Normalised Performance (0-1)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Trial', fontsize=12, fontweight='bold')
        ax.set_title('DQN-Agent Evaluation: Normalised Performance per Sequence',
                      fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, 64)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=11)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                train_dir,
                f'dqn_agent_normalised_performance_{train_date}.png'),
            dpi=150,
            bbox_inches='tight')
        plt.close()
        list_item(f"Saved: dqn_agent_normalised_performance_{train_date}.png")

        cumperfs = numpy.cumsum(perfs)
        Domain = numpy.arange(1, 1 + 64)
        _fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(cumperfs / Domain, color='#8c564b', linewidth=2.5, marker='^', markersize=5, label='DQN-Agent')
        ax.set_ylabel('Cumulative Normalised Performance', fontsize=12, fontweight='bold')
        ax.set_xlabel('Trial', fontsize=12, fontweight='bold')
        ax.set_title('DQN-Agent Evaluation: Cumulative Performance Trend',
                      fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0.5, 1.05)
        ax.set_xlim(0, 64)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=11)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                train_dir,
                f'dqn_agent_cumulative_performance_{train_date}.png'),
            dpi=150,
            bbox_inches='tight')
        plt.close()
        list_item(f"Saved: dqn_agent_cumulative_performance_{train_date}.png")

        _fig, ax = plt.subplots(figsize=(14, 5))
        ax.scatter(
            numpy.asarray(
                range(
                    len(confsrl))),
            confsrl,
            color='#1f77b4',
            s=40,
            alpha=0.6,
            edgecolors='navy',
            linewidth=0.5)
        ax.set_title('DQN-Agent: Decision Confidence Scores During Evaluation',
                      fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('Confidence', fontsize=12, fontweight='bold')
        ax.set_xlabel('Trial Number', fontsize=12, fontweight='bold')
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(-10, 360)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(train_dir, f'dqn_agent_confidences_{train_date}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        list_item(f"Saved: dqn_agent_confidences_{train_date}.png")

        confsrl_arr = numpy.asarray(confsrl, dtype=numpy.float32)

        val_confidences = numpy.arange(11, dtype=numpy.float32) / 10.0
        _confsrl_hist = numpy.histogram(confsrl_arr, bins=val_confidences)

        plot_confidences(confsrl_arr, 'Confidences', Show=False)

        numpy.save(os.path.join(train_dir, f'confsrl_{train_date}.npy'), confsrl_arr)

        confsrl_arr = confsrl_arr[confsrl_arr != -1]

        print(confsrl_arr)

        I = confsrl_arr
        rescaled = (I - numpy.min(I)) * ((1.0 - 0.0) / (numpy.max(I) - numpy.min(I))) + 0.0
        remapconfrl = numpy.clip(rescaled, 0.0, 1.0)

        print(remapconfrl.shape)

        _fig, ax = plt.subplots(figsize=(14, 5))
        ax.scatter(
            numpy.asarray(
                range(
                    remapconfrl.shape[0])),
            remapconfrl,
            color='#2ca02c',
            s=40,
            alpha=0.6,
            edgecolors='darkgreen',
            linewidth=0.5)
        ax.set_ylabel('Remapped Confidence (0-1)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Trial Number', fontsize=12, fontweight='bold')
        ax.set_title('DQN-Agent: Normalised Confidence Scores', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(-10, 360)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                train_dir,
                f'dqn_agent_remapped_confidences_{train_date}.png'),
            dpi=150,
            bbox_inches='tight')
        plt.close()

        plot_confidences(remapconfrl, 'Remapped Confidences', Show=False)

    section("Training Complete", width=80)
    success("DQN-Agent training pipeline finished successfully!")
    info(f"Output directory: {train_dir}")
    print()
#
# END OF 'main()


if __name__ == '__main__':
    main()
