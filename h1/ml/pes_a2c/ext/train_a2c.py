'''
pes_a2c - Pandemic Experiment Scenario: A2C Training Pipeline

Trains an Advantage Actor-Critic (A2C) agent on the Pandemic environment
and evaluates it against a random-player baseline.

Pipeline stages
---------------
1. Load training data (initial_severity.csv, sequence_lengths.csv)
2. Run random-player baseline and save performance plots
3. Train A2C agent (default 250 000 episodes, configurable via CLI)
   Hyperparameters come from Bayesian optimisation trial #90 (2026-04-23,
   mean_perf = 0.887236).
4. Save Keras Actor/Critic models, rewards history, and training config
   to a dated directory
5. Evaluate trained agent on the same sequences and generate
   performance / confidence visualisations

Usage
-----
::

    # 1. Train with the hyperparameters baked into CONFIG.py
    python -m ml.pes_a2c.ext.train_a2c [num_episodes]

    # 2. Reproduce the best trial of a Bayesian optimisation study
    #    (reads params + per-trial seed from the SQLite DB written by
    #     `python -m ml.pes_a2c.ext.optimize_a2c`).  Use this after copying
    #    the optimisation directory from Colab to your local machine.
    python -m ml.pes_a2c.ext.train_a2c --from-best 2026-04-20

The ``--from-best`` flag is the supported way to make local ``mean_perf``
match the Colab ``mean_perf`` of the best trial.  Bit-exact reproduction
requires identical TF version AND identical hardware (CPU vs GPU may differ
in the LSB).  When that is impossible, copy the Actor ``.keras`` file
directly from Colab to ``pes_a2c/inputs/ac_actor.keras``.
'''

##########################
##  Imports externos    ##
##########################
import glob
import json
import os
import sys
import numpy
import warnings
from datetime import datetime

# Default: pin TF to CPU.  Set MPES_USE_GPU=1 (e.g. on Colab) to use GPU.
if os.environ.get('MPES_USE_GPU', '0') != '1':
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf  # pylint: disable=unused-import  # noqa: E402, F401
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes as MplAxes  # noqa: E402

##########################
##  Imports internos    ##
##########################
from .pandemic import Pandemic, ac_agent_meta_cognitive, run_experiment, A2CTraining
from .ac_model import normalize_state
from ..src.terminal_utils import header, section, success, info, list_item
from .tools import plot_confidences, convert_globalseq_to_seqs
from ..config.CONFIG import (SEED, AC_ACTOR_HIDDEN_UNITS, AC_CRITIC_HIDDEN_UNITS,
                             AC_ACTOR_LR, AC_CRITIC_LR, AC_DISCOUNT,
                             AC_ENTROPY_COEFF, AC_EPSILON_INITIAL,
                             AC_EPSILON_MIN, AC_MODEL_ACTOR_FILE,
                             AC_WARMUP_RATIO, AC_TARGET_RATIO,
                             AC_PENALTY_COEFF, AC_GAE_LAMBDA,
                             AC_MAX_GRAD_NORM, AC_LR_MIN_RATIO,
                             AC_SPEND_COST_COEFF, AC_LAST_ACTION_BIAS)
from .. import INPUTS_PATH


# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*Box.*precision lowered.*')
warnings.filterwarnings('ignore', message='.*A NumPy version.*SciPy.*')


###################################
##   Best-trial loader helpers  ##
###################################
import re as _re  # noqa: E402  (local helper, isolated namespace)
_DATE_RE = _re.compile(r'^\d{4}-\d{2}-\d{2}$')


def _parse_cli_args():
    """
    Parse ``train_a2c`` CLI arguments.

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


def _find_latest_opt_date():
    """Return the newest ``YYYY-MM-DD`` for which a Bayesian-opt directory
    exists under ``INPUTS_PATH`` that contains either a
    ``best_params_<date>.json`` sidecar **or** an
    ``optuna_study_<date>.db`` SQLite database.

    Resolution precedence (both are accepted; JSON preferred by
    :func:`_load_best_trial`):

    1. ``<date>_BAYESIAN_OPT/best_params_<date>.json``
    2. ``<date>_BAYESIAN_OPT/optuna_study_<date>.db``

    Returns
    -------
    str or None
        The lexicographically highest valid date, or ``None`` if no
        qualifying directory exists.  Used by ``main()`` to auto-resolve
        ``from_best_date`` so users do not need to type
        ``--from-best YYYY-MM-DD`` for the most recent run.
    """
    candidates: list[str] = []
    # Priority 1 — JSON sidecar (written by optimize_a2c at every new best)
    for path in glob.glob(os.path.join(INPUTS_PATH, '*_BAYESIAN_OPT', 'best_params_*.json')):
        date = os.path.basename(os.path.dirname(path)).split('_', 1)[0]
        if _DATE_RE.match(date):
            candidates.append(date)
    if candidates:
        return max(candidates)
    # Priority 2 — SQLite DB alone (optimization still running / JSON not yet written)
    for path in glob.glob(os.path.join(INPUTS_PATH, '*_BAYESIAN_OPT', 'optuna_study_*.db')):
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

      1. ``pes_a2c/inputs/<opt_date>_BAYESIAN_OPT/best_params_<opt_date>.json``
         (small JSON sidecar written by ``optimize_a2c.main``).
      2. ``pes_a2c/inputs/<opt_date>_BAYESIAN_OPT/optuna_study_<opt_date>.db``
         (full SQLite study DB; required if the JSON is absent).

    Parameters
    ----------
    opt_date : str
        Run date in ``YYYY-MM-DD`` form.

    Returns
    -------
    dict
        Mapping with keys: ``params`` (dict of hyperparameters),
        ``trial_seed`` (int), ``trial_number`` (int), ``mean_perf`` (float).
    """
    opt_dir = os.path.join(INPUTS_PATH, f'{opt_date}_BAYESIAN_OPT')
    params_file = os.path.join(opt_dir, f'best_params_{opt_date}.json')

    # 1. Prefer the JSON sidecar — no optuna dependency, copy-friendly.
    if os.path.isfile(params_file):
        with open(params_file, 'r', encoding='utf-8') as _f:
            payload = json.load(_f)
        return {
            'params':       dict(payload['hyperparameters']),
            'trial_seed':   int(payload['trial_seed']),
            'trial_number': int(payload['best_trial_number']),
            'mean_perf':    float(payload['mean_perf']),
        }

    # 2. Fall back to the SQLite study DB (legacy / partial copies).
    import optuna  # local import — only needed for --from-best
    db_file = os.path.join(opt_dir, f'optuna_study_{opt_date}.db')
    if not os.path.isfile(db_file):
        raise FileNotFoundError(
            f"Neither best_params_{opt_date}.json nor optuna_study_{opt_date}.db "
            f"found in {opt_dir}.\nCopy the {opt_date}_BAYESIAN_OPT directory "
            f"(or at least best_params_{opt_date}.json) from Colab into "
            f"pes_a2c/inputs/ first."
        )
    study = optuna.load_study(
        study_name=f'a2c_opt_{opt_date}',
        storage=f'sqlite:///{db_file}',
    )
    best = study.best_trial
    trial_seed = best.user_attrs.get('trial_seed')
    if trial_seed is None:
        # Pre-2026-04 studies didn't persist trial_seed; reconstruct it
        # using the per-trial seed formula in optimize_a2c.objective.
        trial_seed = SEED + int(best.number) + 1
    return {
        'params': dict(best.params),
        'trial_seed': int(trial_seed),
        'trial_number': int(best.number),
        'mean_perf': float(best.user_attrs.get('mean_perf', best.value)),
    }

def _load_direct_best_params() -> dict | None:
    """Load hyperparameters from the top-level ``inputs/best_params.json``.

    This file is written by ``optimize_a2c.main()`` each time a new best
    trial is found (mirrored from the dated copy inside the
    ``<date>_BAYESIAN_OPT/`` directory).  It can also be generated on
    demand with ``python -m ml.pes_a2c.ext.optimize_a2c --export-best``.

    Returns
    -------
    dict or None
        Same structure as :func:`_load_best_trial`:  keys ``params``
        (dict), ``trial_seed`` (int), ``trial_number`` (int),
        ``mean_perf`` (float).  Returns ``None`` if the file is absent.
    """
    direct = os.path.join(INPUTS_PATH, 'best_params.json')
    if not os.path.isfile(direct):
        return None
    with open(direct, 'r', encoding='utf-8') as _f:
        payload = json.load(_f)
    return {
        'params':       dict(payload['hyperparameters']),
        'trial_seed':   int(payload['trial_seed']),
        'trial_number': int(payload['best_trial_number']),
        'mean_perf':    float(payload['mean_perf']),
    }


###################################
##             Main             ###
###################################


def main():
    """Run the full A2C training and evaluation pipeline."""

    header("A2C TRAINING PIPELINE", width=80)

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
    train_dir = os.path.join(INPUTS_PATH, f'{train_date}_A2C_TRAIN')
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
    assert isinstance(ax, MplAxes)
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
    assert isinstance(ax, MplAxes)
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

    # ---- A2C Training ----
    section("A2C Training", width=80)

    # Parse CLI: optional num_episodes (positional int) and --from-best <date>.
    cli_num_episodes, from_best_date = _parse_cli_args()

    best_info: dict | None = None

    if from_best_date is None:
        # Priority 0 — top-level inputs/best_params.json written/mirrored
        # by optimize_a2c at every new best (or via --export-best).
        _direct = _load_direct_best_params()
        if _direct is not None:
            info("Auto-loading best params from inputs/best_params.json")
            info("  regenerate with: python -m ml.pes_a2c.ext.optimize_a2c --export-best")
            info("  override with:   python -m ml.pes_a2c.ext.train_a2c --from-best YYYY-MM-DD")
            best_info = _direct
        else:
            # Priority 1 / 2 — dated directory with JSON sidecar or SQLite DB.
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
        actor_lr         = bp.get('actor_lr',        AC_ACTOR_LR)
        critic_lr        = bp.get('critic_lr',       AC_CRITIC_LR)
        discount_factor  = bp.get('discount_factor', AC_DISCOUNT)
        entropy_coeff    = bp.get('entropy_coeff',   AC_ENTROPY_COEFF)
        # Optuna fixes the legacy ε-greedy params to neutral values (on-policy
        # sampling is used).  We propagate the SAME neutral values here so the
        # training trajectory is bit-identical to the Optuna trial.
        epsilon_initial  = bp.get('epsilon_initial', 0.0)
        epsilon_min      = bp.get('epsilon_min',     0.0)
        warmup_ratio     = bp.get('warmup_ratio',    0.0)
        target_ratio     = bp.get('target_ratio',    1.0)
        penalty_coeff    = bp.get('penalty_coeff',   AC_PENALTY_COEFF)
        gae_lambda       = bp.get('gae_lambda',      AC_GAE_LAMBDA)
        max_grad_norm    = bp.get('max_grad_norm',   AC_MAX_GRAD_NORM)
        lr_min_ratio     = bp.get('lr_min_ratio',    AC_LR_MIN_RATIO)
        spend_cost_coeff = bp.get('spend_cost_coeff', AC_SPEND_COST_COEFF)
        last_action_bias = bp.get('last_action_bias', AC_LAST_ACTION_BIAS)
        _ahd = bp.get('actor_hidden_dim',  AC_ACTOR_HIDDEN_UNITS[0])
        _chd = bp.get('critic_hidden_dim', AC_CRITIC_HIDDEN_UNITS[0])
        _nhl = bp.get('n_hidden_layers',   len(AC_ACTOR_HIDDEN_UNITS))
        actor_hidden_units  = [_ahd] * _nhl
        critic_hidden_units = [_chd] * _nhl
        num_episodes = cli_num_episodes if cli_num_episodes is not None \
                                        else int(bp.get('num_episodes', 250000))
        info(f"  Reproducing trial #{best_info['trial_number'] + 1}  "
             f"(reported mean_perf = {best_info['mean_perf']:.6f})")
        info(f"  Trial seed = {train_seed}")
    else:
        # Use hyperparameters from CONFIG.py (last best trial baked in).
        train_seed       = SEED
        actor_lr         = AC_ACTOR_LR
        critic_lr        = AC_CRITIC_LR
        discount_factor  = AC_DISCOUNT
        entropy_coeff    = AC_ENTROPY_COEFF
        epsilon_initial  = AC_EPSILON_INITIAL
        epsilon_min      = AC_EPSILON_MIN
        warmup_ratio     = AC_WARMUP_RATIO
        target_ratio     = AC_TARGET_RATIO
        penalty_coeff    = AC_PENALTY_COEFF
        gae_lambda       = AC_GAE_LAMBDA
        max_grad_norm    = AC_MAX_GRAD_NORM
        lr_min_ratio     = AC_LR_MIN_RATIO
        actor_hidden_units  = AC_ACTOR_HIDDEN_UNITS
        critic_hidden_units = AC_CRITIC_HIDDEN_UNITS
        spend_cost_coeff    = AC_SPEND_COST_COEFF
        last_action_bias    = AC_LAST_ACTION_BIAS
        num_episodes = cli_num_episodes if cli_num_episodes is not None else 250000

    info(f"Starting A2C training ({num_episodes:,} episodes)...")
    info(f"  Actor hidden units:  {actor_hidden_units}")
    info(f"  Critic hidden units: {critic_hidden_units}")
    info(f"  Actor LR:            {actor_lr}")
    info(f"  Critic LR:           {critic_lr}")
    info(f"  Entropy coeff:       {entropy_coeff}")
    info(f"  Discount (γ):        {discount_factor}")
    info(f"  Warmup ratio:        {warmup_ratio}")
    info(f"  Target ratio:        {target_ratio}")
    info(f"  PBRS coeff (β):      {penalty_coeff}")
    info(f"  GAE(λ):              {gae_lambda}")
    info(f"  Max grad norm:       {max_grad_norm}")
    info(f"  LR min ratio:        {lr_min_ratio}")
    info(f"  Seed:                {train_seed}")
    info("(This may take a while on CPU)")
    print()

    rewards, actor_model, _confsrl = A2CTraining(
        env, actor_lr, critic_lr,
        discount_factor, entropy_coeff,
        epsilon_initial, epsilon_min, num_episodes,
        actor_hidden=actor_hidden_units,
        critic_hidden=critic_hidden_units,
        seed=train_seed,
        compute_confidence=False,
        warmup_ratio=warmup_ratio,
        target_ratio=target_ratio,
        penalty_coeff=penalty_coeff,
        gae_lambda=gae_lambda,
        max_grad_norm=max_grad_norm,
        lr_min_ratio=lr_min_ratio,
        spend_cost_coeff=spend_cost_coeff,
        last_action_bias=last_action_bias,
    )
    print()
    success("Training completed")
    list_item(f"Actor parameters: {actor_model.count_params()}")
    list_item(f"Rewards history length: {len(rewards)}")

    info("Saving trained models...")

    # Save Keras models and rewards with date stamp
    actor_file = os.path.join(train_dir, f'ac_actor_{train_date}.keras')
    rewards_file = os.path.join(train_dir, f'rewards_{train_date}.npy')
    config_file = os.path.join(train_dir, f'training_config_{train_date}.txt')

    actor_model.save(actor_file)
    numpy.save(rewards_file, rewards)

    # Also save to the standard paths consumed by __main__.py / pygameMediator
    actor_model.save(os.path.join(INPUTS_PATH, AC_MODEL_ACTOR_FILE))
    numpy.save(os.path.join(INPUTS_PATH, 'rewards.npy'), rewards)

    # Create configuration file
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("A2C TRAINING CONFIGURATION\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Training Date: {train_date}\n")
        f.write(f"Training Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("A2C HYPERPARAMETERS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Actor Learning Rate:         {actor_lr}\n")
        f.write(f"Critic Learning Rate:        {critic_lr}\n")
        f.write(f"Discount Factor (γ):         {discount_factor}\n")
        f.write(f"Entropy Coefficient:         {entropy_coeff}\n")
        f.write(f"Initial Epsilon (ε):         {epsilon_initial}\n")
        f.write(f"Minimum Epsilon (ε_min):     {epsilon_min}\n")
        f.write(f"Number of Episodes:          {num_episodes:,}\n")
        f.write(f"Epsilon Decay:               Exponential (warm-up {warmup_ratio:.0%}, target {target_ratio:.0%})\n")
        f.write(f"PBRS Coefficient (β):        {penalty_coeff}\n")
        f.write(f"GAE(λ):                      {gae_lambda}\n")
        f.write(f"Max Gradient Norm:           {max_grad_norm}\n")
        f.write(f"LR Min Ratio (cosine):       {lr_min_ratio}\n")
        f.write(f"Actor Hidden Units:          {actor_hidden_units}\n")
        f.write(f"Critic Hidden Units:         {critic_hidden_units}\n")
        f.write(f"Training Seed:               {train_seed}\n\n")

        f.write("NETWORK ARCHITECTURE\n")
        f.write("-" * 80 + "\n")
        f.write(f"State Dimension:             3 (resources, trial, severity)\n")
        f.write(f"Action Dimension:            {actor_model.output_shape[-1]}\n")
        f.write(f"Actor Parameters:            {actor_model.count_params()}\n")
        f.write(f"Input normalisation:         [res/30, trial/10, sev/9]\n\n")

        f.write("OUTPUT FILES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Actor Model File:            ac_actor_{train_date}.keras\n")
        f.write(f"Rewards File:                rewards_{train_date}.npy\n")
        f.write(f"Configuration File:          training_config_{train_date}.txt\n\n")

        f.write("DESCRIPTION\n")
        f.write("-" * 80 + "\n")
        f.write("Advantage Actor-Critic (A2C) trained on the Pandemic Scenario.\n")
        f.write("The Actor maps normalised (resources, trial, severity) states\n")
        f.write("to action probabilities via softmax for 11 resource allocations.\n")
        f.write("The Critic estimates state-value V(s) for advantage computation.\n")
        f.write("The rewards file contains average reward progression every 10,000 episodes.\n")

    success(f"Actor model saved to ac_actor_{train_date}.keras")
    success(f"Rewards saved to rewards_{train_date}.npy")
    success(f"Configuration saved to training_config_{train_date}.txt")
    list_item(f"Training Directory: {train_dir}")
    print()

    section("Training Performance Analysis", width=80)
    info("Generating reward history visualization...")

    __fig, ax = plt.subplots(figsize=(12, 6))
    assert isinstance(ax, MplAxes)
    ax.plot(100 * (numpy.arange(len(rewards)) + 1), rewards, color='#2ca02c', linewidth=2.5, label='Average Reward')
    ax.fill_between(100 * (numpy.arange(len(rewards)) + 1), rewards, alpha=0.2, color='#2ca02c')
    ax.set_xlabel('Episodes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
    ax.set_title('A2C Training: Average Reward Progression', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(train_dir, f'ac_agent_rewards_vs_episodes_{train_date}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    list_item(f"Saved: ac_agent_rewards_vs_episodes_{train_date}.png")
    print()

    if True:  # pylint: disable=using-constant-test
        section("A2C Agent Evaluation", width=80)

        # Dimension limits for normalisation (must match training)
        _max_res = env.available_resources_states - 1
        _max_tri = env.trial_no_states - 1
        _max_sev = env.severity_states - 1

        # ── Parity evaluation: identical to optimize_a2c.objective.qf ──────
        # Pure argmax over masked probabilities, no RNG calls.  This is the
        # function whose mean_perf is reported as the Optuna trial value, so
        # it must be used here to verify Colab → local reproducibility.
        info("Running parity evaluation (matches Optuna trial mean_perf)...")

        def parity_qf(_env, state, _seqid):
            s_norm = normalize_state(state, _max_res, _max_tri, _max_sev)
            probs = actor_model(s_norm[numpy.newaxis, :], training=False)[0].numpy()
            options = probs.copy()
            o = numpy.arange(len(options), dtype=numpy.float32)
            options[o > state[0]] = 0.00001
            return int(numpy.argmax(options))

        _, parity_perfs, _ = run_experiment(env, parity_qf, False, trials_per_sequence, sevs)
        parity_mean = float(numpy.mean(parity_perfs))
        list_item(f"Parity mean_perf (matches Optuna): {parity_mean:.6f}")
        if from_best_date is not None:
            _expected = best_info['mean_perf']  # noqa: F821
            _delta = abs(parity_mean - _expected)
            list_item(f"Optuna reported mean_perf:         {_expected:.6f}")
            list_item(f"|Δ| = {_delta:.6f}  "
                      f"({'OK' if _delta < 1e-6 else 'within float tolerance' if _delta < 1e-3 else 'MISMATCH — check TF version / hardware'})")
        print()

        info("Running evaluation experiment with trained agent (with confidences)...")
        confsrl = []

        def eval_qf(_env, state, _seqid):
            """Select the best action using the A2C Actor with meta-cognitive confidence."""
            s_norm = normalize_state(state, _max_res, _max_tri, _max_sev)
            probs = actor_model(s_norm[numpy.newaxis, :], training=False)[0].numpy()

            _response, confidence, _rt_hold, _rt_release = ac_agent_meta_cognitive(
                probs, state[0], 10000
            )

            if state[0] == 0:
                confidence = -1.0

            confsrl.append(confidence)
            return _response

        seqs, perfs, _ = run_experiment(env, eval_qf, False, trials_per_sequence, sevs)
        success("Evaluation experiment completed")

        info("Generating performance visualizations...")

        _fig, ax = plt.subplots(figsize=(12, 6))
        assert isinstance(ax, MplAxes)
        ax.plot(seqs, color='#d62728', linewidth=2.5, marker='o', markersize=5, label='A2C Agent')
        ax.set_xlabel('Trial', fontsize=12, fontweight='bold')
        ax.set_ylabel('Final Severity Achieved', fontsize=12, fontweight='bold')
        ax.set_title('A2C Agent Evaluation: Severity per Sequence', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=11)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                train_dir,
                f'ac_agent_sequence_performance_{train_date}.png'),
            dpi=150,
            bbox_inches='tight')
        plt.close()
        list_item(f"Saved: ac_agent_sequence_performance_{train_date}.png")

        _fig, ax = plt.subplots(figsize=(12, 6))
        assert isinstance(ax, MplAxes)
        ax.plot(perfs, color='#9467bd', linewidth=2.5, marker='s', markersize=5, label='A2C Agent')
        ax.set_ylabel('Normalised Performance (0-1)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Trial', fontsize=12, fontweight='bold')
        ax.set_title('A2C Agent Evaluation: Normalised Performance', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, 64)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=11)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                train_dir,
                f'ac_agent_normalised_performance_{train_date}.png'),
            dpi=150,
            bbox_inches='tight')
        plt.close()
        list_item(f"Saved: ac_agent_normalised_performance_{train_date}.png")

        cumperfs = numpy.cumsum(perfs)
        Domain = numpy.arange(1, 1 + 64)
        _fig, ax = plt.subplots(figsize=(12, 6))
        assert isinstance(ax, MplAxes)
        ax.plot(cumperfs / Domain, color='#8c564b', linewidth=2.5, marker='^', markersize=5, label='A2C Agent')
        ax.set_ylabel('Cumulative Normalised Performance', fontsize=12, fontweight='bold')
        ax.set_xlabel('Trial', fontsize=12, fontweight='bold')
        ax.set_title('A2C Agent Evaluation: Cumulative Performance Trend', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0.5, 1.05)
        ax.set_xlim(0, 64)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=11)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                train_dir,
                f'ac_agent_cumulative_performance_{train_date}.png'),
            dpi=150,
            bbox_inches='tight')
        plt.close()
        list_item(f"Saved: ac_agent_cumulative_performance_{train_date}.png")

        _fig, ax = plt.subplots(figsize=(14, 5))
        assert isinstance(ax, MplAxes)
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
        ax.set_title('A2C Agent: Decision Confidence Scores During Evaluation', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('Confidence', fontsize=12, fontweight='bold')
        ax.set_xlabel('Trial Number', fontsize=12, fontweight='bold')
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(-10, 360)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(train_dir, f'ac_agent_confidences_{train_date}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        list_item(f"Saved: ac_agent_confidences_{train_date}.png")

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
        assert isinstance(ax, MplAxes)
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
        ax.set_title('A2C Agent: Normalised Confidence Scores', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(-10, 360)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                train_dir,
                f'ac_agent_remapped_confidences_{train_date}.png'),
            dpi=150,
            bbox_inches='tight')
        plt.close()

        plot_confidences(remapconfrl, 'Remapped Confidences', Show=False)

    section("Training Complete", width=80)
    success("A2C training pipeline finished successfully!")
    info(f"Output directory: {train_dir}")
    print()
#
# END OF 'main()


if __name__ == '__main__':
    main()
