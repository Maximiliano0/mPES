'''
pes_a2c - Pandemic Experiment Scenario

Bayesian Optimization of A2C hyperparameters using Optuna.

Optimizes (configurable via AC_OPTIMIZE_MODE or --mode flag):
    - **full**: 14 hyperparameters (8 base + 6 improvements). The legacy ε
      schedule (`epsilon_initial`, `epsilon_min`, `warmup_ratio`,
      `target_ratio`) is fixed at neutral values because A2C training is
      now pure on-policy (softmax sampling).
    - **improvements_only**: 6 improvement params only (8 base params fixed
      at the CONFIG.py values that come from trial #90).
Objective: maximize mean normalised performance over the 64 evaluation sequences.

The evaluation uses infeasible-action masking (actions > available resources are
suppressed before argmax) so that the metric matches the behaviour of the A2C
agent in ``__main__.py``.  The best Actor model found during the search is
preserved in memory and saved directly, avoiding a lossy re-training step.

Usage:
    python3 -m ml.pes_a2c.ext.optimize_a2c [n_trials] [--resume YYYY-MM-DD] [--mode full|improvements_only]

    n_trials : int, optional
        Number of Bayesian optimization trials (default: 30).
    --resume YYYY-MM-DD : str, optional
        Resume a previous optimization run stored under that date.
    --mode : str, optional
        Optimization mode: 'full' or 'improvements_only' (default from CONFIG).

Search space (full mode — base params, 8 sampled):
    actor_lr             ∈ [1e-4, 1e-2]         (log scale)
    critic_lr            ∈ [1e-4, 1e-2]         (log scale)
    discount_factor      ∈ [0.85, 0.995]
    entropy_coeff        ∈ [0.0, 0.1]           (linear; 0 disables bonus)
    num_episodes         ∈ [50000, 250000]      (step=25000)
    actor_hidden_dim     ∈ {32, 64, 128, 256}   (categorical)
    critic_hidden_dim    ∈ {32, 64, 128, 256}   (categorical)
    n_hidden_layers      ∈ {1, 2, 3}
    epsilon_initial      = 0.0  (fixed; on-policy training)
    epsilon_min          = 0.0  (fixed; on-policy training)
Search space (improvement params — always sampled, 6):
    penalty_coeff        ∈ [0.0, 0.3]           (linear; 0 disables PBRS)
    gae_lambda           ∈ [0.90, 0.99]
    max_grad_norm        ∈ [0.3, 1.5]
    lr_min_ratio         ∈ [0.05, 0.25]
    spend_cost_coeff     ∈ [0.0, 0.05]          (per-action training-only cost)
    last_action_bias     ∈ [-2.0, 0.0]          (init logit of "spend max" action)
    warmup_ratio         = 0.0  (fixed; legacy ε schedule unused)
    target_ratio         = 1.0  (fixed; legacy ε schedule unused)

Outputs (saved to INPUTS_PATH/<date>_BAYESIAN_OPT/):
    - ac_best_<date>.keras             : Keras Actor model from the best trial
    - rewards_best_<date>.npy          : Reward history of the best training run
    - optimization_results_<date>.txt  : Full report (1-based trial #)
    - optimization_history_<date>.png  : Convergence plot (1-based trial #)
    - hyperparameter_importances_<date>.png: Parameter importance plot
    - optuna_study_<date>.db           : SQLite database for resumable studies

Note:
    Trial numbering in reports and plots uses 1-based indexing to match
    the trial_id in the SQLite database.  Optuna internally uses 0-based
    trial.number; the +1 offset is applied at report-generation time.
    Each trial uses an independent seed ``SEED + trial.number + 1`` so
    repeated sampling of the same configuration produces independent
    stochastic replicates.
'''

##########################
##  Imports externos    ##
##########################
import gc
import json
import os
import sys
import time
import numpy
import warnings
import optuna
from datetime import datetime

# Default: pin TF to CPU.  Set MPES_USE_GPU=1 (e.g. on Colab) to use GPU.
if os.environ.get('MPES_USE_GPU', '0') != '1':
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
os.environ.setdefault('TF_CUDNN_DETERMINISTIC', '1')

import tensorflow as tf  # noqa: E402  (after env var)
import matplotlib  # noqa: E402
matplotlib.use('Agg')  # non-interactive backend; prevents NoneType canvas crash on headless servers
import matplotlib.pyplot as plt  # noqa: E402  (must be after TF on Windows)
from matplotlib.axes import Axes as MplAxes  # noqa: E402

# Suppress TF retracing warnings (expected: each Optuna trial creates a fresh
# tf.function wrapper to isolate the graph per trial) and deprecation notices
# from keras.backend.clear_session() calling legacy tf.reset_default_graph.
import logging as _logging
_logging.getLogger('tensorflow').setLevel(_logging.ERROR)

##########################
##  Imports internos    ##
##########################
from .pandemic import Pandemic, run_experiment, A2CTraining
from .ac_model import build_actor, normalize_state
from ..src.terminal_utils import header, section, success, info, list_item
from .tools import convert_globalseq_to_seqs
from ..config.CONFIG import (SEED, AC_ACTOR_LR, AC_CRITIC_LR, AC_DISCOUNT,
                             AC_ENTROPY_COEFF, AC_EPSILON_INITIAL,
                             AC_EPSILON_MIN,
                             AC_ACTOR_HIDDEN_UNITS, AC_CRITIC_HIDDEN_UNITS,
                             AC_OPTIMIZE_MODE,
                             AC_WARMUP_RATIO, AC_TARGET_RATIO,
                             AC_PENALTY_COEFF, AC_GAE_LAMBDA,
                             AC_MAX_GRAD_NORM, AC_LR_MIN_RATIO)
from .. import INPUTS_PATH

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*Box.*precision lowered.*')
warnings.filterwarnings('ignore', message='.*A NumPy version.*SciPy.*')


###################################
##    Global evaluation data     ##
###################################
# Loaded once at startup and reused by every trial
_trials_per_sequence = None
_sevs = None
_number_cities_prob = None
_severity_prob = None

# Store best model weights/rewards during optimization to avoid lossy retraining
_best_artifacts: dict = {'weights': None, 'hidden_units': None,
                         'critic_hidden_units': None,
                         'rewards': None, 'value': float('-inf'),
                         'trial_seed': None, 'trial_number': None}

# Set by main() so objective() can persist _best_artifacts to disk
_opt_dir: str = ''


###################################
##   Pickle-free persistence    ##
###################################
# CWE-502 hardening: never deserialise the Optuna best-artifact via pickle.
# Actor weights are split into named ndarrays inside a single .npz (loaded
# with allow_pickle=False) and the scalar metadata into a sibling .json file.
# Loaders refuse legacy ``_best_weights.pkl`` files.
_BEST_NPZ_BASENAME = '_best_artifacts.npz'
_BEST_META_BASENAME = '_best_artifacts.json'


def _save_best_artifacts(opt_dir: str, artifacts: dict) -> None:
    """Persist *artifacts* to ``opt_dir`` without using ``pickle``."""
    npz_path = os.path.join(opt_dir, _BEST_NPZ_BASENAME)
    meta_path = os.path.join(opt_dir, _BEST_META_BASENAME)
    weights = artifacts['weights'] or []
    arrays = {f'w_{i}': numpy.asarray(w) for i, w in enumerate(weights)}
    arrays['rewards'] = numpy.asarray(artifacts['rewards'])
    # numpy stubs mis-type savez **kwds; runtime accepts ndarrays just fine.
    numpy.savez(npz_path, **arrays)  # type: ignore[arg-type]
    with open(meta_path, 'w', encoding='utf-8') as _f:
        json.dump({
            'value': float(artifacts['value']),
            'hidden_units': list(artifacts['hidden_units'] or []),
            'critic_hidden_units': list(artifacts['critic_hidden_units'] or []),
            'trial_seed': int(artifacts.get('trial_seed') or 0),
            'trial_number': int(artifacts.get('trial_number') or -1),
            'n_weights': len(weights),
        }, _f, indent=2)


def _load_best_artifacts(opt_dir: str, artifacts: dict) -> bool:
    """Load best artifacts from ``opt_dir`` (mutates *artifacts*).

    Returns ``True`` on success. Loaded with ``allow_pickle=False``.
    """
    npz_path = os.path.join(opt_dir, _BEST_NPZ_BASENAME)
    meta_path = os.path.join(opt_dir, _BEST_META_BASENAME)
    if not (os.path.isfile(npz_path) and os.path.isfile(meta_path)):
        return False
    with open(meta_path, 'r', encoding='utf-8') as _f:
        meta = json.load(_f)
    n = int(meta.get('n_weights', 0))
    with numpy.load(npz_path, allow_pickle=False) as _data:
        artifacts['weights'] = [_data[f'w_{i}'] for i in range(n)]
        artifacts['rewards'] = list(_data['rewards'])
    artifacts['value'] = float(meta['value'])
    artifacts['hidden_units'] = list(meta.get('hidden_units', []))
    artifacts['critic_hidden_units'] = list(meta.get('critic_hidden_units', []))
    artifacts['trial_seed'] = int(meta.get('trial_seed', 0))
    artifacts['trial_number'] = int(meta.get('trial_number', -1))
    return True

# Optimization mode: set by main() from AC_OPTIMIZE_MODE / --mode flag,
# read by objective() to decide which hyperparameters to sample.
_optimize_mode: str = ''


def _load_evaluation_data():
    """Load sequence lengths, severities and their probability distributions."""
    global _trials_per_sequence, _sevs, _number_cities_prob, _severity_prob

    _trials_per_sequence = numpy.loadtxt(
        os.path.join(INPUTS_PATH, 'sequence_lengths.csv'), delimiter=','
    )
    all_severities = numpy.loadtxt(
        os.path.join(INPUTS_PATH, 'initial_severity.csv'), delimiter=','
    )
    _sevs = convert_globalseq_to_seqs(_trials_per_sequence, all_severities)

    val_cities, count_cities = numpy.unique(_trials_per_sequence, return_counts=True)
    _number_cities_prob = numpy.asarray((val_cities, count_cities / len(_trials_per_sequence))).T

    val_severity, count_severity = numpy.unique(all_severities, return_counts=True)
    _severity_prob = numpy.asarray((val_severity, count_severity / len(all_severities))).T


###################################
##     Objective function        ##
###################################
def objective(trial: optuna.Trial) -> float:
    """Train an A2C agent with sampled hyperparameters and return mean normalised performance.

    Called by Optuna on each trial.  In ``'full'`` mode, samples all 16
    hyperparameters; in ``'improvements_only'`` mode, fixes the 8 base
    hyperparameters at CONFIG values and samples only the 6 improvement
    parameters.  Trains a fresh A2C agent via :func:`A2CTraining`, evaluates
    it on the 64 fixed sequences with infeasible-action masking, and returns
    the mean normalised performance.

    The best model weights are cached in ``_best_artifacts`` to avoid a lossy
    retraining step at the end of the study.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object used for hyperparameter sampling.

    Returns
    -------
    float
        Mean normalised performance, higher is better; maximised by the study.

    Notes
    -----
    - ``tf.keras.backend.clear_session()`` and ``gc.collect()`` are called
      after each trial to prevent memory growth over 100+ trials.
    - Additional statistics (std, min, max performance) are stored as user
      attributes on the trial for later reporting.
    """
    # --- Sample hyperparameters (base) ---
    if _optimize_mode == 'full':
        actor_lr = trial.suggest_float('actor_lr', 1e-4, 1e-2, log=True)
        critic_lr = trial.suggest_float('critic_lr', 1e-4, 1e-2, log=True)
        discount_factor = trial.suggest_float('discount_factor', 0.85, 0.995)
        # Linear scale (includes 0) so the optimizer can disable the
        # entropy bonus entirely if it harms performance.
        entropy_coeff = trial.suggest_float('entropy_coeff', 0.0, 0.1)
        # ε-greedy is disabled (training is now pure on-policy via softmax
        # sampling).  These values are forwarded for API compatibility but
        # are unused inside A2CTraining.
        epsilon_initial = 0.0
        epsilon_min = 0.0
        # Wider episode budget so the optimizer can find policies that need
        # longer training horizons.
        # Tightened ceiling so screening trials finish in <30 min on
        # Colab CPU.  The 250 000-episode upper bound is the previous
        # best trial's value; longer horizons can be searched separately
        # by re-running with a custom CLI override after a champion is
        # identified.  The 25 000 step keeps the discrete grid coarse
        # enough for TPE to model.
        num_episodes = trial.suggest_int('num_episodes', 50000, 250000, step=25000)
        # Categorical widths so the optimizer can pick larger nets when
        # 32/64-unit single-layer models bottleneck performance.
        actor_hidden_dim = trial.suggest_categorical('actor_hidden_dim', [32, 64, 128, 256])
        critic_hidden_dim = trial.suggest_categorical('critic_hidden_dim', [32, 64, 128, 256])
        n_hidden_layers = trial.suggest_int('n_hidden_layers', 1, 3)
    else:
        # 'improvements_only': fix base hyperparameters at CONFIG values (trial #90)
        actor_lr = AC_ACTOR_LR
        critic_lr = AC_CRITIC_LR
        discount_factor = AC_DISCOUNT
        entropy_coeff = AC_ENTROPY_COEFF
        epsilon_initial = 0.0
        epsilon_min = 0.0
        num_episodes = 250000
        actor_hidden_dim = AC_ACTOR_HIDDEN_UNITS[0]
        critic_hidden_dim = AC_CRITIC_HIDDEN_UNITS[0]
        n_hidden_layers = len(AC_ACTOR_HIDDEN_UNITS)

    actor_hidden = [actor_hidden_dim] * n_hidden_layers
    critic_hidden = [critic_hidden_dim] * n_hidden_layers

    # --- Sample improvement hyperparameters (always sampled) ---
    # warmup_ratio / target_ratio are kept for API compatibility but unused
    # in the on-policy path.  Fix them so they don't waste search budget.
    warmup_ratio = 0.0
    target_ratio = 1.0
    # Linear scale including 0 so PBRS can be disabled if it biases the
    # policy away from the unshaped objective.
    penalty_coeff = trial.suggest_float('penalty_coeff', 0.0, 0.3)
    gae_lambda = trial.suggest_float('gae_lambda', 0.90, 0.99)
    max_grad_norm = trial.suggest_float('max_grad_norm', 0.3, 1.5)
    lr_min_ratio = trial.suggest_float('lr_min_ratio', 0.05, 0.25)

    # --- Symmetry-breaking & spending-cost (always sampled) ---
    # The naive ``argmax(softmax) → max-feasible`` collapse yields the
    # constant 0.857961 baseline observed in earlier trials.  These
    # two knobs let Optuna search the regime where the policy must
    # actively trade off severity reduction against resource spending.
    spend_cost_coeff = trial.suggest_float('spend_cost_coeff', 0.0, 0.05)
    last_action_bias = trial.suggest_float('last_action_bias', -2.0, 0.0)

    # --- Train ---
    env = Pandemic()
    env.number_cities_prob = _number_cities_prob  # type: ignore[assignment]
    env.severity_prob = _severity_prob  # type: ignore[assignment]
    env.verbose = False

    # Per-trial seed: distinct from SEED so trials are independent
    # replicates (Optuna otherwise sees zero per-config variance and
    # cannot estimate stochastic noise on the objective surface).
    trial_seed = SEED + int(trial.number) + 1

    # Optuna intermediate-value reporting for MedianPruner: receives
    # (avg_reward, episode) every 10 000 episodes. Returning True from
    # the callback aborts training when Optuna decides to prune.
    _pruned = {'flag': False}

    def _progress(avg_reward: float, step: int) -> bool:
        trial.report(avg_reward, step)
        if trial.should_prune():
            _pruned['flag'] = True
            return True
        return False

    rewards, actor_model, _ = A2CTraining(
        env, actor_lr, critic_lr,
        discount_factor, entropy_coeff,
        epsilon_initial, epsilon_min, num_episodes,
        actor_hidden=actor_hidden,
        critic_hidden=critic_hidden,
        seed=trial_seed,
        compute_confidence=False,
        verbose=False,
        warmup_ratio=warmup_ratio,
        target_ratio=target_ratio,
        penalty_coeff=penalty_coeff,
        gae_lambda=gae_lambda,
        max_grad_norm=max_grad_norm,
        lr_min_ratio=lr_min_ratio,
        spend_cost_coeff=spend_cost_coeff,
        last_action_bias=last_action_bias,
        progress_callback=_progress,
    )

    if _pruned['flag']:
        raise optuna.TrialPruned()

    # --- Evaluate on fixed sequences ---
    env_eval = Pandemic()
    env_eval.verbose = False

    max_res = env_eval.available_resources_states - 1
    max_tri = env_eval.trial_no_states - 1
    max_sev = env_eval.severity_states - 1

    # Compile the eval-time inference call to avoid per-step eager
    # dispatch overhead (~640 actor calls per trial × ~1 ms eager).
    @tf.function(reduce_retracing=True)
    def _actor_predict_eval(state_batch):
        return actor_model(state_batch, training=False)

    # ── Stochastic multi-replicate evaluation ──────────────────────
    # The previous deterministic ``argmax(masked_softmax)`` eval is
    # invariant to the *shape* of the policy distribution: any actor
    # whose mode falls on the last feasible action yields the same
    # 0.857961 score (verified on an untrained network with seed=7).
    # We replace it with stochastic sampling from the masked softmax,
    # averaged over N independent replicates with deterministic per-
    # trial seeds.  The objective now reflects the *full* distribution
    # the agent learned, exposing differences in entropy, calibration
    # and tail behaviour that argmax hides.
    n_eval_replicates = 8
    eval_rng = numpy.random.default_rng(int(trial_seed))
    action_dim = env_eval.action_space.n  # type: ignore[attr-defined]

    def _make_qf(rng):
        def qf(_env, state, _seqid):
            s_norm = normalize_state(state, max_res, max_tri, max_sev)
            probs = _actor_predict_eval(s_norm[numpy.newaxis, :])[0].numpy()
            mask = numpy.zeros(action_dim, dtype=numpy.float32)
            mask[:min(int(state[0]), action_dim - 1) + 1] = 1.0
            masked = probs * mask
            total = masked.sum()
            if total > 1e-8:
                masked = masked / total
            else:
                masked = mask / mask.sum()
            return int(rng.choice(action_dim, p=masked))
        return qf

    perfs_all: list[float] = []
    for _ in range(n_eval_replicates):
        sub_rng = numpy.random.default_rng(int(eval_rng.integers(0, 2**31 - 1)))
        qf = _make_qf(sub_rng)
        _, perfs, _ = run_experiment(env_eval, qf, False,
                                     _trials_per_sequence, _sevs,
                                     verbose=False)
        perfs_all.extend(perfs)
    mean_perf = float(numpy.mean(perfs_all))
    perfs_arr = numpy.asarray(perfs_all, dtype=numpy.float64)

    # Store extra info for later analysis
    trial.set_user_attr('mean_perf', mean_perf)
    trial.set_user_attr('std_perf', float(numpy.std(perfs_arr)))
    trial.set_user_attr('min_perf', float(numpy.min(perfs_arr)))
    trial.set_user_attr('max_perf', float(numpy.max(perfs_arr)))
    trial.set_user_attr('n_eval_replicates', int(n_eval_replicates))
    # Persist the per-trial seed so the trial can be reproduced bit-exact
    # (on the same TF version + hardware) by retraining with this seed.
    trial.set_user_attr('trial_seed', int(trial_seed))

    # Preserve the best model weights to avoid lossy retraining at the end
    global _best_artifacts
    if mean_perf > _best_artifacts['value']:
        _best_artifacts['weights'] = actor_model.get_weights()
        _best_artifacts['hidden_units'] = actor_hidden
        _best_artifacts['critic_hidden_units'] = critic_hidden
        _best_artifacts['rewards'] = list(rewards)
        _best_artifacts['value'] = mean_perf
        _best_artifacts['trial_seed'] = int(trial_seed)
        _best_artifacts['trial_number'] = int(trial.number)

        # Persist to disk so that --resume can recover without retraining.
        # Pickle-free format (CWE-502 hardening).
        if _opt_dir:
            _save_best_artifacts(_opt_dir, _best_artifacts)

    # ── Memory cleanup ──────────────────────────────────────────────
    # Each Optuna trial builds new Keras models and tf.function traces.
    # Without explicit cleanup TensorFlow retains stale graphs and
    # variables, causing gradual memory growth over 100+ trials.
    del actor_model, rewards, env, env_eval
    tf.keras.backend.clear_session()
    gc.collect()

    return mean_perf


###################################
##        Reporting              ##
###################################
def _save_report(study, opt_dir, opt_date, best_model, best_rewards):
    """Generate and save optimization results report and visualizations.

    Trial numbers are converted to 1-based indexing (trial.number + 1)
    so they match the trial_id column in the Optuna SQLite database.
    """

    best = study.best_trial

    # --- Text report ---
    report_file = os.path.join(opt_dir, f'optimization_results_{opt_date}.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BAYESIAN OPTIMIZATION RESULTS — A2C HYPERPARAMETERS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Date:              {opt_date}\n")
        f.write(f"Total trials:      {len(study.trials)}\n")
        f.write(f"Best trial:        #{best.number + 1}\n")
        f.write(f"Best objective:    {best.value:.6f}\n")
        f.write(f"  Mean perf:       {best.user_attrs.get('mean_perf', best.value):.6f}\n\n")

        f.write("BEST HYPERPARAMETERS\n")
        f.write("-" * 80 + "\n")
        for name, val in best.params.items():
            f.write(f"  {name:<25s} = {val}\n")
        f.write("\n")

        f.write("BEST TRIAL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Mean performance:   {best.user_attrs['mean_perf']:.6f}\n")
        f.write(f"  Std  performance:   {best.user_attrs['std_perf']:.6f}\n")
        f.write(f"  Min  performance:   {best.user_attrs['min_perf']:.6f}\n")
        f.write(f"  Max  performance:   {best.user_attrs['max_perf']:.6f}\n\n")

        # Reproducibility block — everything needed to reproduce mean_perf
        # locally with `python -m ml.pes_a2c.ext.train_a2c`.
        best_seed = best.user_attrs.get('trial_seed', 'UNKNOWN (pre-2026-04 study)')
        f.write("REPRODUCIBILITY (Colab → local PC)\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Best trial seed:    {best_seed}\n")
        f.write("  To reproduce this trial's mean_perf locally, run:\n")
        f.write(f"      python -m ml.pes_a2c.ext.train_a2c --from-best {opt_date}\n")
        f.write("  This loads the best params + seed from the Optuna study DB\n")
        f.write("  and trains with EXACTLY the same configuration.\n")
        f.write("  Caveat: bit-exact reproduction requires identical TF version\n")
        f.write("  AND identical hardware (CPU↔GPU floats may differ in the LSB).\n")
        f.write("  For guaranteed match, copy the Actor .keras file directly:\n")
        f.write(f"      cp <out>/ac_best_{opt_date}.keras pes_a2c/inputs/ac_actor.keras\n\n")

        f.write("ALL TRIALS\n")
        f.write("-" * 100 + "\n")
        # Header and columns depend on optimization mode
        param_names = list(study.best_trial.params.keys())
        header_parts = [f"{'#':>4s}", f"{'objective':>10s}", f"{'perf':>10s}"]
        for pn in param_names:
            header_parts.append(f"{pn:>14s}")
        f.write("  ".join(header_parts) + "\n")
        f.write("-" * 100 + "\n")
        for t in sorted(study.trials, key=lambda t: t.value if t.value is not None else -1, reverse=True):
            if t.value is None:
                continue
            p = t.params
            t_perf = t.user_attrs.get('mean_perf', t.value)
            row_parts = [f"{t.number + 1:4d}", f"{t.value:10.6f}", f"{t_perf:10.6f}"]
            for pn in param_names:
                val = p.get(pn, float('nan'))
                if isinstance(val, int):
                    row_parts.append(f"{val:14d}")
                else:
                    row_parts.append(f"{val:14.6f}")
            f.write("  ".join(row_parts) + "\n")

    success(f"Report saved: optimization_results_{opt_date}.txt")

    # --- Convergence plot ---
    try:
        plt.style.use('ggplot')
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(12, 6))
    assert isinstance(ax, MplAxes)
    trial_numbers = [t.number + 1 for t in study.trials if t.value is not None]
    trial_values = [t.value for t in study.trials if t.value is not None]

    # Running best
    running_best: list[float] = []
    current_best = -1.0
    for v in trial_values:
        current_best = max(current_best, v)
        running_best.append(current_best)

    ax.scatter(trial_numbers, trial_values, color='#1f77b4', s=50, alpha=0.6,
               edgecolors='navy', linewidth=0.5, label='Trial performance', zorder=3)
    ax.plot(trial_numbers, running_best, color='#d62728', linewidth=2.5,
            label='Best so far', zorder=4)
    ax.set_xlabel('Trial number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean normalised performance', fontsize=12, fontweight='bold')
    ax.set_title('Bayesian Optimisation: Convergence — A2C', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(opt_dir, f'optimization_history_{opt_date}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    list_item(f"Saved: optimization_history_{opt_date}.png")

    # --- Hyperparameter importance ---
    try:
        importances = optuna.importance.get_param_importances(study)
        names = list(importances.keys())
        values = list(importances.values())

        fig, ax = plt.subplots(figsize=(10, 5))
        assert isinstance(ax, MplAxes)
        ax.barh(names[::-1], values[::-1], color='#2ca02c', edgecolor='darkgreen', linewidth=0.5)
        ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
        ax.set_title('A2C Hyperparameter Importance', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        fig.savefig(os.path.join(opt_dir, f'hyperparameter_importances_{opt_date}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        list_item(f"Saved: hyperparameter_importances_{opt_date}.png")
    except Exception as e:
        info(f"Could not compute importances: {e}")

    # --- Save best model and rewards ---
    best_model.save(os.path.join(opt_dir, f'ac_best_{opt_date}.keras'))
    numpy.save(os.path.join(opt_dir, f'rewards_best_{opt_date}.npy'), best_rewards)
    success(f"Best Actor model saved: ac_best_{opt_date}.keras")
    success(f"Best rewards saved: rewards_best_{opt_date}.npy")

    # --- Sidecar JSON for cross-package uniformity (mirrors pes_ql/pes_dql) ---
    # train_a2c.py --from-best <date> reads this in preference to the SQLite DB
    # so users can reproduce mean_perf by copying just one small file.
    bp = best.params
    actor_hidden  = _best_artifacts.get('hidden_units')
    critic_hidden = _best_artifacts.get('critic_hidden_units')
    trial_seed = int(best.user_attrs.get('trial_seed', SEED + int(best.number) + 1))
    best_params_payload = {
        'opt_date':            opt_date,
        'best_trial_number':   int(best.number),
        'mean_perf':           float(best.user_attrs.get('mean_perf', best.value)),
        'std_perf':            float(best.user_attrs.get('std_perf', float('nan'))),
        'min_perf':            float(best.user_attrs.get('min_perf', float('nan'))),
        'max_perf':            float(best.user_attrs.get('max_perf', float('nan'))),
        'trial_seed':          trial_seed,
        'actor_hidden_units':  list(actor_hidden)  if actor_hidden  else None,
        'critic_hidden_units': list(critic_hidden) if critic_hidden else None,
        'hyperparameters':     {k: (int(v) if isinstance(v, (bool, int)) else
                                    (float(v) if isinstance(v, float) else v))
                                for k, v in bp.items()},
    }
    params_file = os.path.join(opt_dir, f'best_params_{opt_date}.json')
    with open(params_file, 'w', encoding='utf-8') as _f:
        json.dump(best_params_payload, _f, indent=2)
    success(f"Best params saved: best_params_{opt_date}.json")

    # --- Mirror to the standard input path so train_a2c.py picks it up
    # without requiring an explicit --from-best YYYY-MM-DD flag
    # (matches the pes_ql / pes_dql behavior). ---
    std_params = os.path.join(INPUTS_PATH, 'best_params.json')
    with open(std_params, 'w', encoding='utf-8') as _f:
        json.dump(best_params_payload, _f, indent=2)
    success(f"Mirrored to {std_params}")


###################################
##             Main              ##
###################################

import re as _re_opt  # noqa: E402  (local helper, avoid polluting module namespace)
_DATE_RE_OPT = _re_opt.compile(r'^\d{4}-\d{2}-\d{2}$')


def _export_best_params(opt_dir: str, opt_date: str, storage_override: str | None = None) -> None:
    """Read the best trial from an existing SQLite study and write ``best_params`` JSON files.

    Writes two files:

    * ``<opt_dir>/best_params_<opt_date>.json`` — dated sidecar (same format as
      the one written at optimization end).
    * ``<INPUTS_PATH>/best_params.json`` — top-level mirror consumed by
      ``train_a2c.py`` auto-load.

    Called by ``main()`` when ``--export-best`` is passed, so users can
    generate the JSON from a partially-completed (or finished) study at any time::

        python -m ml.pes_a2c.ext.optimize_a2c --export-best 2026-04-23

    Parameters
    ----------
    opt_dir : str
        Path to the ``<date>_BAYESIAN_OPT`` directory.
    opt_date : str
        Study date in ``YYYY-MM-DD`` format.
    storage_override : str or None
        Custom SQLite storage URL; defaults to ``sqlite:///<opt_dir>/optuna_study_<opt_date>.db``.
    """
    db_path = os.path.join(opt_dir, f'optuna_study_{opt_date}.db')
    storage = storage_override or f'sqlite:///{db_path}'

    if not storage_override and not os.path.isfile(db_path):
        raise FileNotFoundError(
            f"SQLite DB not found: {db_path}\n"
            f"Make sure the {opt_date}_BAYESIAN_OPT directory has been copied "
            f"from Colab into pes_a2c/inputs/ first."
        )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    summaries = optuna.get_all_study_summaries(storage)
    if not summaries:
        raise RuntimeError(f"No study found in {storage}")
    study_name = summaries[0].study_name
    study = optuna.load_study(study_name=study_name, storage=storage)

    try:
        best = study.best_trial
    except ValueError as exc:
        raise RuntimeError(
            f"Study {study_name!r} has no completed trials yet — "
            f"cannot export best params."
        ) from exc

    bp = best.params
    _ahd = bp.get('actor_hidden_dim', AC_ACTOR_HIDDEN_UNITS[0])
    _chd = bp.get('critic_hidden_dim', AC_CRITIC_HIDDEN_UNITS[0])
    _nhl = bp.get('n_hidden_layers', len(AC_ACTOR_HIDDEN_UNITS))
    actor_hidden  = [_ahd] * _nhl
    critic_hidden = [_chd] * _nhl
    trial_seed = int(best.user_attrs.get('trial_seed', SEED + int(best.number) + 1))

    payload = {
        'opt_date':            opt_date,
        'best_trial_number':   int(best.number),
        'mean_perf':           float(best.user_attrs.get('mean_perf', best.value)),
        'std_perf':            float(best.user_attrs.get('std_perf', float('nan'))),
        'min_perf':            float(best.user_attrs.get('min_perf', float('nan'))),
        'max_perf':            float(best.user_attrs.get('max_perf', float('nan'))),
        'trial_seed':          trial_seed,
        'actor_hidden_units':  actor_hidden,
        'critic_hidden_units': critic_hidden,
        'hyperparameters':     {k: (int(v) if isinstance(v, (bool, int)) else
                                    (float(v) if isinstance(v, float) else v))
                                for k, v in bp.items()},
    }

    dated_file = os.path.join(opt_dir, f'best_params_{opt_date}.json')
    with open(dated_file, 'w', encoding='utf-8') as _f:
        json.dump(payload, _f, indent=2)
    success(f"Written: {dated_file}")

    std_file = os.path.join(INPUTS_PATH, 'best_params.json')
    with open(std_file, 'w', encoding='utf-8') as _f:
        json.dump(payload, _f, indent=2)
    success(f"Mirrored: {std_file}")

    info(f"  Study: {study_name}")
    info(f"  Best trial: #{best.number}  mean_perf={payload['mean_perf']:.6f}")
    info(f"  trial_seed: {trial_seed}")


def main():
    """Run Bayesian optimisation of A2C hyperparameters using Optuna."""

    header("BAYESIAN OPTIMISATION — A2C HYPERPARAMETERS", width=80)

    # Parse arguments: [n_trials] [--resume YYYY-MM-DD] [--mode MODE] [--out-dir PATH] [--storage URL]
    # Special shorthand: --export-best [YYYY-MM-DD]  — read existing DB and write best_params.json
    n_trials = 30
    opt_date = datetime.now().strftime("%Y-%m-%d")
    out_dir_override: str | None = None
    storage_override: str | None = None
    export_best_only: bool = False

    global _optimize_mode
    _optimize_mode = AC_OPTIMIZE_MODE

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--resume' and i + 1 < len(args):
            opt_date = args[i + 1]
            i += 2
        elif args[i] == '--export-best':
            export_best_only = True
            # Accept an optional date: --export-best 2026-04-23
            if i + 1 < len(args) and _DATE_RE_OPT.match(args[i + 1]):
                opt_date = args[i + 1]
                i += 2
            else:
                i += 1
        elif args[i] == '--mode' and i + 1 < len(args):
            _optimize_mode = args[i + 1]
            i += 2
        elif args[i] == '--out-dir' and i + 1 < len(args):
            out_dir_override = args[i + 1]
            i += 2
        elif args[i] == '--storage' and i + 1 < len(args):
            storage_override = args[i + 1]
            i += 2
        else:
            try:
                n_trials = int(args[i])
            except ValueError:
                pass
            i += 1

    opt_dir = out_dir_override or os.path.join(INPUTS_PATH, f'{opt_date}_BAYESIAN_OPT')
    os.makedirs(opt_dir, exist_ok=True)

    global _opt_dir
    _opt_dir = opt_dir

    # --export-best: read existing DB, write best_params JSON files, then exit.
    if export_best_only:
        _export_best_params(opt_dir, opt_date, storage_override)
        return

    info(f"Output directory: {opt_dir}")
    info(f"Target number of trials: {n_trials}")
    print()

    # --- Load data ---
    section("Loading Evaluation Data", width=80)
    _load_evaluation_data()
    assert _trials_per_sequence is not None and _sevs is not None
    assert _number_cities_prob is not None and _severity_prob is not None
    list_item(f"Sequence lengths shape: {_trials_per_sequence.shape}")
    list_item(f"Sequences loaded: {len(_sevs)}")
    print()

    # --- Run optimisation ---
    section("Running Bayesian Optimisation", width=80)
    if _optimize_mode == 'full':
        info("Optimization mode: FULL (8 base + 4 improvement hyperparameters)")
        list_item("actor_lr             ∈ [1e-4, 1e-2]            (log scale)")
        list_item("critic_lr            ∈ [1e-4, 1e-2]            (log scale)")
        list_item("discount_factor      ∈ [0.85, 0.995]")
        list_item("entropy_coeff        ∈ [0.0, 0.1]              (linear; 0 disables bonus)")
        list_item("num_episodes         ∈ [50000, 250000]         (step=25000)")
        list_item("actor_hidden_dim     ∈ {32, 64, 128, 256}      (categorical)")
        list_item("critic_hidden_dim    ∈ {32, 64, 128, 256}      (categorical)")
        list_item("n_hidden_layers      ∈ {1, 2, 3}")
        list_item("epsilon_initial / epsilon_min   fixed at 0.0 (on-policy softmax sampling)")
    else:
        info("Optimization mode: IMPROVEMENTS ONLY (base hyperparams fixed at CONFIG)")
    info("Improvement search space (always sampled):")
    list_item("penalty_coeff        ∈ [0.0, 0.3]              (linear; 0 disables PBRS)")
    list_item("gae_lambda           ∈ [0.90, 0.99]")
    list_item("max_grad_norm        ∈ [0.3, 1.5]")
    list_item("lr_min_ratio         ∈ [0.05, 0.25]")
    list_item("warmup_ratio / target_ratio     fixed at 0.0 / 1.0 (legacy ε-decay disabled)")
    print()

    # Suppress Optuna's verbose default logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Use SQLite storage so the study survives machine suspensions/crashes
    db_path = os.path.join(opt_dir, f'optuna_study_{opt_date}.db')
    storage = storage_override or f'sqlite:///{db_path}'

    study = optuna.create_study(
        direction='maximize',
        study_name=f'a2c_opt_{opt_date}',
        sampler=optuna.samplers.TPESampler(seed=SEED),
        # Moderate pruning: n_startup_trials=10 gives TPE enough completed
        # trials (≥10) before pruning activates, reducing the 83% prune rate
        # seen with n_startup_trials=2.  n_warmup_steps=2 requires two
        # intermediate reports (~6 min) before a trial can be pruned, giving
        # TPE more signal from slower-starting configurations.
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
        storage=storage,
        load_if_exists=True,   # Resume from previous run if DB exists
    )

    # Warm-start: enqueue the best hyperparameters from the previous
    # Bayesian optimisation (2026-04-23, trial #90, mean_perf = 0.887236,
    # 100 trials) so that TPE has a strong baseline from trial #1.
    # If the study already has completed trials (resume) this is harmless
    # — Optuna deduplicates.
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if completed == 0:
        warmstart: dict = {}
        if _optimize_mode == 'full':
            warmstart.update({
                'actor_lr': AC_ACTOR_LR,
                'critic_lr': AC_CRITIC_LR,
                'discount_factor': AC_DISCOUNT,
                'entropy_coeff': AC_ENTROPY_COEFF,
                'epsilon_initial': AC_EPSILON_INITIAL,
                'epsilon_min': AC_EPSILON_MIN,
                'num_episodes': 250000,
                'actor_hidden_dim': AC_ACTOR_HIDDEN_UNITS[0],
                'critic_hidden_dim': AC_CRITIC_HIDDEN_UNITS[0],
                'n_hidden_layers': len(AC_ACTOR_HIDDEN_UNITS),
            })
        warmstart.update({
            'warmup_ratio': AC_WARMUP_RATIO,
            'target_ratio': AC_TARGET_RATIO,
            'penalty_coeff': AC_PENALTY_COEFF,
            'gae_lambda': AC_GAE_LAMBDA,
            'max_grad_norm': AC_MAX_GRAD_NORM,
            'lr_min_ratio': AC_LR_MIN_RATIO,
        })
        study.enqueue_trial(warmstart)
        info(f"Warm-start: CONFIG defaults enqueued ({_optimize_mode} mode)")

    # Calculate how many trials remain (allows seamless resume)
    remaining = max(0, n_trials - completed)
    if completed > 0:
        info(f"Resuming: {completed} trials already completed, {remaining} remaining")
    else:
        info(f"Starting fresh: {n_trials} trials to run")

    t_start = time.time()

    # Callback to print progress.  Robust against:
    #   - pruned/failed trials whose ``trial.value`` is ``None``
    #   - the early phase where no trial has COMPLETE state yet, in which
    #     case ``study.best_value`` raises ``ValueError``.
    def _progress_callback(study, trial):
        done = len([t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE])
        elapsed = time.time() - t_start
        try:
            best_val = study.best_value
            best_str = f"{best_val:.4f}"
        except ValueError:
            best_val = None
            best_str = "  n/a  "
        perf = trial.user_attrs.get('mean_perf', 0.0)
        obj_str = f"{trial.value:.4f}" if trial.value is not None else "   n/a"
        print(
            f"  Trial {done:3d}/{n_trials}  |  "
            f"obj={obj_str}  perf={perf:.4f}  |  "
            f"best={best_str}  |  elapsed={elapsed:.0f}s"
        )

    if remaining > 0:
        _prev_err = numpy.seterr(under='ignore')
        try:
            study.optimize(objective, n_trials=remaining, callbacks=[_progress_callback])
        finally:
            numpy.seterr(**_prev_err)
    else:
        info("All trials already completed. Generating reports from stored results.")

    elapsed_total = time.time() - t_start
    total_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print()
    success(f"Optimisation finished in {elapsed_total:.1f}s ({elapsed_total / 60:.1f} min)")
    info(f"Total completed trials: {total_completed}")
    print()

    # --- Best trial summary ---
    best = study.best_trial
    section("Best Hyperparameters Found", width=80)
    for name, val in best.params.items():
        list_item(f"{name:<25s} = {val}")
    info(f"Best objective (mean perf): {best.value:.6f}")
    info(f"  Mean normalised perf:     {best.user_attrs.get('mean_perf', best.value):.6f}")
    print()

    # --- Use best model from optimization, or retrain only if resuming ---
    section("Best Model", width=80)

    bp = best.params
    _ahd = bp.get('actor_hidden_dim', AC_ACTOR_HIDDEN_UNITS[0])
    _nhl = bp.get('n_hidden_layers', len(AC_ACTOR_HIDDEN_UNITS))
    best_actor_hidden = [_ahd] * _nhl

    # Try in-memory artifacts first, then disk fallback from --resume
    pkl_path = os.path.join(opt_dir, '_best_weights.pkl')
    if _best_artifacts['weights'] is None:
        if _load_best_artifacts(opt_dir, _best_artifacts):
            info(f"Loaded best weights from disk (perf={_best_artifacts['value']:.6f})")
        elif os.path.isfile(pkl_path):
            # Legacy pickle artifact from a pre-2026-04-20 run — refused for
            # security (CWE-502). Falls through to deterministic retrain.
            info(f"Found legacy pickle artifact at {pkl_path} — ignored.")

    if _best_artifacts['weights'] is not None and _best_artifacts['value'] >= best.value:
        # Reconstruct model and load cached weights
        best_model = build_actor(3, 11, _best_artifacts['hidden_units'])
        best_model(tf.zeros((1, 3)))  # build
        best_model.set_weights(_best_artifacts['weights'])
        best_rewards = numpy.array(_best_artifacts['rewards'])
        success("Using Actor from best optimization trial (no retraining needed)")
    else:
        # Reproduce the best trial bit-exact: use the SAME per-trial seed
        # that produced the reported mean_perf (SEED + best.number + 1).
        # Falling back to CONFIG.SEED would silently produce a different
        # model with a different mean_perf.
        best_trial_seed = int(best.user_attrs.get('trial_seed', SEED + int(best.number) + 1))
        info("Retraining with best hyperparameters (resumed study, original weights not in memory)...")
        info(f"  trial seed = {best_trial_seed}  (reproduces trial #{best.number + 1})")
        _chd = bp.get('critic_hidden_dim', AC_CRITIC_HIDDEN_UNITS[0])
        best_critic_hidden = [_chd] * _nhl

        env_final = Pandemic()
        env_final.number_cities_prob = _number_cities_prob  # type: ignore[assignment]
        env_final.severity_prob = _severity_prob  # type: ignore[assignment]
        env_final.verbose = False

        best_rewards_list, best_model, _ = A2CTraining(
            env_final,
            bp.get('actor_lr', AC_ACTOR_LR),
            bp.get('critic_lr', AC_CRITIC_LR),
            bp.get('discount_factor', AC_DISCOUNT),
            bp.get('entropy_coeff', AC_ENTROPY_COEFF),
            # ε-greedy / warm-up / target are FIXED to neutral values inside
            # objective() (on-policy training); using AC_* CONFIG fallbacks
            # here would silently change the trajectory.
            0.0,        # epsilon_initial (fixed in objective)
            0.0,        # epsilon_min     (fixed in objective)
            bp.get('num_episodes', 250000),
            actor_hidden=best_actor_hidden,
            critic_hidden=best_critic_hidden,
            seed=best_trial_seed,
            compute_confidence=False,
            warmup_ratio=0.0,   # fixed in objective
            target_ratio=1.0,   # fixed in objective
            penalty_coeff=bp.get('penalty_coeff', AC_PENALTY_COEFF),
            gae_lambda=bp.get('gae_lambda', AC_GAE_LAMBDA),
            max_grad_norm=bp.get('max_grad_norm', AC_MAX_GRAD_NORM),
            lr_min_ratio=bp.get('lr_min_ratio', AC_LR_MIN_RATIO),
        )
        best_rewards = numpy.array(best_rewards_list)
        success(f"Retrained model (deterministic — seed = {best_trial_seed})")

    list_item(f"Actor parameters: {best_model.count_params()}")
    print()

    # --- Save everything ---
    section("Saving Results", width=80)
    _save_report(study, opt_dir, opt_date, best_model, best_rewards)

    print()
    section("Optimisation Complete", width=80)
    success("All outputs saved!")
    info(f"Output directory: {opt_dir}")
    print()


if __name__ == '__main__':
    main()
