'''
pes_dqn - Pandemic Experiment Scenario

Bayesian Optimization of DQN hyperparameters using Optuna.

Optimizes: learning_rate, discount_factor, epsilon_initial, epsilon_min,
           num_episodes, hidden_units, batch_size, buffer_size, target_sync_freq,
           max_grad_norm, penalty_coeff (PBRS), warmup_ratio, target_ratio
Objective: maximize mean normalised performance over the 64 evaluation sequences.

The evaluation uses infeasible-action masking (actions > available resources are
suppressed before argmax) so that the metric matches the behaviour of the DQN agent
in __main__.py.  The best model found during the search is preserved in memory
and saved directly, avoiding a lossy re-training step.

Optimisation features:
    - **MedianPruner**: early stops trials whose intermediate reward is worse
      than the median of completed trials, saving significant compute.
    - **Warm-start**: the first trial uses CONFIG.py defaults as a seed,
      guaranteeing at least one reasonable baseline.
    - **PBRS** (Potential-Based Reward Shaping): optional shaped reward
      ``r' = r + β·(γ·Φ(s') − Φ(s))``, with β optimised.
    - **compute_confidence=False**: skips the extra forward pass for
      meta-cognitive confidence during training (~33 % speed-up).

Usage:
    python3 -m ml.pes_dqn.ext.optimize_dqn [n_trials] [--resume YYYY-MM-DD]

    n_trials : int, optional
        Number of Bayesian optimization trials (default: 60).
    --resume YYYY-MM-DD : str, optional
        Resume a previous optimization run stored under that date.

Search space (16 parameters):
    learning_rate        ∈ [1e-4, 5e-3]      (log scale)
    discount_factor      ∈ [0.92, 0.995]
    epsilon_initial      ∈ [0.80, 1.0]
    epsilon_min          ∈ [0.01, 0.20]
    num_episodes         ∈ [40000, 100000]   (step=20000, opt-time only)
    hidden_layer_size    ∈ {32, 64, 96, 128}
    num_hidden_layers    ∈ {1, 2, 3}
    batch_size           ∈ {32, 64, 128, 256}
    buffer_size          ∈ [20000, 100000]   (step=10000)
    target_sync_freq     ∈ [500, 5000]       (step=500)
    max_grad_norm        ∈ [0.5, 5.0]
    use_pbrs             ∈ {True, False}
    penalty_coeff        ∈ [1e-4, 0.1]       (log scale, only when use_pbrs=True)
    warmup_ratio         ∈ [0.05, 0.30]      (ε-warmup fraction)
    target_ratio         ∈ [0.50, 0.95]      (ε-decay target fraction)
    learning_starts_frac ∈ [0.05, 0.25]      (replay-buffer warm-up fraction)

Note: ``num_episodes`` is intentionally low during optimisation so each trial
fits in <1h on Colab CPU. The winning hyperparameter set is then retrained
at the FULL ``DQN_EPISODES`` count (default 175 000) inside the optimisation
script itself, before saving ``dqn_best_<date>.keras``.

Outputs (saved to INPUTS_PATH/<date>_BAYESIAN_OPT/):
    - dqn_best_<date>.keras                   : Model from the best optimization trial
    - rewards_best_<date>.npy                 : Reward history of the best training run
    - optimization_results_<date>.txt         : Full report of the optimization (1-based trial #)
    - optimization_history_<date>.png         : Convergence plot (1-based trial #)
    - hyperparameter_importances_<date>.png   : Parameter importance plot
    - optuna_study_<date>.db                  : SQLite database for resumable studies

Note:
    Trial numbering in reports and plots uses 1-based indexing to match
    the trial_id in the SQLite database.  Optuna internally uses 0-based
    trial.number; the +1 offset is applied at report-generation time.
'''

##########################
##  Imports internos    ##
##########################
from .pandemic import Pandemic, run_experiment, DQNTraining, dqn_agent_meta_cognitive
from .dqn_model import build_q_network, normalize_state
from ..src.terminal_utils import header, section, success, info, list_item
from .tools import convert_globalseq_to_seqs
from ..config.CONFIG import (SEED, DQN_LEARNING_RATE, DQN_DISCOUNT,
                             DQN_EPSILON_INITIAL, DQN_EPSILON_MIN,
                             DQN_EPISODES, DQN_HIDDEN_UNITS, DQN_BATCH_SIZE,
                             DQN_REPLAY_BUFFER_SIZE, DQN_TARGET_SYNC_FREQ,
                             DQN_WARMUP_RATIO, DQN_TARGET_RATIO,
                             DQN_MAX_GRAD_NORM, DQN_PENALTY_COEFF,
                             DQN_LEARNING_STARTS_FRAC)
from .. import INPUTS_PATH

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
import tensorflow as tf
import matplotlib  # noqa: E402
matplotlib.use('Agg', force=True)  # non-interactive backend; prevents NoneType canvas crash on headless servers
import matplotlib.pyplot as plt
from datetime import datetime

# Device selection is centralised in pes_dqn/__init__.py via MPES_USE_GPU.
# Re-asserting the CPU pin here only as a defensive fallback when this module
# is launched in isolation (e.g. ``python ext/optimize_dqn.py``) without the
# package having been imported first. Honour MPES_USE_GPU=1 to allow GPU runs.
if os.environ.get('MPES_USE_GPU', '0') != '1':
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*Box.*precision lowered.*')
warnings.filterwarnings('ignore', message='.*A NumPy version.*SciPy.*')
warnings.filterwarnings('ignore', category=SyntaxWarning, message='.*invalid escape sequence.*')

# Suppress TF retracing warnings (expected: each Optuna trial creates a fresh
# tf.function wrapper to isolate the graph per trial) and deprecation notices
# from keras.backend.clear_session() calling legacy tf.reset_default_graph.
import logging as _logging  # noqa: E402
_logging.getLogger('tensorflow').setLevel(_logging.ERROR)

##########################
##  Internal imports    ##
##########################




# Nombre del paquete para las notificaciones push
_PKG_NAME = __package__.split('.')[1] if (__package__ and '.' in __package__) else (__package__ or 'mPES')


###################################
##    Global evaluation data     ##
###################################
# Loaded once at startup and reused by every trial
_trials_per_sequence = None
_sevs = None
_number_cities_prob = None
_severity_prob = None

# Store best model weights/rewards during optimization to avoid lossy retraining
_best_artifacts: dict = {'weights': None, 'rewards': None, 'value': float('-inf'),
                         'hidden_units': None, 'trial_seed': None,
                         'trial_number': None}

# Set by main() so objective() can persist _best_artifacts to disk
_opt_dir: str = ''


###################################
##   Pickle-free persistence    ##
###################################
# CWE-502 hardening: never deserialise the Optuna best-artifact via pickle.
# Keras weight tensors are split into named ndarrays inside a single .npz
# (loaded with allow_pickle=False) and the scalar metadata into a sibling
# .json file. Loaders refuse legacy ``_best_weights.pkl`` files.
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
    artifacts['trial_seed'] = int(meta.get('trial_seed', 0))
    artifacts['trial_number'] = int(meta.get('trial_number', -1))
    return True


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
    """
    Optuna objective: train DQN with sampled hyperparameters,
    evaluate on the fixed 64 sequences, and return mean normalised performance.
    """
    # --- Sample hyperparameters ---
    # Widened search space (post Double-DQN + masking + learning_starts fix).
    # Rationale documented in pes_dqn/doc/ — boundaries opened where the
    # previous study (2026-04-18) pegged the limit.
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
    discount_factor = trial.suggest_float('discount_factor', 0.92, 0.995)
    epsilon_initial = trial.suggest_float('epsilon_initial', 0.80, 1.0)
    epsilon_min = trial.suggest_float('epsilon_min', 0.01, 0.20)
    # Opt-phase episode budget: kept low so each trial finishes in <1h on
    # Colab CPU. The winning hyperparameter set is retrained at the FULL
    # ``DQN_EPISODES`` count after ``study.optimize`` returns.
    num_episodes = trial.suggest_int('num_episodes', 40_000, 100_000, step=20_000)
    hidden_layer_size = trial.suggest_categorical(
        'hidden_layer_size', [32, 64, 96, 128])
    num_hidden_layers = trial.suggest_int('num_hidden_layers', 1, 3)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    buffer_size = trial.suggest_int('buffer_size', 20_000, 100_000, step=10_000)
    target_sync_freq = trial.suggest_int('target_sync_freq', 500, 5_000, step=500)
    max_grad_norm = trial.suggest_float('max_grad_norm', 0.5, 5.0)
    # PBRS coefficient: includes 0 so the optimiser can disable shaping.
    use_pbrs = trial.suggest_categorical('use_pbrs', [True, False])
    if use_pbrs:
        penalty_coeff = trial.suggest_float('penalty_coeff', 1e-4, 0.1, log=True)
    else:
        penalty_coeff = 0.0
    warmup_ratio = trial.suggest_float('warmup_ratio', 0.05, 0.30)
    target_ratio = trial.suggest_float('target_ratio', 0.50, 0.95)
    # Fraction of buffer_size that must accumulate before training starts
    # (DQN warm-up). Converted to absolute count below.
    learning_starts_frac = trial.suggest_float('learning_starts_frac', 0.05, 0.25)
    learning_starts = max(int(learning_starts_frac * buffer_size), int(batch_size))

    hidden_units = [hidden_layer_size] * num_hidden_layers

    # --- Pruning callback (reports avg reward every 10k episodes) ---
    _step_counter = [0]

    def _pruning_cb(episode_idx, avg_reward):
        _step_counter[0] += 1
        trial.report(avg_reward, _step_counter[0])
        return trial.should_prune()

    # --- Train ---
    env = Pandemic()
    env.number_cities_prob = _number_cities_prob  # type: ignore[assignment]
    env.severity_prob = _severity_prob  # type: ignore[assignment]
    env.verbose = False

    # Per-trial seed: matches the formula used by sibling pes_a2c / pes_ql
    # so the local train script can reproduce the trial bit-exact via
    # ``train_dqn.py --from-best <date>`` (reads trial_seed user_attr below).
    trial_seed = SEED + int(trial.number) + 1

    rewards, model, _ = DQNTraining(
        env, learning_rate, discount_factor,
        epsilon_initial, epsilon_min, num_episodes,
        hidden_units=hidden_units, batch_size=batch_size,
        buffer_size=buffer_size, target_sync_freq=target_sync_freq,
        max_grad_norm=max_grad_norm, seed=trial_seed,
        penalty_coeff=penalty_coeff,
        compute_confidence=False,
        pruning_callback=_pruning_cb,
        warmup_ratio=warmup_ratio, target_ratio=target_ratio,
        learning_starts=learning_starts,
    )

    # --- Evaluate on fixed sequences ---
    env_eval = Pandemic()
    env_eval.verbose = False
    max_res = env_eval.max_resources
    max_seq = env_eval.max_seq_length
    max_sev = env_eval.max_severity

    def qf(_env, state, _seqid):
        norm_s = normalize_state(state, max_res, max_seq, max_sev)
        q_vals = model(norm_s[numpy.newaxis], training=False).numpy()[0].copy()
        response, _conf, _rt_h, _rt_r = dqn_agent_meta_cognitive(
            q_vals, state[0], 10000
        )
        return response

    _, perfs, _ = run_experiment(env_eval, qf, False, _trials_per_sequence, _sevs)
    mean_perf = float(numpy.mean(perfs))

    # Store extra info for later analysis
    trial.set_user_attr('mean_perf', mean_perf)
    trial.set_user_attr('std_perf', float(numpy.std(perfs)))
    trial.set_user_attr('min_perf', float(numpy.min(perfs)))
    trial.set_user_attr('max_perf', float(numpy.max(perfs)))
    trial.set_user_attr('hidden_units', hidden_units)
    # Persist the per-trial seed so train_dqn.py --from-best can reproduce it
    trial.set_user_attr('trial_seed', int(trial_seed))

    # Preserve the best model weights to avoid lossy retraining at the end
    global _best_artifacts
    if mean_perf > _best_artifacts['value']:
        _best_artifacts['weights'] = model.get_weights()
        _best_artifacts['rewards'] = list(rewards)
        _best_artifacts['value'] = mean_perf
        _best_artifacts['hidden_units'] = hidden_units
        _best_artifacts['trial_seed'] = int(trial_seed)
        _best_artifacts['trial_number'] = int(trial.number)

        # Persist to disk so that --resume can recover without retraining.
        # Pickle-free format (CWE-502 hardening).
        if _opt_dir:
            _save_best_artifacts(_opt_dir, _best_artifacts)

    # ── Memory / state cleanup ─────────────────────────────────
    # Each Optuna trial builds new Keras models and tf.function traces.
    # Without explicit cleanup TensorFlow retains stale graphs, variables
    # and op/name counters, which (a) causes gradual memory growth, and
    # (b) makes the best trial's weight initialization irreproducible
    # outside the optimization process. Clearing the session here ensures
    # every trial — and any subsequent fresh call to ``train_dqn.py`` —
    # starts from the same pristine global TF state, so retraining with
    # the best hyperparameters reproduces the optimization mean performance.
    del model, rewards, env, env_eval
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
        f.write("BAYESIAN OPTIMIZATION RESULTS — DQN HYPERPARAMETERS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Date:              {opt_date}\n")
        f.write(f"Total trials:      {len(study.trials)}\n")
        f.write(f"Best trial:        #{best.number + 1}\n")
        f.write(f"Best mean perf:    {best.value:.6f}\n\n")

        f.write("BEST HYPERPARAMETERS\n")
        f.write("-" * 80 + "\n")
        for name, val in best.params.items():
            f.write(f"  {name:<25s} = {val}\n")
        if 'hidden_units' in best.user_attrs:
            f.write(f"  {'hidden_units':<25s} = {best.user_attrs['hidden_units']}\n")
        f.write("\n")

        # Copy-paste-ready CONFIG.py snippet so train_dqn.py on the local PC
        # reproduces exactly the same mean_perf as the optimisation trial.
        bp = best.params
        hidden = [bp['hidden_layer_size']] * bp['num_hidden_layers']
        use_pbrs = bool(bp.get('use_pbrs', bp.get('penalty_coeff', 0.0) > 0))
        penalty = float(bp.get('penalty_coeff', 0.0)) if use_pbrs else 0.0
        f.write("CONFIG.PY SNIPPET (copy-paste into pes_dqn/config/CONFIG.py)\n")
        f.write("-" * 80 + "\n")
        f.write(f"# Best hyperparameters from Bayesian Optimisation trial #{best.number + 1} ({opt_date}).\n")
        f.write(f"# Performance: mean_perf = {best.value:.6f} over 64 fixed sequences.\n")
        # ``num_episodes`` was capped during the search for speed; the local PC
        # should always retrain at the full ``DQN_EPISODES`` to match the saved
        # ``dqn_best_<date>.keras`` (which is also retrained at full episodes).
        full_episodes = max(int(DQN_EPISODES), int(bp['num_episodes']))
        f.write(f"DQN_HIDDEN_UNITS = {hidden}\n")
        f.write(f"DQN_LEARNING_RATE = {bp['learning_rate']}\n")
        f.write(f"DQN_BATCH_SIZE = {bp['batch_size']}\n")
        f.write(f"DQN_REPLAY_BUFFER_SIZE = {bp['buffer_size']}\n")
        f.write(f"DQN_TARGET_SYNC_FREQ = {bp['target_sync_freq']}\n")
        f.write(f"DQN_DISCOUNT = {bp['discount_factor']}\n")
        f.write(f"DQN_EPSILON_INITIAL = {bp['epsilon_initial']}\n")
        f.write(f"DQN_EPSILON_MIN = {bp['epsilon_min']}\n")
        f.write(f"DQN_EPISODES = {full_episodes}  # full retrain length\n")
        f.write(f"DQN_MAX_GRAD_NORM = {bp.get('max_grad_norm', 1.0)}\n")
        f.write(f"DQN_PENALTY_COEFF = {penalty}\n")
        f.write(f"DQN_WARMUP_RATIO = {bp.get('warmup_ratio', 0.05)}\n")
        f.write(f"DQN_TARGET_RATIO = {bp.get('target_ratio', 0.60)}\n")
        f.write(f"DQN_LEARNING_STARTS_FRAC = {bp.get('learning_starts_frac', 0.1)}\n")
        f.write("\n")

        f.write("BEST TRIAL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Mean performance:   {best.user_attrs['mean_perf']:.6f}\n")
        f.write(f"  Std  performance:   {best.user_attrs['std_perf']:.6f}\n")
        f.write(f"  Min  performance:   {best.user_attrs['min_perf']:.6f}\n")
        f.write(f"  Max  performance:   {best.user_attrs['max_perf']:.6f}\n\n")

        f.write("ALL TRIALS\n")
        f.write("-" * 100 + "\n")
        f.write(
            f"{'#':>4s}  {'mean_perf':>10s}  {'lr':>10s}  "
            f"{'gamma':>8s}  {'eps0':>6s}  {'eps_min':>7s}  "
            f"{'episodes':>8s}  {'batch':>5s}  {'buf_sz':>6s}  "
            f"{'sync':>5s}\n"
        )
        f.write("-" * 100 + "\n")
        for t in sorted(study.trials, key=lambda t: t.value if t.value is not None else -1, reverse=True):
            if t.value is None:
                continue
            p = t.params
            f.write(
                f"{t.number + 1:4d}  {t.value:10.6f}  "
                f"{p['learning_rate']:10.6f}  {p['discount_factor']:8.4f}  "
                f"{p['epsilon_initial']:6.3f}  {p['epsilon_min']:7.4f}  "
                f"{p['num_episodes']:8d}  {p['batch_size']:5d}  "
                f"{p['buffer_size']:6d}  {p['target_sync_freq']:5d}\n"
            )

    success(f"Report saved: optimization_results_{opt_date}.txt")

    # --- Convergence plot ---
    try:
        plt.style.use('ggplot')
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(12, 6))
    trial_numbers = [t.number + 1 for t in study.trials if t.value is not None]
    trial_values = [t.value for t in study.trials if t.value is not None]

    # Running best
    running_best = []
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
    ax.set_title('Bayesian Optimisation: Convergence (DQN)', fontsize=14, fontweight='bold', pad=20)
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
        ax.barh(names[::-1], values[::-1], color='#2ca02c', edgecolor='darkgreen', linewidth=0.5)
        ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
        ax.set_title('DQN Hyperparameter Importance', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        fig.savefig(os.path.join(opt_dir, f'hyperparameter_importances_{opt_date}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        list_item(f"Saved: hyperparameter_importances_{opt_date}.png")
    except Exception as e:
        info(f"Could not compute importances: {e}")

    # --- Save best model and rewards ---
    best_model.save(os.path.join(opt_dir, f'dqn_best_{opt_date}.keras'))
    numpy.save(os.path.join(opt_dir, f'rewards_best_{opt_date}.npy'), best_rewards)
    success(f"Best model saved: dqn_best_{opt_date}.keras")
    success(f"Best rewards saved: rewards_best_{opt_date}.npy")

    # --- Sidecar JSON for cross-package uniformity (mirrors pes_ql/pes_dql) ---
    # train_dqn.py --from-best <date> reads this in preference to the SQLite DB
    # so users can reproduce mean_perf by copying just one small file.
    bp = best.params
    hidden = best.user_attrs.get('hidden_units')
    if hidden is None and 'hidden_layer_size' in bp:
        hidden = [bp['hidden_layer_size']] * bp.get('num_hidden_layers', 1)
    trial_seed = int(best.user_attrs.get('trial_seed', SEED + int(best.number) + 1))
    best_params_payload = {
        'opt_date':           opt_date,
        'best_trial_number':  int(best.number),
        'mean_perf':          float(best.user_attrs.get('mean_perf', best.value)),
        'std_perf':           float(best.user_attrs.get('std_perf', float('nan'))),
        'min_perf':           float(best.user_attrs.get('min_perf', float('nan'))),
        'max_perf':           float(best.user_attrs.get('max_perf', float('nan'))),
        'trial_seed':         trial_seed,
        'hidden_units':       list(hidden) if hidden else None,
        'hyperparameters':    {k: (int(v) if isinstance(v, (bool, int)) else
                                   (float(v) if isinstance(v, float) else v))
                               for k, v in bp.items()},
    }
    params_file = os.path.join(opt_dir, f'best_params_{opt_date}.json')
    with open(params_file, 'w', encoding='utf-8') as _f:
        json.dump(best_params_payload, _f, indent=2)
    success(f"Best params saved: best_params_{opt_date}.json")

    # --- Mirror to the standard input path so train_dqn.py picks it up
    # without requiring an explicit --from-best YYYY-MM-DD flag
    # (matches the pes_ql / pes_dql behavior). ---
    std_params = os.path.join(INPUTS_PATH, 'best_params.json')
    with open(std_params, 'w', encoding='utf-8') as _f:
        json.dump(best_params_payload, _f, indent=2)
    success(f"Mirrored to {std_params}")
def main():
    """Run Bayesian optimisation of DQN hyperparameters using Optuna."""

    header("BAYESIAN OPTIMISATION — DQN HYPERPARAMETERS", width=80)

    # Parse arguments: [n_trials] [--resume YYYY-MM-DD] [--out-dir PATH] [--storage URL]
    n_trials = 60
    opt_date = datetime.now().strftime("%Y-%m-%d")
    out_dir_override: str | None = None
    storage_override: str | None = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--resume' and i + 1 < len(args):
            opt_date = args[i + 1]
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
    info("Search space (16 parameters):")
    list_item("learning_rate        ∈ [1e-4, 5e-3]      (log scale)")
    list_item("discount_factor      ∈ [0.92, 0.995]")
    list_item("epsilon_initial      ∈ [0.80, 1.0]")
    list_item("epsilon_min          ∈ [0.01, 0.20]")
    list_item("num_episodes         ∈ [40000, 100000]   (step=20000, opt-time only)")
    list_item("hidden_layer_size    ∈ {32, 64, 96, 128}")
    list_item("num_hidden_layers    ∈ {1, 2, 3}")
    list_item("batch_size           ∈ {32, 64, 128, 256}")
    list_item("buffer_size          ∈ [20000, 100000]   (step=10000)")
    list_item("target_sync_freq     ∈ [500, 5000]       (step=500)")
    list_item("max_grad_norm        ∈ [0.5, 5.0]")
    list_item("use_pbrs             ∈ {True, False}")
    list_item("penalty_coeff        ∈ [1e-4, 0.1]       (log scale, only when use_pbrs=True)")
    list_item("warmup_ratio         ∈ [0.05, 0.30]")
    list_item("target_ratio         ∈ [0.50, 0.95]")
    list_item("learning_starts_frac ∈ [0.05, 0.25]")
    info("Final winner is retrained at full DQN_EPISODES after study.optimize().")
    print()

    # Suppress Optuna's verbose default logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Use SQLite storage so the study survives machine suspensions/crashes
    db_path = os.path.join(opt_dir, f'optuna_study_{opt_date}.db')
    storage = storage_override or f'sqlite:///{db_path}'

    study = optuna.create_study(
        direction='maximize',
        study_name=f'dqn_opt_{opt_date}',
        sampler=optuna.samplers.TPESampler(seed=SEED),
        # n_startup_trials=10 ensures TPE has enough completed trials before
        # pruning activates, avoiding the >80% prune rate seen with lower values.
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
        storage=storage,
        load_if_exists=True,   # Resume from previous run if DB exists
    )

    # Warm-start: enqueue a trial with CONFIG.py defaults to guarantee a baseline.
    # Values are clamped to the (narrower) opt-time ranges so Optuna does not
    # warn about out-of-distribution fixed parameters.
    if len(study.trials) == 0:
        # num_episodes during opt is capped at 100k; the final winner will be
        # retrained at full DQN_EPISODES afterwards.
        warm_episodes = int(min(max(DQN_EPISODES, 40_000), 100_000))
        # Snap to the 20k step grid declared in trial.suggest_int().
        warm_episodes = (warm_episodes // 20_000) * 20_000
        warm_hidden = int(DQN_HIDDEN_UNITS[0]) if DQN_HIDDEN_UNITS[0] in (32, 64, 96, 128) else 64
        warm_eps_min = float(min(max(DQN_EPSILON_MIN, 0.01), 0.20))
        study.enqueue_trial({
            'learning_rate': DQN_LEARNING_RATE,
            'discount_factor': DQN_DISCOUNT,
            'epsilon_initial': DQN_EPSILON_INITIAL,
            'epsilon_min': warm_eps_min,
            'num_episodes': warm_episodes,
            'hidden_layer_size': warm_hidden,
            'num_hidden_layers': len(DQN_HIDDEN_UNITS),
            'batch_size': DQN_BATCH_SIZE,
            'buffer_size': DQN_REPLAY_BUFFER_SIZE,
            'target_sync_freq': DQN_TARGET_SYNC_FREQ,
            'max_grad_norm': DQN_MAX_GRAD_NORM,
            'use_pbrs': bool(DQN_PENALTY_COEFF > 0),
            'penalty_coeff': DQN_PENALTY_COEFF,
            'warmup_ratio': DQN_WARMUP_RATIO,
            'target_ratio': DQN_TARGET_RATIO,
            'learning_starts_frac': DQN_LEARNING_STARTS_FRAC,
        })
        info("Warm-start: first trial uses CONFIG.py defaults")

    # Calculate how many trials remain (allows seamless resume)
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
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
        value_str = f"{trial.value:.4f}" if trial.value is not None else "   n/a"
        print(
            f"  Trial {done:3d}/{n_trials}  |  "
            f"value={value_str}  |  best={best_str}  |  "
            f"elapsed={elapsed:.0f}s"
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
    info(f"Mean normalised performance: {best.value:.6f}")
    print()

    # --- Use best model from optimization, or retrain only if resuming ---
    section("Best DQN Model", width=80)

    pkl_path = os.path.join(opt_dir, '_best_weights.pkl')
    if _best_artifacts['weights'] is None:
        if _load_best_artifacts(opt_dir, _best_artifacts):
            info(f"Loaded best model weights from disk (perf={_best_artifacts['value']:.6f})")
        elif os.path.isfile(pkl_path):
            # Legacy pickle artifact from a pre-2026-04-20 run — refused for
            # security (CWE-502). Falls through to deterministic retrain.
            info(f"Found legacy pickle artifact at {pkl_path} — ignored.")

    # Reuse the in-memory best per-trial model whenever its score matches the
    # study's best.value.  The previous third clause required
    # ``num_episodes >= DQN_EPISODES`` (175 000), which the search space caps
    # at 100 000 — so it was always False and forced an unwanted full retrain
    # at 175 000 episodes after every Colab/Optuna run.  For a longer-horizon
    # model, use ``train_dqn.py --from-best <date>`` on the local PC.
    if _best_artifacts['weights'] is not None and _best_artifacts['value'] >= best.value:
        # Rebuild model with preserved architecture and weights
        hidden_units = _best_artifacts['hidden_units']
        best_model = build_q_network(3, 11, hidden_units)
        best_model(tf.zeros((1, 3)))  # Build the model
        best_model.set_weights(_best_artifacts['weights'])
        best_rewards = numpy.array(_best_artifacts['rewards'])
        success("Using model from best optimization trial (no retraining needed)")
    else:
        info(f"Retraining best hyperparameters at full DQN_EPISODES={DQN_EPISODES:,} "
             f"(opt-trial used {best.params.get('num_episodes', '?')} episodes)...")
        bp = best.params
        hidden_units = [bp['hidden_layer_size']] * bp['num_hidden_layers']
        # Reuse the per-trial seed so the retrain reproduces the original
        # objective() value bit-exact (subject to TF/HW LSB).
        best_trial_seed = int(best.user_attrs.get('trial_seed', SEED + int(best.number) + 1))
        env_final = Pandemic()
        env_final.number_cities_prob = _number_cities_prob  # type: ignore[assignment]
        env_final.severity_prob = _severity_prob  # type: ignore[assignment]
        env_final.verbose = False

        best_rewards_list, best_model, _ = DQNTraining(
            env_final,
            bp['learning_rate'],
            bp['discount_factor'],
            bp['epsilon_initial'],
            bp['epsilon_min'],
            int(DQN_EPISODES),
            hidden_units=hidden_units,
            batch_size=bp['batch_size'],
            buffer_size=bp['buffer_size'],
            target_sync_freq=bp['target_sync_freq'],
            max_grad_norm=bp.get('max_grad_norm', DQN_MAX_GRAD_NORM),
            seed=best_trial_seed,
            penalty_coeff=bp.get('penalty_coeff', DQN_PENALTY_COEFF if bp.get('use_pbrs', True) else 0.0),
            compute_confidence=False,
            warmup_ratio=bp.get('warmup_ratio', DQN_WARMUP_RATIO),
            target_ratio=bp.get('target_ratio', DQN_TARGET_RATIO),
            learning_starts=max(
                int(bp.get('learning_starts_frac', 0.1) * bp['buffer_size']),
                int(bp['batch_size']),
            ),
        )
        best_rewards = numpy.array(best_rewards_list)
        success(f"Retrained at full episodes (deterministic — seed = {best_trial_seed})")

    list_item(f"Network parameters: {best_model.count_params():,}")
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
