'''
pes_dql - Pandemic Experiment Scenario (Q-Learning v2)

Bayesian Optimization of Q-Learning hyperparameters using Optuna.

Optimizes: learning_rate, discount_factor, epsilon_initial, epsilon_min, num_episodes,
          warmup_ratio, target_ratio, penalty_coeff
Objective: maximize mean normalised performance over the 64 evaluation sequences.

The evaluation uses infeasible-action masking (actions > available resources are
suppressed before argmax) so that the metric matches the behaviour of the RL agent
in __main__.py.  The best Q-table found during the search is preserved in memory
and saved directly, avoiding a lossy re-training step.

Usage:
    python3 -m pes_dql.ext.optimize_rl [n_trials] [--resume YYYY-MM-DD]
                                       [--out-dir PATH] [--storage URL]

    n_trials : int, optional
        Number of Bayesian optimization trials (default: 50).
    --resume YYYY-MM-DD : str, optional
        Resume a previous optimization run stored under that date.
    --out-dir PATH : str, optional
        Override the default output directory
        (``inputs/<date>_BAYESIAN_OPT/``).
    --storage URL : str, optional
        Override the default Optuna storage URL
        (``sqlite:///<opt_dir>/optuna_study_<date>.db``).

Search space:
    learning_rate    ∈ [0.05, 0.30]      (log scale)
    discount_factor  ∈ [0.90, 0.999]
    epsilon_initial  ∈ [0.50, 1.00]
    epsilon_min      ∈ [0.01, 0.10]      (log scale)
    num_episodes     ∈ [150000, 500000]  (step=10000)
    warmup_ratio     ∈ [0.02, 0.15]
    target_ratio     ∈ [0.40, 0.80]
    penalty_coeff    ∈ [1e-4, 0.30]      (log scale)  — PBRS coefficient

Sampler:
    TPESampler(seed=SEED, n_startup_trials=10, multivariate=True, group=True)
    Each trial seeds the QLearning training with SEED + trial.number + 1 so that
    successive trials explore different stochastic trajectories while remaining
    individually reproducible.

Outputs (saved to INPUTS_PATH/<date>_BAYESIAN_OPT/):
    - q_best_<date>.npy              : Q-table from the best optimization trial
    - rewards_best_<date>.npy        : Reward history of the best training run
    - optimization_results_<date>.txt: Full report of the optimization (1-based trial #)
    - optimization_history_<date>.png: Convergence plot (1-based trial #)
    - hyperparameter_importances_<date>.png: Parameter importance plot
    - optuna_study_<date>.db         : SQLite database for resumable studies

Note:
    Trial numbering in reports and plots uses 1-based indexing to match
    the trial_id in the SQLite database.  Optuna internally uses 0-based
    trial.number; the +1 offset is applied at report-generation time.
'''

##########################
##  External imports    ##
##########################
import os
# Force matplotlib to use a non-interactive backend BEFORE the library
# is imported anywhere (this env var is honoured by matplotlib at first
# import, even when something else imported it first via IPython).
os.environ.setdefault('MPLBACKEND', 'Agg')

import json
import sys
import time
import numpy
import warnings
import optuna
import matplotlib
# Belt-and-suspenders: also call matplotlib.use() in case the env var was
# overridden after import.  optimize_rl is always run headless (Colab,
# server, nohup) and never displays figures — without this, pyplot picks
# the default GUI backend (e.g. ``agg`` is fine, but Colab's matplotlib
# config can resolve to a backend whose FigureManager is None, raising
# ``AttributeError: 'NoneType' object has no attribute 'canvas'`` on
# ``plt.subplots``).
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from datetime import datetime

# Force TensorFlow to use CPU
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*Box.*precision lowered.*')
warnings.filterwarnings('ignore', message='.*A NumPy version.*SciPy.*')
# Optuna flags ``multivariate`` and ``group`` as experimental, but the
# combination is the recommended TPE configuration for correlated search
# spaces and has been stable for years.  Silence the noise.
warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)

##########################
##  Internal imports    ##
##########################
from .. import INPUTS_PATH
from ..config.CONFIG import SEED

from .tools import convert_globalseq_to_seqs
from ..src.terminal_utils import header, section, success, info, list_item
from .pandemic import Pandemic, run_experiment, QLearning
from .repro import save_fingerprint


###################################
##    Global evaluation data     ##
###################################
# Loaded once at startup and reused by every trial
_trials_per_sequence = None
_sevs = None
_number_cities_prob = None
_severity_prob = None

# Store best Q-table/rewards during optimization to avoid lossy retraining
_best_artifacts = {'Q': None, 'rewards': None, 'value': float('-inf'), 'trial_number': -1}


###################################
##   Pickle-free persistence    ##
###################################
# CWE-502 hardening: never deserialise the Optuna best-artifact via pickle.
# Numpy arrays go into a .npz (loaded with allow_pickle=False) and the
# scalar/JSON-serialisable metadata goes into a sibling .json file.
_BEST_NPZ_BASENAME = '_best_artifacts.npz'
_BEST_META_BASENAME = '_best_artifacts.json'


def _save_best_artifacts(opt_dir: str, artifacts: dict) -> None:
    """Persist *artifacts* to ``opt_dir`` without using ``pickle``."""
    npz_path = os.path.join(opt_dir, _BEST_NPZ_BASENAME)
    meta_path = os.path.join(opt_dir, _BEST_META_BASENAME)
    numpy.savez(
        npz_path,
        Q=numpy.asarray(artifacts['Q']),
        rewards=numpy.asarray(artifacts['rewards']),
    )
    with open(meta_path, 'w', encoding='utf-8') as _f:
        json.dump({'value': float(artifacts['value']),
                   'trial_number': int(artifacts.get('trial_number', -1))},
                  _f, indent=2)


def _load_best_artifacts(opt_dir: str, artifacts: dict) -> bool:
    """Load best artifacts from ``opt_dir`` (mutates *artifacts*).

    Returns ``True`` on success. Loaded with ``allow_pickle=False``.
    """
    npz_path = os.path.join(opt_dir, _BEST_NPZ_BASENAME)
    meta_path = os.path.join(opt_dir, _BEST_META_BASENAME)
    if not (os.path.isfile(npz_path) and os.path.isfile(meta_path)):
        return False
    with numpy.load(npz_path, allow_pickle=False) as _data:
        artifacts['Q'] = _data['Q']
        artifacts['rewards'] = list(_data['rewards'])
    with open(meta_path, 'r', encoding='utf-8') as _f:
        meta = json.load(_f)
    artifacts['value'] = float(meta['value'])
    artifacts['trial_number'] = int(meta.get('trial_number', -1))
    return True

# Set by main() so objective() can persist _best_artifacts to disk
_opt_dir: str = ''


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
    Optuna objective: train Q-Learning with sampled hyperparameters,
    evaluate on the fixed 64 sequences, and return mean normalised performance.

    Parameters
    ----------
    trial : optuna.Trial
        The current Optuna trial. Used to sample the eight hyperparameters
        listed in the module docstring.

    Returns
    -------
    float
        Mean normalised final-severity performance over the 64 evaluation
        sequences (higher is better, theoretical maximum 1.0).

    Side Effects
    ------------
    - Updates the module-level ``_best_artifacts`` dict whenever the trial
      improves on the running best, and persists it to ``_best_artifacts.npz``
      (Q-table + rewards) and ``_best_artifacts.json`` (metadata) under
      ``_opt_dir`` so the run is resumable. The legacy ``_best_weights.pkl``
      format from pre-2026-04-20 runs is refused on load (CWE-502 hardening).
    - Sets four ``user_attr`` values on ``trial`` (mean/std/min/max perf).

    Notes
    -----
    - Each trial seeds the training with ``SEED + trial.number + 1`` so that
      different trials explore different stochastic trajectories while
      remaining individually reproducible.
    - ``track_confidence=False`` skips the per-step entropy bookkeeping that
      is only needed for the human-vs-agent reporting in ``__main__``.
    - Evaluation masks infeasible actions with a very negative sentinel
      (``-1e9``) before ``argmax`` so they are never selected; this matches
      the masking applied by ``train_rl.py`` and ``__main__.py``.
    """
    # --- Sample hyperparameters ---
    learning_rate    = trial.suggest_float('learning_rate',    0.05, 0.30,   log=True)
    discount_factor  = trial.suggest_float('discount_factor',  0.90, 0.999)
    epsilon_initial  = trial.suggest_float('epsilon_initial',  0.50, 1.00)
    epsilon_min      = trial.suggest_float('epsilon_min',      0.01, 0.10,   log=True)
    num_episodes     = trial.suggest_int('num_episodes',       150000, 500000, step=10000)
    warmup_ratio     = trial.suggest_float('warmup_ratio',     0.02, 0.15)
    target_ratio     = trial.suggest_float('target_ratio',     0.40, 0.80)
    penalty_coeff    = trial.suggest_float('penalty_coeff',    1e-4, 0.30,   log=True)

    # --- Train ---
    env = Pandemic()
    assert _number_cities_prob is not None and _severity_prob is not None
    env.number_cities_prob = _number_cities_prob
    env.severity_prob = _severity_prob
    env.verbose = False

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

    rewards, Q, _ = QLearning(
        env, learning_rate, discount_factor,
        epsilon_initial, epsilon_min, num_episodes,
        warmup_ratio=warmup_ratio, target_ratio=target_ratio,
        double_q=True, penalty_coeff=penalty_coeff,
        seed=SEED + trial.number + 1,
        track_confidence=False,
        progress_callback=_progress,
    )

    if _pruned['flag']:
        raise optuna.TrialPruned()

    # --- Evaluate on fixed sequences ---
    env_eval = Pandemic()
    env_eval.verbose = False

    def qf(_env, state, _seqid):
        s0 = min(int(state[0]), Q.shape[0] - 1)
        s1 = min(int(state[1]), Q.shape[1] - 1)
        s2 = min(int(state[2]), Q.shape[2] - 1)
        # Mask infeasible actions. Q-values are negative (rewards = -Σ severities),
        # so the previous sentinel +0.00001 made every infeasible action look better
        # than every feasible one, forcing argmax to pick an infeasible index that
        # the env then clamped to "spend all remaining resources". Use a very
        # negative sentinel so infeasible actions are guaranteed to lose argmax.
        options = Q[s0, s1, s2].copy()
        o = numpy.arange(len(options), dtype=numpy.float32)
        options[o > state[0]] = -1e9
        return numpy.argmax(options)

    _, perfs, _ = run_experiment(env_eval, qf, False, _trials_per_sequence, _sevs)
    mean_perf = float(numpy.mean(perfs))

    # Store extra info for later analysis
    trial.set_user_attr('mean_perf', mean_perf)
    trial.set_user_attr('std_perf', float(numpy.std(perfs)))
    trial.set_user_attr('min_perf', float(numpy.min(perfs)))
    trial.set_user_attr('max_perf', float(numpy.max(perfs)))

    # Preserve the best Q-table to avoid lossy retraining at the end
    global _best_artifacts
    if mean_perf > _best_artifacts['value']:
        _best_artifacts['Q'] = Q.copy()
        _best_artifacts['rewards'] = list(rewards)
        _best_artifacts['value'] = mean_perf
        _best_artifacts['trial_number'] = trial.number

        # Persist to disk so that --resume can recover without retraining.
        # Pickle-free format (CWE-502 hardening).
        if _opt_dir:
            _save_best_artifacts(_opt_dir, _best_artifacts)

    return mean_perf


###################################
##        Reporting              ##
###################################
def _save_report(study, opt_dir, opt_date, best_Q, best_rewards):
    """Generate and save optimization results report and visualizations.

    Parameters
    ----------
    study : optuna.Study
        The completed Optuna study to summarise.
    opt_dir : str
        Output directory where the report, plots and ``.npy`` files are written.
    opt_date : str
        Date string (``YYYY-MM-DD``) used in the output filenames.
    best_Q : numpy.ndarray
        Q-table from the best trial; saved as ``q_best_<date>.npy``.
    best_rewards : numpy.ndarray
        Reward history of the best trial; saved as ``rewards_best_<date>.npy``.

    Notes
    -----
    Trial numbers are converted to 1-based indexing (trial.number + 1)
    so they match the trial_id column in the Optuna SQLite database.
    """

    best = study.best_trial

    # Trial-state breakdown.  Without this, "Total trials: 100" hides the
    # fact that MedianPruner may have aborted most of them — the user then
    # sees only ~14 entries with mean_perf in [0,1] in the table below and
    # interprets it as "the run stopped at 14/100" (it did not).
    completed_trials = [t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials    = [t for t in study.trials
                        if t.state == optuna.trial.TrialState.PRUNED]
    failed_trials    = [t for t in study.trials
                        if t.state == optuna.trial.TrialState.FAIL]

    # --- Text report ---
    report_file = os.path.join(opt_dir, f'optimization_results_{opt_date}.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BAYESIAN OPTIMIZATION RESULTS — Q-LEARNING HYPERPARAMETERS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Date:              {opt_date}\n")
        f.write(f"Total trials:      {len(study.trials)}"
                f"  (Completed: {len(completed_trials)}"
                f" | Pruned: {len(pruned_trials)}"
                f" | Failed: {len(failed_trials)})\n")
        f.write(f"Best trial:        #{best.number} (0-indexed; matches best_params.json)\n")
        f.write(f"Best mean perf:    {best.value:.6f}\n\n")

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

        f.write("ALL COMPLETED TRIALS (sorted by mean_perf)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'#':>4s}  {'mean_perf':>10s}  {'lr':>10s}  {'gamma':>8s}  {'eps0':>6s}  {'eps_min':>7s}  {'episodes':>8s}  {'warmup':>6s}  {'target':>6s}  {'beta':>8s}\n")
        f.write("-" * 100 + "\n")
        # IMPORTANT: ``t.value`` for PRUNED trials is the last intermediate
        # ``trial.report()`` value (running training reward, very negative),
        # NOT mean_perf.  Filter to COMPLETE so the column is consistent and
        # read mean_perf from user_attrs (set inside ``objective``).
        for t in sorted(completed_trials,
                        key=lambda t: t.user_attrs.get('mean_perf',
                                                       t.value if t.value is not None else -1.0),
                        reverse=True):
            mean_perf = t.user_attrs.get('mean_perf', t.value)
            if mean_perf is None:
                continue
            p = t.params
            f.write(
                f"{t.number:4d}  {mean_perf:10.6f}  "
                f"{p['learning_rate']:10.5f}  {p['discount_factor']:8.4f}  "
                f"{p['epsilon_initial']:6.3f}  {p['epsilon_min']:7.4f}  "
                f"{p['num_episodes']:8d}  "
                f"{p.get('warmup_ratio', 0.05):6.3f}  "
                f"{p.get('target_ratio', 0.66):6.3f}  "
                f"{p.get('penalty_coeff', 0.0):8.5f}\n"
            )

        if pruned_trials:
            f.write("\n")
            f.write(f"PRUNED TRIALS ({len(pruned_trials)} total) — aborted by MedianPruner\n")
            f.write("-" * 80 + "\n")
            f.write("Trial numbers: " +
                    ", ".join(str(t.number) for t in pruned_trials) + "\n")

    success(f"Report saved: optimization_results_{opt_date}.txt")

    # --- Convergence plot ---
    try:
        plt.style.use('ggplot')
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(12, 6))
    # Same caveat as the text report: ``t.value`` for PRUNED trials is the
    # last running training reward, NOT mean_perf.  Plot only COMPLETE
    # trials, using user_attrs['mean_perf'] for a consistent y-axis.
    trial_numbers = [t.number for t in completed_trials
                     if t.user_attrs.get('mean_perf') is not None]
    trial_values  = [t.user_attrs['mean_perf'] for t in completed_trials
                     if t.user_attrs.get('mean_perf') is not None]
    # Sort by trial number so the running-best curve is monotonic in time.
    if trial_numbers:
        order = sorted(range(len(trial_numbers)), key=lambda i: trial_numbers[i])
        trial_numbers = [trial_numbers[i] for i in order]
        trial_values  = [trial_values[i]  for i in order]

    # Running best
    running_best = []
    current_best = -float('inf')
    for v in trial_values:
        current_best = max(current_best, v)
        running_best.append(current_best)

    ax.scatter(trial_numbers, trial_values, color='#1f77b4', s=50, alpha=0.6,
               edgecolors='navy', linewidth=0.5, label='Trial performance', zorder=3)
    ax.plot(trial_numbers, running_best, color='#d62728', linewidth=2.5,
            label='Best so far', zorder=4)
    ax.set_xlabel('Trial number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean normalised performance', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Bayesian Optimisation: Convergence  '
        f'({len(completed_trials)} completed, {len(pruned_trials)} pruned)',
        fontsize=14, fontweight='bold', pad=20
    )
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(opt_dir, f'optimization_history_{opt_date}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    list_item(f"Saved: optimization_history_{opt_date}.png")

    # --- Hyperparameter importance ---
    try:
        importances = optuna.importance.get_param_importances(study)
        names  = list(importances.keys())
        values = list(importances.values())

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(names[::-1], values[::-1], color='#2ca02c', edgecolor='darkgreen', linewidth=0.5)
        ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
        ax.set_title('Hyperparameter Importance', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        fig.savefig(os.path.join(opt_dir, f'hyperparameter_importances_{opt_date}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        list_item(f"Saved: hyperparameter_importances_{opt_date}.png")
    except Exception as e:
        info(f"Could not compute importances: {e}")

    # --- Save best Q-table and rewards (dated copy under the opt directory) ---
    numpy.save(os.path.join(opt_dir, f'q_best_{opt_date}.npy'), best_Q)
    numpy.save(os.path.join(opt_dir, f'rewards_best_{opt_date}.npy'), best_rewards)
    success(f"Best Q-table saved: q_best_{opt_date}.npy")
    success(f"Best rewards saved: rewards_best_{opt_date}.npy")

    # --- Save best hyperparameters as JSON (consumed by train_rl.py) ---
    # The seed must equal SEED + best_trial_number + 1 so train_rl reproduces
    # the exact Q-table from the best trial, yielding the same mean_perf.
    best_trial_number = _best_artifacts.get('trial_number', best.number)
    best_params_payload = {
        'opt_date':           opt_date,
        'best_trial_number':  int(best_trial_number),
        'mean_perf':          float(best.value),
        'std_perf':           float(best.user_attrs.get('std_perf',  float('nan'))),
        'min_perf':           float(best.user_attrs.get('min_perf',  float('nan'))),
        'max_perf':           float(best.user_attrs.get('max_perf',  float('nan'))),
        # Canonical name (matches pes_dqn/pes_a2c).  Old 'seed' key kept
        # as an alias so sidecars produced before 2026-04-21 still load.
        'trial_seed':         int(SEED + best_trial_number + 1),
        'seed':               int(SEED + best_trial_number + 1),
        'track_confidence':   False,
        'double_q':           True,
        'hyperparameters':    {k: float(v) if not isinstance(v, int) else int(v)
                               for k, v in best.params.items()},
    }
    params_file = os.path.join(opt_dir, f'best_params_{opt_date}.json')
    with open(params_file, 'w', encoding='utf-8') as _f:
        json.dump(best_params_payload, _f, indent=2)
    success(f"Best params saved: best_params_{opt_date}.json")

    # --- Mirror the best artifacts to the standard input paths so that
    # train_rl.py and __main__.py pick them up without manual copying. ---
    std_q       = os.path.join(INPUTS_PATH, 'q.npy')
    std_rewards = os.path.join(INPUTS_PATH, 'rewards.npy')
    std_params  = os.path.join(INPUTS_PATH, 'best_params.json')
    numpy.save(std_q, best_Q)
    numpy.save(std_rewards, best_rewards)
    with open(std_params, 'w', encoding='utf-8') as _f:
        json.dump(best_params_payload, _f, indent=2)
    success(f"Mirrored to {std_q}")
    success(f"Mirrored to {std_rewards}")
    success(f"Mirrored to {std_params}")

    # --- Reproducibility fingerprint (numpy/python/CSV-hashes/SEED/git) ---
    # Used by train_rl.py to warn if the local env diverges from the one
    # that produced the best trial.  Mirrors the pes_ql workflow.
    fp_path = save_fingerprint(opt_dir, opt_date)
    success(f"Repro fingerprint saved: {os.path.basename(fp_path)}")


###################################
##             Main              ##
###################################
def main():
    """Run Bayesian optimisation of Q-Learning hyperparameters via Optuna."""

    header("BAYESIAN OPTIMISATION — Q-LEARNING HYPERPARAMETERS", width=80)

    # Parse arguments: [n_trials] [--resume YYYY-MM-DD] [--out-dir PATH] [--storage URL]
    n_trials = 50
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

    opt_dir  = out_dir_override or os.path.join(INPUTS_PATH, f'{opt_date}_BAYESIAN_OPT')
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
    list_item(f"Sequence lengths shape: {_trials_per_sequence.shape}")
    list_item(f"Sequences loaded: {len(_sevs)}")
    print()

    # --- Run optimisation ---
    section("Running Bayesian Optimisation", width=80)
    info("Search space:")
    list_item("learning_rate    ∈ [0.05, 0.30]   (log scale)")
    list_item("discount_factor  ∈ [0.90, 0.999]")
    list_item("epsilon_initial  ∈ [0.50, 1.00]")
    list_item("epsilon_min      ∈ [0.01, 0.10]   (log scale)")
    list_item("num_episodes     ∈ [150000, 500000]  (step=10000)")
    list_item("warmup_ratio     ∈ [0.02, 0.15]")
    list_item("target_ratio     ∈ [0.40, 0.80]")
    list_item("penalty_coeff    ∈ [1e-4, 0.30]   (log scale, PBRS)")
    print()

    # Suppress Optuna's verbose default logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Use SQLite storage so the study survives machine suspensions/crashes
    db_path = os.path.join(opt_dir, f'optuna_study_{opt_date}.db')
    storage = storage_override or f'sqlite:///{db_path}'

    study = optuna.create_study(
        direction='maximize',
        study_name=f'qlearning_opt_{opt_date}',
        sampler=optuna.samplers.TPESampler(
            seed=SEED,
            n_startup_trials=10,
            multivariate=True,
            group=True,
        ),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=4),
        storage=storage,
        load_if_exists=True,   # Resume from previous run if DB exists
    )

    # Calculate how many trials remain (allows seamless resume)
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - completed)
    if completed > 0:
        info(f"Resuming: {completed} trials already completed, {remaining} remaining")
    else:
        info(f"Starting fresh: {n_trials} trials to run")

    t_start = time.time()

    # Callback to print progress.  Robust against:
    #   - pruned / failed trials whose ``trial.value`` is ``None``
    #   - the early phase where no trial has COMPLETE state yet, in which
    #     case ``study.best_value`` raises ``ValueError``.
    def _progress_callback(study, trial):
        done = len([t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE])
        elapsed = time.time() - t_start

        # Format this trial's value.  IMPORTANT: ``trial.value`` is NOT
        # ``mean_perf`` for pruned/failed trials — it is the last
        # intermediate value reported via ``trial.report(avg_reward, step)``
        # for the MedianPruner, which is the running training reward
        # (very negative).  For COMPLETE trials we read the real
        # ``mean_perf`` from user_attrs so the log/heartbeat shows the same
        # metric Optuna is actually maximising.
        state = trial.state
        if state == optuna.trial.TrialState.COMPLETE:
            mean_perf = trial.user_attrs.get('mean_perf', trial.value)
            value_str = f"{mean_perf:8.4f}" if mean_perf is not None else "    n/a "
        else:
            state_label = state.name if state is not None else 'UNKNOWN'
            value_str = f"{state_label:>8s}"

        # best_value() raises if no trial has completed yet
        try:
            best_val = study.best_value
            best_str = f"{best_val:.4f}"
        except ValueError:
            best_val = None
            best_str = "  n/a  "

        print(
            f"  Trial {done:3d}/{n_trials}  |  "
            f"perf={value_str}  |  best={best_str}  |  "
            f"elapsed={elapsed:.0f}s"
        )

    if remaining > 0:
        # Ensure underflow is ignored during optimisation (Optuna's TPE
        # sampler computes exp() of very negative values, which underflow
        # harmlessly to 0.0)
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
    success(f"Optimisation finished in {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    info(f"Total completed trials: {total_completed}")
    print()

    # --- Best trial summary ---
    best = study.best_trial
    section("Best Hyperparameters Found", width=80)
    for name, val in best.params.items():
        list_item(f"{name:<25s} = {val}")
    info(f"Mean normalised performance: {best.value:.6f}")
    print()

    # --- Use best Q-table from optimization, or retrain only if resuming ---
    section("Best Q-Table", width=80)

    # Try in-memory artifacts first, then disk fallback from --resume
    pkl_path = os.path.join(opt_dir, '_best_weights.pkl')
    if _best_artifacts['Q'] is None:
        if _load_best_artifacts(opt_dir, _best_artifacts):
            if _best_artifacts.get('trial_number', -1) < 0:
                _best_artifacts['trial_number'] = best.number
            info(f"Loaded best Q-table from disk (perf={_best_artifacts['value']:.6f})")
        elif os.path.isfile(pkl_path):
            # Legacy pickle artifact from a pre-2026-04-20 run — refused for
            # security (CWE-502). Falls through to deterministic replay below.
            info(f"Found legacy pickle artifact at {pkl_path} — ignored.")

    if _best_artifacts['Q'] is not None and _best_artifacts['value'] >= best.value:
        best_Q = _best_artifacts['Q']
        best_rewards = numpy.array(_best_artifacts['rewards'])
        success("Using Q-table from best optimization trial (no retraining needed)")
    else:
        info("Retraining with best hyperparameters (resumed study, original Q-table not in memory)...")
        bp = best.params
        env_final = Pandemic()
        assert _number_cities_prob is not None and _severity_prob is not None
        env_final.number_cities_prob = _number_cities_prob
        env_final.severity_prob = _severity_prob
        env_final.verbose = False

        best_rewards, best_Q, _ = QLearning(
            env_final,
            bp['learning_rate'],
            bp['discount_factor'],
            bp['epsilon_initial'],
            bp['epsilon_min'],
            bp['num_episodes'],
            warmup_ratio=bp.get('warmup_ratio', 0.05),
            target_ratio=bp.get('target_ratio', 0.66),
            double_q=True,
            penalty_coeff=bp.get('penalty_coeff', 0.0),
            seed=SEED + best.number + 1,
            track_confidence=False,
        )
        # Record so the JSON written by _save_report uses the right trial number
        _best_artifacts['trial_number'] = best.number
        success(f"Retrained Q-table (deterministic — seed = {SEED + best.number + 1})")

    list_item(f"Q-table shape: {best_Q.shape}")
    print()

    # --- Save everything ---
    section("Saving Results", width=80)
    _save_report(study, opt_dir, opt_date, best_Q, best_rewards)

    print()
    section("Optimisation Complete", width=80)
    success("All outputs saved!")
    info(f"Output directory: {opt_dir}")
    print()


if __name__ == '__main__':
    main()
