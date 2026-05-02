'''
pes_ql - Pandemic Experiment Scenario

Bayesian Optimization of Q-Learning hyperparameters using Optuna.

Optimizes: learning_rate, discount_factor, epsilon_initial, epsilon_min, num_episodes
Objective: maximize mean normalised performance over the 64 evaluation sequences.

The evaluation uses infeasible-action masking (actions > available resources are
suppressed before argmax) so that the metric matches the behaviour of the RL agent
in __main__.py.  The best Q-table found during the search is preserved in memory
and saved directly, avoiding a lossy re-training step.

Usage:
    python3 -m pes_ql.ext.optimize_rl [n_trials] [--resume YYYY-MM-DD]

    n_trials : int, optional
        Number of Bayesian optimization trials (default: 100).
    --resume YYYY-MM-DD : str, optional
        Resume a previous optimization run stored under that date.

Search space:
    learning_rate    ∈ [0.05, 0.40]       (log scale)
    discount_factor  ∈ [0.85, 0.999]
    epsilon_initial  ∈ [0.50, 1.00]
    epsilon_min      ∈ [0.01, 0.15]
    num_episodes     ∈ [500000, 1200000]  (step=50000)

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
import json
import os
import sys
import time
import numpy
import warnings
import optuna
import matplotlib  # noqa: E402
matplotlib.use('Agg', force=True)  # non-interactive backend; prevents NoneType canvas crash on headless servers
import matplotlib.pyplot as plt
from datetime import datetime

# Force TensorFlow to use CPU
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*Box.*precision lowered.*')
warnings.filterwarnings('ignore', message='.*A NumPy version.*SciPy.*')

##########################
##  Internal imports    ##
##########################
from .pandemic import Pandemic, run_experiment, QLearning
from .repro import save_artifacts as save_repro_artifacts
from ..src.terminal_utils import header, section, success, info, list_item
from .tools import convert_globalseq_to_seqs
from ..config.CONFIG import SEED
from .. import INPUTS_PATH

try:
    from utils.scripts.notify import notify
except ImportError:
    def notify(*_args, **_kwargs):
        """No-op fallback when ``utils.scripts.notify`` is not importable."""
        return None

# Nombre del paquete para las notificaciones push
_PKG_NAME = __package__.split('.', maxsplit=1)[0] if __package__ else 'mPES'


###################################
##    Global evaluation data     ##
###################################
# Loaded once at startup and reused by every trial
_trials_per_sequence = None
_sevs = None
_number_cities_prob = None
_severity_prob = None

# Store best Q-table/rewards during optimization to avoid lossy retraining
_best_artifacts = {'Q': None, 'rewards': None, 'value': float('-inf')}

# Set by main() so objective() can persist _best_artifacts to disk
_opt_dir: str = ''


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
        json.dump({'value': float(artifacts['value'])}, _f, indent=2)


def _load_best_artifacts(opt_dir: str, artifacts: dict) -> bool:
    """Load best artifacts from ``opt_dir`` (mutates *artifacts*).

    Returns ``True`` on success, ``False`` if no .npz/.json pair is present.
    Loaded with ``allow_pickle=False`` to block CWE-502.
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
    """Optuna objective for Q-Learning hyperparameter search.

    Samples ``learning_rate``, ``discount_factor``, ``epsilon_initial``,
    ``epsilon_min`` and ``num_episodes``, trains a Q-table with
    ``track_confidence=False`` using the per-trial seed
    ``SEED + trial.number + 1``, evaluates the greedy policy with
    infeasible-action masking on the fixed 64 sequences, sanitises
    ``NaN``/``Inf`` to 0 and clips to ``[0, 1]`` so the TPE surrogate
    is not poisoned, then updates ``_best_artifacts`` (and persists it
    to ``_opt_dir`` without ``pickle``) when the score improves.

    Parameters
    ----------
    trial : optuna.Trial
        The Optuna trial driving the hyperparameter sampling and
        intermediate-value reporting (used by ``MedianPruner``).

    Returns
    -------
    float
        Mean normalised performance over the 64 evaluation sequences,
        clipped to ``[0, 1]``.

    Raises
    ------
    optuna.TrialPruned
        Raised when ``MedianPruner`` decides to abort the trial via
        ``trial.should_prune()`` during intermediate reporting.
    """
    # --- Sample hyperparameters ---
    learning_rate = trial.suggest_float('learning_rate', 0.05, 0.40, log=True)
    discount_factor = trial.suggest_float('discount_factor', 0.85, 0.999)
    epsilon_initial = trial.suggest_float('epsilon_initial', 0.50, 1.00)
    epsilon_min = trial.suggest_float('epsilon_min', 0.01, 0.15)
    num_episodes = trial.suggest_int('num_episodes', 500_000, 1_200_000, step=50_000)

    # --- Train ---
    env = Pandemic()
    assert _number_cities_prob is not None and _severity_prob is not None
    env.number_cities_prob = _number_cities_prob
    env.severity_prob = _severity_prob
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

    rewards, Q, _ = QLearning(
        env, learning_rate, discount_factor,
        epsilon_initial, epsilon_min, num_episodes,
        seed=trial_seed,
        progress_callback=_progress,
        track_confidence=False,
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
        # so a small positive sentinel would dominate every feasible Q-value and
        # make argmax pick an infeasible index that the env then clamps to
        # "spend all remaining resources". Use a very negative sentinel.
        options = Q[s0, s1, s2].copy()
        o = numpy.arange(len(options), dtype=numpy.float32)
        options[o > state[0]] = -1e9
        return numpy.argmax(options)

    _, perfs, _ = run_experiment(env_eval, qf, False, _trials_per_sequence, _sevs)
    mean_perf = float(numpy.mean(perfs))

    # Sanitise objective: NaN/Inf can appear if any sequence is degenerate
    # (WorstCase == BestCase ⇒ divide-by-zero in the normalised metric),
    # and a barely-trained Q-table can produce values < 0 or > 1. Either
    # case poisons Optuna's TPE surrogate, so clamp to the valid range.
    if not numpy.isfinite(mean_perf):
        mean_perf = 0.0
    mean_perf = float(numpy.clip(mean_perf, 0.0, 1.0))

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

        # Persist to disk so that --resume can recover without retraining.
        # Pickle-free format (CWE-502 hardening): the ndarray payload goes
        # into _best_artifacts.npz (loaded with allow_pickle=False) and the
        # scalar metadata into _best_artifacts.json.
        if _opt_dir:
            _save_best_artifacts(_opt_dir, _best_artifacts)

    return mean_perf


###################################
##        Reporting              ##
###################################
def _save_report(study, opt_dir, opt_date, best_Q, best_rewards):
    """Generate and save optimization results report and visualizations.

    Trial numbers are converted to 1-based indexing (trial.number + 1)
    so they match the trial_id column in the Optuna SQLite database.
    """

    best = study.best_trial

    # --- Text report ---
    report_file = os.path.join(opt_dir, f'optimization_results_{opt_date}.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BAYESIAN OPTIMIZATION RESULTS — Q-LEARNING HYPERPARAMETERS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Date:              {opt_date}\n")
        f.write(f"Total trials:      {len(study.trials)}\n")
        f.write(f"Best trial:        #{best.number + 1}\n")
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

        f.write("ALL TRIALS\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'#':>4s}  {'mean_perf':>10s}  {'lr':>10s}  "
            f"{'gamma':>8s}  {'eps0':>6s}  {'eps_min':>7s}  "
            f"{'episodes':>8s}\n"
        )
        f.write("-" * 80 + "\n")
        for t in sorted(study.trials, key=lambda t: t.value if t.value is not None else -1, reverse=True):
            if t.value is None:
                continue
            p = t.params
            f.write(
                f"{t.number + 1:4d}  {t.value:10.6f}  "
                f"{p['learning_rate']:10.5f}  {p['discount_factor']:8.4f}  "
                f"{p['epsilon_initial']:6.3f}  {p['epsilon_min']:7.4f}  "
                f"{p['num_episodes']:8d}\n"
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
    ax.set_title('Bayesian Optimisation: Convergence', fontsize=14, fontweight='bold', pad=20)
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
        ax.set_title('Hyperparameter Importance', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        fig.savefig(os.path.join(opt_dir, f'hyperparameter_importances_{opt_date}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        list_item(f"Saved: hyperparameter_importances_{opt_date}.png")
    except Exception as e:
        info(f"Could not compute importances: {e}")

    # --- Save best Q-table and rewards ---
    numpy.save(os.path.join(opt_dir, f'q_best_{opt_date}.npy'), best_Q)
    numpy.save(os.path.join(opt_dir, f'rewards_best_{opt_date}.npy'), best_rewards)
    success(f"Best Q-table saved: q_best_{opt_date}.npy")
    success(f"Best rewards saved: rewards_best_{opt_date}.npy")


###################################
##             Main              ##
###################################
def main():
    """Run Bayesian optimisation of Q-Learning hyperparameters.

    Parses CLI arguments (``[n_trials] [--resume YYYY-MM-DD]
    [--out-dir PATH] [--storage URL]``), loads the fixed evaluation
    data once, creates or resumes an Optuna study backed by SQLite,
    runs ``objective`` for the remaining trials with a TPE sampler
    (seeded with ``SEED``) and a ``MedianPruner``, and finally writes
    the best Q-table, reward history, hyperparameter sidecar,
    reproducibility fingerprint, optimisation report and convergence
    plots to ``inputs/<date>_BAYESIAN_OPT/``. The standard input paths
    (``q.npy``, ``rewards.npy``, ``best_params.json``) are mirrored
    automatically so ``train_rl.py`` and ``__main__.py`` can consume
    them without manual copying.

    Returns
    -------
    None
        All artefacts are written to disk; nothing is returned.
    """

    header("BAYESIAN OPTIMISATION — Q-LEARNING HYPERPARAMETERS", width=80)

    # Parse arguments: [n_trials] [--resume YYYY-MM-DD] [--out-dir PATH] [--storage URL]
    n_trials = 100
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
    info("Search space:")
    list_item("learning_rate    ∈ [0.05, 0.40]      (log scale)")
    list_item("discount_factor  ∈ [0.85, 0.999]")
    list_item("epsilon_initial  ∈ [0.50, 1.00]")
    list_item("epsilon_min      ∈ [0.01, 0.15]")
    list_item("num_episodes     ∈ [500000, 1200000]  (step=50000)")
    print()

    # Suppress Optuna's verbose default logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Use SQLite storage so the study survives machine suspensions/crashes
    db_path = os.path.join(opt_dir, f'optuna_study_{opt_date}.db')
    storage = storage_override or f'sqlite:///{db_path}'

    study = optuna.create_study(
        direction='maximize',
        study_name=f'qlearning_opt_{opt_date}',
        sampler=optuna.samplers.TPESampler(seed=SEED),
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
        # Notificar cada 10 trials completados
        if done > 0 and done % 10 == 0:
            best_msg = f"{best_val:.6f}" if best_val is not None else "n/a"
            notify(
                f"[{_PKG_NAME}] {done}/{n_trials} trials",
                f"Se completaron {done} de {n_trials} trials.\n"
                f"Mejor valor hasta ahora: {best_msg}\n"
                f"Último trial: value={value_str}\n"
                f"Tiempo transcurrido: {elapsed:.0f}s ({elapsed / 60:.1f} min)",
                tags="chart_with_upwards_trend"
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

    # --- Best Q-table from optimisation (or disk fallback for --resume) ---
    # We deliberately persist the Q-table from the WINNING TRIAL (trained with
    # ``seed = SEED + trial.number + 1``). train_rl.py is configured to use the
    # same per-trial seed, so a local re-train with the same hyperparameters
    # produces a bit-for-bit identical Q-table and thus identical mean_perf.
    section("Best Q-Table", width=80)

    pkl_path = os.path.join(opt_dir, '_best_weights.pkl')
    if _best_artifacts['Q'] is None:
        if _load_best_artifacts(opt_dir, _best_artifacts):
            info(f"Loaded best Q-table from disk (perf={_best_artifacts['value']:.6f})")
        elif os.path.isfile(pkl_path):
            # Legacy pickle artifact from a pre-2026-04-20 run — refused for
            # security (CWE-502). Falls through to deterministic replay below.
            info(f"Found legacy pickle artifact at {pkl_path} — ignored.")

    if _best_artifacts['Q'] is not None and _best_artifacts['value'] >= best.value:
        best_Q = _best_artifacts['Q']
        best_rewards = numpy.array(_best_artifacts['rewards'])
        success("Using Q-table from best optimisation trial (no retraining needed)")
    else:
        # Resumed study and the in-memory cache is empty: replay the winning
        # trial deterministically using the SAME seed Optuna used for it.
        info("Replaying best trial deterministically (resumed study)...")
        bp = best.params
        env_final = Pandemic()
        env_final.number_cities_prob = _number_cities_prob  # type: ignore[assignment]
        env_final.severity_prob = _severity_prob  # type: ignore[assignment]
        env_final.verbose = False

        replay_seed = SEED + int(best.number) + 1
        best_rewards, best_Q, _ = QLearning(
            env_final,
            bp['learning_rate'],
            bp['discount_factor'],
            bp['epsilon_initial'],
            bp['epsilon_min'],
            bp['num_episodes'],
            seed=replay_seed,
            track_confidence=False,
        )
        success(f"Replayed Q-table (seed={replay_seed} = SEED + {best.number} + 1)")

    list_item(f"Q-table shape: {best_Q.shape}")
    list_item(f"Best trial number (0-based): {best.number}")
    list_item(f"Training seed for replay:    {SEED + int(best.number) + 1}")
    list_item(f"Expected mean_perf:          {best.value:.6f}")
    print()

    # --- Save everything ---
    section("Saving Results", width=80)
    _save_report(study, opt_dir, opt_date, best_Q, best_rewards)

    # --- Persist hyperparameters + reproducibility fingerprint ---
    # ``best.value`` is ``float | None`` per Optuna's stubs; the surrounding code
    # only enters this branch after at least one completed trial, so the cast is safe.
    _best_value = float(best.value) if best.value is not None else float('nan')
    params_path, fp_path = save_repro_artifacts(
        opt_dir, opt_date, best.params,
        trial_number=int(best.number),
        expected_mean_perf=_best_value,
    )
    success(f"Hyperparameters saved: {os.path.basename(params_path)}")
    success(f"Repro fingerprint saved: {os.path.basename(fp_path)}")

    # --- Mirror to standard input paths so train_rl.py / __main__.py pick
    # them up without manual copying (matches pes_dql/dqn/a2c workflow). ---
    import shutil  # local import: only used by this one-shot mirror block
    std_params  = os.path.join(INPUTS_PATH, 'best_params.json')
    std_q       = os.path.join(INPUTS_PATH, 'q.npy')
    std_rewards = os.path.join(INPUTS_PATH, 'rewards.npy')
    shutil.copyfile(params_path, std_params)
    numpy.save(std_q, best_Q)
    numpy.save(std_rewards, best_rewards)
    success(f"Mirrored to {std_params}")
    success(f"Mirrored to {std_q}")
    success(f"Mirrored to {std_rewards}")

    print()
    section("Optimisation Complete", width=80)
    success("All outputs saved!")
    info(f"Output directory: {opt_dir}")
    print()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        import traceback
        notify(
            f"[{_PKG_NAME}] ERROR en optimización",
            f"Se produjo un error durante la optimización:\n\n"
            f"{traceback.format_exc()}",
            priority="urgent", tags="rotating_light"
        )
        raise
