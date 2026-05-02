'''
pes_ql - Recovery script for an interrupted Bayesian optimisation.

Use this when ``optimize_rl.py`` reached the final ``_save_report`` step
but crashed before writing the Q-table, ``best_params.json``,
``repro_fingerprint.json`` and the mirrored ``inputs/`` files
(typical cause: matplotlib backend failure on Colab/headless runs —
see commit on 2026-04-22).

The script reads:
    - ``optuna_study_<date>.db`` (Optuna SQLite storage, with all trials)
    - ``_best_artifacts.npz`` + ``_best_artifacts.json`` (best Q-table cache)

…and writes everything that ``optimize_rl.main()`` would have written:
    - ``q_best_<date>.npy``
    - ``rewards_best_<date>.npy``
    - ``best_params_<date>.json``
    - ``repro_fingerprint_<date>.json``
    - ``optimization_results_<date>.txt``  (overwritten with COMPLETE-only filter)
    - ``optimization_history_<date>.png``
    - ``hyperparameter_importances_<date>.png``
    - mirror to ``pes_ql/inputs/{q.npy, rewards.npy, best_params.json}``

Usage::

    python3 -m pes_ql.ext.recover_optimization <opt_dir> [--date YYYY-MM-DD]

Example::

    python3 -m pes_ql.ext.recover_optimization \\
        pes_ql/inputs/2026-04-22_BAYESIAN_OPT
'''

##########################
##  External imports    ##
##########################
import os
import sys
import re
import shutil
import matplotlib
matplotlib.use('Agg')   # headless-safe, must precede pyplot import
import numpy
import optuna

##########################
##  Internal imports    ##
##########################
from . import optimize_rl as _opt
from .repro import save_artifacts as save_repro_artifacts
from ..src.terminal_utils import header, section, success, info, list_item
from .. import INPUTS_PATH


def _infer_date_from_dir(opt_dir: str) -> str:
    """Extract ``YYYY-MM-DD`` from a directory name like ``2026-04-22_BAYESIAN_OPT``."""
    base = os.path.basename(os.path.normpath(opt_dir))
    m = re.match(r'(\d{4}-\d{2}-\d{2})', base)
    if not m:
        raise ValueError(
            f"Cannot infer date from {base!r}. Pass --date YYYY-MM-DD explicitly."
        )
    return m.group(1)


def main():
    """Reconstruct missing artefacts from a partial Bayesian optimisation run."""
    header("BAYESIAN OPTIMISATION — ARTEFACT RECOVERY", width=80)

    # --- Parse arguments ---
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    opt_dir = os.path.abspath(sys.argv[1])
    opt_date = None
    if '--date' in sys.argv:
        opt_date = sys.argv[sys.argv.index('--date') + 1]
    if opt_date is None:
        opt_date = _infer_date_from_dir(opt_dir)

    info(f"Optimisation dir: {opt_dir}")
    info(f"Optimisation date: {opt_date}")
    print()

    # --- Locate inputs ---
    db_path  = os.path.join(opt_dir, f'optuna_study_{opt_date}.db')
    npz_path = os.path.join(opt_dir, '_best_artifacts.npz')
    meta_path = os.path.join(opt_dir, '_best_artifacts.json')

    section("Locating Inputs", width=80)
    for label, p in [('SQLite DB', db_path),
                     ('Best Q-table .npz', npz_path),
                     ('Best metadata .json', meta_path)]:
        if not os.path.isfile(p):
            print(f"  ✗ {label} MISSING: {p}")
            sys.exit(2)
        list_item(f"{label}: {os.path.basename(p)}")
    print()

    # --- Load study ---
    section("Loading Optuna Study", width=80)
    storage = f'sqlite:///{db_path}'
    study = optuna.load_study(study_name=f'qlearning_opt_{opt_date}', storage=storage)
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials
              if t.state == optuna.trial.TrialState.PRUNED]
    list_item(f"Total trials: {len(study.trials)}")
    list_item(f"Completed:    {len(completed)}")
    list_item(f"Pruned:       {len(pruned)}")
    best = study.best_trial
    list_item(f"Best trial:   #{best.number} (1-based: #{best.number + 1})")
    list_item(f"Best value:   {best.value:.6f}")
    print()

    # --- Load best Q-table from disk cache (no retraining) ---
    section("Loading Best Q-table", width=80)
    artifacts: dict = {'Q': None, 'rewards': None, 'value': float('-inf')}
    ok = _opt._load_best_artifacts(opt_dir, artifacts)
    if not ok:
        print(f"  ✗ Could not load {npz_path}")
        sys.exit(3)
    best_Q = artifacts['Q']
    best_rewards = numpy.asarray(artifacts['rewards'])
    list_item(f"Q-table shape: {best_Q.shape}")
    list_item(f"Cached value:  {artifacts['value']:.6f}")
    _best_db_value = float(best.value) if best.value is not None else float('nan')
    if abs(artifacts['value'] - _best_db_value) > 1e-9:
        print(
            f"  ⚠ Cached value {artifacts['value']:.6f} differs from "
            f"study.best_value {_best_db_value:.6f} — using cached Q-table anyway."
        )
    print()

    # --- Persist hyperparameters + repro fingerprint ---
    section("Saving Results", width=80)
    _best_value = float(best.value) if best.value is not None else float('nan')
    params_path, fp_path = save_repro_artifacts(
        opt_dir, opt_date, best.params,
        trial_number=int(best.number),
        expected_mean_perf=_best_value,
    )
    success(f"Hyperparameters saved: {os.path.basename(params_path)}")
    success(f"Repro fingerprint saved: {os.path.basename(fp_path)}")

    # --- Mirror to inputs/ ---
    std_params  = os.path.join(INPUTS_PATH, 'best_params.json')
    std_q       = os.path.join(INPUTS_PATH, 'q.npy')
    std_rewards = os.path.join(INPUTS_PATH, 'rewards.npy')
    shutil.copyfile(params_path, std_params)
    numpy.save(std_q, best_Q)
    numpy.save(std_rewards, best_rewards)
    success(f"Mirrored to {std_params}")
    success(f"Mirrored to {std_q}")
    success(f"Mirrored to {std_rewards}")

    # --- Generate report + plots (best-effort) ---
    try:
        _opt._save_report(study, opt_dir, opt_date, best_Q, best_rewards)
    except Exception as e:
        info(f"Report/plot generation failed: {e}")

    print()
    section("Recovery Complete", width=80)
    success("All outputs reconstructed!")
    info(f"Output directory: {opt_dir}")
    print()


if __name__ == '__main__':
    main()
