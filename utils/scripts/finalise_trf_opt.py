"""
Finalise an interrupted ``pes_trf`` Bayesian-optimisation study **locally**.

Use this when a Colab Optuna run was killed by the runtime timeout and only
the resumable artifacts (``optuna_study_<date>.db``, ``_best_artifacts.*``)
were synced to ``pes_trf/inputs/<date>_BAYESIAN_OPT/``.  The script extracts
the best trial from the SQLite study and writes the lightweight artifacts
that ``train_transformer.py`` and ``pes_trf.__main__`` need:

  - ``best_params_<date>.json``                (dated sidecar)
  - ``pes_trf/inputs/best_params.json``        (canonical mirror)
  - ``optimization_results_<date>.txt``        (human-readable report)
  - ``optimization_history_<date>.png``        (convergence plot)
  - ``hyperparameter_importances_<date>.png``  (importance plot)

The model weights (``trf_best_<date>.keras``) and reward curve
(``rewards_best_<date>.npy``) are intentionally **not** produced — they would
require a full ``TRF_EPISODES`` retrain.  Generate them later with::

    python -m pes_trf.ext.train_transformer --from-best <date>

Usage
-----
::

    python utils/scripts/finalise_trf_opt.py 2026-04-29
"""
##########################
##  Imports externos    ##
##########################
import argparse
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import optuna

##########################
##  Imports internos    ##
##########################
# Resolve workspace root + package without importing TensorFlow.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# Skip the heavy ``pes_trf/__init__.py`` (which imports TF) — load CONFIG.py
# directly as a standalone module so we only pull the plain Python constants.
import importlib.util  # noqa: E402

_CONFIG_PATH = os.path.join(_ROOT, 'pes_trf', 'config', 'CONFIG.py')
_spec = importlib.util.spec_from_file_location('_pes_trf_config', _CONFIG_PATH)
assert _spec is not None and _spec.loader is not None
_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)
SEED: int = _cfg.SEED
TRF_EPISODES: int = _cfg.TRF_EPISODES
_INPUTS_PATH = os.path.join(_ROOT, 'pes_trf', 'inputs')

_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')


def _open_study(opt_dir: str, opt_date: str) -> optuna.Study:
    db_file = os.path.join(opt_dir, f'optuna_study_{opt_date}.db')
    if not os.path.isfile(db_file):
        raise FileNotFoundError(f"SQLite study not found: {db_file}")
    return optuna.load_study(
        study_name=f'trf_opt_{opt_date}',
        storage=f'sqlite:///{db_file}',
    )


def _write_sidecar(study: optuna.Study, opt_dir: str, opt_date: str) -> dict:
    best = study.best_trial
    bp = best.params
    hidden = best.user_attrs.get('hidden_units')
    if hidden is None and 'hidden_layer_size' in bp:
        hidden = [bp['hidden_layer_size']] * bp.get('num_hidden_layers', 1)
    trial_seed = int(best.user_attrs.get('trial_seed', SEED + int(best.number) + 1))

    payload = {
        'opt_date':           opt_date,
        'best_trial_number':  int(best.number),
        'mean_perf':          float(best.user_attrs.get('mean_perf', best.value)),
        'std_perf':           float(best.user_attrs.get('std_perf', float('nan'))),
        'min_perf':           float(best.user_attrs.get('min_perf', float('nan'))),
        'max_perf':           float(best.user_attrs.get('max_perf', float('nan'))),
        'trial_seed':         trial_seed,
        'hidden_units':       list(hidden) if hidden else None,
        'hyperparameters':    {
            k: (int(v) if isinstance(v, bool) or isinstance(v, int)
                else (float(v) if isinstance(v, float) else v))
            for k, v in bp.items()
        },
    }

    dated = os.path.join(opt_dir, f'best_params_{opt_date}.json')
    with open(dated, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f"  [OK]  {dated}")

    canonical = os.path.join(_INPUTS_PATH, 'best_params.json')
    with open(canonical, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f"  [OK]  {canonical}  (canonical mirror)")

    return payload


def _write_text_report(study: optuna.Study, opt_dir: str, opt_date: str) -> None:
    best = study.best_trial
    bp = best.params
    hidden = [bp['hidden_layer_size']] * bp['num_hidden_layers']
    use_pbrs = bool(bp.get('use_pbrs', bp.get('penalty_coeff', 0.0) > 0))
    penalty = float(bp.get('penalty_coeff', 0.0)) if use_pbrs else 0.0
    full_episodes = max(int(TRF_EPISODES), int(bp['num_episodes']))

    report_file = os.path.join(opt_dir, f'optimization_results_{opt_date}.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BAYESIAN OPTIMIZATION RESULTS - TRF HYPERPARAMETERS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Date:              {opt_date}\n")
        f.write(f"Total trials:      {len(study.trials)}\n")
        f.write(f"Best trial:        #{best.number + 1}\n")
        f.write(f"Best mean perf:    {best.value:.6f}\n")
        f.write("Note:              Finalised offline with finalise_trf_opt.py\n")
        f.write("                   (no .keras / .npy regenerated; use\n")
        f.write("                   `train_transformer.py --from-best` for those).\n\n")

        f.write("BEST HYPERPARAMETERS\n")
        f.write("-" * 80 + "\n")
        for name, val in best.params.items():
            f.write(f"  {name:<25s} = {val}\n")
        if 'hidden_units' in best.user_attrs:
            f.write(f"  {'hidden_units':<25s} = {best.user_attrs['hidden_units']}\n")
        f.write("\n")

        f.write("CONFIG.PY SNIPPET (copy-paste into pes_trf/config/CONFIG.py)\n")
        f.write("-" * 80 + "\n")
        f.write(f"# Best hyperparameters from Bayesian Optimisation trial #{best.number + 1} ({opt_date}).\n")
        f.write(f"# Performance: mean_perf = {best.value:.6f} over 64 fixed sequences.\n")
        f.write(f"TRF_HIDDEN_UNITS = {hidden}\n")
        f.write(f"TRF_LEARNING_RATE = {bp['learning_rate']}\n")
        f.write(f"TRF_BATCH_SIZE = {bp['batch_size']}\n")
        f.write(f"TRF_REPLAY_BUFFER_SIZE = {bp['buffer_size']}\n")
        f.write(f"TRF_TARGET_SYNC_FREQ = {bp['target_sync_freq']}\n")
        f.write(f"TRF_DISCOUNT = {bp['discount_factor']}\n")
        f.write(f"TRF_EPSILON_INITIAL = {bp['epsilon_initial']}\n")
        f.write(f"TRF_EPSILON_MIN = {bp['epsilon_min']}\n")
        f.write(f"TRF_EPISODES = {full_episodes}  # full retrain length\n")
        f.write(f"TRF_MAX_GRAD_NORM = {bp.get('max_grad_norm', 1.0)}\n")
        f.write(f"TRF_PENALTY_COEFF = {penalty}\n")
        f.write(f"TRF_WARMUP_RATIO = {bp.get('warmup_ratio', 0.05)}\n")
        f.write(f"TRF_TARGET_RATIO = {bp.get('target_ratio', 0.60)}\n")
        f.write(f"TRF_LEARNING_STARTS_FRAC = {bp.get('learning_starts_frac', 0.1)}\n\n")

        f.write("BEST TRIAL STATISTICS\n")
        f.write("-" * 80 + "\n")
        for k in ('mean_perf', 'std_perf', 'min_perf', 'max_perf'):
            v = best.user_attrs.get(k)
            f.write(f"  {k:<20s} {v:.6f}\n" if v is not None else f"  {k:<20s} n/a\n")
        f.write("\n")

        f.write("ALL TRIALS\n")
        f.write("-" * 100 + "\n")
        f.write(
            f"{'#':>4s}  {'mean_perf':>10s}  {'lr':>10s}  "
            f"{'gamma':>8s}  {'eps0':>6s}  {'eps_min':>7s}  "
            f"{'episodes':>8s}  {'batch':>5s}  {'buf_sz':>6s}  "
            f"{'sync':>5s}\n"
        )
        f.write("-" * 100 + "\n")
        for t in sorted(
            study.trials,
            key=lambda t: t.value if t.value is not None else -1,
            reverse=True,
        ):
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

    print(f"  [OK]  {report_file}")


def _write_convergence_plot(study: optuna.Study, opt_dir: str, opt_date: str) -> None:
    try:
        plt.style.use('ggplot')
    except Exception:  # pylint: disable=broad-except
        pass
    fig, ax = plt.subplots(figsize=(12, 6))
    trial_numbers = [t.number + 1 for t in study.trials if t.value is not None]
    trial_values = [t.value for t in study.trials if t.value is not None]
    running_best = []
    current_best = -1.0
    for v in trial_values:
        current_best = max(current_best, v)
        running_best.append(current_best)
    ax.scatter(trial_numbers, trial_values, color='#1f77b4', s=50, alpha=0.6,
               edgecolors='navy', linewidth=0.5,
               label='Trial performance', zorder=3)
    ax.plot(trial_numbers, running_best, color='#d62728', linewidth=2.5,
            label='Best so far', zorder=4)
    ax.set_xlabel('Trial number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean normalised performance', fontsize=12, fontweight='bold')
    ax.set_title('Bayesian Optimisation: Convergence (TRF)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(opt_dir, f'optimization_history_{opt_date}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK]  {out}")


def _write_importance_plot(study: optuna.Study, opt_dir: str, opt_date: str) -> None:
    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"  [WARN] Could not compute importances: {exc}")
        return
    names = list(importances.keys())
    values = list(importances.values())
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(names[::-1], values[::-1], color='#2ca02c',
            edgecolor='darkgreen', linewidth=0.5)
    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_title('TRF Hyperparameter Importance',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    out = os.path.join(opt_dir, f'hyperparameter_importances_{opt_date}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK]  {out}")


def main() -> None:
    """Parse the run date from argv and write all offline finalisation artifacts."""
    parser = argparse.ArgumentParser(
        description="Offline finaliser for an interrupted pes_trf Optuna run.",
    )
    parser.add_argument(
        'opt_date', help="Run date in YYYY-MM-DD form (e.g. 2026-04-29)",
    )
    args = parser.parse_args()

    if not _DATE_RE.match(args.opt_date):
        raise SystemExit(f"opt_date must be YYYY-MM-DD, got {args.opt_date!r}")

    opt_dir = os.path.join(_INPUTS_PATH, f'{args.opt_date}_BAYESIAN_OPT')
    if not os.path.isdir(opt_dir):
        raise SystemExit(f"Directory not found: {opt_dir}")

    print(f"[finalise_trf_opt] opt_dir = {opt_dir}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = _open_study(opt_dir, args.opt_date)
    n_complete = len([t for t in study.trials
                      if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"[finalise_trf_opt] {n_complete} completed trials in study")
    if n_complete == 0:
        raise SystemExit("No completed trials — nothing to finalise.")
    best = study.best_trial
    print(f"[finalise_trf_opt] best trial #{best.number + 1}  "
          f"value={best.value:.6f}\n")

    payload = _write_sidecar(study, opt_dir, args.opt_date)
    _write_text_report(study, opt_dir, args.opt_date)
    _write_convergence_plot(study, opt_dir, args.opt_date)
    _write_importance_plot(study, opt_dir, args.opt_date)

    print(
        "\n[finalise_trf_opt] DONE.\n"
        f"  best_trial_number = {payload['best_trial_number']}\n"
        f"  mean_perf         = {payload['mean_perf']:.6f}\n"
        f"  trial_seed        = {payload['trial_seed']}\n"
        f"  hidden_units      = {payload['hidden_units']}\n"
        "\nNext step (optional, regenerates the .keras + rewards.npy):\n"
        f"  python -m pes_trf.ext.train_transformer --from-best {args.opt_date}\n"
    )


if __name__ == '__main__':
    main()
