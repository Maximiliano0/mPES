"""Colab progress checker for mPES Bayesian-Optimisation runs.

Reads the Optuna SQLite DB on Drive, auto-detects the study name, and reports:
  * Trial counts (total / COMPLETE / RUNNING / FAIL / PRUNED)
  * Best value + best trial number
  * Throughput (trials/h) and ETA to N_TARGET
  * Process liveness (PID from optimize.pid via /proc/<pid>)
  * Tail of bayesian_opt.log and bayesian_opt_err.log

Usage from a Colab cell:
    !python utils/colab/check_progress.py --pkg pes_a2c --date 2026-04-22
    !python utils/colab/check_progress.py --pkg pes_dqn --date 2026-04-22 --tail 20
    !python utils/colab/check_progress.py --pkg pes_a2c --date 2026-04-22 --watch 30

Environment overrides:
    DRIVE_ROOT   default: /content/drive/MyDrive/mPES
"""
##########################
##  Imports externos    ##
##########################
import argparse
import datetime as _dt
import math
import os
import sys
import time

import optuna


##########################
##  Helpers             ##
##########################
def _proc_alive(pid: int) -> bool:
    """Return True iff /proc/<pid> exists (Linux/Colab)."""
    try:
        return os.path.isdir(f'/proc/{pid}')
    except OSError:
        return False


def _read_pid(pid_file: str) -> int | None:
    if not os.path.isfile(pid_file):
        return None
    try:
        with open(pid_file, encoding='utf-8') as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def _tail(path: str, n_lines: int) -> list[str]:
    if not os.path.isfile(path):
        return [f'(missing) {path}']
    try:
        with open(path, encoding='utf-8', errors='replace') as fh:
            return fh.readlines()[-n_lines:]
    except OSError as exc:
        return [f'(error reading {path}: {exc})']


def _fmt_eta(seconds: float) -> str:
    """Format ``seconds`` as ``H:MM:SS``; return ``'n/a'`` for non-positive or NaN."""
    if seconds <= 0 or math.isnan(seconds) or math.isinf(seconds):
        return 'n/a'
    td = _dt.timedelta(seconds=int(seconds))
    return str(td)


##########################
##  Core report         ##
##########################
def report(pkg: str, date: str, drive_root: str, tail: int, n_target: int) -> int:
    """Print a one-shot progress report for ``pkg`` / ``date``; return exit code."""
    pkg_dir = os.path.join(drive_root, pkg, f'{date}_BAYESIAN_OPT')
    db_path = os.path.join(pkg_dir, f'optuna_study_{date}.db')
    pid_file = os.path.join(pkg_dir, 'optimize.pid')
    log_file = os.path.join(pkg_dir, 'bayesian_opt.log')
    err_file = os.path.join(pkg_dir, 'bayesian_opt_err.log')

    print('=' * 72)
    print(f'  {pkg}  |  run {date}')
    print('=' * 72)
    print(f'  dir : {pkg_dir}')

    # ---- process liveness
    pid = _read_pid(pid_file)
    if pid is None:
        print('  pid : (no optimize.pid found)')
    else:
        alive = _proc_alive(pid)
        status = 'ALIVE' if alive else 'DEAD'
        print(f'  pid : {pid}  [{status}]')

    # ---- DB progress
    if not os.path.isfile(db_path):
        print(f'  DB  : NOT FOUND ({db_path})')
        return 1

    storage = 'sqlite:///' + db_path.replace(os.sep, '/')
    summaries = optuna.get_all_study_summaries(storage)
    if not summaries:
        print('  DB  : no studies inside')
        return 1
    study_name = summaries[0].study_name
    study = optuna.load_study(study_name=study_name, storage=storage)

    by_state: dict[str, list] = {}
    for trial in study.trials:
        by_state.setdefault(trial.state.name, []).append(trial)

    n_done = len(by_state.get('COMPLETE', []))
    n_run = len(by_state.get('RUNNING', []))
    n_fail = len(by_state.get('FAIL', []))
    n_prune = len(by_state.get('PRUNED', []))

    print(f'  study      : {study_name}')
    print(f'  trials     : total={len(study.trials)}  '
          f'COMPLETE={n_done}  RUNNING={n_run}  FAIL={n_fail}  PRUNED={n_prune}')

    if n_done:
        print(f'  best value : {study.best_value:.6f}  (trial #{study.best_trial.number})')

    # ---- throughput / ETA from completed trial timestamps
    completed = sorted(by_state.get('COMPLETE', []),
                       key=lambda t: t.datetime_complete or _dt.datetime.min)
    if len(completed) >= 2:
        t0 = completed[0].datetime_start
        t1 = completed[-1].datetime_complete
        if t0 is not None and t1 is not None:
            span = (t1 - t0).total_seconds()
            if span > 0:
                rate = n_done / span                              # trials/sec
                rate_h = rate * 3600.0
                remaining = max(n_target - n_done, 0)
                eta = remaining / rate if rate > 0 else float('inf')
                print(f'  throughput : {rate_h:.2f} trials/h  '
                      f'(avg {span/n_done:.1f}s/trial)')
                print(f'  ETA to {n_target}: {_fmt_eta(eta)}  '
                      f'({remaining} trials remaining)')

    # ---- log tails
    print('-' * 72)
    print(f'  Last {tail} lines of bayesian_opt.log:')
    for line in _tail(log_file, tail):
        print('    ' + line.rstrip())

    err_lines = _tail(err_file, tail)
    non_empty = [ln for ln in err_lines if ln.strip()]
    if non_empty:
        print('-' * 72)
        print(f'  Last {tail} lines of bayesian_opt_err.log:')
        for line in err_lines:
            print('    ' + line.rstrip())
    print()
    return 0


##########################
##  CLI                 ##
##########################
def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pkg', required=True,
                        choices=['pes_ql', 'pes_dql', 'pes_dqn',
                                 'pes_rdqn', 'pes_a2c', 'pes_trf'],
                        help='Package whose study to inspect')
    parser.add_argument('--date', required=True, help='Run date (YYYY-MM-DD)')
    parser.add_argument('--tail', type=int, default=15,
                        help='Number of trailing log lines to print (default: 15)')
    parser.add_argument('--target', type=int, default=100,
                        help='Target number of trials for ETA calculation (default: 100)')
    parser.add_argument('--watch', type=int, default=0,
                        help='If >0, refresh every WATCH seconds (Ctrl-C to stop)')
    parser.add_argument('--drive-root',
                        default=os.environ.get('DRIVE_ROOT', '/content/drive/MyDrive/mPES'),
                        help='Drive root for mPES (default: /content/drive/MyDrive/mPES)')
    args = parser.parse_args()

    if args.watch <= 0:
        return report(args.pkg, args.date, args.drive_root, args.tail, args.target)

    try:
        while True:
            os.system('clear')
            print(f'(refresh every {args.watch}s — Ctrl-C to stop)\n')
            report(args.pkg, args.date, args.drive_root, args.tail, args.target)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print('\n[stopped]')
        return 0


if __name__ == '__main__':
    sys.exit(main())
