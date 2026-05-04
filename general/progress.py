"""Inspect benchmark sweep progress.

Reads ``general/results/raw/<pkg>__<scenario>.json`` and prints, per
package, how many of the expected scenarios have completed plus the
overall ETA based on the median wallclock observed so far.

Usage
-----
.. code-block:: powershell

    python -m general.progress           # one-shot snapshot
    python -m general.progress --watch   # refresh every 30 s
"""
##########################
##  Imports externos    ##
##########################
import argparse
import json
import os
import statistics
import time
from datetime import datetime, timedelta

##########################
##  Imports internos    ##
##########################
from .runner import ALL_PACKAGES, RAW_RESULTS_DIR, _find_baseline_paths
from .scenarios import build_scenarios

# ANSI colors (keep self-contained; matches ext/__init__.py palette).
_GREEN = '\x1b[92m'
_YELLOW = '\x1b[93m'
_RED = '\x1b[91m'
_DIM = '\x1b[2m'
_RST = '\x1b[0m'


def _bar(done: int, total: int, width: int = 24) -> str:
    """Return a unicode progress bar of fixed ``width`` characters."""
    if total == 0:
        return '[' + ' ' * width + ']'
    filled = int(round(width * done / total))
    return '[' + '█' * filled + '·' * (width - filled) + ']'


def _color_for(frac: float) -> str:
    """Return ANSI color escape based on completion fraction."""
    if frac >= 1.0:
        return _GREEN
    if frac >= 0.5:
        return _YELLOW
    return _RED


def _snapshot(scenario_ids: list[str]) -> None:
    """Print one progress snapshot."""
    raw_dir = RAW_RESULTS_DIR
    os.makedirs(raw_dir, exist_ok=True)
    files = [f for f in os.listdir(raw_dir) if f.endswith('.json')]

    # Bucket completed cells per package.
    done_per_pkg: dict[str, set[str]] = {p: set() for p in ALL_PACKAGES}
    timestamps: list[float] = []
    last_pkg, last_scen, last_ts = None, None, 0.0
    for fname in files:
        stem = fname[:-5]                     # strip '.json'
        if '__' not in stem:
            continue
        pkg, sid = stem.split('__', 1)
        if pkg not in done_per_pkg:
            continue
        if sid not in scenario_ids:
            continue                          # stale (e.g. legacy scenario id)
        done_per_pkg[pkg].add(sid)
        path = os.path.join(raw_dir, fname)
        try:
            ts = os.path.getmtime(path)
            timestamps.append(ts)
            if ts > last_ts:
                last_ts, last_pkg, last_scen = ts, pkg, sid
        except OSError:
            continue

    n_scen = len(scenario_ids)
    total = len(ALL_PACKAGES) * n_scen
    done = sum(len(s) for s in done_per_pkg.values())

    # Header.
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    overall_frac = done / total if total else 0.0
    overall_col = _color_for(overall_frac)
    print(f"{_DIM}{now}{_RST}  "
          f"{overall_col}{_bar(done, total)}{_RST}  "
          f"{done}/{total} cells "
          f"({overall_frac * 100:5.1f}%)")

    # Per-package breakdown.
    for pkg in ALL_PACKAGES:
        n = len(done_per_pkg[pkg])
        frac = n / n_scen if n_scen else 0.0
        col = _color_for(frac)
        print(f"  {pkg:<10s} {col}{_bar(n, n_scen, width=20)}{_RST} "
              f"{n:>2}/{n_scen}")

    # ETA based on median per-cell wallclock.
    if len(timestamps) >= 2:
        ts_sorted = sorted(timestamps)
        deltas = [b - a for a, b in zip(ts_sorted[:-1], ts_sorted[1:])]
        # Trim outliers (>3x median) caused by package warm-up / TF imports.
        med = statistics.median(deltas) if deltas else 0.0
        trimmed = [d for d in deltas if d <= max(med * 3.0, 1.0)]
        per_cell = statistics.median(trimmed) if trimmed else med
        remaining = total - done
        eta_sec = remaining * per_cell
        eta_str = str(timedelta(seconds=int(eta_sec)))
        finish = (datetime.now() + timedelta(seconds=eta_sec)
                  ).strftime('%H:%M:%S')
        print(f"  {_DIM}per-cell median: {per_cell:5.1f}s  "
              f"remaining: {remaining}  "
              f"ETA: {eta_str} (≈ {finish}){_RST}")
    if last_pkg is not None:
        last_when = datetime.fromtimestamp(last_ts).strftime('%H:%M:%S')
        print(f"  {_DIM}last completed: {last_pkg} :: {last_scen} "
              f"@ {last_when}{_RST}")

    # Quick failure / error scan in the latest few cells.
    bad = []
    for fname in files:
        path = os.path.join(raw_dir, fname)
        try:
            with open(path, encoding='utf-8') as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if data.get('parse_error') or data.get('returncode', 0) != 0:
            bad.append((fname, data.get('parse_error'),
                        data.get('returncode')))
    if bad:
        print(f"  {_RED}{len(bad)} cell(s) flagged with errors:{_RST}")
        for fname, err, rc in bad[:5]:
            print(f"    - {fname}  rc={rc}  err={err!r}")
        if len(bad) > 5:
            print(f"    ... and {len(bad) - 5} more")


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true',
                        help='Refresh every --interval seconds.')
    parser.add_argument('--interval', type=float, default=30.0)
    parser.add_argument('--reference-pkg', default='pes_dqn')
    args = parser.parse_args()

    sev_path, len_path = _find_baseline_paths(args.reference_pkg)
    catalogue = build_scenarios(sev_path, len_path)
    scenario_ids = [s.scenario_id for s in catalogue]

    if not args.watch:
        _snapshot(scenario_ids)
        return

    try:
        while True:
            print()
            _snapshot(scenario_ids)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\n[progress] stopped.')


if __name__ == '__main__':
    _main()
