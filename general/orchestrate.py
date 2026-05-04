"""Drive the full Cartesian product (model x scenario) sweep.

By default executes ``len(ALL_PACKAGES) * len(catalogue)`` cells
sequentially (single-process to avoid GPU contention) and writes one
JSON per cell into ``general/results/raw/``.

Usage
-----
.. code-block:: powershell

    python -m general.orchestrate                # full sweep
    python -m general.orchestrate --pkg pes_dqn  # single model
    python -m general.orchestrate --scenario sev_empirical  # single scenario
    python -m general.orchestrate --force        # rerun even if cell exists
"""
##########################
##  Imports externos    ##
##########################
import argparse
import sys
import time

##########################
##  Imports internos    ##
##########################
from .runner import (ALL_PACKAGES, _find_baseline_paths, run_cell)
from .scenarios import build_scenarios


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pkg', choices=ALL_PACKAGES, action='append',
                        help='Restrict to one or more packages (repeatable).')
    parser.add_argument('--scenario', action='append',
                        help='Restrict to one or more scenario IDs (repeatable).')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--reference-pkg', default='pes_dqn',
                        help=('Package whose inputs/ provides the empirical '
                              'baseline CSVs (defaults to pes_dqn).'))
    args = parser.parse_args()

    sev_path, len_path = _find_baseline_paths(args.reference_pkg)
    catalogue = build_scenarios(sev_path, len_path)

    pkgs = args.pkg or ALL_PACKAGES
    if args.scenario:
        catalogue = [s for s in catalogue if s.scenario_id in args.scenario]
        if not catalogue:
            sys.exit(f"No scenarios match: {args.scenario}")

    total = len(pkgs) * len(catalogue)
    print(f"[orchestrate] {total} cells: {len(pkgs)} pkgs x {len(catalogue)} scenarios")

    t0 = time.time()
    done = 0
    for pkg in pkgs:
        for scen in catalogue:
            done += 1
            tag = f"[{done:>3}/{total}] {pkg} :: {scen.scenario_id}"
            print(tag, flush=True)
            try:
                run_cell(pkg, scen, seed=args.seed, force=args.force)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"  !! FAILED: {exc}", flush=True)
    print(f"[orchestrate] done in {time.time() - t0:.1f}s")


if __name__ == '__main__':
    _main()
