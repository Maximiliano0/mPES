"""Benchmark execution for the mPES Under Stress Experiments.

Merges the per-cell runner, the full sweep driver and the live progress
monitor in a single module, and owns the canonical results layout::

    results/<suite>/cells/<model>__<scenario>.json
    results/<suite>/matrices/<metric>.csv
    results/<suite>/figures/...
    results/<suite>/summary.json
    results/<suite>/report.md

Usage
-----
.. code-block:: powershell

    python -m general.scripts.benchmark run --suite both
    python -m general.scripts.benchmark run --pkg pes_dqn --scenario sev_empirical
    python -m general.scripts.benchmark progress --suite individual --watch
"""
##########################
##  Imports externos    ##
##########################
import argparse
import datetime as _dt
import glob
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy

##########################
##  Imports internos    ##
##########################
from .scenarios import Scenario, build_scenarios, materialise_scenario


###############
##  Constants
###############
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GENERAL_ROOT = os.path.join(WORKSPACE_ROOT, 'general')
WORK_ROOT = os.path.join(GENERAL_ROOT, 'work')
RESULTS_ROOT = os.path.join(GENERAL_ROOT, 'results')

INDIVIDUAL_PACKAGE_GROUPS = {
    'pes_base': 'tabular',
    'pes_ql':   'tabular',
    'pes_dql':  'tabular',
    'pes_dqn':  'ml',
    'pes_rdqn': 'ml',
    'pes_a2c':  'ml',
    'pes_trf':  'ml',
}
ENSEMBLE_PACKAGE_GROUPS = {
    'pes_ens':                 'ens',
    'pes_ens_sprb':            'ens',
    'pes_ens_accq':            'ens',
    'pes_ens_consensus':       'ens',
    'pes_ens_consensus_prior': 'ens',
    'pes_ens_trf_guard':       'ens',
}
SUITE_PACKAGE_GROUPS = {
    'individual': INDIVIDUAL_PACKAGE_GROUPS,
    'ensemble':   ENSEMBLE_PACKAGE_GROUPS,
}
PACKAGE_GROUPS = {**INDIVIDUAL_PACKAGE_GROUPS, **ENSEMBLE_PACKAGE_GROUPS}
SUITE_PACKAGES = {suite: list(packages) for suite, packages in SUITE_PACKAGE_GROUPS.items()}
ALL_PACKAGES = list(PACKAGE_GROUPS.keys())
SUITES = tuple(SUITE_PACKAGES)
REFERENCE_MODEL = 'pes_base'
REFERENCE_SCENARIO = 'sev_empirical'

# Every package prints ``Sequence <idx>: Performance = <float>`` through
# ``log_utils.tee``; the line is ANSI-coloured by ``terminal_utils``.
_PERF_RE = re.compile(r'Sequence\s+(\d+)\s*:\s*Performance\s*=\s*([\-0-9\.eE+]+)')
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')

_GREEN, _YELLOW, _RED, _DIM, _RST = '\x1b[92m', '\x1b[93m', '\x1b[91m', '\x1b[2m', '\x1b[0m'


###############
##  Layout
###############
def suite_dir(suite: str) -> str:
    """Root directory holding every artefact of one benchmark suite."""
    return os.path.join(RESULTS_ROOT, suite)


def cells_dir(suite: str) -> str:
    """Directory with one JSON payload per ``(model, scenario)`` cell."""
    return os.path.join(suite_dir(suite), 'cells')


def matrices_dir(suite: str) -> str:
    """Directory with the aggregated ``model x scenario`` CSV matrices."""
    return os.path.join(suite_dir(suite), 'matrices')


def figures_dir(suite: str) -> str:
    """Directory with the rendered PNG/PDF figures."""
    return os.path.join(suite_dir(suite), 'figures')


def summary_path(suite: str) -> str:
    """Machine-readable consolidation of every cell in the suite."""
    return os.path.join(suite_dir(suite), 'summary.json')


def report_path(suite: str) -> str:
    """Executive Markdown report of the suite."""
    return os.path.join(suite_dir(suite), 'report.md')


def comparison_metrics_path(suite: str) -> str:
    """Pairwise comparison metrics of the suite."""
    return os.path.join(suite_dir(suite), 'comparison_metrics.json')


def cell_path(suite: str, pkg: str, scenario_id: str) -> str:
    """Path of one cell payload inside its suite."""
    return os.path.join(cells_dir(suite), f'{pkg}__{scenario_id}.json')


def suite_for_package(pkg: str) -> str:
    """Return the comparison suite owning ``pkg``."""
    for suite, packages in SUITE_PACKAGES.items():
        if pkg in packages:
            return suite
    raise ValueError(f'Unknown benchmark package: {pkg}')


def load_cells(suite: str) -> dict:
    """Load ``{model: {scenario: cell}}`` for one suite."""
    out: dict = {}
    for path in glob.glob(os.path.join(cells_dir(suite), '*.json')):
        with open(path, 'r', encoding='utf-8') as handle:
            cell = json.load(handle)
        out.setdefault(cell['model'], {})[cell['scenario']] = cell
    return out


def find_baseline_paths(pkg: str) -> "tuple[str, str]":
    """Return the empirical severity and length CSVs of ``pkg``."""
    inputs = _pkg_inputs(pkg)
    return (os.path.join(inputs, 'initial_severity.csv'),
            os.path.join(inputs, 'sequence_lengths.csv'))


def scenario_catalogue(reference_pkg: str = 'pes_dqn') -> "list[Scenario]":
    """Build the scenario catalogue from a reference package's inputs."""
    severity_path, lengths_path = find_baseline_paths(reference_pkg)
    return build_scenarios(severity_path, lengths_path)


###############
##  Helpers
###############
def _pkg_dir(pkg: str) -> str:
    return os.path.join(WORKSPACE_ROOT, PACKAGE_GROUPS[pkg], pkg)


def _pkg_inputs(pkg: str) -> str:
    return os.path.join(_pkg_dir(pkg), 'inputs')


def _scenario_dir(pkg: str, scenario_id: str) -> str:
    return os.path.join(WORK_ROOT, pkg, 'scenarios', scenario_id)


def _scenario_outputs_dir(pkg: str, scenario_id: str) -> str:
    return os.path.join(WORK_ROOT, pkg, 'outputs', scenario_id)


def _stash_then_copy(scenario_csvs: "tuple[str, str]", pkg: str):
    """Back up the package CSVs in place and copy the scenario CSVs over them."""
    severity_source, lengths_source = scenario_csvs
    inputs = _pkg_inputs(pkg)
    pairs = [
        (severity_source, os.path.join(inputs, 'initial_severity.csv')),
        (lengths_source, os.path.join(inputs, 'sequence_lengths.csv')),
    ]
    stashes = []
    for source, destination in pairs:
        if os.path.exists(destination):
            stash = destination + '.bench_stash'
            shutil.move(destination, stash)
            stashes.append((stash, destination))
        shutil.copyfile(source, destination)
    return stashes


def _restore(stashes) -> None:
    """Restore the package CSVs stashed by :func:`_stash_then_copy`."""
    for stash, destination in stashes:
        if os.path.exists(destination):
            try:
                os.remove(destination)
            except OSError:
                pass
        if os.path.exists(stash):
            shutil.move(stash, destination)


def _spawn(pkg: str, env_overrides: dict, log_path: str) -> int:
    """Run ``python -m <group>.<pkg>`` capturing stdout and stderr."""
    env = os.environ.copy()
    env.update(env_overrides)
    env.setdefault('PYTHONIOENCODING', 'utf-8')
    env.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
    command = [sys.executable, '-m', f'{PACKAGE_GROUPS[pkg]}.{pkg}']
    with open(log_path, 'wb') as log_file:
        process = subprocess.run(command, cwd=WORKSPACE_ROOT, env=env,
                                 stdout=log_file, stderr=subprocess.STDOUT,
                                 check=False)
    return process.returncode


def _latest(pattern: str, outputs_dir: str) -> Optional[str]:
    """Return the most recent file matching ``pattern`` under ``outputs_dir``."""
    matches = glob.glob(os.path.join(outputs_dir, '**', pattern), recursive=True)
    if not matches:
        return None
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def _find_responses_file(outputs_dir: str) -> Optional[str]:
    """Locate the per-trial responses file emitted by the package."""
    return (_latest('*responses_*.txt', outputs_dir)
            or _latest('*responses_*.csv', outputs_dir))


def _parse_responses(path: str) -> "list[int]":
    """Return the per-trial action vector, tolerating ``#`` header lines."""
    actions: "list[int]" = []
    with open(path, 'r', encoding='utf-8', errors='replace') as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [part.strip() for part in line.split(',')]
            if len(parts) < 2:
                continue
            try:
                actions.append(int(float(parts[1])))
            except ValueError:
                continue
    return actions


def _action_distribution(actions: "list[int]", n_actions: int = 11) -> numpy.ndarray:
    """Empirical PMF over the discrete allocation actions."""
    counts = numpy.bincount(numpy.asarray(actions, dtype=int), minlength=n_actions)
    counts = counts[:n_actions].astype(float)
    total = counts.sum()
    if total <= 0:
        return numpy.ones(n_actions) / n_actions
    return counts / total


def _collect_performance(log_path: str, outputs_dir: str, metrics: dict) -> "list[float]":
    """Extract the per-sequence performance vector for one finished cell."""
    per_sequence: "list[float]" = []
    if os.path.isfile(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8', errors='replace') as handle:
                for line in handle:
                    match = _PERF_RE.search(_ANSI_RE.sub('', line))
                    if match:
                        per_sequence.append(float(match.group(2)))
        except OSError as error:
            metrics['log_parse_error'] = str(error)
    if per_sequence:
        return per_sequence
    # Ensemble evaluators persist the vector instead of printing it.
    npy_path = _latest('*performances*.npy', outputs_dir)
    if npy_path:
        try:
            return numpy.asarray(numpy.load(npy_path), dtype=float).tolist()
        except (OSError, ValueError):
            return []
    return []


###############
##  Cell execution
###############
def run_cell(pkg: str, scenario: Scenario, *,
             seed: int = 42, force: bool = False) -> str:
    """Run one ``(pkg, scenario)`` cell and return the written JSON path."""
    suite = suite_for_package(pkg)
    out_path = cell_path(suite, pkg, scenario.scenario_id)
    if not force and os.path.isfile(out_path):
        return out_path
    os.makedirs(cells_dir(suite), exist_ok=True)

    scenario_csvs = materialise_scenario(
        scenario, _scenario_dir(pkg, scenario.scenario_id), seed=seed)
    stashes = _stash_then_copy(scenario_csvs, pkg)

    outputs_dir = _scenario_outputs_dir(pkg, scenario.scenario_id)
    os.makedirs(outputs_dir, exist_ok=True)
    log_path = os.path.join(outputs_dir, '_subprocess.log')
    env_overrides = {
        'MPES_OUTPUTS_PATH':  outputs_dir,
        'MPES_NUM_BLOCKS':    str(scenario.num_blocks),
        'MPES_NUM_SEQUENCES': str(scenario.num_sequences_per_block),
        'VIRTUAL_ENV': os.environ.get('VIRTUAL_ENV', ''),
    }

    started = time.time()
    try:
        return_code = _spawn(pkg, env_overrides, log_path)
    finally:
        _restore(stashes)

    metrics = {
        'model': pkg,
        'suite': suite,
        'scenario': scenario.scenario_id,
        'family': scenario.family,
        'is_baseline': scenario.is_baseline,
        'num_blocks': scenario.num_blocks,
        'num_sequences_per_block': scenario.num_sequences_per_block,
        'seed': seed,
        'wallclock_s': round(time.time() - started, 3),
        'returncode': return_code,
        'timestamp': _dt.datetime.now(_dt.timezone.utc).isoformat(),
        'subprocess_log': os.path.relpath(log_path, WORKSPACE_ROOT),
    }

    report_json = _latest('*results_*.json', outputs_dir)
    if report_json:
        try:
            with open(report_json, 'r', encoding='utf-8') as handle:
                report = json.load(handle)
            stats = report.get('performance_statistics', {})
            metrics['report_json'] = os.path.relpath(report_json, WORKSPACE_ROOT)
            metrics['report_overall_mean'] = stats.get('overall_mean')
            metrics['report_overall_std'] = stats.get('overall_std')
        except (OSError, json.JSONDecodeError) as error:
            metrics['parse_error'] = f'results JSON: {error}'

    per_sequence = _collect_performance(log_path, outputs_dir, metrics)
    if per_sequence:
        values = numpy.asarray(per_sequence, dtype=float)
        metrics.update({
            'n_sequences':       int(values.size),
            'per_sequence_perf': values.tolist(),
            'global_mean_perf':  float(values.mean()),
            'std_perf':          float(values.std(ddof=0)),
            'min_perf':          float(values.min()),
            'max_perf':          float(values.max()),
        })
    else:
        metrics['n_sequences'] = 0
        metrics['parse_error'] = metrics.get(
            'parse_error', 'no per-sequence performance vector found')

    responses_path = _find_responses_file(outputs_dir)
    if responses_path:
        actions = _parse_responses(responses_path)
        if actions:
            metrics['per_trial_actions'] = actions
            metrics['action_distribution'] = _action_distribution(actions).tolist()
            metrics['responses_file'] = os.path.relpath(responses_path, WORKSPACE_ROOT)

    with open(out_path, 'w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)
    return out_path


def run_sweep(packages: "list[str]", catalogue: "list[Scenario]", *,
              seed: int = 42, force: bool = False) -> None:
    """Execute the Cartesian product of ``packages`` and ``catalogue``."""
    total = len(packages) * len(catalogue)
    print(f'[benchmark] {total} cells: {len(packages)} pkgs x {len(catalogue)} scenarios')
    started = time.time()
    done = 0
    for pkg in packages:
        for scenario in catalogue:
            done += 1
            print(f'[{done:>3}/{total}] {pkg} :: {scenario.scenario_id}', flush=True)
            try:
                run_cell(pkg, scenario, seed=seed, force=force)
            except Exception as error:  # pylint: disable=broad-except
                print(f'  !! FAILED: {error}', flush=True)
    print(f'[benchmark] done in {time.time() - started:.1f}s')


###############
##  Progress
###############
def _bar(done: int, total: int, width: int = 24) -> str:
    """Return a fixed-width unicode progress bar."""
    if total == 0:
        return '[' + ' ' * width + ']'
    filled = int(round(width * done / total))
    return '[' + '█' * filled + '·' * (width - filled) + ']'


def _colour_for(fraction: float) -> str:
    """ANSI colour matching a completion fraction."""
    if fraction >= 1.0:
        return _GREEN
    return _YELLOW if fraction >= 0.5 else _RED


def snapshot(suite: str, scenario_ids: "list[str]") -> None:
    """Print one progress snapshot for ``suite``."""
    directory = cells_dir(suite)
    os.makedirs(directory, exist_ok=True)
    packages = SUITE_PACKAGES[suite]
    done_per_pkg: "dict[str, set[str]]" = {pkg: set() for pkg in packages}
    timestamps: "list[float]" = []
    last = (None, None, 0.0)
    failures = []

    for path in glob.glob(os.path.join(directory, '*.json')):
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                cell = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        pkg, scenario = cell.get('model'), cell.get('scenario')
        if pkg not in done_per_pkg or scenario not in scenario_ids:
            continue
        done_per_pkg[pkg].add(scenario)
        modified = os.path.getmtime(path)
        timestamps.append(modified)
        if modified > last[2]:
            last = (pkg, scenario, modified)
        if cell.get('parse_error') or cell.get('returncode', 0) != 0:
            failures.append((os.path.basename(path), cell.get('parse_error'),
                             cell.get('returncode')))

    n_scenarios = len(scenario_ids)
    total = len(packages) * n_scenarios
    done = sum(len(items) for items in done_per_pkg.values())
    fraction = done / total if total else 0.0
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{_DIM}{now}{_RST}  {_colour_for(fraction)}{_bar(done, total)}{_RST}  '
          f'{done}/{total} cells ({fraction * 100:5.1f}%)')
    for pkg in packages:
        completed = len(done_per_pkg[pkg])
        pkg_fraction = completed / n_scenarios if n_scenarios else 0.0
        print(f'  {pkg:<24s} {_colour_for(pkg_fraction)}'
              f'{_bar(completed, n_scenarios, width=20)}{_RST} {completed:>2}/{n_scenarios}')

    if len(timestamps) >= 2:
        ordered = sorted(timestamps)
        deltas = [second - first for first, second in zip(ordered[:-1], ordered[1:])]
        median = statistics.median(deltas) if deltas else 0.0
        trimmed = [delta for delta in deltas if delta <= max(median * 3.0, 1.0)]
        per_cell = statistics.median(trimmed) if trimmed else median
        remaining = total - done
        eta = timedelta(seconds=int(remaining * per_cell))
        finish = (datetime.now() + eta).strftime('%H:%M:%S')
        print(f'  {_DIM}per-cell median: {per_cell:5.1f}s  remaining: {remaining}  '
              f'ETA: {eta} (~ {finish}){_RST}')
    if last[0] is not None:
        when = datetime.fromtimestamp(last[2]).strftime('%H:%M:%S')
        print(f'  {_DIM}last completed: {last[0]} :: {last[1]} @ {when}{_RST}')
    if failures:
        print(f'  {_RED}{len(failures)} cell(s) flagged with errors:{_RST}')
        for name, error, code in failures[:5]:
            print(f'    - {name}  rc={code}  err={error!r}')


###############
##  CLI
###############
def _resolve_packages(args) -> "list[str]":
    """Return the package list selected by the CLI arguments."""
    if args.pkg:
        return args.pkg
    if args.suite == 'both':
        return SUITE_PACKAGES['individual'] + SUITE_PACKAGES['ensemble']
    return SUITE_PACKAGES[args.suite]


def _command_run(args) -> None:
    """Execute the requested cells."""
    catalogue = scenario_catalogue(args.reference_pkg)
    if args.scenario:
        catalogue = [s for s in catalogue if s.scenario_id in args.scenario]
        if not catalogue:
            sys.exit(f'No scenarios match: {args.scenario}')
    run_sweep(_resolve_packages(args), catalogue, seed=args.seed, force=args.force)


def _command_progress(args) -> None:
    """Print progress once or refresh it periodically."""
    scenario_ids = [s.scenario_id for s in scenario_catalogue(args.reference_pkg)]
    if not args.watch:
        snapshot(args.suite, scenario_ids)
        return
    try:
        while True:
            print()
            snapshot(args.suite, scenario_ids)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\n[benchmark] progress stopped.')


def main() -> None:
    """Command-line entry point of the benchmark harness."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)

    run_parser = subparsers.add_parser('run', help='Execute benchmark cells.')
    run_parser.add_argument('--suite', choices=(*SUITES, 'both'), default='individual')
    run_parser.add_argument('--pkg', choices=ALL_PACKAGES, action='append')
    run_parser.add_argument('--scenario', action='append')
    run_parser.add_argument('--seed', type=int, default=42)
    run_parser.add_argument('--force', action='store_true')
    run_parser.add_argument('--reference-pkg', default='pes_dqn')
    run_parser.set_defaults(handler=_command_run)

    progress_parser = subparsers.add_parser('progress', help='Show sweep progress.')
    progress_parser.add_argument('--suite', choices=SUITES, default='individual')
    progress_parser.add_argument('--watch', action='store_true')
    progress_parser.add_argument('--interval', type=float, default=30.0)
    progress_parser.add_argument('--reference-pkg', default='pes_dqn')
    progress_parser.set_defaults(handler=_command_progress)

    args = parser.parse_args()
    args.handler(args)


if __name__ == '__main__':
    main()
