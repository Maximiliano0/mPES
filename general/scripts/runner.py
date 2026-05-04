"""Per-cell benchmark runner.

Executes ONE (model, scenario) cell by:

1. Materialising the scenario CSVs into ``general/work/<pkg>/scenarios/<sid>/``.
2. Stashing the package's existing CSVs in a temporary back-up.
3. Copying the scenario CSVs over the package's working ``inputs/``.
4. Spawning ``python -m <group>.<pkg>`` with environment overrides
   (``MPES_OUTPUTS_PATH``, ``MPES_NUM_BLOCKS``, ``MPES_NUM_SEQUENCES``).
5. Restoring the original CSVs (always, via ``try/finally``).
6. Locating the run's results JSON + responses TXT and computing the
   benchmark metric block.
7. Writing ``general/results/raw/<model>__<sid>.json``.

Idempotent and resumable: a cell whose result JSON already exists is
skipped unless ``force=True``.
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
import subprocess
import sys
import time
from typing import Optional

import numpy

# Pattern matching ``Sequence <idx>: Performance = <float>`` lines that
# every package emits via ``log_utils.tee`` (verified across all 8).
# The line is ANSI-coloured by ``terminal_utils``, so we strip escape
# sequences before matching.
_PERF_RE = re.compile(r'Sequence\s+(\d+)\s*:\s*Performance\s*=\s*([\-0-9\.eE+]+)')
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')

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
RAW_RESULTS_DIR = os.path.join(GENERAL_ROOT, 'results', 'raw')

# Group of every benchmarked package.
PACKAGE_GROUPS = {
    'pes_ql':   'tabular',
    'pes_dql':  'tabular',
    'pes_dqn':  'ml',
    'pes_rdqn': 'ml',
    'pes_a2c':  'ml',
    'pes_trf':  'ml',
    'pes_ens':  'ml',
}
ALL_PACKAGES = list(PACKAGE_GROUPS.keys())


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


def _result_json_path(pkg: str, scenario_id: str) -> str:
    return os.path.join(RAW_RESULTS_DIR, f'{pkg}__{scenario_id}.json')


def _stash_then_copy(scenario_csvs: "tuple[str, str]", pkg: str):
    """Backup the pkg's CSVs in-place; copy scenario CSVs over them.

    Returns the list of stash paths so the caller can restore them.
    """
    sev_src, len_src = scenario_csvs
    inputs = _pkg_inputs(pkg)
    pairs = [
        (sev_src, os.path.join(inputs, 'initial_severity.csv')),
        (len_src, os.path.join(inputs, 'sequence_lengths.csv')),
    ]
    stashes = []
    for src, dst in pairs:
        if os.path.exists(dst):
            stash = dst + '.bench_stash'
            shutil.move(dst, stash)
            stashes.append((stash, dst))
        shutil.copyfile(src, dst)
    return stashes


def _restore(stashes):
    for stash, dst in stashes:
        if os.path.exists(dst):
            try:
                os.remove(dst)
            except OSError:
                pass
        if os.path.exists(stash):
            shutil.move(stash, dst)


def _spawn(pkg: str, env_overrides: dict, log_path: str) -> int:
    env = os.environ.copy()
    env.update(env_overrides)
    env.setdefault('PYTHONIOENCODING', 'utf-8')
    env.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
    cmd = [sys.executable, '-m', f'{PACKAGE_GROUPS[pkg]}.{pkg}']
    with open(log_path, 'wb') as logf:
        proc = subprocess.run(cmd, cwd=WORKSPACE_ROOT, env=env,
                              stdout=logf, stderr=subprocess.STDOUT,
                              check=False)
    return proc.returncode


def _find_results_json(outputs_dir: str) -> Optional[str]:
    """Locate the most recent ``*results_*.json`` produced by the run."""
    matches = glob.glob(os.path.join(outputs_dir, '**', '*results_*.json'),
                        recursive=True)
    if not matches:
        return None
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def _find_responses_txt(outputs_dir: str) -> Optional[str]:
    matches = glob.glob(os.path.join(outputs_dir, '**', '*responses_*.txt'),
                        recursive=True)
    if not matches:
        matches = glob.glob(os.path.join(outputs_dir, '**', '*responses_*.csv'),
                            recursive=True)
    if not matches:
        return None
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def _parse_responses(path: str) -> "list[int]":
    """Return the per-trial action vector. Tolerates header line(s) starting with '#'."""
    actions: "list[int]" = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 2:
                continue
            try:
                actions.append(int(float(parts[1])))
            except ValueError:
                continue
    return actions


def _action_distribution(actions: "list[int]", n_actions: int = 11) -> numpy.ndarray:
    counts = numpy.bincount(numpy.asarray(actions, dtype=int), minlength=n_actions)
    counts = counts[:n_actions].astype(float)
    total = counts.sum()
    if total <= 0:
        return numpy.ones(n_actions) / n_actions
    return counts / total


###############
##  Public API
###############
def run_cell(pkg: str, scenario: Scenario, *,
             seed: int = 42, force: bool = False) -> str:
    """Run one (pkg, scenario) cell. Returns the path to the result JSON."""
    out_path = _result_json_path(pkg, scenario.scenario_id)
    if not force and os.path.isfile(out_path):
        return out_path

    os.makedirs(RAW_RESULTS_DIR, exist_ok=True)

    # 1. Materialise scenario CSVs.
    scen_dir = _scenario_dir(pkg, scenario.scenario_id)
    csvs = materialise_scenario(scenario, scen_dir, seed=seed)

    # 2. Stash + 3. copy.
    stashes = _stash_then_copy(csvs, pkg)

    # 4. Spawn subprocess.
    out_dir = _scenario_outputs_dir(pkg, scenario.scenario_id)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, '_subprocess.log')
    env_overrides = {
        'MPES_OUTPUTS_PATH':  out_dir,
        'MPES_NUM_BLOCKS':    str(scenario.num_blocks),
        'MPES_NUM_SEQUENCES': str(scenario.num_sequences_per_block),
        'VIRTUAL_ENV': os.environ.get('VIRTUAL_ENV', ''),
    }

    t0 = time.time()
    try:
        rc = _spawn(pkg, env_overrides, log_path)
    finally:
        _restore(stashes)
    wallclock = time.time() - t0

    # 6. Parse outputs.
    metrics = {
        'model': pkg,
        'scenario': scenario.scenario_id,
        'family': scenario.family,
        'is_baseline': scenario.is_baseline,
        'num_blocks': scenario.num_blocks,
        'num_sequences_per_block': scenario.num_sequences_per_block,
        'seed': seed,
        'wallclock_s': round(wallclock, 3),
        'returncode': rc,
        'timestamp': _dt.datetime.utcnow().isoformat() + 'Z',
        'subprocess_log': os.path.relpath(log_path, WORKSPACE_ROOT),
    }

    json_path = _find_results_json(out_dir)
    per_seq: "list[float]" = []

    # Primary source: parse stdout log for ``Sequence X: Performance = Y.YYYY``.
    if os.path.isfile(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    clean = _ANSI_RE.sub('', line)
                    m = _PERF_RE.search(clean)
                    if m:
                        per_seq.append(float(m.group(2)))
        except OSError as exc:
            metrics['log_parse_error'] = str(exc)

    # Secondary: pull aggregate stats from the result JSON for cross-check.
    if json_path:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                rep = json.load(f)
            stats = rep.get('performance_statistics', {})
            metrics['report_json'] = os.path.relpath(json_path, WORKSPACE_ROOT)
            metrics['report_overall_mean'] = stats.get('overall_mean')
            metrics['report_overall_std'] = stats.get('overall_std')
        except (OSError, json.JSONDecodeError) as exc:
            metrics['parse_error'] = f'results JSON: {exc}'

    if per_seq:
        arr = numpy.asarray(per_seq, dtype=float)
        metrics.update({
            'n_sequences':      int(arr.size),
            'per_sequence_perf': arr.tolist(),
            'global_mean_perf': float(arr.mean()),
            'std_perf':         float(arr.std(ddof=0)),
            'min_perf':         float(arr.min()),
            'max_perf':         float(arr.max()),
        })
    else:
        metrics['n_sequences'] = 0
        metrics['parse_error'] = metrics.get(
            'parse_error', 'no per-sequence performance vector found')

    # Action distribution.
    resp_path = _find_responses_txt(out_dir)
    if resp_path:
        actions = _parse_responses(resp_path)
        if actions:
            dist = _action_distribution(actions)
            metrics['per_trial_actions'] = actions
            metrics['action_distribution'] = dist.tolist()
            metrics['responses_file'] = os.path.relpath(resp_path, WORKSPACE_ROOT)

    # 7. Write result.
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    return out_path


###############
##  CLI
###############
def _find_baseline_paths(pkg: str) -> "tuple[str, str]":
    inputs = _pkg_inputs(pkg)
    return (os.path.join(inputs, 'initial_severity.csv'),
            os.path.join(inputs, 'sequence_lengths.csv'))


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pkg', required=True, choices=ALL_PACKAGES)
    parser.add_argument('--scenario', required=True,
                        help='Scenario ID or "all".')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    sev_path, len_path = _find_baseline_paths(args.pkg)
    if not (os.path.isfile(sev_path) and os.path.isfile(len_path)):
        sys.exit(
            f"Missing baseline CSVs for {args.pkg}: "
            f"{sev_path} / {len_path}. Train the model first."
        )
    catalogue = build_scenarios(sev_path, len_path)

    if args.scenario == 'all':
        targets = catalogue
    else:
        targets = [s for s in catalogue if s.scenario_id == args.scenario]
        if not targets:
            sys.exit(f"Unknown scenario: {args.scenario}")

    for scen in targets:
        print(f"[runner] {args.pkg} :: {scen.scenario_id}")
        out = run_cell(args.pkg, scen, seed=args.seed, force=args.force)
        print(f"  -> {out}")


if __name__ == '__main__':
    _main()
