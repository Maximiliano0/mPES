'''
pes_ql - Pandemic Experiment Scenario

Reproducibility utilities: capture the runtime fingerprint that determines
the outcome of Q-Learning training (numpy/python versions, CSV hashes,
git commit, CONFIG.SEED) so that ``train_rl.py`` can verify the local
environment matches the one used during Bayesian optimisation.

Two artifacts are produced/consumed by this module:

* ``best_params_<date>.json``  - best hyperparameters from optimisation,
  ready to be loaded as kwargs to ``QLearning``.
* ``repro_fingerprint_<date>.json`` - environment fingerprint (versions,
  hashes, seed, commit) used to validate cross-machine reproducibility.
'''

##########################
##  External imports    ##
##########################
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy

##########################
##  Internal imports    ##
##########################
from .. import INPUTS_PATH
from ..config.CONFIG import SEED


##############################
##  Filename helpers        ##
##############################
PARAMS_BASENAME = 'best_params'         # best_params_<date>.json
FINGERPRINT_BASENAME = 'repro_fingerprint'   # repro_fingerprint_<date>.json

# CSV files that affect training/evaluation outcomes
_TRACKED_CSVS = ('sequence_lengths.csv', 'initial_severity.csv')


##############################
##  Fingerprint capture     ##
##############################
def _sha256(path: str) -> str:
    """Return SHA-256 hex digest of *path* (or 'missing')."""
    if not os.path.isfile(path):
        return 'missing'
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    """Return the current git commit hash, or 'unknown' if not a git repo."""
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'


def capture_fingerprint() -> dict:
    """Snapshot the current environment in a dict.

    The fingerprint contains every variable that, if changed, would alter
    the bit-for-bit output of ``QLearning(seed=SEED)``.

    Returns
    -------
    dict
        Keys: ``numpy_version``, ``python_version``, ``platform``,
        ``seed``, ``git_commit``, ``csv_sha256`` (mapping filename -> digest).
    """
    return {
        'numpy_version': numpy.__version__,
        'python_version': '.'.join(map(str, sys.version_info[:3])),
        'platform': platform.system(),
        'seed': int(SEED),
        'git_commit': _git_commit(),
        'csv_sha256': {
            csv: _sha256(os.path.join(INPUTS_PATH, csv))
            for csv in _TRACKED_CSVS
        },
    }


##############################
##  Persistence             ##
##############################
def save_artifacts(
    opt_dir: str,
    opt_date: str,
    best_params: dict,
    trial_number: int,
    expected_mean_perf: float,
) -> tuple[str, str]:
    """Persist best hyperparameters and runtime fingerprint to ``opt_dir``.

    Parameters
    ----------
    opt_dir : str
        Output directory (typically ``inputs/<date>_BAYESIAN_OPT/``).
    opt_date : str
        Date stamp used in filenames (``YYYY-MM-DD``).
    best_params : dict
        Best hyperparameters as returned by Optuna's ``best_trial.params``.
    trial_number : int
        0-based trial index of the best trial. The training seed used by
        Optuna for this trial was ``SEED + trial_number + 1`` and must be
        replicated exactly by ``train_rl.py`` to reproduce ``mean_perf``.
    expected_mean_perf : float
        ``best_trial.value`` reported by Optuna; train_rl.py will assert
        its evaluation matches this within a tight tolerance.

    Returns
    -------
    tuple of str
        ``(params_path, fingerprint_path)`` written to disk.
    """
    params_payload = {
        'hyperparameters': dict(best_params),
        # --- Canonical sidecar field names (shared with pes_dql/dqn/a2c) ---
        'best_trial_number': int(trial_number),
        'mean_perf':         float(expected_mean_perf),
        'trial_seed':        int(SEED) + int(trial_number) + 1,
        # --- Legacy aliases retained for backward compatibility with
        #     existing best_params_*.json files generated before 2026-04-21. ---
        'trial_number':       int(trial_number),
        'expected_mean_perf': float(expected_mean_perf),
        'training_seed':      int(SEED) + int(trial_number) + 1,
    }

    params_path = os.path.join(opt_dir, f'{PARAMS_BASENAME}_{opt_date}.json')
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(params_payload, f, indent=2, sort_keys=True)

    fp_path = os.path.join(opt_dir, f'{FINGERPRINT_BASENAME}_{opt_date}.json')
    with open(fp_path, 'w', encoding='utf-8') as f:
        json.dump(capture_fingerprint(), f, indent=2, sort_keys=True)

    return params_path, fp_path


def find_latest_artifacts() -> tuple[str, str] | None:
    """Locate the most recent ``best_params_*.json`` + ``repro_fingerprint_*.json``.

    Searches every ``<date>_BAYESIAN_OPT/`` directory under ``INPUTS_PATH``
    and returns the pair from the lexicographically highest date.

    Returns
    -------
    tuple of str or None
        ``(params_path, fingerprint_path)`` if found, else ``None``.
    """
    if not os.path.isdir(INPUTS_PATH):
        return None
    candidates = []
    for entry in os.listdir(INPUTS_PATH):
        full = os.path.join(INPUTS_PATH, entry)
        if not os.path.isdir(full) or not entry.endswith('_BAYESIAN_OPT'):
            continue
        date = entry.split('_', maxsplit=1)[0]
        params = os.path.join(full, f'{PARAMS_BASENAME}_{date}.json')
        fp = os.path.join(full, f'{FINGERPRINT_BASENAME}_{date}.json')
        if os.path.isfile(params) and os.path.isfile(fp):
            candidates.append((date, params, fp))
    if not candidates:
        return None
    candidates.sort()
    _, params, fp = candidates[-1]
    return params, fp


def load_params(params_path: str) -> dict:
    """Load hyperparameters from a ``best_params_*.json`` file."""
    with open(params_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_fingerprint(fp_path: str) -> dict:
    """Load a saved environment fingerprint."""
    with open(fp_path, 'r', encoding='utf-8') as f:
        return json.load(f)


##############################
##  Verification            ##
##############################
def diff_fingerprints(reference: dict, current: dict) -> list[str]:
    """Compare two fingerprints and return a list of human-readable mismatches.

    Parameters
    ----------
    reference : dict
        Fingerprint captured during optimisation (loaded from JSON).
    current : dict
        Fingerprint of the current runtime (from ``capture_fingerprint``).

    Returns
    -------
    list of str
        Empty if the two fingerprints are equivalent for reproducibility
        purposes; otherwise one entry per differing field.
    """
    diffs: list[str] = []

    if reference.get('numpy_version') != current.get('numpy_version'):
        diffs.append(
            f"numpy: reference={reference.get('numpy_version')} "
            f"vs local={current.get('numpy_version')}"
        )

    ref_py = (reference.get('python_version') or '').split('.')[:2]
    cur_py = (current.get('python_version') or '').split('.')[:2]
    if ref_py != cur_py:
        diffs.append(
            f"python: reference={'.'.join(ref_py)} "
            f"vs local={'.'.join(cur_py)}"
        )

    if int(reference.get('seed', -1)) != int(current.get('seed', -2)):
        diffs.append(
            f"SEED: reference={reference.get('seed')} "
            f"vs local={current.get('seed')}"
        )

    ref_commit = reference.get('git_commit', 'unknown')
    cur_commit = current.get('git_commit', 'unknown')
    if ref_commit != cur_commit and 'unknown' not in (ref_commit, cur_commit):
        diffs.append(
            f"git commit: reference={ref_commit[:10]} "
            f"vs local={cur_commit[:10]}"
        )

    ref_csv = reference.get('csv_sha256', {}) or {}
    cur_csv = current.get('csv_sha256', {}) or {}
    for csv in _TRACKED_CSVS:
        if ref_csv.get(csv) != cur_csv.get(csv):
            diffs.append(
                f"{csv}: reference={(ref_csv.get(csv) or 'missing')[:12]} "
                f"vs local={(cur_csv.get(csv) or 'missing')[:12]}"
            )

    return diffs
