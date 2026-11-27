'''
pes_dql - Pandemic Experiment Scenario (Double Q-Learning)

Reproducibility utilities: capture the runtime fingerprint that determines
the outcome of Double Q-Learning training (numpy/python versions, CSV
hashes, git commit, CONFIG.SEED) so that ``train_rl.py`` can verify the
local environment matches the one used during Bayesian optimisation.

Two artifacts are produced/consumed alongside ``best_params_<date>.json``:

* ``best_params_<date>.json``  - best hyperparameters from optimisation
  (written inline by ``optimize_rl.py``; this module does NOT touch it).
* ``repro_fingerprint_<date>.json`` - environment fingerprint (versions,
  hashes, seed, commit) used to validate cross-machine reproducibility.

This module is the tabular-RL equivalent of ``pes_ql/ext/repro.py``.  Deep
packages (``pes_dqn``, ``pes_a2c``) intentionally do NOT use a fingerprint
because TF/CUDA reductions are not bit-exact deterministic.
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
    the bit-for-bit output of ``QLearning(seed=SEED, double_q=True)``.

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
def save_fingerprint(opt_dir: str, opt_date: str) -> str:
    """Write ``repro_fingerprint_<date>.json`` to *opt_dir*.

    Parameters
    ----------
    opt_dir : str
        Output directory (typically ``inputs/<date>_BAYESIAN_OPT/``).
    opt_date : str
        Date stamp used in the filename (``YYYY-MM-DD``).

    Returns
    -------
    str
        Absolute path of the file written.
    """
    fp_path = os.path.join(opt_dir, f'{FINGERPRINT_BASENAME}_{opt_date}.json')
    with open(fp_path, 'w', encoding='utf-8') as f:
        json.dump(capture_fingerprint(), f, indent=2, sort_keys=True)
    return fp_path


def load_fingerprint(fp_path: str) -> dict:
    """Load a saved environment fingerprint."""
    with open(fp_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_fingerprint_for(params_path: str) -> str | None:
    """Locate the ``repro_fingerprint_<date>.json`` matching *params_path*.

    Parameters
    ----------
    params_path : str
        Full path to a ``best_params_<date>.json`` file (or to the mirrored
        ``inputs/best_params.json``).

    Returns
    -------
    str or None
        Path to the sibling ``repro_fingerprint_<date>.json`` if it exists,
        otherwise ``None``.  Returns ``None`` for the mirrored copy at
        ``inputs/best_params.json`` because no fingerprint is mirrored
        there (the date stamp is not preserved).
    """
    directory = os.path.dirname(params_path)
    basename = os.path.basename(params_path)
    if not basename.startswith('best_params_') or not basename.endswith('.json'):
        return None
    date = basename[len('best_params_'):-len('.json')]
    candidate = os.path.join(directory, f'{FINGERPRINT_BASENAME}_{date}.json')
    return candidate if os.path.isfile(candidate) else None


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
