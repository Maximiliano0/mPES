"""utils/colab/monitor.py — unified live-monitoring helper for Colab runs.

Works for any Bayesian-optimisation run launched by `utils/colab/run_colab.sh`.
The script writes a fixed set of artifacts to the per-run directory on Drive::

    /content/drive/MyDrive/mPES/<pkg>/<date>_BAYESIAN_OPT/
        run_meta.json          machine-readable launch metadata + pid
        bayesian_opt.log       stdout (per-trial progress lines)
        bayesian_opt_err.log   stderr
        optimize.pid           pid (also embedded in run_meta.json)
        optuna_study_<date>.db Optuna SQLite storage

This module reads those artifacts and prints a uniform status report
regardless of which package is running.

Usage from a Colab cell::

    from utils.colab.monitor import monitor, follow

    monitor()                       # one-shot snapshot of the active run
    monitor(pkg='dqn', date='2026-04-22')   # specific run
    monitor(n_log_lines=30)         # show more recent log lines
    follow(refresh=15)              # auto-refresh every 15 s (Ctrl-C to stop)
"""

##########################
##  Imports externos    ##
##########################
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Optional

##########################
##  Constantes          ##
##########################
DRIVE_ROOT_DEFAULT = '/content/drive/MyDrive/mPES'

# Map Cell-1 PKG aliases (`ql`, `dql`, `dqn`, `ac`/`a2c`, `rdqn`,
# `transformer`/`tr`) to the full package directory name used on Drive.
ALIAS_TO_PKG = {
    'ql':   'pes_ql',  'pes_ql':   'pes_ql',
    'dql':  'pes_dql', 'pes_dql':  'pes_dql',
    'dqn':  'pes_dqn', 'pes_dqn':  'pes_dqn',
    'ac':   'pes_a2c', 'a2c':      'pes_a2c', 'pes_a2c':  'pes_a2c',
    'rdqn': 'pes_rdqn', 'recurrent': 'pes_rdqn', 'pes_rdqn': 'pes_rdqn',
    'tr':   'pes_trf', 'transformer': 'pes_trf', 'pes_trf':  'pes_trf',
}

OPTUNA_TRIAL_STATES = {
    0: 'RUNNING', 1: 'COMPLETE', 2: 'PRUNED', 3: 'FAIL', 4: 'WAITING',
}


##########################
##  Funciones internas  ##
##########################

def _resolve_pkg(pkg: Optional[str]) -> str:
    """Resolve the full package directory name from an alias or env var.

    Parameters
    ----------
    pkg : str or None
        Alias (``ql``/``dql``/``dqn``/``rdqn``/``ac``/``a2c``/``tr``
        /``transformer``) or full name
        (``pes_ql``/...). When ``None``, falls back to ``os.environ['PKG']``.

    Returns
    -------
    str
        Full package directory name (e.g. ``"pes_ql"``).

    Raises
    ------
    ValueError
        If ``pkg`` cannot be resolved.
    """
    raw = pkg or os.environ.get('PKG', '')
    raw = raw.strip().lower()
    if raw not in ALIAS_TO_PKG:
        raise ValueError(
            f"Unknown package '{pkg}'. Set PKG in cell 1 or pass pkg= explicitly. "
            f"Valid: {sorted(set(ALIAS_TO_PKG))}."
        )
    return ALIAS_TO_PKG[raw]


def _latest_run_dir(pkg_dir: str) -> Optional[str]:
    """Return the most recently modified ``*_BAYESIAN_OPT`` subdirectory.

    Parameters
    ----------
    pkg_dir : str
        Absolute path to ``<DRIVE_ROOT>/<pkg>``.

    Returns
    -------
    str or None
        Absolute path to the latest run directory, or ``None`` if none exist.
    """
    if not os.path.isdir(pkg_dir):
        return None
    candidates = [
        os.path.join(pkg_dir, name) for name in os.listdir(pkg_dir)
        if name.endswith('_BAYESIAN_OPT')
        and os.path.isdir(os.path.join(pkg_dir, name))
    ]
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def _resolve_run_dir(pkg: Optional[str], date: Optional[str],
                     drive_root: str) -> str:
    """Resolve the run directory for a given (pkg, date) pair.

    Parameters
    ----------
    pkg : str or None
        Package alias or full name. ``None`` reads ``os.environ['PKG']``.
    date : str or None
        ``YYYY-MM-DD`` date. ``None`` selects the most recent run.
    drive_root : str
        Drive root (default ``/content/drive/MyDrive/mPES``).

    Returns
    -------
    str
        Absolute path to ``<drive_root>/<pkg>/<date>_BAYESIAN_OPT/``.
    """
    pkg_full = _resolve_pkg(pkg)
    pkg_dir = os.path.join(drive_root, pkg_full)
    if date:
        return os.path.join(pkg_dir, f'{date}_BAYESIAN_OPT')
    latest = _latest_run_dir(pkg_dir)
    if latest is None:
        raise FileNotFoundError(
            f"No *_BAYESIAN_OPT directory found under {pkg_dir}. "
            f"Has the optimisation been launched yet?"
        )
    return latest


def _is_alive(pid: int) -> bool:
    """Return ``True`` if a process with the given PID is currently running.

    Parameters
    ----------
    pid : int
        Process ID to probe.

    Returns
    -------
    bool
        ``True`` if the process exists, ``False`` otherwise.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _read_meta(run_dir: str) -> dict:
    """Read the ``run_meta.json`` sidecar, returning ``{}`` if absent.

    Parameters
    ----------
    run_dir : str
        Absolute path to the run directory.

    Returns
    -------
    dict
        Parsed metadata, or an empty dict.
    """
    meta_path = os.path.join(run_dir, 'run_meta.json')
    if not os.path.isfile(meta_path):
        return {}
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _read_pid(run_dir: str, meta: dict) -> int:
    """Read the optimisation PID from metadata or the legacy pid file.

    Parameters
    ----------
    run_dir : str
        Absolute path to the run directory.
    meta : dict
        Parsed ``run_meta.json`` contents.

    Returns
    -------
    int
        PID, or 0 if unknown.
    """
    if isinstance(meta.get('pid'), int):
        return int(meta['pid'])
    pid_path = os.path.join(run_dir, 'optimize.pid')
    if not os.path.isfile(pid_path):
        return 0
    try:
        with open(pid_path, 'r', encoding='utf-8') as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return 0


def _query_study(run_dir: str, meta: dict) -> dict:
    """Read trial counts and best value directly from the Optuna SQLite DB.

    Parameters
    ----------
    run_dir : str
        Absolute path to the run directory.
    meta : dict
        Parsed ``run_meta.json`` contents (used to locate the DB).

    Returns
    -------
    dict
        Keys: ``counts`` (state name → int), ``best_value`` (float or None),
        ``best_trial`` (int or None), ``first_ts`` / ``last_ts``
        (datetime or None) for completed trials, ``error`` (str or None).
    """
    out = {'counts': {}, 'best_value': None, 'best_trial': None,
           'first_ts': None, 'last_ts': None, 'error': None}
    run_date = meta.get('run_date') or os.path.basename(run_dir).split('_')[0]
    db_path = os.path.join(run_dir, f'optuna_study_{run_date}.db')
    if not os.path.isfile(db_path):
        out['error'] = f"DB not found: {os.path.basename(db_path)}"
        return out
    try:
        # Read-only via URI to avoid lock contention with the live writer.
        uri = f'file:{db_path}?mode=ro&immutable=0'
        con = sqlite3.connect(uri, uri=True, timeout=2.0)
        cur = con.cursor()
        cur.execute('SELECT state, COUNT(*) FROM trials GROUP BY state')
        for state, cnt in cur.fetchall():
            name = OPTUNA_TRIAL_STATES.get(state, f'STATE_{state}')
            out['counts'][name] = cnt
        cur.execute(
            "SELECT trial_id, number, datetime_start, datetime_complete "
            "FROM trials WHERE state=1 ORDER BY datetime_complete ASC"
        )
        rows = cur.fetchall()
        if rows:
            out['first_ts'] = rows[0][2]
            out['last_ts']  = rows[-1][3]
        cur.execute(
            "SELECT t.number, v.value FROM trial_values v "
            "JOIN trials t ON t.trial_id = v.trial_id "
            "WHERE t.state=1 ORDER BY v.value DESC LIMIT 1"
        )
        row = cur.fetchone()
        if row:
            out['best_trial'] = int(row[0])
            out['best_value'] = float(row[1])
        con.close()
    except sqlite3.Error as exc:
        out['error'] = f"sqlite: {exc}"
    return out


def _tail_lines(path: str, n: int) -> list:
    """Return the last ``n`` lines of a UTF-8 text file.

    Parameters
    ----------
    path : str
        File path.
    n : int
        Maximum number of lines to return.

    Returns
    -------
    list of str
        The trailing lines (without newline characters), or ``[]`` if the
        file is missing or unreadable.
    """
    if not os.path.isfile(path):
        return []
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 8192
            data = b''
            while size > 0 and data.count(b'\n') <= n:
                read = min(block, size)
                size -= read
                f.seek(size)
                data = f.read(read) + data
        text = data.decode('utf-8', errors='replace')
        return text.splitlines()[-n:]
    except OSError:
        return []


def _filter_trial_lines(lines: list) -> list:
    """Keep only the per-trial progress lines emitted by all optimisers.

    Parameters
    ----------
    lines : list of str
        Raw log lines.

    Returns
    -------
    list of str
        Lines matching the canonical ``  Trial NNN/TTT  |  value=...  |
        best=...  |  elapsed=...`` format.
    """
    return [ln for ln in lines if 'Trial ' in ln and '|' in ln and 'value=' in ln]


def _humanise_seconds(seconds: float) -> str:
    """Format a duration in seconds as ``HhMMmSSs`` or ``MMmSSs``.

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        Compact human-readable string.
    """
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f'{h}h{m:02d}m{s:02d}s'
    return f'{m}m{s:02d}s'


def _parse_iso(ts: Optional[str]):
    """Parse an ISO-8601 timestamp produced by Optuna or the launcher.

    Parameters
    ----------
    ts : str or None
        ISO-8601 string, possibly with trailing ``Z``.

    Returns
    -------
    datetime or None
        Parsed timestamp, or ``None`` on failure.
    """
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace('Z', '+00:00'))
    except ValueError:
        return None


##########################
##  API pública         ##
##########################

def monitor(pkg: Optional[str] = None, date: Optional[str] = None,
            n_log_lines: int = 15, n_err_lines: int = 5,
            drive_root: str = DRIVE_ROOT_DEFAULT) -> dict:
    """Print a unified status snapshot for a Bayesian-optimisation run.

    Parameters
    ----------
    pkg : str or None, optional
        Package alias (``ql``/``dql``/``dqn``/``rdqn``/``ac``/``tr``) or full name. When
        ``None`` (default), reads ``os.environ['PKG']`` set by Cell 1.
    date : str or None, optional
        Run date as ``YYYY-MM-DD``. When ``None`` (default), the most
        recently modified run directory is used.
    n_log_lines : int, optional
        Number of trailing per-trial log lines to display. Default ``15``.
    n_err_lines : int, optional
        Number of trailing stderr lines to display. Default ``5``.
    drive_root : str, optional
        Drive workspace root. Default ``/content/drive/MyDrive/mPES``.

    Returns
    -------
    dict
        The same data printed, in machine-readable form: ``run_dir``,
        ``meta``, ``alive``, ``study`` (counts + best), ``last_trial_lines``,
        ``last_err_lines``, ``throughput_per_hour``, ``eta_seconds``.
    """
    run_dir = _resolve_run_dir(pkg, date, drive_root)
    meta = _read_meta(run_dir)
    pid = _read_pid(run_dir, meta)
    alive = _is_alive(pid)
    study = _query_study(run_dir, meta)

    log_path = os.path.join(run_dir, 'bayesian_opt.log')
    err_path = os.path.join(run_dir, 'bayesian_opt_err.log')
    log_tail = _tail_lines(log_path, max(n_log_lines * 6, 60))
    trial_lines = _filter_trial_lines(log_tail)[-n_log_lines:]
    err_tail = _tail_lines(err_path, n_err_lines)

    # ── Throughput + ETA from DB timestamps ─────────────────────────────
    completed = study['counts'].get('COMPLETE', 0)
    n_trials = int(meta.get('n_trials', 0)) or completed
    throughput = None
    eta_seconds = None
    avg_seconds = None
    first_ts = _parse_iso(study['first_ts'])
    last_ts  = _parse_iso(study['last_ts'])
    if first_ts and last_ts and completed > 1:
        elapsed = (last_ts - first_ts).total_seconds()
        if elapsed > 0:
            throughput = completed / (elapsed / 3600.0)
            remaining = max(0, n_trials - completed)
            avg_seconds = elapsed / completed
            eta_seconds = remaining * avg_seconds

    # ── Pretty print ────────────────────────────────────────────────────
    bar = '═' * 72
    now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')
    print(bar)
    print(f"  mPES Bayesian Optimisation — status @ {now_str}")
    print(bar)
    print(f"  Package      : {meta.get('package', os.path.basename(os.path.dirname(run_dir)))}")
    print(f"  Module       : {meta.get('module', '?')}")
    print(f"  Run date     : {meta.get('run_date', '?')}")
    print(f"  Run dir      : {run_dir}")
    print(f"  Git          : {meta.get('git_branch', '?')}@{meta.get('git_sha', '?')}")
    print(f"  Launched     : {meta.get('launch_ts', '?')}")
    print(f"  PID          : {pid or '?'}   ({'ALIVE' if alive else 'NOT RUNNING'})")
    print(bar)

    counts = study['counts']
    target = meta.get('n_trials', '?')
    print(f"  Trials       : {completed} / {target} completed"
          f"   (pruned={counts.get('PRUNED', 0)}"
          f", running={counts.get('RUNNING', 0)}"
          f", failed={counts.get('FAIL', 0)})")
    if study['best_value'] is not None:
        print(f"  Best         : trial #{study['best_trial']}  →  value={study['best_value']:.6f}")
    else:
        print("  Best         : (no completed trials yet)")
    if throughput is not None and avg_seconds is not None:
        print(f"  Throughput   : {throughput:.2f} trials/h"
              f"   (avg {avg_seconds:.1f}s/trial)")
    if eta_seconds is not None and eta_seconds > 0:
        eta_dt = datetime.now(timezone.utc).timestamp() + eta_seconds
        eta_iso = datetime.fromtimestamp(eta_dt, timezone.utc).strftime('%Y-%m-%d %H:%MZ')
        print(f"  ETA          : ~{_humanise_seconds(eta_seconds)} (≈ {eta_iso})")
    if study['error']:
        print(f"  DB note      : {study['error']}")
    print(bar)

    print(f"  Last {len(trial_lines)} trial line(s) from bayesian_opt.log:")
    if trial_lines:
        for ln in trial_lines:
            print(f"    {ln}")
    else:
        print("    (no per-trial lines yet — optimisation may still be warming up)")
    print(bar)

    if err_tail and any(ln.strip() for ln in err_tail):
        print(f"  Last {len(err_tail)} stderr line(s):")
        for ln in err_tail:
            print(f"    {ln}")
        print(bar)

    return {
        'run_dir':              run_dir,
        'meta':                 meta,
        'pid':                  pid,
        'alive':                alive,
        'study':                study,
        'last_trial_lines':     trial_lines,
        'last_err_lines':       err_tail,
        'throughput_per_hour':  throughput,
        'eta_seconds':          eta_seconds,
    }


def follow(pkg: Optional[str] = None, date: Optional[str] = None,
           refresh: int = 10, n_log_lines: int = 10,
           drive_root: str = DRIVE_ROOT_DEFAULT) -> None:
    """Auto-refresh the snapshot every ``refresh`` seconds until interrupted.

    Parameters
    ----------
    pkg : str or None, optional
        Package alias or full name. Default reads ``os.environ['PKG']``.
    date : str or None, optional
        Run date. Default selects the most recent run directory.
    refresh : int, optional
        Seconds between refreshes. Default ``10``.
    n_log_lines : int, optional
        Trailing per-trial lines to show on each refresh. Default ``10``.
    drive_root : str, optional
        Drive workspace root.

    Notes
    -----
    Press Ctrl-C in the Colab cell to stop the loop. The optimisation
    process keeps running independently — `follow` only reads files.
    """
    try:
        while True:
            try:
                from IPython.display import clear_output  # type: ignore[import-not-found]
                clear_output(wait=True)
            except ImportError:
                pass
            monitor(pkg=pkg, date=date, n_log_lines=n_log_lines,
                    drive_root=drive_root)
            time.sleep(max(2, int(refresh)))
    except KeyboardInterrupt:
        print("\n  follow() stopped — optimisation process is unaffected.")
