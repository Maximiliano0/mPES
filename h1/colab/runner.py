"""Run long mPES optimisations and trainings from Google Colab.

The runner executes on Colab's local disk and uses Google Drive only as a
persistent bucket. Benchmark commands are intentionally not supported.
"""
from __future__ import annotations

import argparse
import datetime as datetime_module
import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any


PACKAGE_REGISTRY: dict[str, dict[str, dict[str, str | None]]] = {
    'h1': {
        'pes_ql': {'group': 'tabular', 'optimize': 'tabular.pes_ql.ext.optimize_rl', 'train': 'tabular.pes_ql.ext.train_rl'},
        'pes_dql': {'group': 'tabular', 'optimize': 'tabular.pes_dql.ext.optimize_rl', 'train': 'tabular.pes_dql.ext.train_rl'},
        'pes_dqn': {'group': 'ml', 'optimize': 'ml.pes_dqn.ext.optimize_dqn', 'train': 'ml.pes_dqn.ext.train_dqn'},
        'pes_rdqn': {'group': 'ml', 'optimize': 'ml.pes_rdqn.ext.optimize_rdqn', 'train': 'ml.pes_rdqn.ext.train_rdqn'},
        'pes_a2c': {'group': 'ml', 'optimize': 'ml.pes_a2c.ext.optimize_a2c', 'train': 'ml.pes_a2c.ext.train_a2c'},
        'pes_trf': {'group': 'ml', 'optimize': 'ml.pes_trf.ext.optimize_tr', 'train': 'ml.pes_trf.ext.train_transformer'},
        'pes_ens_sprb': {'group': 'ens', 'optimize': 'ens.pes_ens_sprb.ext.optimize_ens', 'train': None},
        'pes_ens_accq': {'group': 'ens', 'optimize': 'ens.pes_ens_accq.ext.optimize_ens', 'train': None},
    },
    'h2': {
        'ql_conf': {'group': 'tabular_conf', 'optimize': 'tabular_conf.ql_conf.ext.optimize_rl', 'train': 'tabular_conf.ql_conf.ext.train_rl'},
    },
    'h3': {
        'ql_uq': {'group': 'tabular_uq', 'optimize': 'tabular_uq.ql_uq.ext.optimize_rl', 'train': 'tabular_uq.ql_uq.ext.train_rl'},
    },
}
PACKAGE_DEPENDENCIES = {
    'pes_ens_sprb': ('pes_dqn', 'pes_rdqn', 'pes_trf'),
    'pes_ens_accq': ('pes_dqn', 'pes_rdqn', 'pes_trf'),
}

TRIAL_RE = re.compile(r'Trial\s+(\d+)(?:/|\s).*?best[= ]+([0-9.eE+-]+)')


def _timestamp() -> str:
    """Return an UTC timestamp suitable for metadata files."""
    return datetime_module.datetime.now(datetime_module.timezone.utc).isoformat()


def _copy_tree(source: Path, target: Path) -> None:
    """Merge one directory tree into another when the source exists."""
    if source.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, dirs_exist_ok=True)


def _package_dir(repo_root: Path, line: str, package: str) -> Path:
    """Return the package directory for a registered line and package."""
    group = PACKAGE_REGISTRY[line][package]['group']
    assert group is not None
    return repo_root / line / group / package


def _pull_package(repo_root: Path, drive_root: Path, line: str, package: str) -> None:
    """Restore package inputs and outputs from the Drive bucket."""
    packages = (package,) + PACKAGE_DEPENDENCIES.get(package, ())
    for dependency in packages:
        local = _package_dir(repo_root, line, dependency)
        remote = drive_root / line / str(PACKAGE_REGISTRY[line][dependency]['group']) / dependency
        _copy_tree(remote / 'inputs', local / 'inputs')
        _copy_tree(remote / 'outputs', local / 'outputs')


def _push_package(repo_root: Path, drive_root: Path, line: str, package: str) -> None:
    """Persist package inputs and outputs into the Drive bucket."""
    packages = (package,) + PACKAGE_DEPENDENCIES.get(package, ())
    for dependency in packages:
        local = _package_dir(repo_root, line, dependency)
        remote = drive_root / line / str(PACKAGE_REGISTRY[line][dependency]['group']) / dependency
        _copy_tree(local / 'inputs', remote / 'inputs')
        _copy_tree(local / 'outputs', remote / 'outputs')


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON metadata atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    temporary.replace(path)


def _git_metadata(repo_root: Path) -> dict[str, str]:
    """Return the repository revision used by the Colab execution."""
    metadata = {'repository': str(repo_root), 'git_revision': 'unavailable', 'git_remote': 'unavailable'}
    try:
        metadata['git_revision'] = subprocess.check_output(
            ['git', '-C', str(repo_root), 'rev-parse', 'HEAD'], text=True).strip()
        metadata['git_remote'] = subprocess.check_output(
            ['git', '-C', str(repo_root), 'config', '--get', 'remote.origin.url'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return metadata


def _reader(process: subprocess.Popen[str], output: queue.Queue[str | None]) -> None:
    """Forward subprocess output to a queue from a dedicated reader thread."""
    assert process.stdout is not None
    for line in process.stdout:
        output.put(line)
    output.put(None)


def _command(line: str, package: str, operation: str, trials: int, resume_date: str | None) -> list[str]:
    """Build the package command for the selected operation."""
    module = PACKAGE_REGISTRY[line][package][operation]
    if module is None:
        raise ValueError(f'{package} no tiene una operacion {operation}; los ensembles solo se optimizan.')
    command = [sys.executable, '-m', module]
    if operation == 'optimize':
        command.append(str(trials))
        if resume_date:
            command.extend(['--resume', resume_date])
    elif resume_date and line == 'h1' and package in {'pes_dqn', 'pes_rdqn', 'pes_a2c', 'pes_trf'}:
        command.extend(['--from-best', resume_date])
    return command


def _run_package(repo_root: Path, drive_root: Path, line: str, package: str,
                 operation: str, trials: int, resume_date: str | None,
                 run_dir: Path, sync_interval: int) -> int:
    """Run one package, stream its output, and periodically sync its files."""
    _pull_package(repo_root, drive_root, line, package)
    command = _command(line, package, operation, trials, resume_date)
    log_path = run_dir / line / package / f'{operation}.log'
    status_path = run_dir / line / package / 'status.json'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {
        'line': line, 'package': package, 'operation': operation,
        'state': 'running', 'requested_trials': trials, 'started_at': _timestamp(),
        'command': command,
    }
    _write_json(status_path, status)
    environment = os.environ.copy()
    environment.update({'PYTHONIOENCODING': 'utf-8', 'TF_ENABLE_ONEDNN_OPTS': '0', 'PYTHONUNBUFFERED': '1'})
    process = subprocess.Popen(command, cwd=repo_root / line, env=environment,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    lines: queue.Queue[str | None] = queue.Queue()
    threading.Thread(target=_reader, args=(process, lines), daemon=True).start()
    finished_reading = False
    last_sync = time.monotonic()
    with log_path.open('a', encoding='utf-8') as log_file:
        while not finished_reading or process.poll() is None:
            try:
                line_output = lines.get(timeout=1.0)
            except queue.Empty:
                line_output = ''
            if line_output is None:
                finished_reading = True
            elif line_output:
                print(line_output, end='', flush=True)
                log_file.write(line_output)
                match = TRIAL_RE.search(line_output)
                if match:
                    status['completed_trials'] = int(match.group(1))
                    status['best_value'] = float(match.group(2))
                    status['last_update'] = _timestamp()
            if time.monotonic() - last_sync >= sync_interval:
                status['last_sync'] = _timestamp()
                _write_json(status_path, status)
                _push_package(repo_root, drive_root, line, package)
                _copy_tree(run_dir, drive_root / 'runs' / run_dir.name)
                last_sync = time.monotonic()
    return_code = process.wait()
    status.update({'state': 'completed' if return_code == 0 else 'failed',
                   'returncode': return_code, 'finished_at': _timestamp(),
                   'last_sync': _timestamp()})
    _write_json(status_path, status)
    _push_package(repo_root, drive_root, line, package)
    _copy_tree(run_dir, drive_root / 'runs' / run_dir.name)
    return return_code


def main() -> None:
    """Run selected optimisations or trainings with Drive persistence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--line', choices=PACKAGE_REGISTRY, required=True)
    parser.add_argument('--packages', required=True, help='Comma-separated package names.')
    parser.add_argument('--operation', choices=('optimize', 'train'), required=True)
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--resume-date')
    parser.add_argument('--repo-root', type=Path, default=Path.cwd())
    parser.add_argument('--drive-root', type=Path, required=True)
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--sync-interval', type=int, default=300)
    arguments = parser.parse_args()
    packages = [item.strip() for item in arguments.packages.split(',') if item.strip()]
    unknown = [package for package in packages if package not in PACKAGE_REGISTRY[arguments.line]]
    if unknown:
        parser.error(f'Paquetes no registrados para {arguments.line}: {", ".join(unknown)}')
    if arguments.sync_interval < 10:
        parser.error('--sync-interval debe ser >= 10 segundos')
    run_id = arguments.run_id or f'{arguments.line}_{arguments.operation}_{datetime_module.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    run_dir = arguments.repo_root / '.colab_runs' / run_id
    manifest = {
        'run_id': run_id, 'line': arguments.line, 'packages': packages,
        'operation': arguments.operation, 'requested_trials': arguments.trials,
        'resume_date': arguments.resume_date, 'started_at': _timestamp(),
        'drive_root': str(arguments.drive_root), 'benchmarks_enabled': False,
    }
    manifest.update(_git_metadata(arguments.repo_root))
    _write_json(run_dir / 'manifest.json', manifest)
    _copy_tree(run_dir, arguments.drive_root / 'runs' / run_id)
    for package in packages:
        code = _run_package(arguments.repo_root, arguments.drive_root, arguments.line, package,
                             arguments.operation, arguments.trials, arguments.resume_date,
                             run_dir, arguments.sync_interval)
        if code != 0:
            raise SystemExit(code)
    manifest.update({'state': 'completed', 'finished_at': _timestamp()})
    _write_json(run_dir / 'manifest.json', manifest)
    _copy_tree(run_dir, arguments.drive_root / 'runs' / run_id)
    print(f'[colab] run completed: {run_id}')


if __name__ == '__main__':
    main()
