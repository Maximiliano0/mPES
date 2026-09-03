"""
Wrapper that redirects stdout/stderr to files **in Python** before
running a module via ``runpy.run_module``.

Why: on Windows, ``Start-Process -RedirectStandardOutput`` creates Win32
pipes for the standard handles.  TensorFlow (and other native DLLs) can
crash (ACCESS_VIOLATION 0xC0000005) when those handles are pipes instead
of a real console.  By redirecting at the Python level the OS process
keeps a normal (hidden) console window and only Python-level I/O is
captured.

Usage (called by run_bayesian_opt.ps1):
    python utils/run_module.py <module> <log_file> <err_file> [args ...]

Example:
    python utils/run_module.py pes_dqn.ext.optimize_dqn ^
        pes_dqn/inputs/bayesian_opt.log ^
        pes_dqn/inputs/bayesian_opt_err.log 110
"""

import io
import os
import runpy
import sys


def main():
    """Redirect stdout/stderr to files and run the target module."""

    # Auto-set environment variables for detached execution (WMI / ShellExecute)
    # so that the process works even without inheriting the parent shell's env.
    venv_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))
    os.environ.setdefault('VIRTUAL_ENV', venv_dir)
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

    if len(sys.argv) < 4:
        print(
            "Usage: python utils/run_module.py <module> <log_file> <err_file> [args ...]",
            file=sys.stderr,
        )
        sys.exit(2)

    module   = sys.argv[1]
    log_file = sys.argv[2]
    err_file = sys.argv[3]

    # Replace sys.argv so the target module sees only its own arguments
    sys.argv = [module] + sys.argv[4:]

    # Ensure CWD is on sys.path (mirrors python -m behaviour)
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    # Ensure parent directories exist
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(err_file)), exist_ok=True)

    # Redirect stdout and stderr to files (line-buffered, UTF-8)
    sys.stdout = io.open(log_file, 'w', encoding='utf-8', buffering=1)  # type: ignore[assignment]
    sys.stderr = io.open(err_file, 'w', encoding='utf-8', buffering=1)  # type: ignore[assignment]

    # Redirect stdin from devnull so that accidental input() calls
    # (e.g. the VIRTUAL_ENV prompt in __init__.py) fail fast with
    # EOFError instead of hanging forever in a hidden window.
    sys.stdin = open(os.devnull, 'r')  # type: ignore[assignment]

    try:
        runpy.run_module(module, run_name='__main__', alter_sys=True)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == '__main__':
    main()
