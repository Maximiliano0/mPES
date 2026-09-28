"""Rebuild the LaTeX manuscript and regenerate the PDF in writings/out.

This helper is intentionally simple and fully reproducible: it invokes the
project's own audit/compile pipeline and leaves the final PDF under the
``writings/out`` directory without requiring any AI or manual editing steps.

Usage
-----
    python writings/auxiliar/scripts/rebuild_thesis.py
    python writings/auxiliar/scripts/rebuild_thesis.py --clean

With ``--clean`` the script first runs ``audit/audit.py --clean``, which deletes
the build artefacts, and then recompiles. The rebuild only succeeds if a PDF
newer than the start of the run appears in ``writings/out``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT_SCRIPT = ROOT / "audit" / "audit.py"
OUT_DIR = ROOT / "out"


def main() -> int:
    """Execute the thesis rebuild pipeline from the writings root."""
    parser = argparse.ArgumentParser(
        description="Recompila la tesis LaTeX y regenera el PDF en writings/out/."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Elimina los artefactos de compilación (audit.py --clean) antes de recompilar.",
    )
    args = parser.parse_args()

    if not AUDIT_SCRIPT.exists():
        raise FileNotFoundError(f"No se encontró el compilador de tesis: {AUDIT_SCRIPT}")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    print(f"Directorio base: {ROOT}")
    started = time.time()

    commands = [[sys.executable, str(AUDIT_SCRIPT)]]
    if args.clean:
        commands.insert(0, [sys.executable, str(AUDIT_SCRIPT), "--clean"])
    for cmd in commands:
        print(f"Ejecutando: {' '.join(cmd)}")
        completed = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
        if completed.returncode != 0:
            print(f"La recompilación terminó con código {completed.returncode}.", file=sys.stderr)
            return completed.returncode

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = [pdf for pdf in OUT_DIR.glob("*.pdf") if pdf.stat().st_mtime >= started]
    if pdfs:
        newest = max(pdfs, key=lambda pdf: pdf.stat().st_mtime)
        print(f"PDF generado en: {newest.relative_to(ROOT)}")
    else:
        print("No se encontró ningún PDF nuevo en writings/out/.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
