"""Rebuild the LaTeX manuscript and regenerate the PDF in writings/out.

This helper is intentionally simple and fully reproducible: it invokes the
project's own audit/compile pipeline and leaves the final PDF under the
``writings/out`` directory without requiring any AI or manual editing steps.

Usage
-----
    python writings/auxiliar/scripts/rebuild_thesis.py
    python writings/auxiliar/scripts/rebuild_thesis.py --clean
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT_SCRIPT = ROOT / "audit" / "audit.py"
OUT_DIR = ROOT / "out"


def main() -> int:
    """Execute the thesis rebuild pipeline from the writings root."""
    parser = argparse.ArgumentParser(
        description="Recompila la tesis LaTeX y regenerar el PDF en writings/out/."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Elimina artefactos de compilación antes de recompilar.",
    )
    args = parser.parse_args()

    if not AUDIT_SCRIPT.exists():
        raise FileNotFoundError(f"No se encontró el compilador de tesis: {AUDIT_SCRIPT}")

    cmd = [sys.executable, str(AUDIT_SCRIPT)]
    if args.clean:
        cmd.append("--clean")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    print(f"Directorio base: {ROOT}")
    print(f"Ejecutando: {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)

    if completed.returncode != 0:
        print(f"La recompilación terminó con código {completed.returncode}.", file=sys.stderr)
        return completed.returncode

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(OUT_DIR.glob("*.pdf"))
    if pdfs:
        print(f"PDF generado en: {pdfs[-1].relative_to(ROOT)}")
    else:
        print("No se encontró ningún PDF nuevo en writings/out/.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
