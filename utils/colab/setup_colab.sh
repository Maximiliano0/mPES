#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  utils/colab/setup_colab.sh — bootstrap a Colab Pro+ runtime for mPES
# ═══════════════════════════════════════════════════════════════════════════════
#  What it does
#  ────────────
#  1. Installs the Python dependencies declared in utils/config/requirements.txt
#  2. Exports the env vars needed by mPES (VIRTUAL_ENV/PYTHONIOENCODING/oneDNN)
#  3. Verifies that Google Drive is mounted at /content/drive
#
#  Designed to be sourced from a Colab cell **after** the repo has been cloned
#  to /content/Win_mPES and Drive has been mounted from a Python cell with:
#      from google.colab import drive
#      drive.mount('/content/drive')
#
#  Usage (inside a Colab cell):
#      !bash utils/colab/setup_colab.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO_DIR="${REPO_DIR:-/content/Win_mPES}"
DRIVE_DIR="${DRIVE_DIR:-/content/drive/MyDrive/mPES}"
REQ_FILE="${REPO_DIR}/utils/config/requirements.txt"

echo "════════════════════════════════════════════════════════════════════════"
echo "  mPES — Colab Pro+ bootstrap"
echo "════════════════════════════════════════════════════════════════════════"

# --- Sanity checks --------------------------------------------------------
if [[ ! -d "$REPO_DIR" ]]; then
    echo "ERROR: repo not found at $REPO_DIR"
    echo "Clone it first, e.g.:  !git clone https://github.com/<user>/Win_mPES.git $REPO_DIR"
    exit 1
fi

if [[ ! -d "/content/drive/MyDrive" ]]; then
    echo "ERROR: Google Drive is not mounted at /content/drive"
    echo "Run in a Python cell first:"
    echo "    from google.colab import drive"
    echo "    drive.mount('/content/drive')"
    exit 1
fi

mkdir -p "$DRIVE_DIR"
echo "→ Drive workspace: $DRIVE_DIR"

# --- Install Python dependencies -----------------------------------------
echo "────────────────────────────────────────────────────────────────────────"
echo "  Installing Python dependencies"
echo "────────────────────────────────────────────────────────────────────────"
pip install --quiet --upgrade pip
pip install --quiet -r "$REQ_FILE"

# --- Repair Colab pyparsing/httplib2 incompatibility ---------------------
# Colab base image ships pyparsing < 3.1 but httplib2.auth (pulled in by
# googleapiclient → tensorflow.python.distribute) calls pp.DelimitedList
# (added in pyparsing 3.1).  Symptom on `import tensorflow`:
#   AttributeError: module 'pyparsing' has no attribute 'DelimitedList'.
# Pin pyparsing to a compatible recent release.
pip install --quiet --upgrade 'pyparsing>=3.1.0'

# --- Export env vars (written to /etc/profile.d for persistence) ---------
echo "────────────────────────────────────────────────────────────────────────"
echo "  Exporting mPES environment variables"
echo "────────────────────────────────────────────────────────────────────────"
ENV_FILE="/content/mpes_env.sh"
cat > "$ENV_FILE" <<EOF
export VIRTUAL_ENV="${REPO_DIR}"
export PYTHONIOENCODING="utf-8"
export TF_ENABLE_ONEDNN_OPTS="0"
export TF_CPP_MIN_LOG_LEVEL="2"
# Silence benign third-party SyntaxWarnings on Python 3.12 (e.g. matplotlib
# mathtext, optuna helpers) that surface as ``<unknown>:NN: SyntaxWarning:
# invalid escape sequence``.  These come from upstream packages and do not
# affect correctness; suppress to keep the Colab logs readable.
export PYTHONWARNINGS="ignore::SyntaxWarning"
# Determinism: keep TF ops bit-reproducible per seed (CPU and GPU).
export TF_DETERMINISTIC_OPS="1"
export TF_CUDNN_DETERMINISTIC="1"
# GPU policy: forwarded from the launcher cell.  If MPES_USE_GPU=1 we leave
# CUDA_VISIBLE_DEVICES alone so TF can see the Colab GPU; otherwise the
# package itself pins CPU.
export MPES_USE_GPU="${MPES_USE_GPU:-0}"
export PYTHONPATH="${REPO_DIR}:\${PYTHONPATH:-}"
EOF
# shellcheck disable=SC1090
source "$ENV_FILE"
echo "→ Env vars sourced from $ENV_FILE"

echo "════════════════════════════════════════════════════════════════════════"
echo "  Bootstrap complete. Next: run utils/colab/run_colab.sh <PKG> <TRIALS>"
echo "════════════════════════════════════════════════════════════════════════"
