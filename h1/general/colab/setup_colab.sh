#!/usr/bin/env bash
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  utils/colab/setup_colab.sh â€” bootstrap a Colab Pro+ runtime for mPES
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  What it does
#  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/content/mPES}"
REPO_DIR="${REPO_DIR:-$WORKSPACE_DIR}"
H1_DIR="${H1_DIR:-${WORKSPACE_DIR}/h1}"
DRIVE_DIR="${DRIVE_DIR:-/content/drive/MyDrive/mPES}"
REQ_FILE="${REQ_FILE:-${WORKSPACE_DIR}/requirements.txt}"

echo "â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•"
echo "  mPES â€” Colab Pro+ bootstrap"
echo "â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•"

# --- Sanity checks --------------------------------------------------------
if [[ ! -d "$H1_DIR" ]]; then
    echo "ERROR: h1 not found at $H1_DIR"
    exit 1
fi

if [[ ! -f "$REQ_FILE" ]]; then
    echo "ERROR: requirements file not found at $REQ_FILE"
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
echo "â†’ Drive workspace: $DRIVE_DIR"

# --- Install Python dependencies -----------------------------------------
echo "â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€"
echo "  Installing Python dependencies"
echo "â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€"
pip install --quiet --upgrade pip
pip install --quiet -r "$REQ_FILE"

# --- Repair Colab pyparsing/httplib2 incompatibility ---------------------
# Colab base image ships pyparsing < 3.1 but httplib2.auth (pulled in by
# googleapiclient â†’ tensorflow.python.distribute) calls pp.DelimitedList
# (added in pyparsing 3.1).  Symptom on `import tensorflow`:
#   AttributeError: module 'pyparsing' has no attribute 'DelimitedList'.
# Pin pyparsing to a compatible recent release.
pip install --quiet --upgrade 'pyparsing>=3.1.0'

# --- Export env vars (written to /etc/profile.d for persistence) ---------
echo "â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€"
echo "  Exporting mPES environment variables"
echo "â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€"
ENV_FILE="/content/mpes_env.sh"
cat > "$ENV_FILE" <<EOF
export VIRTUAL_ENV="${WORKSPACE_DIR}"
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
export PYTHONPATH="${WORKSPACE_DIR}:${H1_DIR}:\${PYTHONPATH:-}"
EOF
# shellcheck disable=SC1090
source "$ENV_FILE"
echo "â†’ Env vars sourced from $ENV_FILE"

echo "â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•"
echo "  Bootstrap complete. Next: run utils/colab/run_colab.sh <PKG> <TRIALS>"
echo "â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•"
