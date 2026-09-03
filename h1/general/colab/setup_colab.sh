#!/usr/bin/env bash
#
#  utils/colab/setup_colab.sh  bootstrap a Colab Pro+ runtime for mPES
#
#  What it does
#
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
#
set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/content/mPES}"
REPO_DIR="${REPO_DIR:-$WORKSPACE_DIR}"
H_DIR="${H_DIR:-${WORKSPACE_DIR}/h1}"
DRIVE_DIR="${DRIVE_DIR:-/content/drive/MyDrive/mPES}"

echo ""
echo "  mPES  Colab Pro+ bootstrap"
echo ""

# --- Sanity checks --------------------------------------------------------
if [[ ! -d "$H_DIR" ]]; then
    echo "ERROR: h1 not found at $H_DIR"
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
echo " Drive workspace: $DRIVE_DIR"

# --- Install Python dependencies -----------------------------------------
echo ""
echo "  Installing Python dependencies"
echo ""
echo "  Checking pinned optimisation and training dependencies"
runtime_packages="$(python3 - <<'PY'
from importlib.metadata import PackageNotFoundError, version

required = {
    'gymnasium': 'gymnasium==1.2.3',
    'keras': 'keras==3.13.2',
    'matplotlib': 'matplotlib==3.10.8',
    'numpy': 'numpy==2.4.3',
    'optuna': 'optuna==4.7.0',
    'tensorflow': 'tensorflow==2.21.0',
}

install = []
for distribution, requirement in required.items():
    try:
        installed = version(distribution)
    except PackageNotFoundError:
        install.append(requirement)
    else:
        expected = requirement.split('==', maxsplit=1)[1]
        if installed != expected:
            install.append(requirement)
print(' '.join(install))
PY
)"
if [[ -n "$runtime_packages" ]]; then
    echo "  Installing pinned runtime packages: $runtime_packages"
    pip install --quiet --prefer-binary $runtime_packages
else
    echo "  Required runtime packages already match project versions."
fi

# --- Export env vars (written to /etc/profile.d for persistence) ---------
echo ""
echo "  Exporting mPES environment variables"
echo ""
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
export PYTHONPATH="${WORKSPACE_DIR}:${H_DIR}:\${PYTHONPATH:-}"
EOF
# shellcheck disable=SC1090
source "$ENV_FILE"
echo " Env vars sourced from $ENV_FILE"

echo ""
echo "  Bootstrap complete. Next: run utils/colab/run_colab.sh <PKG> <TRIALS>"
echo ""
