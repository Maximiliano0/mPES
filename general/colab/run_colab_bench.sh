#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  general/run_colab_bench.sh — launch the OOD benchmark sweep on Colab Pro+
# ═══════════════════════════════════════════════════════════════════════════════
#  What it does
#  ────────────
#  1. Mirrors trained model artefacts from a Drive snapshot into the cloned
#     repo's <group>/<pkg>/inputs/ directories (so general.scripts.runner can find
#     the .keras / .npy / best_params.json files).
#  2. Mirrors general/results/ from Drive (resume support).
#  3. Runs python -m general.scripts.orchestrate (then aggregate / plot / report).
#  4. Mirrors general/results/ back to Drive.
#
#  Designed to be invoked from the Colab launcher notebook
#  (general/colab_bench.ipynb), AFTER setup_colab.sh has prepared the
#  environment.
#
#  Usage
#  ─────
#      bash general/run_colab_bench.sh            # all 7 pkgs, all scenarios
#      bash general/run_colab_bench.sh pes_dqn    # restrict to one pkg
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

PKG_FILTER="${1:-}"

REPO_DIR="${REPO_DIR:-/content/Win_mPES}"
DRIVE_DIR="${DRIVE_DIR:-/content/drive/MyDrive/mPES}"
DRIVE_BENCH="${DRIVE_DIR}/_benchmark"
DRIVE_RESULTS="${DRIVE_BENCH}/results"

# shellcheck disable=SC1091
[[ -f /content/mpes_env.sh ]] && source /content/mpes_env.sh

cd "$REPO_DIR"
mkdir -p "$DRIVE_BENCH" "$DRIVE_RESULTS"

# ─── Mirror trained artefacts from Drive into the cloned repo ────────────
# Each package's Bayesian-opt run on Drive lives at:
#   $DRIVE_DIR/<PKG>/<DATE>_<TYPE>/inputs/  (best_params.json, *.keras, *.npy)
# We copy the most recent snapshot for every PKG into <group>/<pkg>/inputs/.
declare -A PKG_GROUP=(
    [pes_ql]=tabular  [pes_dql]=tabular
    [pes_dqn]=ml      [pes_rdqn]=ml      [pes_a2c]=ml
    [pes_trf]=ml      [pes_ens]=ml
)

echo "════════════════════════════════════════════════════════════════════════"
echo "  Mirroring trained artefacts from Drive into repo"
echo "════════════════════════════════════════════════════════════════════════"
for pkg in "${!PKG_GROUP[@]}"; do
    [[ -n "$PKG_FILTER" && "$PKG_FILTER" != "$pkg" ]] && continue
    group="${PKG_GROUP[$pkg]}"
    src=""
    # Preferred: a flat snapshot at $DRIVE_BENCH/inputs/<pkg>/
    if [[ -d "$DRIVE_BENCH/inputs/$pkg" ]]; then
        src="$DRIVE_BENCH/inputs/$pkg"
    else
        # Fallback: most-recent Bayesian-opt run that contains an inputs/ dir.
        latest=$(ls -1d "$DRIVE_DIR/$pkg"/*/inputs 2>/dev/null | sort -r | head -n1 || true)
        [[ -n "$latest" ]] && src="$latest"
    fi
    if [[ -z "$src" ]]; then
        echo "  ! $pkg : no Drive snapshot found (skip; expects repo to have artefacts)"
        continue
    fi
    dst="$REPO_DIR/$group/$pkg/inputs"
    mkdir -p "$dst"
    cp -nu "$src"/* "$dst/" 2>/dev/null || true
    echo "  ✓ $pkg : $src -> $dst"
done

# ─── Pull existing results from Drive (resume) ───────────────────────────
if [[ -d "$DRIVE_RESULTS/raw" ]]; then
    echo "  ✓ resuming from $DRIVE_RESULTS/raw -> general/results/raw/"
    mkdir -p "$REPO_DIR/general/results/raw"
    cp -nu "$DRIVE_RESULTS/raw"/*.json "$REPO_DIR/general/results/raw/" 2>/dev/null || true
fi

# ─── Run the sweep ───────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════════════"
echo "  Launching general.scripts.orchestrate"
echo "════════════════════════════════════════════════════════════════════════"
ORCH_ARGS=()
[[ -n "$PKG_FILTER" ]] && ORCH_ARGS+=( --pkg "$PKG_FILTER" )
python3 -m general.scripts.orchestrate "${ORCH_ARGS[@]}" 2>&1 | tee -a "$DRIVE_BENCH/orchestrate.log"

# ─── Aggregate + plot + report ───────────────────────────────────────────
echo "  → aggregate"
python3 -m general.scripts.aggregate    2>&1 | tee -a "$DRIVE_BENCH/aggregate.log"
echo "  → plot"
python3 -m general.scripts.plot_matrix  2>&1 | tee -a "$DRIVE_BENCH/plot.log"
echo "  → report"
python3 -m general.scripts.report       2>&1 | tee -a "$DRIVE_BENCH/report.log"

# ─── Push results to Drive ───────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════════════"
echo "  Syncing results to $DRIVE_RESULTS"
echo "════════════════════════════════════════════════════════════════════════"
mkdir -p "$DRIVE_RESULTS"
cp -ru "$REPO_DIR/general/results/." "$DRIVE_RESULTS/" 2>/dev/null || true
echo "  ✓ done"
