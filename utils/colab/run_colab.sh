#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  utils/colab/run_colab.sh — launch a Bayesian optimisation on Colab Pro+
# ═══════════════════════════════════════════════════════════════════════════════
#  What it does
#  ────────────
#  1. Resolves the package alias to (PKG, OPT_MODULE) using the same map as
#     utils/linux/run_bayesian_opt.sh.
#  2. Stores the Optuna SQLite DB on Google Drive so the study survives Colab
#     instance shutdowns (24h limit, idle reclaim, network drops).
#  3. Launches the optimisation with `nohup` so it stays alive even if the
#     foreground Colab cell loses its kernel link.
#
#  Usage (inside a Colab cell, after setup_colab.sh):
#      !bash utils/colab/run_colab.sh <PKG_ALIAS> [N_TRIALS] [RESUME_DATE]
#
#  Examples:
#      !bash utils/colab/run_colab.sh dql 100
#      !bash utils/colab/run_colab.sh dqn 60
#      !bash utils/colab/run_colab.sh rdqn 60
#      !bash utils/colab/run_colab.sh ac  100
#      !bash utils/colab/run_colab.sh transformer 60
#      !bash utils/colab/run_colab.sh ql  150 2026-04-20
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ─── Inputs ──────────────────────────────────────────────────────────────
PKG_ALIAS="${1:-}"
N_TRIALS="${2:-30}"
RESUME_DATE="${3:-}"

if [[ -z "$PKG_ALIAS" ]]; then
    echo "Usage: $0 <PKG_ALIAS> [N_TRIALS] [RESUME_DATE]"
    echo "Aliases: ql|bayesian|1, dql|2, dqn|3, ac|a2c|4, rdqn|recurrent|6, transformer|tr|5"
    exit 1
fi

REPO_DIR="${REPO_DIR:-/content/Win_mPES}"
DRIVE_DIR="${DRIVE_DIR:-/content/drive/MyDrive/mPES}"

# shellcheck disable=SC1091
source /content/mpes_env.sh
cd "$REPO_DIR"

# ─── Resolve alias → (PKG, OPT_MODULE) ───────────────────────────────────
case "$PKG_ALIAS" in
    ql|bayesian|1)         PKG="pes_ql";   OPT_MODULE="pes_ql.ext.optimize_rl" ;;
    dql|2)                 PKG="pes_dql";  OPT_MODULE="pes_dql.ext.optimize_rl" ;;
    dqn|3)                 PKG="pes_dqn";  OPT_MODULE="pes_dqn.ext.optimize_dqn" ;;
    ac|a2c|4)              PKG="pes_a2c";  OPT_MODULE="pes_a2c.ext.optimize_a2c" ;;
    transformer|tr|5)      PKG="pes_trf";  OPT_MODULE="pes_trf.ext.optimize_tr" ;;
    rdqn|recurrent|6)      PKG="pes_rdqn"; OPT_MODULE="pes_rdqn.ext.optimize_rdqn" ;;
    *)                     echo "Unknown alias: $PKG_ALIAS"; exit 1 ;;
esac

# ─── Resolve run date and Drive paths ────────────────────────────────────
RUN_DATE="${RESUME_DATE:-$(date +%Y-%m-%d)}"
PKG_DRIVE_DIR="${DRIVE_DIR}/${PKG}/${RUN_DATE}_BAYESIAN_OPT"
mkdir -p "$PKG_DRIVE_DIR"

DB_FILE="${PKG_DRIVE_DIR}/optuna_study_${RUN_DATE}.db"
STORAGE_URL="sqlite:///${DB_FILE}"
LOG_FILE="${PKG_DRIVE_DIR}/bayesian_opt.log"
ERR_FILE="${PKG_DRIVE_DIR}/bayesian_opt_err.log"
PID_FILE="${PKG_DRIVE_DIR}/optimize.pid"
META_FILE="${PKG_DRIVE_DIR}/run_meta.json"

# ─── Collect launch metadata ──────────────────────────────────────────────
LAUNCH_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
GIT_SHA="$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
GIT_BRANCH_NAME="$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
PY_VERSION="$(python3 -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo 'unknown')"
HOSTNAME_STR="$(hostname 2>/dev/null || echo 'unknown')"

echo "════════════════════════════════════════════════════════════════════════"
echo "  Launching Bayesian optimisation on Colab Pro+"
echo "════════════════════════════════════════════════════════════════════════"
echo "  Package         : $PKG"
echo "  Module          : $OPT_MODULE"
echo "  Trials          : $N_TRIALS"
echo "  Run date        : $RUN_DATE"
echo "  Output dir      : $PKG_DRIVE_DIR"
echo "  Storage         : $STORAGE_URL"
echo "  Git             : ${GIT_BRANCH_NAME}@${GIT_SHA}"
echo "  Python          : $PY_VERSION"
echo "  GPU mode        : ${MPES_USE_GPU:-0}"
[[ -n "$RESUME_DATE" ]] && echo "  Resuming        : $RESUME_DATE"
echo "════════════════════════════════════════════════════════════════════════"

# ─── Build optimise command ───────────────────────────────────────────────
OPT_ARGS=(
    "$N_TRIALS"
    --out-dir "$PKG_DRIVE_DIR"
    --storage "$STORAGE_URL"
)
[[ -n "$RESUME_DATE" ]] && OPT_ARGS+=( --resume "$RESUME_DATE" )

# Optional pass-through for package-specific flags (e.g. pes_a2c's --mode).
# Example: EXTRA_ARGS="--mode improvements_only" bash run_colab.sh ac 100
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    OPT_ARGS+=( ${EXTRA_ARGS} )
fi

# ─── Write a structured banner into the log itself ────────────────────────
# This means `tail` of the log always carries the run context, even when
# inspected days later from a different machine.
{
    echo "════════════════════════════════════════════════════════════════════════"
    echo "  mPES Bayesian-Optimisation run — ${LAUNCH_TS}"
    echo "════════════════════════════════════════════════════════════════════════"
    echo "  package      : $PKG"
    echo "  module       : $OPT_MODULE"
    echo "  n_trials     : $N_TRIALS"
    echo "  run_date     : $RUN_DATE"
    echo "  out_dir      : $PKG_DRIVE_DIR"
    echo "  storage      : $STORAGE_URL"
    echo "  git          : ${GIT_BRANCH_NAME}@${GIT_SHA}"
    echo "  python       : $PY_VERSION"
    echo "  host         : $HOSTNAME_STR"
    echo "  gpu_mode     : ${MPES_USE_GPU:-0}"
    echo "  resume_date  : ${RESUME_DATE:-none}"
    echo "  extra_args   : ${EXTRA_ARGS:-none}"
    echo "  cmd          : python3 -u -m $OPT_MODULE ${OPT_ARGS[*]}"
    echo "════════════════════════════════════════════════════════════════════════"
} >> "$LOG_FILE"

# ─── Write machine-readable metadata sidecar ──────────────────────────────
cat > "$META_FILE" <<EOF
{
  "package":     "$PKG",
  "module":      "$OPT_MODULE",
  "n_trials":    $N_TRIALS,
  "run_date":    "$RUN_DATE",
  "resume_date": "${RESUME_DATE}",
  "launch_ts":   "$LAUNCH_TS",
  "git_branch":  "$GIT_BRANCH_NAME",
  "git_sha":     "$GIT_SHA",
  "python":      "$PY_VERSION",
  "host":        "$HOSTNAME_STR",
  "gpu_mode":    "${MPES_USE_GPU:-0}",
  "out_dir":     "$PKG_DRIVE_DIR",
  "storage":     "$STORAGE_URL",
  "log_file":    "$LOG_FILE",
  "err_file":    "$ERR_FILE",
  "pid_file":    "$PID_FILE",
  "extra_args":  "${EXTRA_ARGS:-}"
}
EOF

# ─── Reattach to a still-running optimiser, if any ────────────────────────
# Prevents a second cell-run from launching a duplicate process when the
# user just refreshed the browser or re-ran Cell 4 to re-establish the
# foreground stream after a network blip.
EXISTING_PID=""
if [[ -f "$PID_FILE" ]]; then
    _pid_candidate="$(tr -dc '0-9' < "$PID_FILE")"
    if [[ -n "$_pid_candidate" ]] && kill -0 "$_pid_candidate" 2>/dev/null; then
        EXISTING_PID="$_pid_candidate"
    fi
fi

if [[ -n "$EXISTING_PID" ]]; then
    echo "════════════════════════════════════════════════════════════════════════"
    echo "  ► Found a running optimiser (PID $EXISTING_PID) for this run."
    echo "    Reattaching to its log instead of launching a duplicate."
    echo "════════════════════════════════════════════════════════════════════════"
    OPT_PID="$EXISTING_PID"
    # Skip launch + autorestart loop; jump straight to the foreground waiter.
    SKIP_LAUNCH=1
else
    SKIP_LAUNCH=0
fi

# ─── Launch optimisation under nohup (with auto-restart loop) ─────────────
# MPES_USE_GPU is honoured by pes_*/__init__.py: 1 → use GPU, 0 → pin CPU.
# PYTHONUNBUFFERED guarantees line-buffered stdout/stderr even when the OS
# detects the file descriptor as a regular file (Drive-backed log).
#
# AUTO-RESTART: if the child process dies (Colab idle reclaim, transient
# OOM, etc.), the supervisor relaunches it with ``--resume RUN_DATE`` so
# Optuna picks up from the last completed trial. Up to MAX_RESTARTS times.
MAX_RESTARTS="${MAX_RESTARTS:-10}"

if [[ "$SKIP_LAUNCH" == "0" ]]; then
    # Make sure --resume is present so every restart is idempotent.
    HAS_RESUME=0
    for _a in "${OPT_ARGS[@]}"; do [[ "$_a" == "--resume" ]] && HAS_RESUME=1; done
    if [[ "$HAS_RESUME" == "0" ]]; then
        OPT_ARGS+=( --resume "$RUN_DATE" )
    fi

    SUPERVISOR_LOG="${PKG_DRIVE_DIR}/supervisor.log"
    {
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] supervisor starting"
        echo "  cmd : python3 -u -m $OPT_MODULE ${OPT_ARGS[*]}"
        echo "  max_restarts: $MAX_RESTARTS"
    } >> "$SUPERVISOR_LOG"

    # Run the optimiser + restart loop in the background as one supervisor.
    (
        attempt=1
        while (( attempt <= MAX_RESTARTS + 1 )); do
            echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] launching attempt $attempt" \
                >> "$SUPERVISOR_LOG"
            env MPES_USE_GPU="${MPES_USE_GPU:-0}" PYTHONUNBUFFERED=1 \
                python3 -u -m "$OPT_MODULE" "${OPT_ARGS[@]}" \
                >> "$LOG_FILE" 2>> "$ERR_FILE"
            rc=$?
            if (( rc == 0 )); then
                echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] optimiser exited cleanly (attempt $attempt)" \
                    >> "$SUPERVISOR_LOG"
                exit 0
            fi
            echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] optimiser died with code $rc on attempt $attempt" \
                >> "$SUPERVISOR_LOG"
            if (( attempt > MAX_RESTARTS )); then
                echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] giving up after $MAX_RESTARTS restarts" \
                    >> "$SUPERVISOR_LOG"
                exit "$rc"
            fi
            attempt=$(( attempt + 1 ))
            sleep 5
        done
    ) >/dev/null 2>&1 &
    disown
    OPT_PID=$!
    echo "$OPT_PID" > "$PID_FILE"
fi

# Patch the metadata file with the actual PID now that we have it.
python3 - "$META_FILE" "$OPT_PID" <<'PY'
import json, sys
path, pid = sys.argv[1], int(sys.argv[2])
with open(path, 'r', encoding='utf-8') as f:
    meta = json.load(f)
meta['pid'] = pid
with open(path, 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2)
PY

echo "→ Optimisation PID    : $OPT_PID"
echo "→ stdout              : $LOG_FILE"
echo "→ stderr              : $ERR_FILE"
echo "→ metadata            : $META_FILE"

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  Optimisation supervisor launched."
echo "  ▸ Cell BLOCKS in foreground tailing the Drive log so Colab Pro+"
echo "    Background Execution keeps the VM alive after browser close."
echo "  ▸ If the python child dies, the supervisor relaunches it with"
echo "    --resume (up to MAX_RESTARTS=${MAX_RESTARTS:-10} times)."
echo "  ▸ If you re-run this cell while a previous PID is still alive,"
echo "    it just reattaches — no duplicate processes."
echo ""
echo "  Required: Runtime → Manage sessions → Background execution = ON."
echo "  Safe to close the browser once that toggle is on."
echo "════════════════════════════════════════════════════════════════════════"
echo "  Live monitoring (any other Colab cell, any machine):"
echo "      from utils.colab.monitor import monitor; monitor()"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

# ─── Foreground waiter ────────────────────────────────────────────────────
# Why block the cell:
#   1. Colab Pro+ Background Execution only persists the VM while at least
#      one cell is actively executing. A detached nohup that returns
#      immediately makes the notebook look idle → VM reclaimed within
#      minutes of the browser disconnecting → process killed.
#   2. Streaming the Drive-backed log to the cell stdout gives a live view
#      that any monitor() / tail-from-another-machine call also sees.
#
# OPT_PID points at either the still-alive previous python process
# (reattach path) or the supervisor script (fresh launch path). In both
# cases, the cell exits when that PID exits, which only happens when the
# optimisation has either finished or burned through all restart attempts.
#
# Set FOREGROUND=0 to skip the wait (legacy detached behaviour) — use only
# if you intend to keep at least one OTHER cell running for the whole run.
FOREGROUND="${FOREGROUND:-1}"

if [[ "$FOREGROUND" == "1" ]]; then
    # Compact live status panel.  Repaints every PROGRESS_REFRESH seconds
    # using the same data sources as utils.colab.monitor (Optuna SQLite +
    # log tail), instead of streaming the raw log line by line.  The cell
    # exits as soon as OPT_PID dies, which only happens when the
    # optimisation finished or all restart attempts were exhausted.
    PROGRESS_REFRESH="${PROGRESS_REFRESH:-30}"

    REPO_DIR="$REPO_DIR" \
    PKG="$PKG" \
    PKG_DRIVE_DIR="$PKG_DRIVE_DIR" \
    OPT_PID="$OPT_PID" \
    LOG_FILE="$LOG_FILE" \
    ERR_FILE="$ERR_FILE" \
    PROGRESS_REFRESH="$PROGRESS_REFRESH" \
    python3 - <<'PY'
import os, sys, time
sys.path.insert(0, os.environ['REPO_DIR'])
from utils.colab.monitor import monitor  # noqa: E402

run_dir  = os.environ['PKG_DRIVE_DIR']
pkg      = os.environ['PKG']
opt_pid  = int(os.environ['OPT_PID'])
err_path = os.environ['ERR_FILE']
refresh  = max(5, int(os.environ['PROGRESS_REFRESH']))
run_date = os.path.basename(run_dir).split('_')[0]

def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True

# Repaint loop. Ctrl-C / kernel interrupt breaks cleanly.
try:
    while _alive(opt_pid):
        # ANSI clear + home so the panel always overwrites itself.
        sys.stdout.write('\x1b[2J\x1b[H')
        sys.stdout.flush()
        try:
            monitor(pkg=pkg, date=run_date, n_log_lines=12, n_err_lines=4)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  (monitor() error: {exc})")
        sys.stdout.write(
            f"\n  Refreshing every {refresh}s — "
            f"close the browser anytime; the supervisor keeps running.\n"
        )
        sys.stdout.flush()
        # Sleep in 1s slices so Ctrl-C is responsive.
        for _ in range(refresh):
            if not _alive(opt_pid):
                break
            time.sleep(1)
except KeyboardInterrupt:
    print("\n  Foreground watcher interrupted — supervisor PID still running.")
PY
    _mon_rc=$?
    if (( _mon_rc != 0 )); then
        echo "  (foreground monitor exited rc=$_mon_rc; supervisor PID $OPT_PID still running in background)"
        echo "  Tail the log directly:  tail -f $LOG_FILE"
    fi

    # Best-effort wait in case the supervisor was a child of this shell.
    wait "$OPT_PID" 2>/dev/null
    EXIT_CODE=$?
    # The supervisor is `disown`-ed, so `wait` typically returns 127 even on
    # success.  Re-derive the true status from process liveness: if the PID
    # is dead by now, treat the run as completed cleanly (the supervisor
    # writes its own success/failure record to supervisor.log).
    if (( EXIT_CODE != 0 )) && ! kill -0 "$OPT_PID" 2>/dev/null; then
        EXIT_CODE=0
    fi

    echo ""
    echo "════════════════════════════════════════════════════════════════════════"
    if [[ $EXIT_CODE -eq 0 ]]; then
        echo "  ✓ Optimisation finished cleanly (supervisor PID $OPT_PID)."
    else
        echo "  ✗ Supervisor exited with code $EXIT_CODE."
        echo "    See $ERR_FILE and ${PKG_DRIVE_DIR}/supervisor.log"
    fi
    echo "════════════════════════════════════════════════════════════════════════"
    exit $EXIT_CODE
fi

echo "  FOREGROUND=0 set — cell returns immediately."
echo "  ⚠ Keep at least one other cell running, or the VM will be reclaimed."
echo "════════════════════════════════════════════════════════════════════════"
