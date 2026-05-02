#!/usr/bin/env bash
# ------------------------------------------------------------------
#  Lanzar optimización Bayesiana
#
#  Script compartido para pes_ql, pes_dql, pes_dqn, pes_rdqn,
#  pes_a2c y pes_trf.  Todas las rutas se resuelven de forma
#  relativa a la ubicación de este script (utils/ → mPES/).
#
#  Funcionalidades:
#    - Lanza la optimización en segundo plano con nohup.
#    - Inhibe la suspensión del sistema (systemd-inhibit) mientras
#      el proceso esté vivo.
#    - Configura GNOME para ignorar el cierre de tapa.
#
#  Uso:
#    chmod +x run_bayesian_opt.sh
#    ./run_bayesian_opt.sh bayesian 100              # pes_ql, corrida nueva
#    ./run_bayesian_opt.sh dql 100                    # pes_dql, corrida nueva
#    ./run_bayesian_opt.sh dqn 30                    # pes_dqn, corrida nueva
#    ./run_bayesian_opt.sh ac 30                     # pes_a2c, corrida nueva
#    ./run_bayesian_opt.sh transformer 30            # pes_trf, corrida nueva
#    ./run_bayesian_opt.sh bayesian 100 2026-02-12   # reanudar desde fecha
# ------------------------------------------------------------------
set -euo pipefail

# ── Rutas base ───────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"       # mPES/ (raíz del workspace)
VENV="$PROJECT_DIR/linux_mpes_env/bin/activate"

# ── Resolver paquete desde primer argumento ──────────────────────
resolve_package() {
    case "${1:-}" in
        bayesian|Bayesian|BAYESIAN|bay|1) echo "pes_ql"        ;;
        dql|DQL|ql|2)                     echo "pes_dql"       ;;
        dqn|DQN|3)                        echo "pes_dqn"             ;;
        rdqn|RDQN|7)                      echo "pes_rdqn"            ;;
        ac|a2c|actor-critic|4)            echo "pes_a2c"    ;;
        transformer|tr|5)                 echo "pes_trf"     ;;
        *) return 1 ;;
    esac
}

# ── Verificar argumentos ────────────────────────────────────────
if [[ $# -lt 2 ]]; then
    echo "Error: Debe especificar el paquete y el número de trials"
    echo ""
    echo "Uso: $0 <paquete> <n_trials> [fecha_resume]"
    echo ""
    echo "  Paquetes: bayesian (pes_ql), dql (pes_dql), dqn (pes_dqn), rdqn (pes_rdqn), ac (pes_a2c), transformer (pes_trf)"
    echo ""
    echo "Ejemplos:"
    echo "  $0 bayesian 100"
    echo "  $0 dql 100"
    echo "  $0 dqn 30"
    echo "  $0 ac 30"
    echo "  $0 transformer 30"
    echo "  $0 bayesian 100 2026-02-12"
    exit 1
fi

PKG_NAME="$(resolve_package "$1")" || {
    echo "Error: Paquete desconocido: '$1'"
    echo "  Opciones válidas: bayesian, dql, dqn, rdqn, ac, transformer"
    exit 1
}
N_TRIALS="$2"

# ── Resolver módulo de optimización por paquete ──────────────────
resolve_module() {
    case "$1" in
        pes_ql)     echo "pes_ql.ext.optimize_rl"  ;;
        pes_dql)    echo "pes_dql.ext.optimize_rl" ;;
        pes_dqn)    echo "pes_dqn.ext.optimize_dqn"     ;;
        pes_rdqn)   echo "pes_rdqn.ext.optimize_rdqn"   ;;
        pes_a2c)     echo "pes_a2c.ext.optimize_a2c" ;;
        pes_trf)    echo "pes_trf.ext.optimize_tr" ;;
    esac
}
OPT_MODULE="$(resolve_module "$PKG_NAME")"

LOG_DIR="$PROJECT_DIR/$PKG_NAME/inputs"

cd "$PROJECT_DIR"
source "$VENV"
export PYTHONIOENCODING="utf-8"
export TF_ENABLE_ONEDNN_OPTS="0"

# Construir argumentos (--resume si se pasa una fecha)
ARGS="$N_TRIALS"
LOG_SUFFIX=""
if [[ ${3:-} != "" ]]; then
    ARGS="$N_TRIALS --resume $3"
    LOG_SUFFIX="_resume_$3"
fi

LOGFILE="$LOG_DIR/bayesian_opt${LOG_SUFFIX}.log"

# Evitar suspensión, apagado y bloqueo de pantalla
gsettings set org.gnome.settings-daemon.plugins.power lid-close-ac-action 'nothing'
gsettings set org.gnome.settings-daemon.plugins.power lid-close-battery-action 'nothing'
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'nothing'
gsettings set org.gnome.desktop.session idle-delay 0
gsettings set org.gnome.desktop.screensaver lock-enabled false

# Lanzar optimización en segundo plano (PYTHONUNBUFFERED para flush inmediato)
nohup env PYTHONUNBUFFERED=1 python3 -m "${OPT_MODULE}" $ARGS > "$LOGFILE" 2>&1 &
OPT_PID=$!
echo "Optimización lanzada  PID=$OPT_PID  trials=$N_TRIALS"
echo "Log: $LOGFILE"

# Inhibir suspensión y apagado mientras el proceso esté vivo
nohup systemd-inhibit \
    --what=idle:sleep:shutdown:handle-lid-switch \
    --who="mPES Bayesian Optimization ($PKG_NAME)" \
    --why="Running $N_TRIALS-trial Bayesian optimization" \
    --mode=block \
    tail --pid=$OPT_PID -f /dev/null > /dev/null 2>&1 &
echo "Inhibidor de suspensión/apagado activo (se desactiva al terminar)"

echo ""
echo "Comandos útiles:"
echo "  Progreso:    grep 'Trial' $LOGFILE | tail -10"
echo "  Tiempo real: tail -f $LOGFILE"
echo "  Vivo?:       pgrep -f '${OPT_MODULE}' -a"