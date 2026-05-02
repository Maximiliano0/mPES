#!/usr/bin/env bash
# ------------------------------------------------------------------
#  Muestra el estado de las optimizaciones Bayesianas en curso.
#
#  Busca procesos de optimización activos (optimize_rl, optimize_dqn,
#  optimize_a2c, optimize_tr).  Para cada paquete detecta:
#    - PID del proceso de optimización.
#    - Último trial completado, mejor valor y tiempo transcurrido.
#    - Últimas líneas del log de errores (si las hay).
#
#  Uso:
#    ./utils/optimization_status.sh              # Todos los paquetes
#    ./utils/optimization_status.sh pes_dqn      # Solo un paquete
#    ./utils/optimization_status.sh ac           # Alias aceptados
# ------------------------------------------------------------------
set -uo pipefail

# ── Rutas base ───────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Colores ──────────────────────────────────────────────────────
BOLD='\033[1m'
RED='\033[91m'
GREEN='\033[92m'
YELLOW='\033[93m'
CYAN='\033[96m'
GRAY='\033[90m'
RESET='\033[0m'

# ── Resolver paquete (opcional) ──────────────────────────────────
resolve_package() {
    case "${1:-}" in
        pes_ql|bayesian|bay|1)              echo "pes_ql"  ;;
        pes_dql|dql|ql|2)                   echo "pes_dql" ;;
        pes_dqn|dqn|3)                      echo "pes_dqn" ;;
        pes_a2c|ac|a2c|actor-critic|4)       echo "pes_a2c"  ;;
        pes_trf|transformer|tr|5)           echo "pes_trf" ;;
        all|"")                             echo "all"     ;;
        *) echo ""; return 1 ;;
    esac
}

FILTER_PKG="$(resolve_package "${1:-all}")" || {
    echo "Error: Paquete desconocido: '${1:-}'"
    echo "  Opciones: pes_ql, pes_dql, pes_dqn, pes_a2c, pes_trf, all"
    exit 1
}

ALL_PACKAGES=(pes_ql pes_dql pes_dqn pes_a2c pes_trf)
if [[ "$FILTER_PKG" != "all" ]]; then
    ALL_PACKAGES=("$FILTER_PKG")
fi

# ── Mapeo paquete → módulo de optimización ───────────────────────
get_opt_module() {
    case "$1" in
        pes_ql)  echo "optimize_rl"  ;;
        pes_dql) echo "optimize_rl"  ;;
        pes_dqn) echo "optimize_dqn" ;;
        pes_a2c)  echo "optimize_a2c"  ;;
        pes_trf) echo "optimize_tr"  ;;
    esac
}
# ── Mapeo paquete → prefijo study_name de Optuna ────────────────
get_study_prefix() {
    case "$1" in
        pes_ql)  echo "qlearning_opt" ;;
        pes_dql) echo "qlearning_opt" ;;
        pes_dqn) echo "dqn_opt"       ;;
        pes_a2c)  echo "a2c_opt"       ;;
        pes_trf) echo "transformer_opt" ;;
    esac
}

# ── Consultar Optuna DB para progreso real ───────────────────────
# Uso: query_optuna_db <pkg>
# Imprime: completed|running|best_value|best_trial_num|study_name
query_optuna_db() {
    local pkg="$1" db_dir db_file prefix
    prefix=$(get_study_prefix "$pkg")
    # Find most recent BAYESIAN_OPT directory
    db_dir=$(find "$PROJECT_DIR/$pkg/inputs" -maxdepth 1 -type d -name '*_BAYESIAN_OPT' 2>/dev/null | sort -r | head -n1)
    [[ -z "$db_dir" ]] && return 1
    db_file=$(find "$db_dir" -maxdepth 1 -name 'optuna_study_*.db' 2>/dev/null | sort -r | head -n1)
    [[ -z "$db_file" || ! -f "$db_file" ]] && return 1
    # Extract date from db filename
    local db_date
    db_date=$(basename "$db_file" | grep -oP '\d{4}-\d{2}-\d{2}')
    local study_name="${prefix}_${db_date}"
    # Query with sqlite3 (fast, no Python import overhead)
    if command -v sqlite3 &>/dev/null; then
        local completed running best_val best_num n_target_from_log
        completed=$(sqlite3 "$db_file" "SELECT COUNT(*) FROM trial_values tv JOIN trials t ON tv.trial_id=t.trial_id WHERE t.state='COMPLETE';" 2>/dev/null)
        running=$(sqlite3 "$db_file" "SELECT COUNT(*) FROM trials WHERE state='RUNNING';" 2>/dev/null)
        best_val=$(sqlite3 "$db_file" "SELECT MAX(tv.value) FROM trial_values tv JOIN trials t ON tv.trial_id=t.trial_id WHERE t.state='COMPLETE';" 2>/dev/null)
        best_num=$(sqlite3 "$db_file" "SELECT t.trial_id FROM trial_values tv JOIN trials t ON tv.trial_id=t.trial_id WHERE t.state='COMPLETE' ORDER BY tv.value DESC LIMIT 1;" 2>/dev/null)
        echo "${completed:-0}|${running:-0}|${best_val:-?}|${best_num:-?}|${study_name}"
        return 0
    fi
    return 1
}
# ── Barra de progreso compacta (ancho fijo 20 chars) ─────────────
# Uso: draw_bar <pct>   (0-100, entero)
draw_bar() {
    local pct="${1:-0}" bar_w=20 filled empty bar_str=""
    filled=$(( bar_w * pct / 100 ))
    empty=$(( bar_w - filled ))
    local i
    for (( i=0; i<filled; i++ )); do bar_str+="━"; done
    for (( i=0; i<empty;  i++ )); do bar_str+="─"; done
    echo -e "${GREEN}${bar_str:0:$filled}${GRAY}${bar_str:$filled}${RESET}"
}

# ── Cabecera ─────────────────────────────────────────────────────
echo ""
echo -e "  ${CYAN}╔════════════════════════════════════════════════╗${RESET}"
echo -e "  ${CYAN}║     ESTADO DE OPTIMIZACIONES BAYESIANAS       ║${RESET}"
echo -e "  ${CYAN}╚════════════════════════════════════════════════╝${RESET}"
echo -e "  ${GRAY}$(date '+%Y-%m-%d %H:%M:%S')  $PROJECT_DIR${RESET}"
echo ""

FOUND_ANY=false

for pkg in "${ALL_PACKAGES[@]}"; do
    opt_module="$(get_opt_module "$pkg")"
    log_dir="$PROJECT_DIR/$pkg/inputs"

    # ── Detectar proceso de optimización ─────────────────────────
    opt_pid=""

    # Buscar PID de optimización (ps aux con grep)
    opt_line=$(ps aux 2>/dev/null | grep "$opt_module" | grep -v grep | head -n1)
    if [[ -n "$opt_line" ]]; then
        opt_pid=$(echo "$opt_line" | awk '{print $2}')
    fi

    # ── Buscar log files ─────────────────────────────────────────
    log_file=""
    err_file=""
    if [[ -d "$log_dir" ]]; then
        # Más reciente primero, excluir _err
        log_file=$(find "$log_dir" -maxdepth 1 -name 'bayesian_opt*.log' ! -name '*_err*' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -n1 | cut -d' ' -f2-)
        err_file=$(find "$log_dir" -maxdepth 1 -name 'bayesian_opt*_err.log' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -n1 | cut -d' ' -f2-)
    fi

    has_process=false
    has_log=false
    [[ -n "$opt_pid" ]] && has_process=true
    [[ -n "$log_file" && -f "$log_file" ]] && has_log=true

    if ! $has_process && ! $has_log; then
        continue
    fi
    FOUND_ANY=true

    # ── Zombie detection helper ──────────────────────────────────
    opt_status="ACTIVO"
    opt_is_zombie=false
    if $has_process; then
        mem_kb=$(ps -p "$opt_pid" -o rss= 2>/dev/null | tr -d ' ')
        cpu_total=$(ps -p "$opt_pid" -o time= 2>/dev/null | tr -d ' ')
        # Parse CPU time (HH:MM:SS or MM:SS) to seconds
        cpu_secs=0
        if [[ -n "$cpu_total" ]]; then
            IFS=: read -ra parts <<< "$cpu_total"
            if [[ ${#parts[@]} -eq 3 ]]; then
                cpu_secs=$(( 10#${parts[0]}*3600 + 10#${parts[1]}*60 + 10#${parts[2]} ))
            elif [[ ${#parts[@]} -eq 2 ]]; then
                cpu_secs=$(( 10#${parts[0]}*60 + 10#${parts[1]} ))
            fi
        fi
        # Zombie heuristic: < 20MB RAM and < 1s CPU after 60s alive
        start_epoch=$(stat -c %Y /proc/"$opt_pid" 2>/dev/null || echo 0)
        now_epoch=$(date +%s)
        age_s=$(( now_epoch - start_epoch ))
        mem_mb_val=0
        [[ -n "$mem_kb" ]] && mem_mb_val=$(awk "BEGIN {printf \"%d\", $mem_kb/1024}")
        if [[ "$mem_mb_val" -lt 20 && "$cpu_secs" -lt 1 && "$age_s" -gt 60 ]]; then
            opt_status="ZOMBIE"
            opt_is_zombie=true
        fi
    fi

    # ── Encabezado del paquete ───────────────────────────────────
    if $has_process && $opt_is_zombie; then
        echo -e "  ${RED}⚠ $pkg${RESET}  ${RED}[$opt_status]${RESET}"
    elif $has_process; then
        echo -e "  ${GREEN}● $pkg${RESET}  ${GREEN}[EN CURSO]${RESET}"
    else
        echo -e "  ${GRAY}○ $pkg${RESET}  ${GRAY}[FINALIZADO]${RESET}"
    fi

    # ── PID info ─────────────────────────────────────────────────
    if $has_process; then
        # Obtener CPU time y memoria
        cpu_time=$(ps -p "$opt_pid" -o time= 2>/dev/null | tr -d ' ')
        mem_kb=$(ps -p "$opt_pid" -o rss= 2>/dev/null | tr -d ' ')
        mem_mb=""
        if [[ -n "$mem_kb" ]]; then
            mem_mb=$(awk "BEGIN {printf \"%.1f\", $mem_kb/1024}")
        fi
        if $opt_is_zombie; then
            echo -e "    ${GRAY}PID optimización:${RESET} $opt_pid  ${GRAY}(CPU: ${cpu_time:-?}, Mem: ${mem_mb:-?} MB, Estado: ${RED}$opt_status${RESET}${GRAY})${RESET}"
            echo -e "    ${RED}*** PROCESO ZOMBIE DETECTADO — debe relanzarse ***${RESET}"
        else
            echo -e "    ${GRAY}PID optimización:${RESET} $opt_pid  ${GRAY}(CPU: ${cpu_time:-?}, Mem: ${mem_mb:-?} MB, Estado: ${GREEN}$opt_status${RESET}${GRAY})${RESET}"
        fi
    fi

    # ── Parsear progreso del log ─────────────────────────────────
    if $has_log; then
        # Target trials
        n_target=$(grep -oP 'Target number of trials:\s*\K\d+' "$log_file" | tail -n1)
        n_target="${n_target:-?}"

        # Último trial completado.  Acepta perf=<numero>, perf=<estado> (PRUNED/FAILED)
        # y best=<numero> o best=n/a.  Mantiene compatibilidad con logs antiguos
        # que usaban la etiqueta 'value=' en vez de 'perf='.
        last_trial=$(grep -oP 'Trial\s+\K\d+/\d+\s+\|\s+(perf|value)=\s*[-\d.A-Za-z/]+\s+\|\s+best=\s*[-\d.A-Za-z/]+\s+\|\s+elapsed=\d+s' "$log_file" | tail -n1)

        if [[ -n "$last_trial" ]]; then
            done_total=$(echo "$last_trial" | grep -oP '^\d+/\d+')
            done_n=$(echo "$done_total" | cut -d'/' -f1)
            total_n=$(echo "$done_total" | cut -d'/' -f2)
            value=$(echo "$last_trial" | grep -oP '(perf|value)=\s*\K[-\d.A-Za-z/]+')
            best=$(echo "$last_trial" | grep -oP 'best=\s*\K[-\d.A-Za-z/]+')
            elapsed=$(echo "$last_trial" | grep -oP 'elapsed=\K\d+')

            pct=$(awk "BEGIN {printf \"%d\", ($done_n/$total_n)*100}")

            hrs=$((elapsed / 3600))
            mins=$(( (elapsed % 3600) / 60 ))
            if [[ $hrs -gt 0 ]]; then
                el_str="${hrs}h ${mins}m"
            else
                el_str="${mins}m"
            fi

            # Barra de progreso compacta (20 chars)
            bar_str=$(draw_bar "$pct")

            echo -e "    ${bar_str}  ${BOLD}${done_n}/${total_n}${RESET} ${GRAY}(${pct}%)${RESET}"
            echo -e "    ${GRAY}Mejor:${RESET}  ${CYAN}${best}${RESET}   ${GRAY}Último:${RESET} ${value}   ${GRAY}Tiempo:${RESET} ${el_str}"

            # ETA
            if [[ "$done_n" -gt 0 ]]; then
                avg=$(awk "BEGIN {printf \"%.0f\", $elapsed/$done_n}")
                remaining=$(( (total_n - done_n) * avg ))
                rem_hrs=$((remaining / 3600))
                rem_mins=$(( (remaining % 3600) / 60 ))
                if [[ $rem_hrs -gt 0 ]]; then
                    eta_str="~${rem_hrs}h ${rem_mins}m"
                else
                    eta_str="~${rem_mins}m"
                fi
                echo -e "    ${GRAY}ETA:${RESET}    ${YELLOW}$eta_str${RESET} ${GRAY}(~${avg}s/trial)${RESET}"
            fi
        else
            # Log vacío o sin trials — consultar DB directamente
            db_info=$(query_optuna_db "$pkg" 2>/dev/null)
            if [[ -n "$db_info" ]]; then
                IFS='|' read -r db_completed db_running db_best db_best_num db_study <<< "$db_info"
                if [[ "$db_completed" -gt 0 ]] 2>/dev/null; then
                    db_total="${n_target:-?}"
                    if [[ "$db_total" == "?" ]]; then
                        # Fallback: try all logs for target
                        for lf in "$log_dir"/bayesian_opt*.log; do
                            [[ -f "$lf" ]] || continue
                            t=$(grep -oP 'Target number of trials:\s*\K\d+' "$lf" 2>/dev/null | tail -n1)
                            [[ -n "$t" ]] && db_total="$t" && break
                        done
                    fi
                    if [[ "$db_total" != "?" ]]; then
                        db_pct=$(awk "BEGIN {printf \"%d\", ($db_completed/$db_total)*100}")
                        bar_str=$(draw_bar "$db_pct")
                        echo -e "    ${bar_str}  ${BOLD}${db_completed}/${db_total}${RESET} ${GRAY}(${db_pct}%) — desde DB${RESET}"
                    else
                        echo -e "    ${BOLD}${db_completed}${RESET} ${GRAY}trials completados — desde DB${RESET}"
                    fi
                    echo -e "    ${GRAY}Mejor:${RESET}  ${CYAN}${db_best}${RESET} ${GRAY}(trial #${db_best_num})${RESET}   ${GRAY}En curso:${RESET} ${db_running}"
                else
                    resume_line=$(grep -E 'Starting fresh|Resuming' "$log_file" | tail -n1)
                    if [[ -n "$resume_line" ]]; then
                        echo -e "    ${GRAY}$(echo "$resume_line" | sed 's/^[[:space:]]*//')${RESET}"
                    else
                        echo -e "    ${YELLOW}Inicializando... (trial en curso)${RESET}"
                    fi
                    echo -e "    ${GRAY}Target: $n_target trials${RESET}"
                fi
            else
                echo -e "    ${YELLOW}Inicializando... (trial en curso)${RESET}"
                echo -e "    ${GRAY}Target: $n_target trials${RESET}"
            fi
        fi

        # ── Log path ────────────────────────────────────────────
        rel_log="${log_file#$PROJECT_DIR/}"
        echo -e "    ${GRAY}Log: $rel_log${RESET}"
    fi

    # ── Errores ──────────────────────────────────────────────────
    if [[ -n "$err_file" && -f "$err_file" ]]; then
        err_count=$(grep -c '\S' "$err_file" 2>/dev/null)
        if [[ "$err_count" -gt 0 ]]; then
            echo -e "    ${YELLOW}⚠ $err_count línea(s) en stderr${RESET}"
            grep '\S' "$err_file" | tail -n3 | while IFS= read -r line; do
                trimmed="${line:0:70}"
                [[ ${#line} -gt 70 ]] && trimmed="${trimmed}…"
                echo -e "      ${YELLOW}$trimmed${RESET}"
            done
        fi
    fi

    echo ""
done

if ! $FOUND_ANY; then
    echo -e "  ${YELLOW}No se detectaron optimizaciones activas ni logs recientes.${RESET}"
    echo ""
fi

# ── Comandos útiles ──────────────────────────────────────────────
echo -e "  ${GRAY}── Comandos ──────────────────────────────────────${RESET}"
echo -e "  ${GRAY}  tail -f <pkg>/inputs/bayesian_opt.log${RESET}"
echo -e "  ${GRAY}  tail -20 <pkg>/inputs/bayesian_opt_err.log${RESET}"
echo ""
